#include "dsv4/cpu_moe.hpp"
#include "dsv4/dequant.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace dsv4 {
struct CpuMoe::Impl {
    CpuPool pool;
    bool use_native;
    NativeCpuFormat fg, fu, fd;
    std::vector<uint8_t> qg, qu;
    std::vector<std::vector<uint8_t>> qd;
    std::vector<float> activation, output;
    uint32_t tg = UINT32_MAX, tu = UINT32_MAX, td = UINT32_MAX;
    int dim = 0, ff = 0, count = 0;
    size_t rg = 0, ru = 0, rd = 0;
    float limit = 0;
    const float* x = nullptr;
    const CpuExpert* experts = nullptr;
    const uint8_t* up_act = nullptr;

    Impl(int threads, bool pin, bool native, bool pin_caller) : pool(threads, pin, pin_caller), use_native(native) {}
    static float dot(bool native, const NativeCpuFormat& f, uint32_t type, const uint8_t* w,
                     const uint8_t* q, const float* x, int n) noexcept {
        if (!native) return row_dot(type, w, x, n);
        float result = 0;
        // Formats and buffers are checked on the producer before workers are released.
        native_cpu_dot(f, w, q, result);
        return result;
    }
    static void gate_up(void* context, int64_t begin, int64_t end) noexcept {
        auto& s = *static_cast<Impl*>(context);
        for (int64_t global = begin; global < end; ++global) {
            const int k = int(global / s.ff), row = int(global % s.ff);
            const CpuExpert& e = s.experts[k];
            float g = dot(s.use_native, s.fg, s.tg, e.gate + size_t(row)*s.rg,
                          s.qg.data(), s.x, s.dim);
            float u = dot(s.use_native, s.fu, s.tu, e.up + size_t(row)*s.ru,
                          s.up_act, s.x, s.dim);
            if (s.limit > 0.f) { g = std::min(g, s.limit); u = std::clamp(u, -s.limit, s.limit); }
            // Match the reference clamped SwiGLU and placement of the routing weight.
            const float a = float(double(g) / (1.0 + std::exp(-double(g)))) * u;
            s.activation[size_t(global)] = a * e.weight;
        }
    }
    static void down(void* context, int64_t begin, int64_t end) noexcept {
        auto& s = *static_cast<Impl*>(context);
        for (int64_t global = begin; global < end; ++global) {
            const int k = int(global / s.dim), row = int(global % s.dim);
            s.output[size_t(global)] = dot(s.use_native, s.fd, s.td,
                s.experts[k].down + size_t(row)*s.rd, s.use_native ? s.qd[size_t(k)].data() : nullptr,
                s.activation.data() + size_t(k)*s.ff, s.ff);
        }
    }
};

CpuMoe::CpuMoe(int threads, bool pin, bool native, bool pin_caller) : p_(new Impl(threads, pin, native, pin_caller)) {}
CpuMoe::~CpuMoe() = default;
int CpuMoe::threads() const { return p_->pool.threads(); }
int CpuMoe::caller_cpu() const { return p_->pool.caller_cpu(); }
CpuPool::CallerScope CpuMoe::scoped_caller() const { return p_->pool.scoped_caller(); }
std::vector<int> CpuMoe::worker_cpus() const { return p_->pool.worker_cpus(); }
bool CpuMoe::native() const { return p_->use_native; }

bool CpuMoe::run(uint32_t tg, uint32_t tu, uint32_t td, int dim, int ff, float limit,
                 const float* x, const CpuExpert* experts, int count, float* sum, std::string& err) {
    auto& s = *p_;
    if (dim <= 0 || ff <= 0 || count < 0 || count > 16 || !x || !sum || (count && !experts)) {
        err = "invalid CPU MoE shape or buffers"; return false;
    }
    if (count == 0) { std::fill(sum, sum + dim, 0.f); return true; }
    if (!dequant_supported(tg) || !dequant_supported(tu) || !dequant_supported(td)) {
        err = "unsupported CPU MoE weight type"; return false;
    }
    const size_t rg = row_bytes(tg, dim), ru = row_bytes(tu, dim), rd = row_bytes(td, ff);
    if (!rg || !ru || !rd) { err = "CPU MoE rows must contain whole quantization blocks"; return false; }
    for (int k = 0; k < count; ++k)
        if (!experts[k].gate || !experts[k].up || !experts[k].down) {
            err = "CPU MoE expert is missing weights"; return false;
        }
    if (s.use_native && (s.tg != tg || s.tu != tu || s.td != td || s.dim != dim || s.ff != ff)) {
        if (!native_cpu_format(tg, dim, s.fg, err) || !native_cpu_format(tu, dim, s.fu, err) ||
            !native_cpu_format(td, ff, s.fd, err)) return false;
        if (s.fg.weight_row_bytes != rg || s.fu.weight_row_bytes != ru || s.fd.weight_row_bytes != rd) {
            err = "ggml and DeepSeek disagree on expert row byte sizes"; return false;
        }
    }
    s.tg = tg; s.tu = tu; s.td = td; s.dim = dim; s.ff = ff; s.count = count;
    s.rg = rg; s.ru = ru; s.rd = rd; s.limit = limit; s.x = x; s.experts = experts;
    s.activation.resize(size_t(count)*ff); s.output.resize(size_t(count)*dim);
    if (s.use_native) {
        if (!native_cpu_prepare(s.fg, x, s.qg, err)) return false;
        if (s.fg.activation_type == s.fu.activation_type) s.up_act = s.qg.data();
        else {
            if (!native_cpu_prepare(s.fu, x, s.qu, err)) return false;
            s.up_act = s.qu.data();
        }
    }
    s.pool.run(int64_t(count)*ff, Impl::gate_up, &s);
    if (s.use_native) {
        s.qd.resize(size_t(count));
        for (int k = 0; k < count; ++k)
            if (!native_cpu_prepare(s.fd, s.activation.data() + size_t(k)*ff, s.qd[size_t(k)], err)) return false;
    }
    s.pool.run(int64_t(count)*dim, Impl::down, &s);
    std::fill(sum, sum + dim, 0.f);
    // Fixed routing order makes results independent of worker count and completion order.
    for (int k = 0; k < count; ++k)
        for (int i = 0; i < dim; ++i) sum[i] += s.output[size_t(k)*dim + i];
    return true;
}
}  // namespace dsv4
