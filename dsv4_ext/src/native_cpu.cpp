#include "dsv4/native_cpu.hpp"

#if defined(DSV4_NATIVE_CPU)
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <deque>
#include <climits>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace dsv4 {

#if defined(DSV4_NATIVE_CPU)
namespace {
struct Ops {
    ggml_type weight_type;
    ggml_type activation_type;
    int64_t n;
    size_t weight_bytes;
    size_t activation_bytes;
    ggml_from_float_t from_float;
    ggml_vec_dot_t vec_dot;
    bool use_avx512_iq1m;
};

enum class Iq1mKernelPolicy { automatic, avx2, avx512, invalid };

Iq1mKernelPolicy iq1m_kernel_policy() noexcept {
    static const Iq1mKernelPolicy policy = [] {
        const char* value = std::getenv("DSV4_CPU_IQ1M_KERNEL");
        if (!value || !*value || std::strcmp(value, "auto") == 0) return Iq1mKernelPolicy::automatic;
        if (std::strcmp(value, "avx2") == 0) return Iq1mKernelPolicy::avx2;
        if (std::strcmp(value, "avx512") == 0) return Iq1mKernelPolicy::avx512;
        return Iq1mKernelPolicy::invalid;
    }();
    return policy;
}

const char* iq1m_kernel_policy_name(Iq1mKernelPolicy policy) noexcept {
    switch (policy) {
        case Iq1mKernelPolicy::automatic: return "auto";
        case Iq1mKernelPolicy::avx2: return "avx2";
        case Iq1mKernelPolicy::avx512: return "avx512";
        case Iq1mKernelPolicy::invalid: return "invalid";
    }
    return "invalid";
}

bool trace_cpu_dispatch() noexcept {
#if defined(DSV4_NATIVE_CPU_AVX512)
    static const bool enabled = [] {
        const char* value = std::getenv("DSV4_CPU_TRACE_DISPATCH");
        return value && std::strcmp(value, "0") != 0 && *value;
    }();
    return enabled;
#else
    return false;
#endif
}

void init_ggml() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

bool has_required_isa() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 1);
    constexpr int fma = 1 << 12, osxsave = 1 << 27, avx = 1 << 28, f16c = 1 << 29;
    if ((regs[2] & (fma | osxsave | avx | f16c)) != (fma | osxsave | avx | f16c)) return false;
    unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    constexpr int avx2 = 1 << 5;
#if defined(DSV4_NATIVE_CPU_BMI2)
    constexpr int bmi2 = 1 << 8;
    return (regs[1] & (avx2 | bmi2)) == (avx2 | bmi2);
#else
    return (regs[1] & avx2) != 0;
#endif
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    const bool base = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                      __builtin_cpu_supports("f16c");
#if defined(DSV4_NATIVE_CPU_BMI2)
    return base && __builtin_cpu_supports("bmi2");
#else
    return base;
#endif
#else
    return false;
#endif
}

bool has_avx512_isa() noexcept {
#if defined(DSV4_NATIVE_CPU_AVX512) && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    if (!has_required_isa()) return false;
    int regs[4]{};
    __cpuidex(regs, 1, 0);
    constexpr int osxsave = 1 << 27, avx = 1 << 28;
    if ((regs[2] & (osxsave | avx)) != (osxsave | avx)) return false;
    const unsigned __int64 xcr0 = _xgetbv(0);
    constexpr unsigned __int64 avx512_state = 0xE6;  // XMM, YMM, opmask, upper ZMM, high ZMM
    if ((xcr0 & avx512_state) != avx512_state) return false;
    __cpuidex(regs, 7, 0);
    constexpr unsigned int avx512f = 1u << 16, avx512dq = 1u << 17, avx512cd = 1u << 28;
    constexpr unsigned int avx512bw = 1u << 30, avx512vl = 1u << 31;
    constexpr int avx512vnni = 1 << 11;  // CPUID.7.0:ECX
    return (static_cast<unsigned int>(regs[1]) & (avx512f | avx512dq | avx512cd | avx512bw | avx512vl)) ==
               (avx512f | avx512dq | avx512cd | avx512bw | avx512vl) &&
           (regs[2] & avx512vnni) != 0;
#elif defined(DSV4_NATIVE_CPU_AVX512) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return has_required_isa() && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512vnni");
#else
    return false;
#endif
}
}  // namespace
#endif

bool native_cpu_available() noexcept {
#if defined(DSV4_NATIVE_CPU)
    return has_required_isa();
#else
    return false;
#endif
}

bool native_cpu_avx512_available() noexcept {
#if defined(DSV4_NATIVE_CPU)
    return has_avx512_isa();
#else
    return false;
#endif
}

std::string native_cpu_description() {
#if defined(DSV4_NATIVE_CPU)
    const char* version = ggml_version();
    const char* commit = ggml_commit();
    const char* isa = "AVX2+FMA+F16C";
#if defined(DSV4_NATIVE_CPU_AVX512)
    static thread_local std::string isa_description;
    isa_description = native_cpu_avx512_available() ? "AVX2+FMA+F16C; IQ1_M AVX512F+BW+VL+VNNI available" :
                                                       "AVX2+FMA+F16C; IQ1_M AVX512F+BW+VL+VNNI unavailable";
    isa_description += "; IQ1_M kernel policy=";
    isa_description += iq1m_kernel_policy_name(iq1m_kernel_policy());
    isa_description += " (auto uses AVX512 for IQ1_M n=4096 when available; AVX2 otherwise)";
    isa = isa_description.c_str();
#endif
#if defined(DSV4_NATIVE_CPU_BMI2)
    isa = "AVX2+FMA+F16C+BMI2";
#if defined(DSV4_NATIVE_CPU_AVX512)
    isa_description = native_cpu_avx512_available() ? "AVX2+FMA+F16C+BMI2; IQ1_M AVX512F+BW+VL+VNNI available" :
                                                       "AVX2+FMA+F16C+BMI2; IQ1_M AVX512F+BW+VL+VNNI unavailable";
    isa_description += "; IQ1_M kernel policy=";
    isa_description += iq1m_kernel_policy_name(iq1m_kernel_policy());
    isa_description += " (auto uses AVX512 for IQ1_M n=4096 when available; AVX2 otherwise)";
    isa = isa_description.c_str();
#endif
#endif
    return std::string("ggml ") + (version ? version : "unknown") + " (" + (commit ? commit : "unknown") +
           "), compiled ISA floor " + isa + ", runtime " +
           (native_cpu_available() ? "available" : "unavailable");
#else
    return "ggml-cpu MoE kernels not enabled at build time";
#endif
}

bool native_cpu_format(uint32_t type, int64_t n, NativeCpuFormat& out, std::string& err) {
    out = {};
#if defined(DSV4_NATIVE_CPU)
    if (!native_cpu_available()) {
#if defined(DSV4_NATIVE_CPU_BMI2)
        err = "native ggml-cpu MoE kernels require AVX2, FMA, F16C, BMI2, and enabled AVX OS state";
#else
        err = "native ggml-cpu MoE kernels require AVX2, FMA, F16C, and enabled AVX OS state";
#endif
        return false;
    }
    if (n <= 0 || n > INT_MAX || type >= GGML_TYPE_COUNT) {
        err = "native ggml-cpu MoE format has invalid type or row width";
        return false;
    }
    const Iq1mKernelPolicy policy = iq1m_kernel_policy();
    if (policy == Iq1mKernelPolicy::invalid) {
        err = "DSV4_CPU_IQ1M_KERNEL must be auto, avx2, or avx512";
        return false;
    }
#if defined(DSV4_NATIVE_CPU_AVX512)
    if (type == GGML_TYPE_IQ1_M && policy == Iq1mKernelPolicy::avx512 && !native_cpu_avx512_available()) {
        err = "DSV4_CPU_IQ1M_KERNEL=avx512 requires AVX512F, AVX512BW, AVX512VL, AVX512VNNI, and enabled ZMM OS state";
        return false;
    }
#else
    if (type == GGML_TYPE_IQ1_M && policy == Iq1mKernelPolicy::avx512) {
        err = "DSV4_CPU_IQ1M_KERNEL=avx512 requires a DSV4_CPU_AVX512=ON build";
        return false;
    }
#endif
    init_ggml();
    const auto wt = static_cast<ggml_type>(type);
    const auto* traits = ggml_get_type_traits_cpu(wt);
    if (!traits || !traits->vec_dot) {
        err = std::string("ggml-cpu has no vec_dot for ") + ggml_type_name(wt);
        return false;
    }
    const auto at = traits->vec_dot_type;
    const auto* act_traits = ggml_get_type_traits_cpu(at);
    if (!act_traits || !act_traits->from_float) {
        err = std::string("ggml-cpu has no activation quantizer for ") + ggml_type_name(at);
        return false;
    }
    const int wb = ggml_blck_size(wt), ab = ggml_blck_size(at);
    if (wb <= 0 || ab <= 0 || n % wb || n % ab) {
        err = "native ggml-cpu MoE row width is not a whole quantization block";
        return false;
    }
    const size_t wbytes = ggml_row_size(wt, n);
    const size_t abytes = ggml_row_size(at, n);
    if (!wbytes || !abytes) {
        err = "ggml-cpu returned an invalid row size";
        return false;
    }
    // The records are static for the common (ggml type, row width) tuples. Avoid a map allocation in this
    // frequently initialized path; the format itself carries the two function pointers in the opaque record.
    static std::mutex cache_mutex;
    static std::deque<Ops> cache;
    const Ops* found = nullptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (const auto& op : cache)
            if (op.weight_type == wt && op.n == n) { found = &op; break; }
        if (!found) {
            bool use_avx512_iq1m = false;
#if defined(DSV4_NATIVE_CPU_AVX512)
            const Iq1mKernelPolicy policy = iq1m_kernel_policy();
            const bool selected = policy == Iq1mKernelPolicy::avx512 ||
                                  (policy == Iq1mKernelPolicy::automatic && n == 4096);
            use_avx512_iq1m = wt == GGML_TYPE_IQ1_M && selected && native_cpu_avx512_available();
#endif
            cache.push_back(Ops{wt, at, n, wbytes, abytes, act_traits->from_float, traits->vec_dot,
                                use_avx512_iq1m});
            found = &cache.back();
#if defined(DSV4_NATIVE_CPU_AVX512)
            if (trace_cpu_dispatch()) {
                std::fprintf(stderr, "native CPU row-dot selected type=%s n=%lld kernel=%s\n",
                             ggml_type_name(wt), static_cast<long long>(n),
                             use_avx512_iq1m ? "AVX512" : "AVX2");
            }
#endif
        }
    }
    out.weight_type = type;
    out.activation_type = static_cast<uint32_t>(at);
    out.n = n;
    out.weight_row_bytes = wbytes;
    out.activation_bytes = abytes;
    out._ops = found;
    return true;
#else
    (void) type; (void) n;
    err = "native ggml-cpu MoE kernels were not enabled at build time";
    return false;
#endif
}

bool native_cpu_prepare(const NativeCpuFormat& format, const float* x, std::vector<uint8_t>& activation,
                        std::string& err) {
#if defined(DSV4_NATIVE_CPU)
    if (!format._ops || !x) { err = "native ggml-cpu activation has no format or input"; return false; }
    const auto* op = static_cast<const Ops*>(format._ops);
    if (op->n != format.n || op->activation_bytes != format.activation_bytes) {
        err = "native ggml-cpu activation format mismatch"; return false;
    }
    try { activation.resize(op->activation_bytes); }
    catch (...) { err = "could not allocate quantized MoE activation"; return false; }
    op->from_float(x, activation.data(), op->n);
    return true;
#else
    (void) format; (void) x; (void) activation;
    err = "native ggml-cpu MoE kernels were not enabled at build time";
    return false;
#endif
}

bool native_cpu_dot(const NativeCpuFormat& format, const uint8_t* weight_row, const uint8_t* activation,
                    float& result) noexcept {
#if defined(DSV4_NATIVE_CPU)
    if (!format._ops || !weight_row || !activation) return false;
    const auto* op = static_cast<const Ops*>(format._ops);
    if (op->n != format.n || op->weight_bytes != format.weight_row_bytes ||
        op->activation_bytes != format.activation_bytes) return false;
#if defined(DSV4_NATIVE_CPU_AVX512)
    if (op->use_avx512_iq1m)
        return native_cpu_dot_iq1m_avx512(weight_row, activation, op->n, result);
#endif
    result = 0.f;
    op->vec_dot(static_cast<int>(op->n), &result, 0, weight_row, 0, activation, 0, 1);
    return true;
#else
    (void) format; (void) weight_row; (void) activation; (void) result;
    return false;
#endif
}

}  // namespace dsv4
