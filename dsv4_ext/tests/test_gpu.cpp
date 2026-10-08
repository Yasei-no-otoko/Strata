// Real-device correctness checks for the DSV4 GPU backend. This test deliberately refuses the
// CPU-emulated backend: passing it is evidence that kernels ran on a CUDA/HIP device.
#include "dsv4/dequant.hpp"
#include "dsv4/gpu.hpp"
#include "dsv4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

uint64_t rng = 0xD5A4F1A5C0FFEE11ULL;
uint64_t next_u64() {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 2685821657736338717ULL;
}

void fill_bytes(std::vector<uint8_t>& bytes) {
    for (uint8_t& b : bytes) b = (uint8_t) next_u64();
}

void put_u16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, sizeof(v)); }

void put_f32(uint8_t* p, float v) { std::memcpy(p, &v, sizeof(v)); }

bool close_enough(float got, float want) {
    if (!std::isfinite(got) || !std::isfinite(want)) return false;
    constexpr double abs_tol = 2e-3;
    constexpr double rel_tol = 3e-4;
    const double diff = std::fabs((double) got - (double) want);
    return diff <= abs_tol + rel_tol * std::fabs((double) want);
}

bool finite_dot(uint32_t type, const uint8_t* row, const std::vector<float>& x, float& y) {
    y = dsv4::row_dot(type, row, x.data(), (int64_t)x.size());
    return std::isfinite(y);
}

// Retain varied packed payload bits but replace each scale with a small finite value. Random bytes
// in scale fields frequently encode half-float NaNs/Infs and are poor GPU correctness fixtures.
void bound_block(uint32_t type, uint8_t* b) {
    switch (type) {
        case dsv4::T_F32:
            for (int i = 0; i < 1; ++i) {
                // F32 is not blocked; handled across the row by make_matrix().
                (void)i;
            }
            break;
        case dsv4::T_BF16:
            break;  // handled across the row by make_matrix().
        case dsv4::T_Q8_0:
            put_u16(b, 0x3000);  // f16 0.125
            break;
        case dsv4::T_Q4_K:
        case dsv4::T_Q5_K:
            put_u16(b, 0x3000);  // d = 0.125
            put_u16(b + 2, 0);   // min = 0
            std::memset(b + 4, 0, 12);
            b[4] = b[5] = b[6] = b[7] = 1;
            b[12] = b[13] = b[14] = b[15] = 1;
            break;
        case dsv4::T_Q6_K:
            std::memset(b + 192, 1, 16);  // signed sub-block scales
            put_u16(b + 208, 0x3000);
            break;
        case dsv4::T_IQ3_XXS:
        case dsv4::T_IQ2_XXS:
            put_u16(b, 0x3000);
            break;
        case dsv4::T_IQ1_M:
            std::memset(b + 48, 0, 8);
            put_u16(b + 54, 0x3000);  // assembled block scale becomes f16 0.125
            break;
        case dsv4::T_MXFP4:
            b[0] = 128;  // e8m0 scale 1.0
            break;
        default:
            break;
    }
}

bool make_matrix(uint32_t type, int64_t rows, int64_t in, std::vector<uint8_t>& w) {
    const size_t rb = dsv4::row_bytes(type, in);
    if (!rb) return false;
    w.resize((size_t)rows * rb);
    fill_bytes(w);
    const int block_elems = dsv4::dequant_block_elems(type);
    if (type == dsv4::T_F32 || type == dsv4::T_BF16) {
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = w.data() + (size_t)r * rb;
            for (int64_t i = 0; i < in; ++i) {
                const float value = (float)((i * 7 + r * 11) % 31 - 15) / 16.f;
                if (type == dsv4::T_F32) put_f32(row + 4 * (size_t)i, value);
                else {
                    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
                    put_u16(row + 2 * (size_t)i, (uint16_t)(bits >> 16));
                }
            }
        }
    } else {
        if (block_elems <= 0 || in % block_elems != 0) return false;
        const size_t block_bytes = rb / (size_t)(in / block_elems);
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = w.data() + (size_t)r * rb;
            for (int64_t block = 0; block < in / block_elems; ++block)
                bound_block(type, row + (size_t)block * block_bytes);
        }
    }

    std::vector<float> decoded((size_t)in);
    for (int64_t r = 0; r < rows; ++r) {
        dsv4::dequant_row(type, w.data() + (size_t)r * rb, in, decoded.data());
        for (float value : decoded)
            if (!std::isfinite(value) || std::fabs(value) > 64.f) {
                std::fprintf(stderr, "FAIL invalid fixture element for type %u: %g\n", type, value);
                return false;
            }
    }
    return true;
}

bool check_matvec(uint32_t type, int64_t in, int64_t rows) {
    const size_t rb = dsv4::row_bytes(type, in);
    std::vector<uint8_t> host_w;
    std::vector<float> x((size_t)in), want((size_t)rows), got((size_t)rows, -98765.f);
    if (!make_matrix(type, rows, in, host_w)) {
        std::fprintf(stderr, "FAIL fixture generation for type %u\n", type);
        return false;
    }
    for (int64_t i = 0; i < in; ++i) x[(size_t)i] = (float)((i * 13 % 101) - 50) / 80.f;
    for (int64_t r = 0; r < rows; ++r)
        if (!finite_dot(type, host_w.data() + (size_t)r * rb, x, want[(size_t)r])) return false;

    void* dw = dsv4::gpu::alloc(host_w.size());
    float* dx = (float*)dsv4::gpu::alloc(x.size() * sizeof(float));
    float* dy = (float*)dsv4::gpu::alloc(got.size() * sizeof(float));
    if (!dw || !dx || !dy) {
        std::fprintf(stderr, "FAIL device allocation for type %u\n", type);
        if (dw) dsv4::gpu::release(dw); if (dx) dsv4::gpu::release(dx); if (dy) dsv4::gpu::release(dy);
        return false;
    }
    dsv4::gpu::h2d(dw, host_w.data(), host_w.size());
    dsv4::gpu::h2d(dx, x.data(), x.size() * sizeof(float));
    const bool launched = dsv4::gpu::matvec(type, (const uint8_t*)dw, rows, in, dx, dy);
    if (launched) dsv4::gpu::d2h(got.data(), dy, got.size() * sizeof(float));
    dsv4::gpu::release(dw); dsv4::gpu::release(dx); dsv4::gpu::release(dy);
    if (!launched) {
        std::fprintf(stderr, "FAIL matvec rejected supported type %u\n", type);
        return false;
    }
    for (int64_t r = 0; r < rows; ++r) {
        if (!close_enough(got[(size_t)r], want[(size_t)r])) {
            std::fprintf(stderr, "FAIL matvec type %u row %lld: gpu=%g cpu=%g\n", type,
                         (long long)r, got[(size_t)r], want[(size_t)r]);
            return false;
        }
    }
    std::printf("ok matvec type %u (%lld rows x %lld)\n", type, (long long)rows, (long long)in);
    return true;
}

bool check_grouped_matvec(uint32_t type, int64_t groups) {
    constexpr int64_t in = 256, rows_per_group = 3;
    const int64_t total_rows = groups * rows_per_group;
    const size_t rb = dsv4::row_bytes(type, in);
    std::vector<uint8_t> host_w;
    std::vector<float> x((size_t)(groups * in)), want((size_t)total_rows), got((size_t)total_rows, -98765.f);
    if (!make_matrix(type, total_rows, in, host_w)) return false;
    for (int64_t group = 0; group < groups; ++group)
        for (int64_t i = 0; i < in; ++i)
            x[(size_t)(group * in + i)] = (float)((i * 13 + group * 19) % 101 - 50) / 80.f;
    for (int64_t row = 0; row < total_rows; ++row) {
        const int64_t group = row / rows_per_group;
        want[(size_t)row] = dsv4::row_dot(type, host_w.data() + (size_t)row * rb, x.data() + group * in, in);
        if (!std::isfinite(want[(size_t)row])) return false;
    }
    void* dw = dsv4::gpu::alloc(host_w.size());
    auto* dx = (float*)dsv4::gpu::alloc(x.size() * sizeof(float));
    auto* dy = (float*)dsv4::gpu::alloc(got.size() * sizeof(float));
    if (!dw || !dx || !dy) {
        std::fprintf(stderr, "FAIL grouped device allocation for type %u\n", type);
        if (dw) dsv4::gpu::release(dw); if (dx) dsv4::gpu::release(dx); if (dy) dsv4::gpu::release(dy);
        return false;
    }
    dsv4::gpu::h2d(dw, host_w.data(), host_w.size());
    dsv4::gpu::h2d(dx, x.data(), x.size() * sizeof(float));
    const bool launched = dsv4::gpu::matvec_grouped(type, (const uint8_t*)dw, groups, rows_per_group, in, dx, dy);
    if (launched) dsv4::gpu::d2h(got.data(), dy, got.size() * sizeof(float));
    dsv4::gpu::release(dw); dsv4::gpu::release(dx); dsv4::gpu::release(dy);
    if (!launched) { std::fprintf(stderr, "FAIL grouped matvec rejected type %u groups %lld\n", type, (long long)groups); return false; }
    for (int64_t row = 0; row < total_rows; ++row) if (!close_enough(got[(size_t)row], want[(size_t)row])) {
        std::fprintf(stderr, "FAIL grouped matvec type %u groups %lld row %lld: gpu=%g cpu=%g\n", type,
                     (long long)groups, (long long)row, got[(size_t)row], want[(size_t)row]);
        return false;
    }
    std::printf("ok grouped matvec type %u groups=%lld rows/group=%lld\n", type,
                (long long)groups, (long long)rows_per_group);
    return true;
}

bool run_matvec_bench(int iterations, int64_t rows) {
    // Opt-in throughput probe for comparing kernel revisions. Weight/input uploads are excluded;
    // each timed call includes launch and a D2H sync, so report it as end-to-end matvec latency.
    constexpr int64_t in = 4096;
    const uint32_t types[] = {dsv4::T_F32, dsv4::T_BF16, dsv4::T_Q8_0, dsv4::T_Q4_K, dsv4::T_Q5_K,
        dsv4::T_Q6_K, dsv4::T_IQ3_XXS, dsv4::T_IQ2_XXS, dsv4::T_IQ1_M, dsv4::T_MXFP4};
    std::vector<float> x((size_t)in), y((size_t)rows);
    for (int64_t i = 0; i < in; ++i) x[(size_t)i] = (float)((i * 13 % 101) - 50) / 80.f;
    for (uint32_t type : types) {
        std::vector<uint8_t> w;
        if (!make_matrix(type, rows, in, w)) return false;
        void* dw = dsv4::gpu::alloc(w.size());
        auto* dx = (float*)dsv4::gpu::alloc(x.size() * sizeof(float));
        auto* dy = (float*)dsv4::gpu::alloc(y.size() * sizeof(float));
        if (!dw || !dx || !dy) {
            std::fprintf(stderr, "FAIL benchmark allocation for type %u\n", type);
            if (dw) dsv4::gpu::release(dw); if (dx) dsv4::gpu::release(dx); if (dy) dsv4::gpu::release(dy);
            return false;
        }
        dsv4::gpu::h2d(dw, w.data(), w.size());
        dsv4::gpu::h2d(dx, x.data(), x.size() * sizeof(float));
        for (int warm = 0; warm < 3; ++warm) {
            if (!dsv4::gpu::matvec(type, (const uint8_t*)dw, rows, in, dx, dy)) return false;
            dsv4::gpu::d2h(y.data(), dy, y.size() * sizeof(float));
        }
        const auto begin = std::chrono::steady_clock::now();
        for (int it = 0; it < iterations; ++it) {
            if (!dsv4::gpu::matvec(type, (const uint8_t*)dw, rows, in, dx, dy)) return false;
            dsv4::gpu::d2h(y.data(), dy, y.size() * sizeof(float));
        }
        const auto end = std::chrono::steady_clock::now();
        for (float value : y) if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL benchmark non-finite output for type %u\n", type);
            return false;
        }
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
        const double gbps = (double)w.size() / (ms * 1.0e6);
        std::printf("bench type=%u shape=%lldx%lld bytes=%zu iterations=%d ms/call=%.4f effective_GB/s=%.2f\n",
                    type, (long long)rows, (long long)in, w.size(), iterations, ms, gbps);
        dsv4::gpu::release(dw); dsv4::gpu::release(dx); dsv4::gpu::release(dy);
    }
    return true;
}

struct Expert {
    std::vector<uint8_t> g, u, d;
    uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr;
};

bool check_moe() {
    constexpr int64_t dim = 256, ff = 256;
    constexpr int n = 3;
    constexpr uint32_t tg = dsv4::T_IQ3_XXS, tu = dsv4::T_Q8_0, td = dsv4::T_IQ1_M;
    const size_t bg = dsv4::row_bytes(tg, dim) * (size_t)ff;
    const size_t bu = dsv4::row_bytes(tu, dim) * (size_t)ff;
    const size_t bd = dsv4::row_bytes(td, ff) * (size_t)dim;
    const size_t bpe = bg + bu + bd;
    std::vector<Expert> experts((size_t)n);
    for (auto& e : experts)
        if (!make_matrix(tg, ff, dim, e.g) || !make_matrix(tu, ff, dim, e.u) ||
            !make_matrix(td, dim, ff, e.d)) {
            std::fprintf(stderr, "FAIL MoE fixture generation\n"); return false;
        }

    std::vector<float> x((size_t)dim);
    for (int64_t i = 0; i < dim; ++i) x[(size_t)i] = (float)((i * 7 % 67) - 33) / 64.f;
    const float weights[n] = {0.5f, 0.3f, 0.2f};
    constexpr float limit = 5.5f;
    std::vector<float> want((size_t)dim, 0.f), got_resident((size_t)dim, -1.f), got_host((size_t)dim, -1.f);
    for (int k = 0; k < n; ++k) {
        std::vector<float> g((size_t)ff), u((size_t)ff), a((size_t)ff), y((size_t)dim);
        for (int64_t r = 0; r < ff; ++r) {
            g[(size_t)r] = dsv4::row_dot(tg, experts[(size_t)k].g.data() + (size_t)r * dsv4::row_bytes(tg, dim), x.data(), dim);
            u[(size_t)r] = dsv4::row_dot(tu, experts[(size_t)k].u.data() + (size_t)r * dsv4::row_bytes(tu, dim), x.data(), dim);
        }
        dsv4::swiglu_clamped(g.data(), u.data(), (int)ff, limit, a.data());
        for (int64_t i = 0; i < ff; ++i) a[(size_t)i] *= weights[k];
        for (int64_t r = 0; r < dim; ++r) {
            y[(size_t)r] = dsv4::row_dot(td, experts[(size_t)k].d.data() + (size_t)r * dsv4::row_bytes(td, ff), a.data(), ff);
            want[(size_t)r] += y[(size_t)r];
        }
    }
    for (float v : want) if (!std::isfinite(v)) { std::fprintf(stderr, "FAIL non-finite CPU MoE reference\n"); return false; }

    void* dxv = dsv4::gpu::alloc(x.size() * sizeof(float));
    float* dg = (float*)dsv4::gpu::alloc((size_t)dsv4::gpu::kMaxHit * ff * sizeof(float));
    float* du = (float*)dsv4::gpu::alloc((size_t)dsv4::gpu::kMaxHit * ff * sizeof(float));
    float* da = (float*)dsv4::gpu::alloc((size_t)dsv4::gpu::kMaxHit * ff * sizeof(float));
    float* dy = (float*)dsv4::gpu::alloc((size_t)dim * sizeof(float));
    const size_t stage_stride = bpe + 64;
    auto* stage = (uint8_t*)dsv4::gpu::alloc((size_t)dsv4::gpu::kMaxHit * stage_stride);
    if (!dxv || !dg || !du || !da || !dy || !stage) {
        std::fprintf(stderr, "FAIL MoE device allocation\n");
        if (dxv) dsv4::gpu::release(dxv); if (dg) dsv4::gpu::release(dg); if (du) dsv4::gpu::release(du);
        if (da) dsv4::gpu::release(da); if (dy) dsv4::gpu::release(dy); if (stage) dsv4::gpu::release(stage);
        return false;
    }
    if (!dsv4::gpu::staging(stage, (size_t)dsv4::gpu::kMaxHit * stage_stride) || !dsv4::gpu::has_staging()) {
        std::fprintf(stderr, "FAIL staging registration\n"); return false;
    }
    dsv4::gpu::h2d(dxv, x.data(), x.size() * sizeof(float));

    dsv4::gpu::ExpPtrs p{}; p.n = n; p.bpe = stage_stride;
    for (int k = 0; k < n; ++k) {
        auto& e = experts[(size_t)k];
        e.dg = (uint8_t*)dsv4::gpu::alloc(bpe);
        if (!e.dg) { std::fprintf(stderr, "FAIL resident expert allocation\n"); return false; }
        e.du = e.dg + bg; e.dd = e.du + bu;
        std::vector<uint8_t> packed; packed.reserve(bpe);
        packed.insert(packed.end(), e.g.begin(), e.g.end()); packed.insert(packed.end(), e.u.begin(), e.u.end());
        packed.insert(packed.end(), e.d.begin(), e.d.end());
        dsv4::gpu::h2d(e.dg, packed.data(), packed.size());
        p.gate[k] = e.dg; p.up[k] = e.du; p.down[k] = e.dd;
        p.hg[k] = e.g.data(); p.hu[k] = e.u.data(); p.hd[k] = e.d.data(); p.w[k] = weights[k];
    }
    const bool resident_ok = dsv4::gpu::experts_hit(p, tg, tu, td, ff, dim, limit, (float*)dxv, dg, du, da, dy);
    if (resident_ok) dsv4::gpu::d2h(got_resident.data(), dy, got_resident.size() * sizeof(float));

    for (int k = 0; k < n; ++k) { p.gate[k] = p.up[k] = p.down[k] = nullptr; }
    const bool host_ok = dsv4::gpu::experts_hit(p, tg, tu, td, ff, dim, limit, (float*)dxv, dg, du, da, dy);
    if (host_ok) dsv4::gpu::d2h(got_host.data(), dy, got_host.size() * sizeof(float));
    if (!resident_ok || !host_ok) { std::fprintf(stderr, "FAIL experts_hit resident=%d host-stage=%d\n", resident_ok, host_ok); return false; }
    for (int64_t i = 0; i < dim; ++i) {
        if (!close_enough(got_resident[(size_t)i], want[(size_t)i]) ||
            !close_enough(got_host[(size_t)i], want[(size_t)i]) ||
            !close_enough(got_resident[(size_t)i], got_host[(size_t)i])) {
            std::fprintf(stderr, "FAIL MoE output %lld: resident=%g host=%g cpu=%g\n", (long long)i,
                         got_resident[(size_t)i], got_host[(size_t)i], want[(size_t)i]);
            return false;
        }
    }

    // Exercise reuse of the same pinned/device staging slots with changing host weights, routing
    // weights, and activations. This catches stale-stage contents across calls, including when
    // DSV4_HIP_ASYNC_STAGE=1 queues copies on a separate stream.
    auto compute_host_reference = [&](const std::vector<float>& input, std::vector<float>& output) {
        output.assign((size_t)dim, 0.f);
        for (int k = 0; k < n; ++k) {
            std::vector<float> g((size_t)ff), u((size_t)ff), a((size_t)ff), y((size_t)dim);
            for (int64_t r = 0; r < ff; ++r) {
                g[(size_t)r] = dsv4::row_dot(tg, experts[(size_t)k].g.data() + (size_t)r * dsv4::row_bytes(tg, dim), input.data(), dim);
                u[(size_t)r] = dsv4::row_dot(tu, experts[(size_t)k].u.data() + (size_t)r * dsv4::row_bytes(tu, dim), input.data(), dim);
            }
            dsv4::swiglu_clamped(g.data(), u.data(), (int)ff, limit, a.data());
            for (int64_t i = 0; i < ff; ++i) a[(size_t)i] *= p.w[k];
            for (int64_t r = 0; r < dim; ++r) {
                y[(size_t)r] = dsv4::row_dot(td, experts[(size_t)k].d.data() + (size_t)r * dsv4::row_bytes(td, ff), a.data(), ff);
                output[(size_t)r] += y[(size_t)r];
            }
        }
        for (float value : output) if (!std::isfinite(value)) return false;
        return true;
    };
    bool repeat_ok = true;
    const char* async_env = std::getenv("DSV4_HIP_ASYNC_STAGE");
    if (async_env && std::strcmp(async_env, "1") == 0) {
        // Specifically test slot reuse before the prior default-stream compute is read back.
        // The second call changes both the host packed matrices and x while reusing the same stages.
        std::vector<float> async_x((size_t)dim), async_want;
        for (int64_t i = 0; i < dim; ++i) async_x[(size_t)i] = (float)((i * 23 % 97) - 48) / 80.f;
        auto* dx_async = (float*)dsv4::gpu::alloc(async_x.size() * sizeof(float));
        if (!dx_async) { std::fprintf(stderr, "FAIL async staging input allocation\n"); repeat_ok = false; }
        if (repeat_ok) {
            dsv4::gpu::h2d(dx_async, async_x.data(), async_x.size() * sizeof(float));
            const bool first_queued = dsv4::gpu::experts_hit(p, tg, tu, td, ff, dim, limit,
                                                              (float*)dxv, dg, du, da, dy);
            if (!first_queued) repeat_ok = false;
            for (int k = 0; k < n; ++k) {
                auto& e = experts[(size_t)k];
                e.g[bg - 1] ^= (uint8_t)(0x31 + k);
                e.u[bu - 1] ^= (uint8_t)(0x27 + k);
                e.d[bd - 9] ^= (uint8_t)(0x19 + k);
                p.w[k] = 0.22f + 0.09f * (float)k;
            }
            p.bpe = bpe;  // smaller stride makes the new expert ranges overlap prior neighbor slots
            if (!compute_host_reference(async_x, async_want)) repeat_ok = false;
            const bool second_queued = repeat_ok && dsv4::gpu::experts_hit(p, tg, tu, td, ff, dim, limit,
                                                                            dx_async, dg, du, da, dy);
            if (second_queued) {
                dsv4::gpu::d2h(got_host.data(), dy, got_host.size() * sizeof(float));
                for (int64_t i = 0; i < dim; ++i) if (!close_enough(got_host[(size_t)i], async_want[(size_t)i])) {
                    std::fprintf(stderr, "FAIL async staged slot reuse row %lld: gpu=%g cpu=%g\n",
                                 (long long)i, got_host[(size_t)i], async_want[(size_t)i]);
                    repeat_ok = false; break;
                }
            } else {
                std::fprintf(stderr, "FAIL async staged double enqueue first=%d second=%d\n", first_queued, second_queued);
                repeat_ok = false;
            }
        }
        if (dx_async) dsv4::gpu::release(dx_async);
    }
    for (int iteration = 0; iteration < 8; ++iteration) {
        if (!repeat_ok) break;
        std::vector<float> iter_x((size_t)dim), iter_want;
        for (int64_t i = 0; i < dim; ++i)
            iter_x[(size_t)i] = (float)((i * 17 + iteration * 29) % 113 - 56) / 96.f;
        for (int k = 0; k < n; ++k) {
            auto& e = experts[(size_t)k];
            e.g[bg - 1] ^= (uint8_t)(iteration * 17 + k + 1); // packed IQ3 payload, scale is untouched
            e.u[bu - 1] ^= (uint8_t)(iteration * 11 + k + 1); // Q8 quant payload
            e.d[bd - 9] ^= (uint8_t)(iteration * 7 + k + 1);  // IQ1 payload, before its final 8 scale bytes
            p.w[k] = 0.15f + 0.07f * (float)((iteration + k * 3) % 7);
        }
        if (!compute_host_reference(iter_x, iter_want)) { repeat_ok = false; break; }
        dsv4::gpu::h2d(dxv, iter_x.data(), iter_x.size() * sizeof(float));
        const bool iteration_ok = dsv4::gpu::experts_hit(p, tg, tu, td, ff, dim, limit,
                                                           (float*)dxv, dg, du, da, dy);
        if (!iteration_ok) { repeat_ok = false; break; }
        dsv4::gpu::d2h(got_host.data(), dy, got_host.size() * sizeof(float));
        for (int64_t i = 0; i < dim; ++i) if (!close_enough(got_host[(size_t)i], iter_want[(size_t)i])) {
            std::fprintf(stderr, "FAIL repeated staged MoE iter %d row %lld: gpu=%g cpu=%g\n", iteration,
                         (long long)i, got_host[(size_t)i], iter_want[(size_t)i]);
            repeat_ok = false; break;
        }
        if (!repeat_ok) break;
    }
    for (auto& e : experts) if (e.dg) dsv4::gpu::release(e.dg);
    dsv4::gpu::release(dxv); dsv4::gpu::release(dg); dsv4::gpu::release(du); dsv4::gpu::release(da);
    dsv4::gpu::release(dy); dsv4::gpu::release(stage);
    if (!repeat_ok) { std::fprintf(stderr, "FAIL repeated staged MoE slot reuse\n"); return false; }
    std::puts("ok MoE gate/up SwiGLU/down for resident and staged host experts");
    return true;
}

uint32_t ordered_float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

uint32_t float_ulp_distance(float a, float b) {
    const uint32_t oa = ordered_float_bits(a), ob = ordered_float_bits(b);
    return oa > ob ? oa - ob : ob - oa;
}

bool check_reference_swiglu_activation() {
    constexpr int n = 256;
    constexpr float limit = 5.5f;
    const size_t matrix_bytes = (size_t)n * n * sizeof(float);
    std::vector<float> x((size_t)n), gate((size_t)n * n, 0.f), up((size_t)n * n, 0.f), down((size_t)n * n, 0.f);
    const float samples[] = {-20.f, -5.5f, -3.f, -1.f, -0.1f, 0.f, 0.1f, 1.f, 3.f, 5.5f, 10.f, 20.f};
    for (int i = 0; i < n; ++i) {
        x[(size_t)i] = samples[i % (int)(sizeof(samples) / sizeof(samples[0]))];
        gate[(size_t)i * n + i] = 1.f;
        up[(size_t)i * n + i] = 2.f;
        down[(size_t)i * n + i] = 1.f;
    }

    auto* d_gate = (uint8_t*)dsv4::gpu::alloc(matrix_bytes);
    auto* d_up = (uint8_t*)dsv4::gpu::alloc(matrix_bytes);
    auto* d_down = (uint8_t*)dsv4::gpu::alloc(matrix_bytes);
    auto* d_x = (float*)dsv4::gpu::alloc((size_t)n * sizeof(float));
    auto* d_g = (float*)dsv4::gpu::alloc((size_t)n * sizeof(float));
    auto* d_u = (float*)dsv4::gpu::alloc((size_t)n * sizeof(float));
    auto* d_a = (float*)dsv4::gpu::alloc((size_t)n * sizeof(float));
    auto* d_y = (float*)dsv4::gpu::alloc((size_t)n * sizeof(float));
    if (!d_gate || !d_up || !d_down || !d_x || !d_g || !d_u || !d_a || !d_y) {
        std::fprintf(stderr, "FAIL reference SwiGLU test allocation\n");
        void* allocations[] = {d_gate, d_up, d_down, d_x, d_g, d_u, d_a, d_y};
        for (void* p : allocations)
            if (p) dsv4::gpu::release(p);
        return false;
    }
    dsv4::gpu::h2d(d_gate, gate.data(), matrix_bytes);
    dsv4::gpu::h2d(d_up, up.data(), matrix_bytes);
    dsv4::gpu::h2d(d_down, down.data(), matrix_bytes);
    dsv4::gpu::h2d(d_x, x.data(), x.size() * sizeof(float));

    dsv4::gpu::ExpPtrs expert{};
    expert.n = 1;
    expert.gate[0] = d_gate;
    expert.up[0] = d_up;
    expert.down[0] = d_down;
    expert.w[0] = 1.f;
    expert.reference_activation = true;
    bool ok = dsv4::gpu::experts_hit(expert, dsv4::T_F32, dsv4::T_F32, dsv4::T_F32, n, n,
                                    limit, d_x, d_g, d_u, d_a, d_y);
    std::vector<float> got_g((size_t)n), got_u((size_t)n), got_a((size_t)n), got_y((size_t)n), expected((size_t)n);
    if (ok) {
        dsv4::gpu::d2h(got_g.data(), d_g, (size_t)n * sizeof(float));
        dsv4::gpu::d2h(got_u.data(), d_u, (size_t)n * sizeof(float));
        dsv4::gpu::d2h(got_a.data(), d_a, (size_t)n * sizeof(float));
        dsv4::gpu::d2h(got_y.data(), d_y, (size_t)n * sizeof(float));
    }
    dsv4::gpu::release(d_gate); dsv4::gpu::release(d_up); dsv4::gpu::release(d_down);
    dsv4::gpu::release(d_x); dsv4::gpu::release(d_g); dsv4::gpu::release(d_u); dsv4::gpu::release(d_a); dsv4::gpu::release(d_y);
    if (!ok) { std::fprintf(stderr, "FAIL reference SwiGLU F32 kernel dispatch\n"); return false; }
    dsv4::swiglu_clamped(got_g.data(), got_u.data(), n, limit, expected.data());

    for (int i = 0; i < n; ++i) {
        if (got_g[(size_t)i] != x[(size_t)i] || got_u[(size_t)i] != 2.f * x[(size_t)i] ||
            !std::isfinite(got_a[(size_t)i]) || !std::isfinite(got_y[(size_t)i]) ||
            float_ulp_distance(got_a[(size_t)i], expected[(size_t)i]) > 2 ||
            float_ulp_distance(got_y[(size_t)i], expected[(size_t)i]) > 2) {
            std::fprintf(stderr, "FAIL reference SwiGLU row %d input=%g g=%g u=%g got=%g y=%g expected=%g ulp=%u\n",
                         i, x[(size_t)i], got_g[(size_t)i], got_u[(size_t)i], got_a[(size_t)i], got_y[(size_t)i],
                         expected[(size_t)i], float_ulp_distance(got_a[(size_t)i], expected[(size_t)i]));
            return false;
        }
    }
    std::puts("ok shared-reference SwiGLU double-exp/cast/multiply (F32, clamped and unclamped range)");
    return true;
}

bool check_unsupported_rejects_without_write() {
    // Q4_0 is intentionally outside the DSV4 backend's supported kernel set.
    if (dsv4::gpu::type_supported(dsv4::T_Q4_0)) return true;
    std::vector<uint8_t> w(dsv4::row_bytes(dsv4::T_Q4_0, 256), 0);
    std::vector<float> x(256, 0.25f), y(2, 1234.5f);
    void* dw = dsv4::gpu::alloc(w.size()); auto* dx = (float*)dsv4::gpu::alloc(x.size() * sizeof(float));
    auto* dy = (float*)dsv4::gpu::alloc(y.size() * sizeof(float));
    if (!dw || !dx || !dy) return false;
    dsv4::gpu::h2d(dw, w.data(), w.size()); dsv4::gpu::h2d(dx, x.data(), x.size() * sizeof(float));
    dsv4::gpu::h2d(dy, y.data(), y.size() * sizeof(float));
    const bool accepted = dsv4::gpu::matvec(dsv4::T_Q4_0, (const uint8_t*)dw, 2, 256, dx, dy);
    dsv4::gpu::d2h(y.data(), dy, y.size() * sizeof(float));
    dsv4::gpu::release(dw); dsv4::gpu::release(dx); dsv4::gpu::release(dy);
    if (accepted || y[0] != 1234.5f || y[1] != 1234.5f) {
        std::fprintf(stderr, "FAIL unsupported type was accepted or modified output\n"); return false;
    }
    std::puts("ok unsupported Q4_0 rejected without writing output");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int bench_iterations = 10;
    int64_t bench_rows = 4096;
    if (argc > 1) {
        if (std::strcmp(argv[1], "--bench") != 0 || argc > 4) {
            std::fprintf(stderr, "usage: test_gpu [--bench [iterations [rows]]]\n"); return 2;
        }
        bench = true;
        if (argc >= 3) {
            char* end = nullptr;
            const long parsed = std::strtol(argv[2], &end, 10);
            if (!end || *end || parsed < 1 || parsed > 10000) {
                std::fprintf(stderr, "benchmark iterations must be 1..10000\n"); return 2;
            }
            bench_iterations = (int)parsed;
        }
        if (argc == 4) {
            char* end = nullptr;
            const long long parsed = std::strtoll(argv[3], &end, 10);
            if (!end || *end || parsed < 1 || parsed > 129280) {
                std::fprintf(stderr, "benchmark rows must be 1..129280\n"); return 2;
            }
            bench_rows = (int64_t)parsed;
        }
    }
    std::string err;
    if (!dsv4::gpu::init(err)) {
        std::fprintf(stderr, "GPU INIT FAILED: %s\n", err.c_str()); return 2;
    }
    if (dsv4::gpu::is_emulated()) {
        std::fprintf(stderr, "GPU TEST REFUSED: backend is CPU-emulated\n"); return 2;
    }
    if (bench) return run_matvec_bench(bench_iterations, bench_rows) ? 0 : 1;
    const uint32_t types[] = {dsv4::T_F32, dsv4::T_BF16, dsv4::T_Q8_0, dsv4::T_Q4_K, dsv4::T_Q5_K,
        dsv4::T_Q6_K, dsv4::T_IQ3_XXS, dsv4::T_IQ2_XXS, dsv4::T_IQ1_M, dsv4::T_MXFP4};
    bool ok = true;
    for (uint32_t type : types) {
        if (!dsv4::gpu::type_supported(type)) {
            std::fprintf(stderr, "FAIL expected DSV4 type %u to be supported\n", type); ok = false; continue;
        }
        ok = check_matvec(type, 256, 5) && ok;
        ok = check_matvec(type, 4096, 5) && ok;
        ok = check_grouped_matvec(type, 1) && ok;
        ok = check_grouped_matvec(type, 8) && ok;
    }
    ok = check_unsupported_rejects_without_write() && ok;
    ok = check_moe() && ok;
    ok = check_reference_swiglu_activation() && ok;
    std::puts(ok ? "ALL REAL GPU TESTS PASSED" : "REAL GPU TESTS FAILED");
    return ok ? 0 : 1;
}
