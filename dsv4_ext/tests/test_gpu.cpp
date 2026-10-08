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

bool check_matvec(uint32_t type) {
    constexpr int64_t in = 256, rows = 5;
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
    auto* stage = (uint8_t*)dsv4::gpu::alloc((size_t)dsv4::gpu::kMaxHit * bpe);
    if (!dxv || !dg || !du || !da || !dy || !stage) {
        std::fprintf(stderr, "FAIL MoE device allocation\n");
        if (dxv) dsv4::gpu::release(dxv); if (dg) dsv4::gpu::release(dg); if (du) dsv4::gpu::release(du);
        if (da) dsv4::gpu::release(da); if (dy) dsv4::gpu::release(dy); if (stage) dsv4::gpu::release(stage);
        return false;
    }
    if (!dsv4::gpu::staging(stage, (size_t)dsv4::gpu::kMaxHit * bpe) || !dsv4::gpu::has_staging()) {
        std::fprintf(stderr, "FAIL staging registration\n"); return false;
    }
    dsv4::gpu::h2d(dxv, x.data(), x.size() * sizeof(float));

    dsv4::gpu::ExpPtrs p{}; p.n = n; p.bpe = bpe;
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
    for (auto& e : experts) if (e.dg) dsv4::gpu::release(e.dg);
    dsv4::gpu::release(dxv); dsv4::gpu::release(dg); dsv4::gpu::release(du); dsv4::gpu::release(da);
    dsv4::gpu::release(dy); dsv4::gpu::release(stage);
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
    std::puts("ok MoE gate/up SwiGLU/down for resident and staged host experts");
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

int main() {
    std::string err;
    if (!dsv4::gpu::init(err)) {
        std::fprintf(stderr, "GPU INIT FAILED: %s\n", err.c_str()); return 2;
    }
    if (dsv4::gpu::is_emulated()) {
        std::fprintf(stderr, "GPU TEST REFUSED: backend is CPU-emulated\n"); return 2;
    }
    const uint32_t types[] = {dsv4::T_F32, dsv4::T_BF16, dsv4::T_Q8_0, dsv4::T_Q4_K, dsv4::T_Q5_K,
        dsv4::T_Q6_K, dsv4::T_IQ3_XXS, dsv4::T_IQ2_XXS, dsv4::T_IQ1_M, dsv4::T_MXFP4};
    bool ok = true;
    for (uint32_t type : types) {
        if (!dsv4::gpu::type_supported(type)) {
            std::fprintf(stderr, "FAIL expected DSV4 type %u to be supported\n", type); ok = false; continue;
        }
        ok = check_matvec(type) && ok;
    }
    ok = check_unsupported_rejects_without_write() && ok;
    ok = check_moe() && ok;
    std::puts(ok ? "ALL REAL GPU TESTS PASSED" : "REAL GPU TESTS FAILED");
    return ok ? 0 : 1;
}
