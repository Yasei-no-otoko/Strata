#include "dsv4/dequant.hpp"
#include "dsv4/native_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace {
int failures = 0;

void check_type(uint32_t type, int64_t n) {
    dsv4::NativeCpuFormat fmt;
    std::string err;
    if (!dsv4::native_cpu_format(type, n, fmt, err)) {
        std::fprintf(stderr, "FAIL native_cpu_format type=%u: %s\n", type, err.c_str());
        ++failures;
        return;
    }
    std::mt19937 rng(0xD54 + type);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x((size_t)n), decoded((size_t)n);
    for (float& v : x) v = dist(rng);
    std::vector<uint8_t> w(fmt.weight_row_bytes);
    for (uint8_t& b : w) b = static_cast<uint8_t>(rng());
    // Keep each block's global half scale finite and in a representative range.
    if (type == dsv4::T_IQ1_M) {
        for (size_t off = 0; off < w.size(); off += 56) {
            w[off + 48] = 0x00; w[off + 49] = 0x30;
            w[off + 50] = 0x00; w[off + 51] = 0x30;
            w[off + 52] = 0x00; w[off + 53] = 0xC0;
            w[off + 54] = 0x00; w[off + 55] = 0x30;
        }
    } else {
        const size_t bsize = type == dsv4::T_IQ2_XXS ? 66 : 98;
        for (size_t off = 0; off < w.size(); off += bsize) { w[off] = 0x00; w[off + 1] = 0x30; }
    }
    std::vector<uint8_t> act;
    if (!dsv4::native_cpu_prepare(fmt, x.data(), act, err) || act.size() != fmt.activation_bytes) {
        std::fprintf(stderr, "FAIL native_cpu_prepare type=%u: %s\n", type, err.c_str());
        ++failures;
        return;
    }
    float got = 0.f;
    if (!dsv4::native_cpu_dot(fmt, w.data(), act.data(), got) || !std::isfinite(got)) {
        std::fprintf(stderr, "FAIL native_cpu_dot type=%u\n", type);
        ++failures;
        return;
    }
    if (act.size() != fmt.activation_bytes) { ++failures; return; }
    dsv4::dequant_row(type, w.data(), n, decoded.data());
    double ref = 0, l1 = 0;
    for (int64_t i = 0; i < n; ++i) {
        ref += (double)decoded[(size_t)i] * x[(size_t)i];
        l1 += std::fabs((double)decoded[(size_t)i] * x[(size_t)i]);
    }
    std::vector<float> quant_x((size_t)n);
    dsv4::dequant_row(fmt.activation_type, act.data(), n, quant_x.data());
    const float quantized_ref = dsv4::row_dot(type, w.data(), quant_x.data(), n);
    const double impl_error = std::fabs((double)got - quantized_ref);
    if (impl_error > std::max(1e-4, 0.003 * l1)) {
        std::fprintf(stderr, "FAIL native_cpu type=%u differs from decoded-activation float-dot by %.6g\n",
                     type, impl_error);
        ++failures;
    }
    const double err_abs = std::fabs((double)got - ref);
    // Quantized activation dots are an explicit fast-mode tradeoff, not bit-exact float-reference math.
    // Bound the observed error relative to the absolute contribution scale to avoid cancellation artifacts.
    if (err_abs > std::max(1e-4, 0.06 * l1)) {
        std::fprintf(stderr, "FAIL native_cpu type=%u quantized-dot error %.6g exceeds 6%% contribution budget %.6g\n",
                     type, err_abs, 0.06 * l1);
        ++failures;
    } else {
        std::printf("ok native_cpu type=%u activation=%u ggml-vs-quantized-float=%.4g quant-vs-float-input=%.4g l1=%.4g\n",
                    type, fmt.activation_type, impl_error, err_abs, l1);
    }
}
}  // namespace

int main() {
    if (!dsv4::native_cpu_available()) {
        std::puts("SKIP native_cpu: build or host does not meet the AVX2 floor");
        return 0;
    }
    check_type(dsv4::T_IQ1_M, 4096);
    check_type(dsv4::T_IQ2_XXS, 4096);
    check_type(dsv4::T_IQ3_XXS, 4096);
    return failures ? 1 : 0;
}
