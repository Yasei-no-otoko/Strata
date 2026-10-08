#include "dsv4/dequant.hpp"
#include "dsv4/native_cpu.hpp"
#if defined(DSV4_NATIVE_CPU)
#include "ggml-cpu.h"
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
#if defined(DSV4_NATIVE_CPU_AVX512)
double avx512_vs_ggml_max = 0.0;
double avx512_vs_ggml_sum_sq = 0.0;
size_t avx512_vs_ggml_rows = 0;
size_t real_q8_min_count = 0;
#endif

int benchmark_iq1m(int64_t n, size_t row_count) {
    constexpr int warmup_rounds = 1;
    constexpr int timed_rounds = 8;
    if (n <= 0 || n % 256 != 0) {
        std::fprintf(stderr, "usage: test_native_cpu --bench-iq1m <width divisible by 256> [rows]\n");
        return 2;
    }
    dsv4::NativeCpuFormat format;
    std::string err;
    if (!dsv4::native_cpu_format(dsv4::T_IQ1_M, n, format, err)) {
        std::fprintf(stderr, "FAIL benchmark format: %s\n", err.c_str());
        return 1;
    }
    std::mt19937 rng(uint32_t(0xA51 + n));
    std::uniform_real_distribution<float> input_dist(-1.f, 1.f);
    std::vector<uint8_t> weights(row_count * format.weight_row_bytes);
    for (uint8_t& byte : weights) byte = uint8_t(rng());
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t off = 0; off < format.weight_row_bytes; off += 56) {
            uint16_t scales[4];
            for (uint16_t& scale : scales) scale = uint16_t(rng() & 0x0fff);
            scales[3] |= 0x3000;
            std::memcpy(weights.data() + row * format.weight_row_bytes + off + 48, scales, sizeof(scales));
        }
    }
    std::vector<float> input(static_cast<size_t>(n));
    for (float& x : input) x = input_dist(rng);
    std::vector<uint8_t> activation;
    if (!dsv4::native_cpu_prepare(format, input.data(), activation, err)) {
        std::fprintf(stderr, "FAIL benchmark activation: %s\n", err.c_str());
        return 1;
    }
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_M);
    if (!traits || !traits->vec_dot) {
        std::fprintf(stderr, "FAIL benchmark could not resolve ggml AVX2 row-dot\n");
        return 1;
    }

    std::vector<float> decoded_weights(static_cast<size_t>(n));
    std::vector<float> decoded_activation(static_cast<size_t>(n));
    dsv4::dequant_row(dsv4::T_IQ1_M, weights.data(), n, decoded_weights.data());
    dsv4::dequant_row(dsv4::T_Q8_K, activation.data(), n, decoded_activation.data());
    const float oracle = dsv4::row_dot(dsv4::T_IQ1_M, weights.data(), decoded_activation.data(), n);
    double l1 = 0.0;
    for (int64_t i = 0; i < n; ++i)
        l1 += std::fabs(double(decoded_weights[static_cast<size_t>(i)]) * decoded_activation[static_cast<size_t>(i)]);
    float checked = 0.f;
    if (!dsv4::native_cpu_dot(format, weights.data(), activation.data(), checked) ||
        std::fabs(double(checked) - oracle) > std::max(1e-4, 0.003 * l1)) {
        std::fprintf(stderr, "FAIL benchmark oracle parity n=%lld got=%.9g oracle=%.9g l1=%.9g\n",
                     static_cast<long long>(n), checked, oracle, l1);
        return 1;
    }
    double max_ggml_delta = 0.0, sum_sq_ggml_delta = 0.0;
    for (size_t row = 0; row < row_count; ++row) {
        float selected = 0.f, ggml_dot = 0.f;
        const uint8_t* row_weights = weights.data() + row * format.weight_row_bytes;
        if (!dsv4::native_cpu_dot(format, row_weights, activation.data(), selected)) return 1;
        traits->vec_dot(static_cast<int>(n), &ggml_dot, 0, row_weights, 0, activation.data(), 0, 1);
        const double delta = double(selected) - ggml_dot;
        max_ggml_delta = std::max(max_ggml_delta, std::fabs(delta));
        sum_sq_ggml_delta += delta * delta;
    }

    volatile double checksum = 0.0;
    auto run_round = [&] {
        double sum = 0.0;
        for (size_t row = 0; row < row_count; ++row) {
            float dot = 0.f;
            if (!dsv4::native_cpu_dot(format, weights.data() + row * format.weight_row_bytes,
                                      activation.data(), dot)) return false;
            sum += dot;
        }
        checksum = checksum + sum;
        return true;
    };
    for (int i = 0; i < warmup_rounds; ++i) {
        if (!run_round()) return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < timed_rounds; ++i) {
        if (!run_round()) return 1;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    const char* policy = std::getenv("DSV4_CPU_IQ1M_KERNEL");
    std::printf("IQ1_M row-dot benchmark: n=%lld rows=%zu rounds=%d policy=%s %s total=%.3f ms ns/row=%.1f checksum=%.9g oracle_error=%.4g ggml_max_abs=%.9g ggml_rms=%.9g\n",
                static_cast<long long>(n), row_count, timed_rounds,
                policy && *policy ? policy : "auto", dsv4::native_cpu_description().c_str(), elapsed_ms,
                elapsed_ms * 1.0e6 / double(row_count * timed_rounds), double(checksum),
                std::fabs(double(checked) - oracle), max_ggml_delta,
                std::sqrt(sum_sq_ggml_delta / double(row_count)));
    return 0;
}

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
            uint16_t scales[4];
            for (uint16_t& scale : scales) scale = uint16_t(rng() & 0x0fff);
            // Encode global f16 scale 0x3000 in the high nibbles while varying all 16
            // local 3-bit scales; random qh bytes below also exercise every delta sign.
            scales[3] |= 0x3000;
            std::memcpy(w.data() + off + 48, scales, sizeof(scales));
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
    double ggml_delta = 0.0;
    size_t q8_min_count = 0;
    if (type == dsv4::T_IQ1_M) {
        const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_M);
        if (!traits || !traits->vec_dot) {
            std::fprintf(stderr, "FAIL resolving ggml AVX2 IQ1_M row-dot\n");
            ++failures;
            return;
        }
        float ggml_dot = 0.f;
        traits->vec_dot(static_cast<int>(n), &ggml_dot, 0, w.data(), 0, act.data(), 0, 1);
        ggml_delta = double(got) - ggml_dot;
        for (int64_t i = 0; i < n; ++i)
            q8_min_count += static_cast<int8_t>(act[4 + static_cast<size_t>(i)]) == -128;
#if defined(DSV4_NATIVE_CPU_AVX512)
        const char* policy = std::getenv("DSV4_CPU_IQ1M_KERNEL");
        if (policy && std::strcmp(policy, "avx512") == 0 && dsv4::native_cpu_avx512_available()) {
            avx512_vs_ggml_max = std::max(avx512_vs_ggml_max, std::fabs(ggml_delta));
            avx512_vs_ggml_sum_sq += ggml_delta * ggml_delta;
            ++avx512_vs_ggml_rows;
            real_q8_min_count += q8_min_count;
        }
#endif
    }
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
        if (type == dsv4::T_IQ1_M) {
            std::printf("ok native_cpu type=%u activation=%u ggml-vs-quantized-float=%.4g quant-vs-float-input=%.4g l1=%.4g selected-vs-ggml=%.4g q8_min_count=%zu\n",
                        type, fmt.activation_type, impl_error, err_abs, l1, ggml_delta, q8_min_count);
        } else {
            std::printf("ok native_cpu type=%u activation=%u ggml-vs-quantized-float=%.4g quant-vs-float-input=%.4g l1=%.4g\n",
                        type, fmt.activation_type, impl_error, err_abs, l1);
        }
    }
}

#if defined(DSV4_NATIVE_CPU_AVX512)
void check_avx512_iq1m_edges() {
    constexpr int64_t n = 256;
    dsv4::NativeCpuFormat fmt;
    std::string err;
    if (!dsv4::native_cpu_format(dsv4::T_IQ1_M, n, fmt, err)) {
        std::fprintf(stderr, "FAIL AVX512 edge format: %s\n", err.c_str());
        ++failures;
        return;
    }
    if (!dsv4::native_cpu_avx512_available()) {
        std::puts("ok AVX512 edge dispatch skipped; guarded AVX2 ggml fallback remains selected");
        return;
    }

    std::mt19937 rng(0xA512);
    std::vector<uint8_t> weights(fmt.weight_row_bytes), activation(fmt.activation_bytes);
    std::vector<float> decoded_weights((size_t)n), decoded_activation((size_t)n);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (uint8_t& byte : weights) byte = (uint8_t)byte_dist(rng);
    for (uint8_t& byte : activation) byte = (uint8_t)byte_dist(rng);

    const uint16_t globals[] = {0x3000, 0x0000, 0xB000};
    const int8_t q8_valid_edges[] = {-127, 0, 127};
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_IQ1_M);
    if (!traits || !traits->vec_dot) {
        std::fprintf(stderr, "FAIL AVX512 edge test could not resolve ggml AVX2 row-dot\n");
        ++failures;
        return;
    }
    double max_ggml_delta = 0.0, sum_sq_ggml_delta = 0.0;
    size_t ggml_comparisons = 0;
    for (int test = 0; test < 12; ++test) {
        uint16_t scales[4];
        for (uint16_t& scale : scales) scale = uint16_t(byte_dist(rng) & 0x0fff);
        scales[3] = uint16_t((scales[3] & 0x0fff) | globals[test % 3]);
        std::memcpy(weights.data() + 48, scales, sizeof(scales));
        // Force low/high ends of the 11-bit grid and all four sign combinations.
        weights[0] = (test & 1) ? 0xff : 0x00;
        weights[32] = uint8_t(((test & 1) ? 0x07 : 0x00) |
                              ((test & 1) ? 0x08 : 0x00) |
                              ((test & 2) ? 0x80 : 0x00));
        weights[33] = uint8_t(((test & 2) ? 0x07 : 0x00) |
                              ((test & 2) ? 0x08 : 0x00) |
                              ((test & 1) ? 0x80 : 0x00));
        float activation_scale = 0.125f;
        std::memcpy(activation.data(), &activation_scale, sizeof(activation_scale));
        for (int i = 0; i < n; ++i) activation[4 + i] = uint8_t(q8_valid_edges[(i + test) % 3]);
        const bool injected_q8_min = test == 11;
        if (injected_q8_min) activation[4] = uint8_t(-128);
        // Q8_K carries per-16-element sums consumed by the row-dot implementation.
        for (int group = 0; group < n / 16; ++group) {
            int16_t sum = 0;
            for (int i = 0; i < 16; ++i)
                sum += static_cast<int8_t>(activation[4 + group * 16 + i]);
            std::memcpy(activation.data() + 4 + n + group * sizeof(sum), &sum, sizeof(sum));
        }

        float got = 0.f;
        if (!dsv4::native_cpu_dot_iq1m_avx512(weights.data(), activation.data(), n, got) || !std::isfinite(got)) {
            std::fprintf(stderr, "FAIL AVX512 IQ1_M edge dot case=%d\n", test);
            ++failures;
            continue;
        }
        float ggml_dot = 0.f;
        traits->vec_dot(static_cast<int>(n), &ggml_dot, 0, weights.data(), 0, activation.data(), 0, 1);
        if (std::memcmp(&got, &ggml_dot, sizeof(got)) != 0) {
            std::fprintf(stderr, "FAIL AVX512 IQ1_M differs from pinned ggml AVX2 case=%d got=%.9g ggml=%.9g\n",
                         test, got, ggml_dot);
            ++failures;
        }
        const double ggml_delta = double(got) - ggml_dot;
        max_ggml_delta = std::max(max_ggml_delta, std::fabs(ggml_delta));
        sum_sq_ggml_delta += ggml_delta * ggml_delta;
        ++ggml_comparisons;
        if (!injected_q8_min) {
            dsv4::dequant_row(dsv4::T_IQ1_M, weights.data(), n, decoded_weights.data());
            dsv4::dequant_row(dsv4::T_Q8_K, activation.data(), n, decoded_activation.data());
            const float oracle = dsv4::row_dot(dsv4::T_IQ1_M, weights.data(), decoded_activation.data(), n);
            double l1 = 0.0;
            for (int i = 0; i < n; ++i)
                l1 += std::fabs((double)decoded_weights[(size_t)i] * decoded_activation[(size_t)i]);
            if (std::fabs((double)got - oracle) > std::max(1e-4, 0.003 * l1)) {
                std::fprintf(stderr, "FAIL AVX512 IQ1_M canonical parity case=%d got=%.9g oracle=%.9g l1=%.9g\n",
                             test, got, oracle, l1);
                ++failures;
            }
        }
    }
    if (ggml_comparisons) {
        std::printf("AVX512 IQ1_M vs ggml AVX2 synthetic Q8_K edge dots (includes one injected -128 wrap-compatibility case): max_abs=%.9g rms=%.9g cases=%zu\n",
                    max_ggml_delta, std::sqrt(sum_sq_ggml_delta / ggml_comparisons), ggml_comparisons);
    }
    if (!failures) std::puts("ok AVX512 IQ1_M parity: independent oracle on valid q8 range; exact ggml compatibility on injected -128");
}
#endif
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench-iq1m") {
        if (argc < 3 || argc > 4) {
            std::fprintf(stderr, "usage: test_native_cpu --bench-iq1m <width divisible by 256> [rows]\n");
            return 2;
        }
        try {
            size_t parsed = 0;
            const auto n = std::stoll(argv[2], &parsed);
            if (parsed != std::strlen(argv[2])) throw std::invalid_argument("trailing chars");
            size_t rows = 2048;
            if (argc == 4) {
                size_t parsed_rows = 0;
                rows = std::stoull(argv[3], &parsed_rows);
                if (parsed_rows != std::strlen(argv[3]) || rows == 0) throw std::invalid_argument("invalid rows");
            }
            return benchmark_iq1m(n, rows);
        } catch (...) {
            std::fprintf(stderr, "invalid IQ1_M benchmark width\n");
            return 2;
        }
    }
    if (!dsv4::native_cpu_available()) {
        std::puts("SKIP native_cpu: build or host does not meet the AVX2 floor");
        return 0;
    }
    check_type(dsv4::T_IQ1_M, 256);
    check_type(dsv4::T_IQ1_M, 768);
    check_type(dsv4::T_IQ1_M, 2048);
    check_type(dsv4::T_IQ1_M, 4096);
    check_type(dsv4::T_IQ2_XXS, 4096);
    check_type(dsv4::T_IQ3_XXS, 4096);
#if defined(DSV4_NATIVE_CPU_AVX512)
    check_avx512_iq1m_edges();
    if (avx512_vs_ggml_rows) {
        std::printf("AVX512 vs ggml AVX2 on native-quantized Q8_K rows: max_abs=%.9g rms=%.9g rows=%zu q8_min_count=%zu\n",
                    avx512_vs_ggml_max, std::sqrt(avx512_vs_ggml_sum_sq / avx512_vs_ggml_rows),
                    avx512_vs_ggml_rows, real_q8_min_count);
    }
#endif
    return failures ? 1 : 0;
}
