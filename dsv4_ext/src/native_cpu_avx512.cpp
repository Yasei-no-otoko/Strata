// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors
// IQ1_M x Q8_K layout and accumulation order adapt pinned ggml CPU code
// (ggml/src/ggml-cpu/quants.c); see its LICENSE and the repository LICENSE.

#include "dsv4/iq_tables.hpp"
#include "dsv4/native_cpu.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include <immintrin.h>

#include <cstring>
#include <limits>

namespace dsv4 {
namespace {

float hsum_float_8(__m256 value) noexcept {
    __m128 sum = _mm256_extractf128_ps(value, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(value));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

}  // namespace

bool native_cpu_dot_iq1m_avx512(const uint8_t* weight_row, const uint8_t* activation, int64_t n,
                                float& result) noexcept {
    static_assert(sizeof(block_iq1_m) == 56);
    static_assert(sizeof(block_q8_K) == 292);
    static_assert(sizeof(uint64_t) == 8);
    static_assert(std::numeric_limits<uint8_t>::is_modulo);
    if (!weight_row || !activation || n <= 0 || n % QK_K != 0) return false;

    const auto* weights = reinterpret_cast<const block_iq1_m*>(weight_row);
    const auto* activations = reinterpret_cast<const block_q8_K*>(activation);
    const int64_t blocks = n / QK_K;

    const __m256i mask7 = _mm256_set1_epi16(0x7);
    const __m256i one16 = _mm256_set1_epi16(1);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i two8 = _mm256_set1_epi8(2);
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);
    const __m256i scale_index1 = _mm256_set1_epi16(0x0100);
    const __m256i scale_index2 = _mm256_add_epi8(scale_index1, _mm256_set1_epi8(8));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i scale_byte_index = _mm256_setzero_si256();

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int64_t block = 0; block < blocks; ++block) {
        const block_iq1_m& w = weights[block];
        const block_q8_K& a = activations[block];
        uint16_t scale_words[4]{};
        std::memcpy(scale_words, w.scales, sizeof(scale_words));
        const uint16_t global_scale = uint16_t((scale_words[0] >> 12) |
                                               ((scale_words[1] >> 8) & 0x00f0) |
                                               ((scale_words[2] >> 4) & 0x0f00) |
                                               (scale_words[3] & 0xf000));

        uint64_t packed_scales = 0;
        std::memcpy(&packed_scales, w.scales, sizeof(packed_scales));
        __m256i scales = _mm256_set1_epi64x(static_cast<int64_t>(packed_scales));
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask7), 1), one16);
        __m256i scales_idx1 = scale_index1;
        __m256i scales_idx2 = scale_index2;
        __m256i sumi1 = zero;
        __m256i sumi2 = zero;

        const int8_t* q8 = a.qs;
        const uint8_t* qs = w.qs;
        const uint8_t* qh = w.qh;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m256i q1b1 = _mm256_set_epi64x(
                static_cast<int64_t>(iq::iq1s_grid[qs[3] | ((uint16_t(qh[1]) << 4) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[2] | ((uint16_t(qh[1]) << 8) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[1] | ((uint16_t(qh[0]) << 4) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[0] | ((uint16_t(qh[0]) << 8) & 0x700)]));
            const __m256i q1b2 = _mm256_set_epi64x(
                static_cast<int64_t>(iq::iq1s_grid[qs[7] | ((uint16_t(qh[3]) << 4) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[6] | ((uint16_t(qh[3]) << 8) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[5] | ((uint16_t(qh[2]) << 4) & 0x700)]),
                static_cast<int64_t>(iq::iq1s_grid[qs[4] | ((uint16_t(qh[2]) << 8) & 0x700)]));

            const __m256i delta1 = _mm256_set_epi64x(
                qh[1] & 0x80 ? -1LL : 0x0101010101010101LL,
                qh[1] & 0x08 ? -1LL : 0x0101010101010101LL,
                qh[0] & 0x80 ? -1LL : 0x0101010101010101LL,
                qh[0] & 0x08 ? -1LL : 0x0101010101010101LL);
            const __m256i delta2 = _mm256_set_epi64x(
                qh[3] & 0x80 ? -1LL : 0x0101010101010101LL,
                qh[3] & 0x08 ? -1LL : 0x0101010101010101LL,
                qh[2] & 0x80 ? -1LL : 0x0101010101010101LL,
                qh[2] & 0x08 ? -1LL : 0x0101010101010101LL);

            const __m256i q8b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
            q8 += 32;
            const __m256i q8b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8));
            q8 += 32;

            const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);
            const __m256i scale_bytes1 = _mm256_shuffle_epi8(scale1, scale_byte_index);
            const __m256i scale_bytes2 = _mm256_shuffle_epi8(scale2, scale_byte_index);

            // Each YMM half has one odd IQ1_M scale for its 16 Q8 values.
            // Weighted 4-byte VNNI sums match ggml's pairwise maddubs+madd lanes.
            const __m256i signed_q1_1 = _mm256_sign_epi8(q8b1, q1b1);
            const __m256i signed_q1_2 = _mm256_sign_epi8(q8b2, q1b2);
            const __m256i dot1 = _mm256_dpbusd_epi32(zero, scale_bytes1, signed_q1_1);
            const __m256i dot2 = _mm256_dpbusd_epi32(zero, scale_bytes2, signed_q1_2);
            const __m256i dot3 = _mm256_dpbusd_epi32(zero, scale_bytes1, _mm256_sign_epi8(q8b1, delta1));
            const __m256i dot4 = _mm256_dpbusd_epi32(zero, scale_bytes2, _mm256_sign_epi8(q8b2, delta2));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(dot1, dot2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(dot3, dot4));
            scales_idx1 = _mm256_add_epi8(scales_idx1, two8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, two8);
            qs += 8;
            qh += 4;
        }

        const __m256 d = _mm256_set1_ps(a.d * ggml_fp16_to_fp32(global_scale));
        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    result = hsum_float_8(accum1) + 0.125f * hsum_float_8(accum2);
    return true;
}

}  // namespace dsv4
