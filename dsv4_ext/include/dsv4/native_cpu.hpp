// Opt-in ggml-cpu activation quantisation and row-dot adapter for MoE host work.
// The quantized activation arithmetic intentionally differs from dsv4::row_dot's float reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

struct NativeCpuFormat {
    uint32_t weight_type = 0;
    uint32_t activation_type = 0;
    int64_t n = 0;
    size_t weight_row_bytes = 0;
    size_t activation_bytes = 0;
    const void* _ops = nullptr;  // opaque, process-lifetime ggml-cpu trait record
};

/// True only in a DSV4_NATIVE_CPU build and when this CPU/OS provides the configured ISA floor.
bool native_cpu_available() noexcept;
/// True when the optional IQ1_M override's AVX-512F/BW/VNNI ISA and ZMM OS state are available.
bool native_cpu_avx512_available() noexcept;
/// Build/version/ISA information suitable for a startup diagnostic.
std::string native_cpu_description();

/// Resolve ggml-cpu's row-dot and activation quantizer for a GGUF type and row width.
bool native_cpu_format(uint32_t weight_type, int64_t n, NativeCpuFormat& out, std::string& err);

/// Quantize one float activation with the format's ggml vec_dot_type. The vector is dynamically sized
/// for this layer; callers may prepare once and reuse it across all rows with the same format.
bool native_cpu_prepare(const NativeCpuFormat& format, const float* x, std::vector<uint8_t>& activation,
                        std::string& err);

/// Compute one row with ggml-cpu's vec_dot. `activation` must be prepared with the same format.
bool native_cpu_dot(const NativeCpuFormat& format, const uint8_t* weight_row, const uint8_t* activation,
                    float& result) noexcept;

/// AVX-512 VNNI IQ1_M row dot; callers must check native_cpu_avx512_available() first.
bool native_cpu_dot_iq1m_avx512(const uint8_t* weight_row, const uint8_t* activation, int64_t n,
                                float& result) noexcept;

}  // namespace dsv4
