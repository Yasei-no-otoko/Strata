// dsv4/gpu.hpp - interchangeable device backends behind one API:
//   src/gpu.cpp       CPU-emulated device (malloc "VRAM", memcpy H2D/D2H, ggml-identical matvecs).
//                     Always buildable, and what the tests run against.
//   src/gpu_cuda.cu   real CUDA device (built by build.sh --cuda / cmake -DDSV4_WITH_CUDA=ON).
//                     The same kernels target AMD with cmake -DDSV4_WITH_HIP=ON.
// Both define exactly the same dsv4::gpu symbols, so a binary links ONE of them; the loader, the memory
// plan and the HIT/MISS tier above this header never know which.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace dsv4 {
namespace gpu {

constexpr int kMaxHit = 8;  // max concurrent resident (HIT) experts per MoE layer

// Pointers into the VRAM expert pool: one slot is [gate | up | down], so the three rows are offsets
// of the same base. w[] is the routing weight already applied by the caller's combine step.
//
// An expert may also be given through the HOST pointers hg/hu/hd. experts_hit() then stages it into
// the device pool registered by staging() and evaluates it on the device like a resident one. That is
// an alternative to the scalar float-reference host path for a MISS. Native quantized CPU
// kernels provide another option; which is faster depends on the CPU and transfer costs.
struct ExpPtrs {
    int n = 0;
    uint8_t* gate[kMaxHit] = {};
    uint8_t* up[kMaxHit] = {};
    uint8_t* down[kMaxHit] = {};
    const uint8_t* hg[kMaxHit] = {};
    const uint8_t* hu[kMaxHit] = {};
    const uint8_t* hd[kMaxHit] = {};
    float w[kMaxHit] = {};
    size_t bpe = 0;
    // Shared experts use the host swiglu_clamped double-exp arithmetic; routed experts keep
    // the faster device float-exp path unless this opt-in field is set.
    bool reference_activation = false;
};

/// Register the device scratch that host experts are staged through. Returns false, and leaves
/// staging unset, if the device cannot take it: experts_hit() then refuses host experts and the
/// caller keeps computing them on the host.
bool staging(uint8_t* device_pool, size_t bytes);

/// True when host experts can be staged. The CPU-emulated backend answers true with no pool at all:
/// its "device" memory is host memory, so staging would be a pointless copy.
bool has_staging();

bool is_emulated();
/// Backend compiled into this binary, also used by logs and the serving protocol.
const char* backend_name();
bool init(std::string& err);
void* alloc(size_t n);
void release(void* p);
void mem_info(size_t* free_bytes, size_t* total_bytes);

/// Can THIS backend compute over weights of ggml type `type`? The emulated device answers yes for every
/// type src/dequant.cpp can decode; the CUDA device answers yes only for the types it has a kernel for.
/// The loader uses this to avoid uploading weights no kernel could read, and to keep those matvecs on the
/// host instead of silently producing zeros.
bool type_supported(uint32_t type);

/// True only when all three matrices of a MoE layer have a kernel. That is the one case where giving the
/// layer VRAM slots is worth anything: a HIT expert is evaluated by experts_hit() and by nothing else.
bool experts_supported(uint32_t type_gate, uint32_t type_up, uint32_t type_down);

// Funzioni di copia memoria e calcolo dei kernel
void h2d(void* dst, const void* src, size_t bytes);
void d2h(void* dst, const void* src, size_t bytes);

/// y = W * x. False means this backend has no kernel for `type` (see type_supported) and wrote NOTHING:
/// the caller must fall back to the host. Never a partial or zero result.
bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y);

/// Batched y[g, :] = W[g, :, :] * x[g, :]. W is a contiguous sequence of `groups` row-major
/// matrices, each with `rows_per_group` rows and `in` elements per row. x and y are contiguous
/// group-major device buffers. False means unsupported/invalid input and writes nothing.
bool matvec_grouped(uint32_t type, const uint8_t* W, int64_t groups, int64_t rows_per_group, int64_t in,
                    const float* d_x, float* d_y);

/// Fused attention projections on a shared input. Workspace holds packed output
/// [q_rows | kv_rows | q_lora] followed by q_lora temporary floats. q_a and kv
/// both read d_x; q_a is weighted-RMS-normalized before q_b. The caller copies
/// the packed prefix back once after this asynchronous operation succeeds.
/// False means invalid geometry or an unsupported backend/type; no fallback
/// work is performed, so callers may run their existing reference path.
bool attention_qkv_bundle(uint32_t q_a_type, const uint8_t* q_a_W, int64_t q_lora, int64_t dim,
                          uint32_t kv_type, const uint8_t* kv_W, int64_t kv_rows,
                          uint32_t q_b_type, const uint8_t* q_b_W, int64_t q_rows,
                          const float* q_a_norm, float eps, const float* d_x,
                          float* d_workspace, size_t workspace_floats);

/// One MoE layer's HIT experts, all resident in VRAM:
///   d_g[k] = Wg[k] * d_x ; d_u[k] = Wu[k] * d_x ; d_a[k] = w_k * swiglu_clamped(d_g[k], d_u[k], limit)
///   d_y    = sum_k Wd[k] * d_a[k]
/// Scratch must hold kMaxHit * ff (d_g, d_u, d_a) and dim (d_y) floats.
/// False: one of the three types has no kernel and d_y was left untouched (see experts_supported).
/// Launches are asynchronous with respect to the host, so the caller can compute the MISS experts on the
/// CPU in the meantime and only pay for the device when it reads d_y back with d2h().
bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y);

}  // namespace gpu
}  // namespace dsv4
