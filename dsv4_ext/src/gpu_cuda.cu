// dsv4/gpu_cuda.cu - the REAL CUDA device layer: the same dsv4::gpu API as src/gpu.cpp, compiled INSTEAD
// of it when the build asks for CUDA (build.sh --cuda, or cmake -DDSV4_WITH_CUDA=ON).
//
// What lives here:
//   * device memory, H2D/D2H, and the driver's real free/total VRAM;
//   * one dequantise-and-dot kernel per ggml type. The decode math is NOT duplicated: it is the shared
//     traits in include/dsv4/dq_traits.hpp, which tests/test_dq_traits.cpp checks element by element
//     against dequant_row()/row_dot() from src/dequant.cpp on the host. Same source, two compilations.
//   * the MoE HIT path: gate/up matvecs, a fused clamped-SwiGLU + routing-weight kernel, then the down
//     matvecs summed into d_y.
//
// Only the ggml types this model actually stores get a kernel (see supported()): Q8_0, Q4_K, Q5_K, Q6_K,
// BF16, F32, IQ3_XXS, IQ2_XXS, IQ1_M and MXFP4 - which is every tensor of DeepSeek-V4-Flash per shapes.txt.
// type_supported() reports anything else as unsupported so the loader can leave those tensors, and those MoE
// layers, on the host instead of uploading bytes no kernel could read. Adding a type = one trait in the
// header plus one case in matvec().
//
// Precision, deliberately: the host reference accumulates a dot product in double and sums left to right;
// the kernel accumulates in float and reduces as a tree. Results agree to float rounding, NOT bit for bit.
// The CPU-emulated device (src/gpu.cpp) is what the golden tests in tests/ are compared against.
#include "dsv4/gpu.hpp"

#include "dsv4/dequant.hpp"
#include "dsv4/iq_tables.hpp"

#include <cstdio>
#include <climits>
#include <cstdlib>
#include <cstring>

#include "dsv4/device_runtime.hpp"

namespace dsv4 {
namespace gpu {
namespace {

// ---------------------------------------------------------------- error handling
// A CUDA error is sticky for the rest of the context, so the first one is the interesting one: printed once,
// remembered, and every later call reports failure instead of spamming the log or dying mid-token.
bool g_cuda_err = false;

bool check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    if (!g_cuda_err) std::fprintf(stderr, "%s error at %s: %s\n", backend_name(), what, cudaGetErrorString(e));
    g_cuda_err = true;
    return false;
}

// ---------------------------------------------------------------- device state
size_t g_total = 0, g_used = 0;
bool g_ready = false;
float* g_down_acc = nullptr;   // kMaxHit * kMaxDim floats: the down projection of each HIT expert

// Staging for MISS experts: a device pool the host blocks are copied into, and a pinned host buffer
// they are copied THROUGH. Pinned matters - an unpinned cudaMemcpyAsync degrades to a staged,
// synchronised copy, and the whole point here is that the transfer queues ahead of the kernels
// instead of stalling the host.
uint8_t* g_stage = nullptr;
size_t g_stage_bytes = 0;
uint8_t* g_stage_pin = nullptr;
bool g_async_stage = false;
cudaStream_t g_stage_stream = nullptr;
cudaEvent_t g_stage_copied[kMaxHit] = {};
cudaEvent_t g_stage_consumed[kMaxHit] = {};
bool g_stage_copied_valid[kMaxHit] = {};
bool g_stage_consumed_valid[kMaxHit] = {};
size_t g_stage_last_bpe = 0;

constexpr int64_t kMaxDim = 16384;
constexpr int kThreads = 256;

void clear_staging_resources() {
    if (g_stage_pin && g_ready)
        check(cudaDeviceSynchronize(), "staging cleanup synchronize");
    for (int k = 0; k < kMaxHit; ++k) {
        if (g_stage_copied[k]) check(cudaEventDestroy(g_stage_copied[k]), "destroy stage-copy event");
        if (g_stage_consumed[k]) check(cudaEventDestroy(g_stage_consumed[k]), "destroy stage-compute event");
        g_stage_copied[k] = g_stage_consumed[k] = nullptr;
        g_stage_copied_valid[k] = g_stage_consumed_valid[k] = false;
    }
    if (g_stage_stream) check(cudaStreamDestroy(g_stage_stream), "destroy stage-copy stream");
    g_stage_stream = nullptr;
    g_async_stage = false;
    g_stage_last_bpe = 0;
    if (g_stage_pin) {
        check(cudaFreeHost(g_stage_pin), "free pinned staging buffer");
        g_stage_pin = nullptr;
    }
    g_stage = nullptr;
    g_stage_bytes = 0;
}

// ---------------------------------------------------------------- codebooks on device
// __constant__ because a warp reads the same grid entry with the same index nearly always, which is exactly
// what the constant cache broadcasts. The G_* macros below point the shared traits at these copies.
__constant__ uint64_t c_iq1s_grid[2048];
__constant__ uint64_t c_iq2xxs_grid[256];
__constant__ uint32_t c_iq3xxs_grid[256];
__constant__ uint8_t c_ksigns_iq2xs[128];
__constant__ uint8_t c_kmask_iq2xs[8];

}  // namespace
}  // namespace gpu
}  // namespace dsv4

// The shared decode traits: with these macros they read device constant memory here, and the plain
// dsv4::iq:: tables when the same header is compiled for the host by tests/test_dq_traits.cpp.
// Device-only on purpose: nothing in this file calls at() from the host, and if the traits were also
// __host__ nvcc compiles a host copy that reads __constant__ memory, which it warns about (#20091-D).
#define DSV4_HOST_DEVICE __device__ __forceinline__
#define G_iq1s_grid dsv4::gpu::c_iq1s_grid
#define G_iq2xxs_grid dsv4::gpu::c_iq2xxs_grid
#define G_iq3xxs_grid dsv4::gpu::c_iq3xxs_grid
#define G_ksigns_iq2xs dsv4::gpu::c_ksigns_iq2xs
#define G_kmask_iq2xs dsv4::gpu::c_kmask_iq2xs
#include "dsv4/dq_traits.hpp"

namespace dsv4 {
namespace gpu {
namespace {

// ---------------------------------------------------------------- matvec kernel
// One block per output row, kThreads threads: each thread walks a strided slice of the row, decoding the
// weights on the fly (the weights are the traffic here, not the activations). rows is gridDim.x, so the
// 129280-row output projection is a single launch.
template <class Tr>
__global__ void matvec_kernel(const uint8_t* __restrict__ W, int64_t rows, int64_t rows_per_group, int in, int rb,
                              const float* __restrict__ x, float* __restrict__ y) {
    const int64_t r = blockIdx.x;
    if (r >= rows) return;
    // The byte stride comes from row_bytes() on the host, the single source of truth shared with
    // dequant.cpp. Checking it against the trait's block size means a trait that disagrees with the type it
    // claims to decode cannot read past the row.
    if (in <= 0 || in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) {
        if (threadIdx.x == 0) y[r] = 0.f;
        return;
    }
    const uint8_t* w = W + (size_t) r * (size_t) rb;
    const float* xg = x + (size_t)(r / rows_per_group) * (size_t)in;
    float acc = 0.f;
    for (int e = threadIdx.x; e < in; e += kThreads) acc += Tr::at(w, e) * xg[e];
    __shared__ float red[kThreads];
    red[threadIdx.x] = acc;
    __syncthreads();
#ifdef DSV4_USE_HIP
    // gfx1030 is wave32. Keep the original balanced-tree association exactly: shared-memory steps
    // combine across waves first (128, 64, 32), then wave shuffles do the original 16..1 steps in
    // lane zero's tree. This removes five CTA barriers without changing the summation order.
    const int wave = (int)warpSize;
    for (int s = kThreads / 2; s >= wave; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < wave) {
        float total = red[threadIdx.x];
        const int lane = (int)threadIdx.x;
        for (int offset = wave / 2; offset > 0; offset >>= 1) {
            // Every lane in the first wave must participate in the shuffle. Only the lower half
            // consumes the partner, matching the original tree's active-lane additions.
#ifdef DSV4_USE_HIP
            const float other = __shfl_down(total, offset, wave);
#else
            const float other = __shfl_down_sync(0xffffffffu, total, offset, wave);
#endif
            if (lane < offset) total += other;
        }
        if (lane == 0) y[r] = total;
    }
#else
    #pragma unroll
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] = red[0];
#endif
}

// ---------------------------------------------------------------- MoE kernels
// a = w_k * swiglu_clamped(g, u, limit). The routing weight is folded in here so the down projection needs
// no scaling pass of its own. w_k is passed BY VALUE: p.w is a host array and must never be read here.
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ a,
                              int64_t n, float limit, float w_k, bool reference_activation) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xa = g[i], ub = u[i];
    if (limit > 0.f) { xa = fminf(xa, limit); ub = fminf(fmaxf(ub, -limit), limit); }
    if (reference_activation) {
        // Preserve src/ops.cpp ordering for the shared expert: double exp/sigmoid, cast
        // sigmoid to float, multiply by up, then apply the (shared=1) expert weight.
        const float sigmoid = (float)(double(xa) / (1.0 + exp(-double(xa))));
        const float activated = sigmoid * ub;
        a[i] = activated * w_k;
    } else {
        a[i] = w_k * (xa / (1.f + expf(-xa)) * ub);
    }
}

// d_y = sum_k tmp[k] in a fixed order: deterministic, and cheap (dim floats).
__global__ void sum_rows_kernel(const float* __restrict__ tmp, int n, int64_t dim, float* __restrict__ y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float s = 0.f;
    for (int k = 0; k < n; ++k) s += tmp[(size_t) k * (size_t) dim + i];
    y[i] = s;
}

// ---------------------------------------------------------------- dispatch
template <class Tr>
bool launch_mv(uint32_t type, const uint8_t* W, int64_t groups, int64_t rows_per_group, int64_t in,
               const float* d_x, float* d_y) {
    const size_t rb = row_bytes(type, in);
    if (!rb || groups <= 0 || rows_per_group <= 0 || in <= 0 || in > (1 << 30) ||
        groups > INT64_MAX / rows_per_group) return false;
    const int64_t rows = groups * rows_per_group;
    if ((uint64_t)rows > UINT_MAX) return false;
    // Checked HERE, on the host, before anything is launched: row_bytes() is the single source of truth
    // shared with dequant.cpp, and if a trait's block size disagrees with the type it claims to decode,
    // the honest answer is false ("wrote nothing, use the host") - not a row of zeros on the device.
    if (in % Tr::BE != 0 || (int64_t)(in / Tr::BE) * (int64_t) Tr::BB != (int64_t) rb) return false;
    matvec_kernel<Tr><<<(unsigned) rows, kThreads>>>(W, rows, rows_per_group, (int) in, (int) rb, d_x, d_y);
    return check(cudaGetLastError(), "matvec launch");
}

/// Every ggml type this file has a kernel for. One table, so type_supported(), experts_supported() and
/// matvec() can never disagree about what is safe to put on the device.
bool supported(uint32_t type) {
    switch (type) {
        case T_F32: case T_BF16: case T_Q8_0: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ3_XXS: case T_IQ2_XXS: case T_IQ1_M: case T_MXFP4:
            return true;
        default: return false;
    }
}

}  // namespace

// ================================================================================================
bool is_emulated() { return false; }
const char* backend_name() {
#ifdef DSV4_USE_HIP
    return "hip";
#else
    return "cuda";
#endif
}

bool init(std::string& err) {
    if (g_ready) return true;
    int ndev = 0;
    if (!check(cudaGetDeviceCount(&ndev), "get device count")) { err = std::string("cannot initialize ") + backend_name(); return false; }
    if (ndev == 0) { err = std::string("no ") + backend_name() + " device visible"; return false; }
    if (!check(cudaSetDevice(0), "select device")) { err = "cannot select device 0"; return false; }
    cudaDeviceProp prop{};
    if (!check(cudaGetDeviceProperties(&prop, 0), "get device properties")) { err = "cannot read device properties"; return false; }
#ifdef DSV4_USE_HIP
    std::fprintf(stderr, "dsv4: HIP device 0: %s (arch %s, wave%d)\n", prop.name, prop.gcnArchName, prop.warpSize);
#else
    std::fprintf(stderr, "dsv4: CUDA device 0: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
#endif
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo")) { err = "cannot read VRAM size"; return false; }
    g_total = total_b; g_used = 0;

    // Codebooks: the same bytes src/iq_tables.cpp gives the host decoder.
    bool ok = true;
    ok &= check(cudaMemcpyToSymbol(c_iq1s_grid, iq::iq1s_grid, sizeof(iq::iq1s_grid)), "copy iq1s_grid");
    ok &= check(cudaMemcpyToSymbol(c_iq2xxs_grid, iq::iq2xxs_grid, sizeof(iq::iq2xxs_grid)), "copy iq2xxs_grid");
    ok &= check(cudaMemcpyToSymbol(c_iq3xxs_grid, iq::iq3xxs_grid, sizeof(iq::iq3xxs_grid)), "copy iq3xxs_grid");
    ok &= check(cudaMemcpyToSymbol(c_ksigns_iq2xs, iq::ksigns_iq2xs, sizeof(iq::ksigns_iq2xs)), "copy ksigns_iq2xs");
    ok &= check(cudaMemcpyToSymbol(c_kmask_iq2xs, iq::kmask_iq2xs, sizeof(iq::kmask_iq2xs)), "copy kmask_iq2xs");
    if (!ok) { err = "cannot upload the IQ codebooks"; return false; }

    if (!check(cudaMalloc((void**) &g_down_acc, (size_t) kMaxHit * kMaxDim * sizeof(float)), "alloc down scratch")) {
        err = "cannot allocate the MoE down-projection scratch"; g_down_acc = nullptr; return false;
    }
    g_ready = true;
    return true;
}

void* alloc(size_t n) {
    if (!g_ready || g_cuda_err || g_used + n > g_total) return nullptr;
    void* p = nullptr;
    if (!check(cudaMalloc(&p, n), "cudaMalloc")) return nullptr;
    g_used += n;
    return p;
}

void release(void* p) {
    if (!p) return;
    if (p == g_stage) clear_staging_resources();
    check(cudaFree(p), "free");
}  // g_used is never decreased: the pool is freed with the model

void mem_info(size_t* free_bytes, size_t* total_bytes) {
    size_t f = 0, t = 0;
    if (g_ready && check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo")) {
        if (free_bytes) *free_bytes = f;
        if (total_bytes) *total_bytes = t;
    } else {
        if (free_bytes) *free_bytes = g_ready && g_total > g_used ? g_total - g_used : 0;
        if (total_bytes) *total_bytes = g_ready ? g_total : 0;
    }
}

bool type_supported(uint32_t type) { return g_ready && !g_cuda_err && supported(type); }
bool experts_supported(uint32_t g, uint32_t u, uint32_t d) {
    return g_ready && !g_cuda_err && supported(g) && supported(u) && supported(d);
}

namespace {

void configure_async_staging() {
#ifdef DSV4_USE_HIP
    const char* enabled = std::getenv("DSV4_HIP_ASYNC_STAGE");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;

    cudaError_t status = cudaStreamCreateWithFlags(&g_stage_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "dsv4: async HIP staging unavailable (%s); using ordered staging\n", cudaGetErrorString(status));
        g_stage_stream = nullptr;
        return;
    }
    int events_created = 0;
    for (int k = 0; k < kMaxHit; ++k) {
        status = cudaEventCreateWithFlags(&g_stage_copied[k], cudaEventDisableTiming);
        if (status != cudaSuccess) break;
        status = cudaEventCreateWithFlags(&g_stage_consumed[k], cudaEventDisableTiming);
        if (status != cudaSuccess) {
            (void)cudaEventDestroy(g_stage_copied[k]);
            g_stage_copied[k] = nullptr;
            break;
        }
        ++events_created;
    }
    if (events_created != kMaxHit) {
        for (int k = 0; k < events_created; ++k) {
            (void)cudaEventDestroy(g_stage_copied[k]);
            (void)cudaEventDestroy(g_stage_consumed[k]);
            g_stage_copied[k] = g_stage_consumed[k] = nullptr;
        }
        (void)cudaStreamDestroy(g_stage_stream);
        g_stage_stream = nullptr;
        std::fprintf(stderr, "dsv4: async HIP staging event creation failed (%s); using ordered staging\n",
                     cudaGetErrorString(status));
        return;
    }
    g_async_stage = true;
    std::fprintf(stderr, "dsv4: HIP async expert staging enabled\n");
#endif
}

}  // namespace

bool staging(uint8_t* device_pool, size_t bytes) {
    if (!g_ready || g_cuda_err || !device_pool || bytes == 0) return false;
    if (g_stage || g_stage_pin || g_stage_stream) clear_staging_resources();
    void* pin = nullptr;
    if (!check(cudaHostAlloc(&pin, bytes, cudaHostAllocDefault), "pinned staging buffer")) return false;
    g_stage = device_pool;
    g_stage_bytes = bytes;
    g_stage_pin = (uint8_t*) pin;
    configure_async_staging();
    return true;
}

bool has_staging() { return g_stage != nullptr && !g_cuda_err; }

void h2d(void* dst, const void* src, size_t bytes) {
    if (!bytes || g_cuda_err) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "H2D copy");
}
void d2h(void* dst, const void* src, size_t bytes) {
    if (!bytes || g_cuda_err) return;
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "D2H copy");  // also the sync point of a launch
}

bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y) {
    if (!g_ready || g_cuda_err || !supported(type)) return false;
    switch (type) {
        case T_F32:     return launch_mv<dqt::F32T>(type, W, 1, rows, in, d_x, d_y);
        case T_BF16:    return launch_mv<dqt::BF16T>(type, W, 1, rows, in, d_x, d_y);
        case T_Q8_0:    return launch_mv<dqt::Q8_0T>(type, W, 1, rows, in, d_x, d_y);
        case T_Q4_K:    return launch_mv<dqt::Q4_KT>(type, W, 1, rows, in, d_x, d_y);
        case T_Q5_K:    return launch_mv<dqt::Q5_KT>(type, W, 1, rows, in, d_x, d_y);
        case T_Q6_K:    return launch_mv<dqt::Q6_KT>(type, W, 1, rows, in, d_x, d_y);
        case T_IQ3_XXS: return launch_mv<dqt::IQ3_XXST>(type, W, 1, rows, in, d_x, d_y);
        case T_IQ2_XXS: return launch_mv<dqt::IQ2_XXST>(type, W, 1, rows, in, d_x, d_y);
        case T_IQ1_M:   return launch_mv<dqt::IQ1_MT>(type, W, 1, rows, in, d_x, d_y);
        case T_MXFP4:   return launch_mv<dqt::MXFP4T>(type, W, 1, rows, in, d_x, d_y);
        default: return false;
    }
}

bool matvec_grouped(uint32_t type, const uint8_t* W, int64_t groups, int64_t rows_per_group, int64_t in,
                    const float* d_x, float* d_y) {
    if (!g_ready || g_cuda_err || !supported(type)) return false;
    switch (type) {
        case T_F32:     return launch_mv<dqt::F32T>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_BF16:    return launch_mv<dqt::BF16T>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_Q8_0:    return launch_mv<dqt::Q8_0T>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_Q4_K:    return launch_mv<dqt::Q4_KT>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_Q5_K:    return launch_mv<dqt::Q5_KT>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_Q6_K:    return launch_mv<dqt::Q6_KT>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_IQ3_XXS: return launch_mv<dqt::IQ3_XXST>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_IQ2_XXS: return launch_mv<dqt::IQ2_XXST>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_IQ1_M:   return launch_mv<dqt::IQ1_MT>(type, W, groups, rows_per_group, in, d_x, d_y);
        case T_MXFP4:   return launch_mv<dqt::MXFP4T>(type, W, groups, rows_per_group, in, d_x, d_y);
        default: return false;
    }
}

bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y) {
    if (p.n <= 0) return true;
    if (!g_ready || g_cuda_err) return false;
    if (!supported(type_g) || !supported(type_u) || !supported(type_d)) return false;
    if (dim > kMaxDim) return false;   // the down scratch was sized for kMaxDim at init

    const int nk = p.n < kMaxHit ? p.n : kMaxHit;

    // Resolve and validate every expert before enqueueing work. Host experts use a distinct pinned
    // segment per slot so the CPU can fill expert k+1 while the default stream computes expert k.
    // H2D and kernels are still ordered on that stream; this overlaps host staging memcpy with device
    // compute without claiming that the transfers themselves overlap the kernels.
    uint8_t* gp[kMaxHit];
    uint8_t* gu[kMaxHit];
    uint8_t* gd[kMaxHit];
    bool host_stage[kMaxHit] = {};
    bool has_host_stage = false;
    const size_t bg = row_bytes(type_g, dim) * (size_t) ff;
    const size_t bu = row_bytes(type_u, dim) * (size_t) ff;
    const size_t bd = row_bytes(type_d, ff) * (size_t) dim;
    if (!bg || !bu || !bd) return false;
    for (int k = 0; k < nk; ++k) {
        gp[k] = p.gate[k];
        gu[k] = p.up[k];
        gd[k] = p.down[k];
        if (gp[k] || gu[k] || gd[k]) {
            if (!gp[k] || !gu[k] || !gd[k]) return false;
            continue;
        }
        if (!p.hg[k] || !p.hu[k] || !p.hd[k]) return false;
        if (p.bpe < bg + bu + bd || (size_t) (k + 1) * p.bpe > g_stage_bytes) return false;
        uint8_t* dst = g_stage + (size_t) k * p.bpe;
        gp[k] = dst;
        gu[k] = dst + bg;
        gd[k] = dst + bg + bu;
        host_stage[k] = true;
        has_host_stage = true;
    }

    if (g_async_stage && has_host_stage) {
        // Layer bpe can change between consecutive experts_hit calls. Their k*bpe ranges can then
        // overlap different prior slots, so fence every prior reader/copy before changing strides.
        if (g_stage_last_bpe != 0 && g_stage_last_bpe != p.bpe) {
            for (int k = 0; k < kMaxHit; ++k) {
                if (g_stage_copied_valid[k] &&
                    !check(cudaEventSynchronize(g_stage_copied[k]), "wait copies before stage-stride change")) return false;
            }
            for (int k = 0; k < kMaxHit; ++k) {
                if (g_stage_consumed_valid[k] &&
                    !check(cudaStreamWaitEvent(g_stage_stream, g_stage_consumed[k], 0), "wait compute before stage-stride change")) return false;
            }
        }
        g_stage_last_bpe = p.bpe;
    }

    for (int k = 0; k < nk; ++k) {
        if (host_stage[k]) {
            uint8_t* pin = g_stage_pin + (size_t) k * p.bpe;
            if (g_async_stage) {
                // The pinned source and device slot are recycled on later tokens. Do not overwrite
                // either until its previous asynchronous reader has finished.
                if (g_stage_copied_valid[k] && !check(cudaEventSynchronize(g_stage_copied[k]), "wait prior stage copy")) return false;
                if (g_stage_consumed_valid[k] &&
                    !check(cudaStreamWaitEvent(g_stage_stream, g_stage_consumed[k], 0), "wait prior stage compute")) return false;
            }
            std::memcpy(pin, p.hg[k], bg);
            std::memcpy(pin + bg, p.hu[k], bu);
            std::memcpy(pin + bg + bu, p.hd[k], bd);
            const cudaStream_t copy_stream = g_async_stage ? g_stage_stream : nullptr;
            if (!check(cudaMemcpyAsync(gp[k], pin, bg + bu + bd, cudaMemcpyHostToDevice, copy_stream), "stage H2D")) return false;
            if (g_async_stage) {
                if (!check(cudaEventRecord(g_stage_copied[k], g_stage_stream), "record stage copy")) return false;
                g_stage_copied_valid[k] = true;
                if (!check(cudaStreamWaitEvent(nullptr, g_stage_copied[k], 0), "wait stage copy")) return false;
            }
        }
        if (!matvec(type_g, gp[k], ff, dim, d_x, d_g + (size_t) k * ff)) return false;
        if (!matvec(type_u, gu[k], ff, dim, d_x, d_u + (size_t) k * ff)) return false;
        swiglu_kernel<<<(unsigned) ((ff + 255) / 256), 256>>>(d_g + (size_t) k * ff, d_u + (size_t) k * ff,
                                                              d_a + (size_t) k * ff, ff, swiglu_limit, p.w[k],
                                                              p.reference_activation);
        if (!check(cudaGetLastError(), "swiglu launch")) return false;
        if (!matvec(type_d, gd[k], dim, ff, d_a + (size_t) k * ff, g_down_acc + (size_t) k * dim)) return false;
        if (host_stage[k] && g_async_stage) {
            if (!check(cudaEventRecord(g_stage_consumed[k], nullptr), "record stage compute")) return false;
            g_stage_consumed_valid[k] = true;
        }
    }

    sum_rows_kernel<<<(unsigned) ((dim + 255) / 256), 256>>>(g_down_acc, nk, dim, d_y);
    return check(cudaGetLastError(), "sum_rows launch");
}

}  // namespace gpu
}  // namespace dsv4
