#pragma once

#include "dsv4/config.hpp"
#include "dsv4/mem_plan.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace dsv4 {

// Definizione completa di RunOpts per permettere a Impl di allocarlo direttamente
struct RunOpts {
    int ctx = 2048;
    bool gpu = true;
    int threads = 0;              // 0: as many as the machine has cores (see Model::load)
    bool qat_sim = true;
    int max_layers = -1;
    double vram_pct = -1.0;
    long long abs_slots = -1;
    long long reserve_mib = 1024;
    long long vram_free_mib = -1;
    long long vram_total_mib = -1;
    std::string profile_in;
    std::string profile_out;
    bool ram_all = false;
    // LRU arena for the experts that are neither in the profile's RAM copy nor in a VRAM slot. Without it every
    // MISS re-reads its 6.8 MiB of weights from the mmap'ed GGUF, which on a model bigger than RAM means disk.
    long long ram_cache_mib = 8192;
    int adapt_swaps = 0;
    bool verify_slots = false;
    bool verbose = true;          // startup report: memory plan, profile, RAM/VRAM expert counts
    bool profile_timing = false;  // optional host wall-clock breakdown; nested fields are not additive
    bool grouped_attention = true; // combine output-projection transfers across attention groups
    bool fused_attention_proj = false; // fuse q_a -> weighted RMSNorm -> q_b and kv input/readback
    bool fused_shared = true;      // keep the shared expert's intermediate activations on the GPU
    int cpu_experts = 0;          // send the last N routed experts to the CPU; remaining experts use HIP/CUDA
    bool cpu_on_cache_miss = false; // route resident experts to GPU and nonresident selected experts to native CPU
    int cpu_threads = 0;          // persistent expert pool participants, 0 follows --threads
    bool cpu_pin = true;          // one worker per physical core, with Windows processor groups
    bool cpu_host_pin = true;     // scoped Windows CPU Set on the reserved caller core
    std::string cpu_moe_kernel = "auto"; // auto/native/reference; native quantizes activations as ggml-cpu does
};

// Statistiche di runtime (lette da dsv4_run per il report finale).
struct Stats {
    uint64_t h2d_bytes = 0;   // resident expert admissions/swaps uploaded to VRAM (not all H2D)
    uint64_t staged_bytes = 0; // expert weights submitted through the device staging pool
    uint64_t cpu_experts = 0;  // expert evaluations actually computed on the CPU
    uint64_t native_cpu_experts = 0; // evaluated with ggml-cpu quantized activation kernels
    uint64_t attention_bundle_calls = 0; // fused q_a/RMSNorm/q_b + kv bundles completed
    uint64_t attention_bundle_fallbacks = 0; // opted-in attention projections using the reference path
    uint64_t admits = 0;      // experts loaded into a free VRAM slot
    uint64_t swaps = 0;       // adaptive evict+load decisions
    uint64_t hits = 0;        // routed expert evaluations served from VRAM
    uint64_t misses = 0;      // routed expert evaluations served from RAM/disk
    uint64_t cache_hits = 0;      // MISS experts already in the RAM LRU arena
    uint64_t cache_loads = 0;     // MISS experts copied into the arena (from mmap)
    uint64_t cache_evictions = 0; // arena slots reused for another expert
    uint64_t cache_bytes = 0;     // bytes copied into the arena
    uint64_t tokens = 0;      // tokens processed
    double total_s = 0;       // forward time
    double cpu_miss_s = 0;    // host staging/submission plus any CPU fallback computation
    double gpu_hit_s = 0;     // tail wait/readback for resident AND staged GPU experts
    double dense_mv_s = 0;    // all dense matvec round trips, nested within attention/MoE/head
    double attention_s = 0;
    double sparse_attention_s = 0; // nested within attention_s
    double moe_s = 0;
    double hyper_s = 0;
};

class Model {
public:
    Model();
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const std::vector<std::string>& shards, const RunOpts& opts, std::string& err_out);
    void reset();
    void forward(int token, int pos, std::vector<float>* logits_out);
    void end_token();
    bool save_profile(std::string& err) const;
    void selfcheck() const;
    
    uint64_t model_hash() const;
    const Dsv4Config& config() const;
    const std::vector<std::string>& tokens() const;

    std::vector<int> last_routing;

    Stats stats;  // p_->st points here (set in the Model constructor)

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// Maps the model's byte-level token strings back to text (see docs / decode_token_text).
std::string decode_token_text(const std::string& tok);

} // namespace dsv4
