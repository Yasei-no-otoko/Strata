// CPU-only tests: synthetic GGUF (shapes only, no weights) -> reader -> config -> expert inventory -> memory plan.
#include "dsv4/config.hpp"
#include "dsv4/gguf_header.hpp"
#include "dsv4/mem_plan.hpp"
#include "dsv4/platform_file.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <winioctl.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while (0)

namespace {
std::string temp_path(const char* suffix) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() / ("dsv4_" + std::to_string(stamp) + "_" + suffix)).string();
}

bool write_sparse_sentinel(const std::string& path, uint64_t offset, const uint8_t* bytes, size_t size) {
#if defined(_WIN32)
    std::wstring wide;
    if (!dsv4::utf8_to_wide(path, wide)) return false;
    HANDLE f = CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    DWORD ignored = 0;
    const bool sparse = DeviceIoControl(f, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ignored, nullptr) != 0;
    LARGE_INTEGER pos{}; pos.QuadPart = static_cast<LONGLONG>(offset + size);
    bool ok = sparse && SetFilePointerEx(f, pos, nullptr, FILE_BEGIN) && SetEndOfFile(f);
    pos.QuadPart = static_cast<LONGLONG>(offset);
    ok = ok && SetFilePointerEx(f, pos, nullptr, FILE_BEGIN);
    DWORD written = 0;
    ok = ok && WriteFile(f, bytes, static_cast<DWORD>(size), &written, nullptr) && written == size;
    CloseHandle(f);
    return ok;
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) return false;
    bool ok = offset <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - size &&
              ftruncate(fd, static_cast<off_t>(offset + size)) == 0;
    size_t done = 0;
    while (ok && done < size) {
        const ssize_t n = pwrite(fd, bytes + done, size - done, static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { ok = false; break; }
        done += static_cast<size_t>(n);
    }
    ::close(fd);
    return ok;
#endif
}

struct W {
    std::vector<uint8_t> b;
    template <class T> void pod(T v) { const uint8_t* p = (const uint8_t*) &v; b.insert(b.end(), p, p + sizeof(T)); }
    void str(const std::string& s) { pod<uint64_t>(s.size()); b.insert(b.end(), s.begin(), s.end()); }
};
struct TInfo { std::string name; std::vector<uint64_t> dims; uint32_t type; };

void kv_u32(W& w, const std::string& k, uint32_t v) { w.str(k); w.pod<uint32_t>(dsv4::GGUF_U32); w.pod<uint32_t>(v); }
void kv_f32(W& w, const std::string& k, float v) { w.str(k); w.pod<uint32_t>(dsv4::GGUF_F32); w.pod<float>(v); }
void kv_bool(W& w, const std::string& k, bool v) { w.str(k); w.pod<uint32_t>(dsv4::GGUF_BOOL); w.pod<uint8_t>(v ? 1 : 0); }
void kv_str(W& w, const std::string& k, const std::string& v) { w.str(k); w.pod<uint32_t>(dsv4::GGUF_STR); w.str(v); }
void kv_arr_i32(W& w, const std::string& k, const std::vector<int32_t>& v) {
    w.str(k); w.pod<uint32_t>(dsv4::GGUF_ARR); w.pod<uint32_t>(dsv4::GGUF_I32); w.pod<uint64_t>(v.size());
    for (int32_t x : v) w.pod<int32_t>(x);
}
void kv_arr_f32(W& w, const std::string& k, const std::vector<float>& v) {
    w.str(k); w.pod<uint32_t>(dsv4::GGUF_ARR); w.pod<uint32_t>(dsv4::GGUF_F32); w.pod<uint64_t>(v.size());
    for (float x : v) w.pod<float>(x);
}
void kv_arr_str(W& w, const std::string& k, size_t n) {
    w.str(k); w.pod<uint32_t>(dsv4::GGUF_ARR); w.pod<uint32_t>(dsv4::GGUF_STR); w.pod<uint64_t>(n);
    for (size_t i = 0; i < n; ++i) w.str("tok" + std::to_string(i));
}

bool write_file(const std::string& path, const W& body_kv, uint64_t n_kv, const std::vector<TInfo>& ts) {
    W w;
    w.b.insert(w.b.end(), {'G', 'G', 'U', 'F'});
    w.pod<uint32_t>(3);
    w.pod<uint64_t>(ts.size());
    w.pod<uint64_t>(n_kv);
    w.b.insert(w.b.end(), body_kv.b.begin(), body_kv.b.end());
    uint64_t off = 0;
    for (const TInfo& t : ts) {
        w.str(t.name);
        w.pod<uint32_t>((uint32_t) t.dims.size());
        for (uint64_t d : t.dims) w.pod<uint64_t>(d);
        w.pod<uint32_t>(t.type);
        w.pod<uint64_t>(off);
        off += 32;
    }
    FILE* f = dsv4::open_binary_file(path, true);
    if (!f) return false;
    std::fwrite(w.b.data(), 1, w.b.size(), f);
    std::fclose(f);
    return true;
}

constexpr uint32_t T_IQ1_M = 29, T_IQ2_XXS = 16, T_F32 = 0;
}  // namespace

int main() {
    const std::string f1 = temp_path("t1.gguf"), f2 = temp_path("t2.gguf"), f3 = temp_path("t3.gguf");
    const std::string bad_path = temp_path("bad.gguf"), overflow_path = temp_path("overflow.gguf"), raw_path = temp_path("mapped.bin");
    const std::string sparse_path = temp_path("sparse.bin");
    const std::string replace_path = temp_path("profile.bin"), replace_tmp = temp_path("profile.bin.tmp");
    // ---- synthetic DeepSeek-V4-Flash header (values copied from the real shard-1 dump) ----
    W kv; uint64_t n = 0;
    kv_str(kv, "general.architecture", "deepseek4"); ++n;
    kv_u32(kv, "deepseek4.block_count", 43); ++n;
    kv_u32(kv, "deepseek4.context_length", 1048576); ++n;
    kv_u32(kv, "deepseek4.embedding_length", 4096); ++n;
    kv_u32(kv, "deepseek4.attention.head_count", 64); ++n;
    kv_u32(kv, "deepseek4.attention.head_count_kv", 1); ++n;
    kv_u32(kv, "deepseek4.attention.key_length", 512); ++n;
    kv_u32(kv, "deepseek4.attention.value_length", 512); ++n;
    kv_u32(kv, "deepseek4.rope.dimension_count", 64); ++n;
    kv_u32(kv, "deepseek4.attention.q_lora_rank", 1024); ++n;
    kv_u32(kv, "deepseek4.attention.output_lora_rank", 1024); ++n;
    kv_u32(kv, "deepseek4.attention.output_group_count", 8); ++n;
    kv_u32(kv, "deepseek4.attention.sliding_window", 128); ++n;
    kv_u32(kv, "deepseek4.expert_count", 256); ++n;
    kv_u32(kv, "deepseek4.expert_used_count", 6); ++n;
    kv_u32(kv, "deepseek4.expert_shared_count", 1); ++n;
    kv_u32(kv, "deepseek4.expert_feed_forward_length", 2048); ++n;
    kv_u32(kv, "deepseek4.expert_gating_func", 4); ++n;
    kv_f32(kv, "deepseek4.expert_weights_scale", 1.5f); ++n;
    kv_bool(kv, "deepseek4.expert_weights_norm", true); ++n;
    kv_u32(kv, "deepseek4.hash_layer_count", 3); ++n;
    kv_u32(kv, "deepseek4.attention.indexer.head_count", 64); ++n;
    kv_u32(kv, "deepseek4.attention.indexer.key_length", 128); ++n;
    kv_u32(kv, "deepseek4.attention.indexer.top_k", 512); ++n;
    kv_u32(kv, "deepseek4.hyper_connection.count", 4); ++n;
    kv_u32(kv, "deepseek4.hyper_connection.sinkhorn_iterations", 20); ++n;
    kv_f32(kv, "deepseek4.hyper_connection.epsilon", 1e-6f); ++n;
    kv_f32(kv, "deepseek4.attention.layer_norm_rms_epsilon", 1e-6f); ++n;
    kv_f32(kv, "deepseek4.rope.freq_base", 10000.f); ++n;
    kv_f32(kv, "deepseek4.attention.compress_rope_freq_base", 160000.f); ++n;
    kv_f32(kv, "deepseek4.rope.scaling.factor", 16.f); ++n;
    kv_f32(kv, "deepseek4.rope.scaling.yarn_beta_fast", 32.f); ++n;
    kv_f32(kv, "deepseek4.rope.scaling.yarn_beta_slow", 1.f); ++n;
    kv_u32(kv, "deepseek4.rope.scaling.original_context_length", 65536); ++n;
    std::vector<int32_t> ratios = {0, 0};
    for (int i = 2; i < 43; ++i) ratios.push_back(i % 2 == 0 ? 4 : 128);
    ratios.push_back(0);  // the extra (MTP) entry
    kv_arr_i32(kv, "deepseek4.attention.compress_ratios", ratios); ++n;
    kv_arr_f32(kv, "deepseek4.swiglu_clamp_exp", std::vector<float>(43, 10.f)); ++n;
    kv_arr_f32(kv, "deepseek4.swiglu_clamp_shexp", std::vector<float>(43, 10.f)); ++n;
    kv_arr_str(kv, "tokenizer.ggml.tokens", 1000); ++n;
    kv_u32(kv, "tokenizer.ggml.bos_token_id", 0); ++n;
    kv_u32(kv, "tokenizer.ggml.eos_token_id", 1); ++n;

    // shard 1: metadata only (as in the real file); shard 2: layers 0..21 + token_embd; shard 3: layers 22..42 + MTP
    auto layer_tensors = [](int l, uint32_t down_type) {
        std::string p = "blk." + std::to_string(l) + ".";
        return std::vector<TInfo>{{p + "ffn_gate_exps.weight", {4096, 2048, 256}, T_IQ1_M},
                                  {p + "ffn_up_exps.weight", {4096, 2048, 256}, T_IQ1_M},
                                  {p + "ffn_down_exps.weight", {2048, 4096, 256}, down_type},
                                  {p + "attn_norm.weight", {4096}, T_F32}};
    };
    std::vector<TInfo> t2 = {{"token_embd.weight", {4096, 1000}, T_F32}}, t3;
    for (int l = 0; l < 22; ++l) for (auto& t : layer_tensors(l, l == 5 ? T_IQ2_XXS : T_IQ1_M)) t2.push_back(t);
    for (int l = 22; l < 44; ++l) for (auto& t : layer_tensors(l, T_IQ1_M)) t3.push_back(t);  // 43 = MTP layer
    W empty;
    CHECK(write_file(f1, kv, n, {}));
    CHECK(write_file(f2, empty, 0, t2));
    CHECK(write_file(f3, empty, 0, t3));

    dsv4::GgufHeader h; std::string err;
    CHECK(dsv4::gguf_read_header({f1, f2, f3}, h, err));
    if (!err.empty()) std::fprintf(stderr, "reader: %s\n", err.c_str());
    CHECK(h.version == 3);
    CHECK(h.tensors.size() == 1 + 22 * 4 + 22 * 4);

    dsv4::Dsv4Config c;
    CHECK(dsv4::config_from_gguf(h, c, err));
    if (!err.empty()) std::fprintf(stderr, "config: %s\n", err.c_str());
    CHECK(c.n_layer == 43 && c.n_expert == 256 && c.n_expert_used == 6 && c.ff_exp == 2048);
    CHECK(c.compress_ratios.size() == 44 && c.n_extra_layers == 1);
    CHECK(c.compress_ratios[2] == 4 && c.compress_ratios[3] == 128 && c.compress_ratios[43] == 0);
    CHECK(c.swiglu_clamp_exp.size() == 43 && c.swiglu_clamp_exp[7] == 10.f);
    CHECK(c.hc_count == 4 && c.hc_sinkhorn_iters == 20 && c.n_hash_layers == 3 && c.exp_w_norm);
    CHECK(c.vocab == 1000 && c.bos == 0 && c.eos == 1);

    // a wrong architecture must be refused
    {
        W bad; uint64_t nb = 0; kv_str(bad, "general.architecture", "qwen3"); ++nb;
        write_file(bad_path, bad, nb, {});
        dsv4::GgufHeader hb; dsv4::Dsv4Config cb; std::string eb;
        CHECK(dsv4::gguf_read_header({bad_path}, hb, eb));
        CHECK(!dsv4::config_from_gguf(hb, cb, eb));
    }
    {
        W malformed; uint64_t nm = 0;
        write_file(overflow_path, malformed, nm, {{"overflow", {UINT64_MAX, 2}, T_F32}});
        dsv4::GgufHeader hm; std::string em;
        CHECK(!dsv4::gguf_read_header({overflow_path}, hm, em));
        CHECK(em.find("dimensions overflow") != std::string::npos);
    }
    {
        const uint8_t bytes[] = {1, 3, 5, 7, 9, 11, 13, 15};
        FILE* f = dsv4::open_binary_file(raw_path, true);
        CHECK(f != nullptr);
        if (f) { CHECK(std::fwrite(bytes, 1, sizeof(bytes), f) == sizeof(bytes)); std::fclose(f); }
        dsv4::ReadOnlyFile mapped; std::string em;
        CHECK(mapped.open(raw_path, em));
        CHECK(mapped.size() == sizeof(bytes));
        uint8_t got[3] = {};
        CHECK(mapped.read_at(2, got, sizeof(got)));
        CHECK(got[0] == 5 && got[1] == 7 && got[2] == 9);
        CHECK(!mapped.read_at(7, got, sizeof(got)));
    }
    {
        // This is a sparse logical file: only the sentinel bytes occupy storage.
        // Reading it exercises offsets beyond the 32-bit boundary on both platforms.
        constexpr uint64_t offset = (1ULL << 32) + 12345;
        const uint8_t sentinel[] = {0x5a, 0xa5, 0x3c, 0xc3};
        CHECK(write_sparse_sentinel(sparse_path, offset, sentinel, sizeof(sentinel)));
        dsv4::ReadOnlyFile mapped; std::string em;
        CHECK(mapped.open(sparse_path, em));
        CHECK(mapped.size() == offset + sizeof(sentinel));
        uint8_t got[sizeof(sentinel)] = {};
        CHECK(mapped.read_at(offset, got, sizeof(got)));
        CHECK(std::memcmp(got, sentinel, sizeof(got)) == 0);
    }
    {
        auto write_bytes = [](const std::string& path, const char* text, size_t size) {
            FILE* f = dsv4::open_binary_file(path, true);
            if (!f) return false;
            bool ok = std::fwrite(text, 1, size, f) == size;
            if (std::fclose(f) != 0) ok = false;
            return ok;
        };
        CHECK(write_bytes(replace_path, "old-profile", 11));
        CHECK(write_bytes(replace_tmp, "new-profile-data", 16));
        CHECK(dsv4::atomic_replace_file(replace_tmp, replace_path));
        FILE* f = dsv4::open_binary_file(replace_path, false);
        CHECK(f != nullptr);
        if (f) {
            char got[16] = {};
            CHECK(std::fread(got, 1, sizeof(got), f) == sizeof(got));
            CHECK(std::memcmp(got, "new-profile-data", sizeof(got)) == 0);
            CHECK(std::fgetc(f) == EOF);
            std::fclose(f);
        }
    }

    dsv4::ExpertInventory inv;
    CHECK(dsv4::inventory_from_gguf(h, c, inv, err));
    // IQ1_M: 4096*2048/256 blocks * 56 B = 1,835,008 B per expert per matrix
    const uint64_t m = 4096ull * 2048 / 256 * 56;
    CHECK(inv.bytes_per_expert[0] == 3 * m);
    CHECK(inv.bytes_per_expert[0] == 5505024ull);
    // layer 5 down is IQ2_XXS: 66 B / 256 elems
    const uint64_t m2 = 4096ull * 2048 / 256 * 66;
    CHECK(inv.bytes_per_expert[5] == 2 * m + m2);
    CHECK(inv.types[5] == "IQ1_M/IQ1_M/IQ2_XXS");
    CHECK(inv.layers_without_experts == 0);
    CHECK(inv.bytes_per_expert_extra.size() == 1 && inv.bytes_per_expert_extra[0] == 3 * m);
    CHECK(inv.extra_expert_bytes == 3 * m * 256);
    CHECK(inv.total_expert_bytes == (42 * 3 * m + (2 * m + m2)) * 256);

    // ---- memory plan ----
    dsv4::MemPlanInput in;
    in.vram_total = 24564LL << 20; in.vram_free = 23100LL << 20; in.reserve = 1024LL << 20;
    in.weights = 4000LL << 20; in.kv = 1000LL << 20; in.act = 500LL << 20; in.prefill = 800LL << 20;
    in.mtp = 200LL << 20; in.scratch = 300LL << 20;
    in.n_expert = c.n_expert; in.bytes_per_expert = inv.bytes_per_expert;
    in.ram_expert_bytes = inv.total_expert_bytes;
    const double pcts[] = {0, 25, 50, 75, 100};
    int64_t prev = -1;
    for (double pct : pcts) {
        in.pct = pct;
        dsv4::MemPlan p = dsv4::plan_expert_memory(in);
        CHECK(p.ok);
        CHECK((int64_t) p.slot_bytes_total <= p.budget_expert);
        CHECK(p.slots_total >= prev);
        prev = p.slots_total;
        for (int l = 0; l < 43; ++l) CHECK(p.slots_layer[(size_t) l] >= 0 && p.slots_layer[(size_t) l] <= 256);
        if (pct == 0) CHECK(p.slots_total == 0 && !p.all_resident);
        if (pct == 100) CHECK(!p.all_resident);  // 57 GiB of experts do not fit in the ~16 GiB left
        std::printf("P=%3.0f slots=%lld per-layer=%d..%d used=%lld MiB of %lld MiB\n", pct, (long long) p.slots_total,
                    *std::min_element(p.slots_layer.begin(), p.slots_layer.end()),
                    *std::max_element(p.slots_layer.begin(), p.slots_layer.end()),
                    (long long) (p.slot_bytes_total >> 20), (long long) (p.budget_expert >> 20));
    }
    {   // everything fits -> all resident, whatever the number of layers
        dsv4::MemPlanInput big = in; big.vram_free = 200000LL << 20; big.pct = 100;
        dsv4::MemPlan p = dsv4::plan_expert_memory(big);
        CHECK(p.ok && p.all_resident && p.slots_total == 43 * 256);
        big.pct = -1;  // auto
        p = dsv4::plan_expert_memory(big);
        CHECK(p.ok && p.all_resident);
    }
    {   // auto when they do not fit -> 90%
        dsv4::MemPlanInput a = in; a.pct = -1;
        dsv4::MemPlan p = dsv4::plan_expert_memory(a);
        CHECK(p.ok && std::abs(p.pct_used - 90.0) < 1e-9);
    }
    {   // --expert-cache N overrides pct, and a request that does not fit is refused with a message
        dsv4::MemPlanInput a = in; a.abs_slots = 430; a.pct = 0;
        dsv4::MemPlan p = dsv4::plan_expert_memory(a);
        CHECK(p.ok && p.slots_total == 430);
        a.abs_slots = 43 * 256;
        p = dsv4::plan_expert_memory(a);
        CHECK(!p.ok && !p.err.empty());
    }
    {   // fixed > free is a clear error
        dsv4::MemPlanInput a = in; a.weights = 40000LL << 20;
        dsv4::MemPlan p = dsv4::plan_expert_memory(a);
        CHECK(!p.ok && p.err.find("max-context") != std::string::npos);
    }
    {
        in.pct = 50;
        dsv4::MemPlan p = dsv4::plan_expert_memory(in);
        std::printf("--- startup report (P=50) ---\n%s", dsv4::plan_report(in, p).c_str());
    }
    std::printf("%s\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED");
    std::error_code ignored;
    for (const std::string& p : {f1, f2, f3, bad_path, overflow_path, raw_path, sparse_path, replace_path, replace_tmp})
        std::filesystem::remove(p, ignored);
    return g_fail ? 1 : 0;
}
