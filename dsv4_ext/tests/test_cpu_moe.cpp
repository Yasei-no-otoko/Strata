// Correctness tests for the persistent two-phase CPU MoE executor.
#include "dsv4/cpu_moe.hpp"
#include "dsv4/dequant.hpp"
#include "dsv4/native_cpu.hpp"
#include "dsv4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ExpertData {
    std::vector<uint8_t> gate, up, down;
    float weight = 0.f;
};

struct BenchmarkSet {
    std::vector<ExpertData> data;
    std::vector<dsv4::CpuExpert> experts;
    std::vector<float> reference;
    size_t weight_bytes = 0;
};

uint64_t rng = 0xA63B9F1248D57C01ULL;
uint64_t next_u64() {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 2685821657736338717ULL;
}

void fill_bytes(std::vector<uint8_t>& data) {
    for (uint8_t& byte : data) byte = (uint8_t)next_u64();
}

void put_u16(uint8_t* p, uint16_t value) { std::memcpy(p, &value, sizeof(value)); }

void bound_block(uint32_t type, uint8_t* block) {
    switch (type) {
        case dsv4::T_Q8_0:
            put_u16(block, 0x3000); // half 0.125
            break;
        case dsv4::T_Q4_K:
        case dsv4::T_Q5_K:
            put_u16(block, 0x3000);
            put_u16(block + 2, 0);
            std::memset(block + 4, 0, 12);
            block[4] = block[5] = block[6] = block[7] = 1;
            block[12] = block[13] = block[14] = block[15] = 1;
            break;
        case dsv4::T_Q6_K:
            std::memset(block + 192, 1, 16);
            put_u16(block + 208, 0x3000);
            break;
        case dsv4::T_IQ3_XXS:
        case dsv4::T_IQ2_XXS:
            put_u16(block, 0x3000);
            break;
        case dsv4::T_IQ1_M:
            std::memset(block + 48, 0, 8);
            put_u16(block + 54, 0x3000);
            break;
        case dsv4::T_MXFP4:
            block[0] = 128; // e8m0 scale 1.0
            break;
        default:
            break;
    }
}

bool make_matrix(uint32_t type, int rows, int in, std::vector<uint8_t>& bytes) {
    const size_t row_size = dsv4::row_bytes(type, in);
    const int block_elements = dsv4::dequant_block_elems(type);
    if (!row_size || block_elements <= 0 || in % block_elements) return false;
    bytes.resize((size_t)rows * row_size);
    fill_bytes(bytes);
    const size_t block_bytes = row_size / (size_t)(in / block_elements);
    for (int row = 0; row < rows; ++row) {
        uint8_t* data = bytes.data() + (size_t)row * row_size;
        for (int block = 0; block < in / block_elements; ++block)
            bound_block(type, data + (size_t)block * block_bytes);
    }

    std::vector<float> decoded((size_t)in);
    for (int row = 0; row < rows; ++row) {
        dsv4::dequant_row(type, bytes.data() + (size_t)row * row_size, in, decoded.data());
        for (float value : decoded)
            if (!std::isfinite(value) || std::fabs(value) > 64.f) return false;
    }
    return true;
}

bool make_experts(uint32_t tg, uint32_t tu, uint32_t td, int dim, int ff, int count,
                  std::vector<ExpertData>& data, std::vector<dsv4::CpuExpert>& ptrs) {
    data.resize((size_t)count);
    ptrs.resize((size_t)count);
    for (int k = 0; k < count; ++k) {
        auto& e = data[(size_t)k];
        if (!make_matrix(tg, ff, dim, e.gate) || !make_matrix(tu, ff, dim, e.up) ||
            !make_matrix(td, dim, ff, e.down)) return false;
        e.weight = 0.17f + 0.11f * (float)((k * 3 + 1) % 7);
        ptrs[(size_t)k] = {e.gate.data(), e.up.data(), e.down.data(), e.weight};
    }
    return true;
}

bool same_floats(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool close_floats(const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return false;
        const double diff = std::fabs((double)got[i] - (double)want[i]);
        if (diff > 2e-4 + 2e-5 * std::fabs((double)want[i])) return false;
    }
    return true;
}

bool float_reference(uint32_t tg, uint32_t tu, uint32_t td, int dim, int ff, float limit,
                     const std::vector<float>& x, const std::vector<dsv4::CpuExpert>& experts,
                     std::vector<float>& sum) {
    sum.assign((size_t)dim, 0.f);
    const size_t rg = dsv4::row_bytes(tg, dim), ru = dsv4::row_bytes(tu, dim), rd = dsv4::row_bytes(td, ff);
    for (const auto& expert : experts) {
        std::vector<float> g((size_t)ff), u((size_t)ff), a((size_t)ff), y((size_t)dim);
        for (int r = 0; r < ff; ++r) {
            g[(size_t)r] = dsv4::row_dot(tg, expert.gate + (size_t)r * rg, x.data(), dim);
            u[(size_t)r] = dsv4::row_dot(tu, expert.up + (size_t)r * ru, x.data(), dim);
        }
        dsv4::swiglu_clamped(g.data(), u.data(), ff, limit, a.data());
        for (float& value : a) value *= expert.weight;
        for (int r = 0; r < dim; ++r) {
            y[(size_t)r] = dsv4::row_dot(td, expert.down + (size_t)r * rd, a.data(), ff);
            sum[(size_t)r] += y[(size_t)r];
        }
    }
    for (float value : sum) if (!std::isfinite(value)) return false;
    return true;
}

bool native_reference(uint32_t tg, uint32_t tu, uint32_t td, int dim, int ff, float limit,
                      const std::vector<float>& x, const std::vector<dsv4::CpuExpert>& experts,
                      std::vector<float>& sum, std::string& err) {
    dsv4::NativeCpuFormat fg, fu, fd;
    if (!dsv4::native_cpu_format(tg, dim, fg, err) || !dsv4::native_cpu_format(tu, dim, fu, err) ||
        !dsv4::native_cpu_format(td, ff, fd, err)) return false;
    std::vector<uint8_t> qg, qu;
    if (!dsv4::native_cpu_prepare(fg, x.data(), qg, err)) return false;
    const uint8_t* qu_ptr = qg.data();
    if (fg.activation_type != fu.activation_type) {
        if (!dsv4::native_cpu_prepare(fu, x.data(), qu, err)) return false;
        qu_ptr = qu.data();
    }
    const size_t rg = dsv4::row_bytes(tg, dim), ru = dsv4::row_bytes(tu, dim), rd = dsv4::row_bytes(td, ff);
    sum.assign((size_t)dim, 0.f);
    for (const auto& expert : experts) {
        std::vector<float> g((size_t)ff), u((size_t)ff), activation((size_t)ff), y((size_t)dim);
        for (int r = 0; r < ff; ++r) {
            if (!dsv4::native_cpu_dot(fg, expert.gate + (size_t)r * rg, qg.data(), g[(size_t)r]) ||
                !dsv4::native_cpu_dot(fu, expert.up + (size_t)r * ru, qu_ptr, u[(size_t)r])) {
                err = "native sequential gate/up dot failed"; return false;
            }
        }
        for (int i = 0; i < ff; ++i) {
            float a = g[(size_t)i], b = u[(size_t)i];
            if (limit > 0.f) { a = std::min(a, limit); b = std::clamp(b, -limit, limit); }
            activation[(size_t)i] = float(double(a) / (1.0 + std::exp(-double(a)))) * b;
            activation[(size_t)i] *= expert.weight;
        }
        std::vector<uint8_t> qd;
        if (!dsv4::native_cpu_prepare(fd, activation.data(), qd, err)) return false;
        for (int r = 0; r < dim; ++r) {
            if (!dsv4::native_cpu_dot(fd, expert.down + (size_t)r * rd, qd.data(), y[(size_t)r])) {
                err = "native sequential down dot failed"; return false;
            }
            sum[(size_t)r] += y[(size_t)r];
        }
    }
    return true;
}

bool run_case(uint32_t tg, uint32_t tu, uint32_t td, int dim, int ff, int count) {
    constexpr float limit = 5.5f;
    std::vector<float> x((size_t)dim);
    for (int i = 0; i < dim; ++i) x[(size_t)i] = (float)((i * 19 + 7) % 127 - 63) / 128.f;
    std::vector<ExpertData> data;
    std::vector<dsv4::CpuExpert> experts;
    if (!make_experts(tg, tu, td, dim, ff, count, data, experts)) {
        std::fprintf(stderr, "FAIL fixture creation: types %u/%u/%u dim=%d ff=%d\n", tg, tu, td, dim, ff);
        return false;
    }
    std::vector<float> want;
    if (!float_reference(tg, tu, td, dim, ff, limit, x, experts, want)) return false;
    std::string err;
    std::vector<float> one((size_t)dim), four((size_t)dim);
    dsv4::CpuMoe pool1(1, false, false), pool4(4, false, false);
    if (!pool1.run(tg, tu, td, dim, ff, limit, x.data(), experts.data(), count, one.data(), err)) {
        std::fprintf(stderr, "FAIL CpuMoe float worker=1: %s\n", err.c_str()); return false;
    }
    if (!pool4.run(tg, tu, td, dim, ff, limit, x.data(), experts.data(), count, four.data(), err)) {
        std::fprintf(stderr, "FAIL CpuMoe float worker=4: %s\n", err.c_str()); return false;
    }
    if (!same_floats(one, four) || !close_floats(one, want)) {
        std::fprintf(stderr, "FAIL float MoE mismatch types=%u/%u/%u dim=%d ff=%d experts=%d\n",
                     tg, tu, td, dim, ff, count); return false;
    }

    if (dsv4::native_cpu_available()) {
        std::vector<float> native_want, native1((size_t)dim), native4((size_t)dim);
        if (!native_reference(tg, tu, td, dim, ff, limit, x, experts, native_want, err)) {
            std::printf("skip native MoE types %u/%u/%u shape %d/%d: %s\n", tg, tu, td, dim, ff, err.c_str());
        } else {
            dsv4::CpuMoe native_pool1(1, false, true), native_pool4(4, false, true);
            if (!native_pool1.run(tg, tu, td, dim, ff, limit, x.data(), experts.data(), count, native1.data(), err) ||
                !native_pool4.run(tg, tu, td, dim, ff, limit, x.data(), experts.data(), count, native4.data(), err)) {
                std::fprintf(stderr, "FAIL native CpuMoe: %s\n", err.c_str()); return false;
            }
            if (!same_floats(native1, native4) || !same_floats(native1, native_want)) {
                std::fprintf(stderr, "FAIL native MoE determinism types=%u/%u/%u dim=%d ff=%d experts=%d\n",
                             tg, tu, td, dim, ff, count); return false;
            }
        }
    }
    std::printf("ok CPU MoE types=%u/%u/%u dim=%d ff=%d experts=%d native=%s\n",
                tg, tu, td, dim, ff, count, dsv4::native_cpu_available() ? "checked" : "unavailable");
    return true;
}

bool run_benchmark(int threads, bool pin, int warmup_iterations, int timed_iterations, int set_count) {
    // This shape models a full DeepSeek V4 expert FFN. With --sets 8, rotating through independent
    // fixtures expands the weight working set from 33.4 MiB to about 267 MiB.
    constexpr uint32_t tg = dsv4::T_IQ1_M, tu = dsv4::T_IQ1_M, td = dsv4::T_IQ2_XXS;
    constexpr int dim = 4096, ff = 2048, count = 6;
    constexpr float limit = 5.5f;
    if (!dsv4::native_cpu_available()) {
        std::fprintf(stderr, "--bench requires a build/runtime with native CPU support (%s)\n",
                     dsv4::native_cpu_description().c_str());
        return false;
    }

    std::vector<BenchmarkSet> sets((size_t)set_count);
    size_t weight_bytes = 0;
    for (BenchmarkSet& set : sets) {
        if (!make_experts(tg, tu, td, dim, ff, count, set.data, set.experts)) {
            std::fprintf(stderr, "FAIL creating full-geometry benchmark fixture\n");
            return false;
        }
        for (const auto& e : set.data)
            set.weight_bytes += e.gate.size() + e.up.size() + e.down.size();
        weight_bytes += set.weight_bytes;
    }

    std::vector<float> x((size_t)dim);
    for (int i = 0; i < dim; ++i) x[(size_t)i] = (float)((i * 19 + 7) % 127 - 63) / 128.f;
    std::vector<float> float_want((size_t)dim), output((size_t)dim);
    std::string err;
    double ref_ms = 0.0;
    for (size_t i = 0; i < sets.size(); ++i) {
        const auto ref_start = std::chrono::steady_clock::now();
        if (!native_reference(tg, tu, td, dim, ff, limit, x, sets[i].experts, sets[i].reference, err)) {
            std::fprintf(stderr, "FAIL native sequential benchmark reference for set %zu: %s\n", i, err.c_str());
            return false;
        }
        ref_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ref_start).count();
    }
    const auto float_start = std::chrono::steady_clock::now();
    if (!float_reference(tg, tu, td, dim, ff, limit, x, sets.front().experts, float_want)) {
        std::fprintf(stderr, "FAIL sequential float benchmark reference\n");
        return false;
    }
    const double float_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - float_start).count();

    dsv4::CpuMoe pool(threads, pin, true);
    auto execute = [&](BenchmarkSet& set) {
        return pool.run(tg, tu, td, dim, ff, limit, x.data(), set.experts.data(), count, output.data(), err);
    };
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        BenchmarkSet& set = sets[(size_t)iteration % sets.size()];
        if (!execute(set)) {
            std::fprintf(stderr, "FAIL native CpuMoe benchmark warmup %d: %s\n", iteration + 1, err.c_str());
            return false;
        }
        if (!same_floats(output, set.reference)) {
            std::fprintf(stderr, "FAIL native benchmark warmup %d differs from sequential reference\n", iteration + 1);
            return false;
        }
    }
    const auto timed_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        BenchmarkSet& set = sets[(size_t)iteration % sets.size()];
        if (!execute(set)) {
            std::fprintf(stderr, "FAIL native CpuMoe benchmark iteration %d: %s\n", iteration + 1, err.c_str());
            return false;
        }
        if (!same_floats(output, set.reference)) {
            std::fprintf(stderr, "FAIL native benchmark iteration %d differs from sequential reference\n", iteration + 1);
            return false;
        }
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - timed_start).count();
    double checksum = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr, "FAIL non-finite output at %zu\n", i);
            return false;
        }
        checksum += (double)output[i] * (double)(i + 1);
    }
    if (!std::isfinite(checksum)) {
        std::fprintf(stderr, "FAIL non-finite benchmark checksum\n");
        return false;
    }
    std::printf("CPU MoE benchmark: dim=%d ff=%d experts=%d types=IQ1_M/IQ1_M/IQ2_XXS threads=%d\n",
                dim, ff, count, pool.threads());
    const std::vector<int> worker_cpus = pool.worker_cpus();
    std::printf("  native CPU: %s\n", dsv4::native_cpu_description().c_str());
    std::printf("  pin=%s requested_threads=%d actual_threads=%d worker_cpus=", pin ? "on" : "off",
                threads, pool.threads());
    for (size_t i = 0; i < worker_cpus.size(); ++i)
        std::printf("%s%d", i ? "," : "", worker_cpus[i]);
    std::printf("\n");
    std::printf("  synthetic weights=%.2f MiB (%d rotating set%s)\n",
                (double)weight_bytes / (1024.0 * 1024.0), set_count, set_count == 1 ? "" : "s");
    std::printf("  sequential native-quantized references=%.2f ms total; sequential original float row_dot=%.2f ms\n",
                ref_ms, float_ms);
    std::printf("  native CpuMoe warmup=%d timed=%d total=%.2f ms average=%.3f ms checksum=%.9g\n",
                warmup_iterations, timed_iterations, total_ms, total_ms / timed_iterations, checksum);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        int threads = 8;
        int warmup_iterations = 1;
        int timed_iterations = 3;
        int set_count = 1;
        bool pin = false;
        bool have_threads = false;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--pin") {
                pin = true;
                continue;
            }
            try {
                if (arg == "--warmup" || arg == "--iterations" || arg == "--sets") {
                    if (++i >= argc) throw std::invalid_argument("missing value");
                    size_t parsed = 0;
                    const int value = std::stoi(argv[i], &parsed);
                    if (parsed != std::strlen(argv[i])) throw std::invalid_argument("trailing characters");
                    if (arg == "--warmup" && value >= 0 && value <= 100000) warmup_iterations = value;
                    else if (arg == "--iterations" && value >= 1 && value <= 1000000) timed_iterations = value;
                    else if (arg == "--sets" && value >= 1 && value <= 8) set_count = value;
                    else throw std::invalid_argument("out of range");
                    continue;
                }
                if (arg.rfind("--", 0) == 0 || have_threads) throw std::invalid_argument("unexpected argument");
                size_t parsed = 0;
                threads = std::stoi(arg, &parsed);
                if (parsed != arg.size() || threads < 1 || threads > 256) throw std::invalid_argument("range");
                have_threads = true;
            } catch (...) {
                std::fprintf(stderr,
                    "usage: test_cpu_moe --bench [threads] [--pin] [--warmup N] [--iterations N] [--sets 1..8]\n");
                return 2;
            }
        }
        return run_benchmark(threads, pin, warmup_iterations, timed_iterations, set_count) ? 0 : 1;
    }
    struct Types { uint32_t g, u, d; };
    const Types mixes[] = {
        {dsv4::T_IQ3_XXS, dsv4::T_Q8_0, dsv4::T_IQ1_M},
        {dsv4::T_Q4_K, dsv4::T_Q6_K, dsv4::T_Q8_0},
        {dsv4::T_Q8_0, dsv4::T_IQ2_XXS, dsv4::T_Q5_K},
    };
    const int shapes[][2] = {{256, 256}, {256, 512}, {512, 256}, {512, 512}};
    const int counts[] = {1, 3, 6};
    bool ok = true;
    for (const auto& types : mixes)
        for (const auto& shape : shapes)
            for (int count : counts)
                ok = run_case(types.g, types.u, types.d, shape[0], shape[1], count) && ok;
    std::puts(ok ? "ALL CPU MOE TESTS PASSED" : "CPU MOE TESTS FAILED");
    return ok ? 0 : 1;
}
