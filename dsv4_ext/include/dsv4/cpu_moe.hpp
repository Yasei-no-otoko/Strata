#pragma once

#include "dsv4/cpu_pool.hpp"
#include "dsv4/native_cpu.hpp"
#include <memory>

namespace dsv4 {
struct CpuExpert {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    float weight = 0.f;
};

/// Shape-independent counterpart of Strata's split native-expert pool. It uses two row phases,
/// reuses a quantized input across experts, and never starts a nested OpenMP/ggml worker team.
class CpuMoe {
public:
    CpuMoe(int threads, bool pin, bool native, bool pin_caller = true);
    ~CpuMoe();
    CpuMoe(const CpuMoe&) = delete;
    CpuMoe& operator=(const CpuMoe&) = delete;
    bool run(uint32_t gate_type, uint32_t up_type, uint32_t down_type, int dim, int ff,
             float limit, const float* x, const CpuExpert* experts, int count,
             float* sum, std::string& err);
    int threads() const;
    int caller_cpu() const;
    CpuPool::CallerScope scoped_caller() const;
    std::vector<int> worker_cpus() const;
    bool native() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace dsv4
