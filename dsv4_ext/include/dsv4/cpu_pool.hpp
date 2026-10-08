#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace dsv4 {

/// Persistent CPU workers for independent row ranges.
///
/// `total_threads` counts the calling thread plus pool workers. When pinning is
/// enabled, workers use one logical processor from each physical core. On
/// Windows the caller can be temporarily selected onto the first available
/// physical core during each run; its prior CPU-set selection is restored.
class CpuPool {
public:
    using RowFunction = void (*)(void* context, int64_t begin, int64_t end) noexcept;

    /// Scoped caller CPU-set placement. On Windows this restores the exact
    /// selected-set list captured on the same thread; other platforms are no-op.
    class CallerScope {
    public:
        CallerScope() noexcept = default;
        ~CallerScope() noexcept;
        CallerScope(const CallerScope&) = delete;
        CallerScope& operator=(const CallerScope&) = delete;
        CallerScope(CallerScope&&) = delete;
        CallerScope& operator=(CallerScope&&) = delete;

    private:
        friend class CpuPool;
        CallerScope(bool enabled, unsigned long target_cpu_set_id);
        std::vector<unsigned long> previous_cpu_sets_;
        bool active_ = false;
    };

    explicit CpuPool(int total_threads, bool pin = true, bool pin_caller = true);
    ~CpuPool();
    CpuPool(const CpuPool&) = delete;
    CpuPool& operator=(const CpuPool&) = delete;

    /// Run a synchronous, balanced partition of [0, rows). Each row belongs
    /// to exactly one participant. `fn` must not throw or call this pool's
    /// `run` recursively. Calls from separate producers are serialized.
    void run(int64_t rows, RowFunction fn, void* context);

    /// Actual number of participants including the calling thread. With pinning
    /// and known topology this is capped at one participant per physical core.
    int threads() const noexcept { return total_threads_; }
    int requested_threads() const noexcept { return requested_threads_; }
    /// Encoded reserved caller CPU on Windows, or -1 if unavailable/disabled.
    /// Other platforms leave the caller's existing placement unchanged.
    int caller_cpu() const noexcept { return caller_cpu_; }
    /// Keep the caller on its reserved CPU across an outer operation. Nested
    /// run() calls detect this selection and avoid repeated set/restore calls.
    CallerScope scoped_caller() const;

    /// Actual encoded logical-processor IDs for workers that were pinned;
    /// group-aware Windows IDs are group * 64 + processor-in-group. -1 means
    /// pinning was disabled or the OS rejected that worker's affinity.
    std::vector<int> worker_cpus() const { return worker_cpus_; }

private:
    void worker_loop(int worker_id, int preferred_cpu) noexcept;
    void stop_and_join() noexcept;

    int requested_threads_ = 0;
    int total_threads_ = 1;
    int worker_count_ = 0;
    int caller_cpu_ = -1;
    bool pin_caller_ = false;
#if defined(_WIN32)
    unsigned long caller_cpu_set_id_ = 0;
    bool caller_cpu_set_valid_ = false;
#endif
    std::vector<std::thread> workers_;
    std::vector<int> worker_cpus_;

    // One producer publishes these fields before releasing the next epoch.
    int64_t rows_ = 0;
    RowFunction fn_ = nullptr;
    void* context_ = nullptr;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> remaining_{0};
    std::atomic<int> started_{0};
    std::atomic<bool> stopping_{false};
    std::mutex run_mutex_;
};

}  // namespace dsv4
