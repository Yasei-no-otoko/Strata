#include "dsv4/cpu_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <fstream>
#include <string>
#endif

namespace dsv4 {
namespace {

// Return one logical processor from each physical core. IDs are encoded as
// group * 64 + processor-in-group on Windows and as Linux CPU IDs on Linux.
std::vector<int> physical_processors() {
    std::vector<int> result;
#if defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    bool topology_query_succeeded = false;
    if (bytes != 0) {
        std::vector<unsigned char> buffer(bytes);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
            topology_query_succeeded = true;
            const unsigned char* p = buffer.data();
            const unsigned char* const end = p + bytes;
            constexpr size_t kProcessorOffset = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
            constexpr size_t kGroupCountEnd = kProcessorOffset + offsetof(PROCESSOR_RELATIONSHIP, GroupCount) +
                                              sizeof(WORD);
            constexpr size_t kGroupMaskOffset = kProcessorOffset + offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
            while ((size_t) (end - p) >= kProcessorOffset) {
                const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
                // These records are variable-sized. On current Windows the
                // outer union is 80 bytes, but a one-group core record is only
                // 48 bytes; requiring sizeof(*info) silently rejects every
                // valid core and falls back to sibling logical processors.
                if (info->Size < kProcessorOffset || (size_t) info->Size > (size_t) (end - p)) {
                    result.clear();
                    break;
                }
                if (info->Relationship == RelationProcessorCore) {
                    if (info->Size < kGroupCountEnd) {
                        result.clear();
                        break;
                    }
                    const size_t group_count = (size_t) info->Processor.GroupCount;
                    if (info->Size < kGroupMaskOffset ||
                        group_count > ((size_t) info->Size - kGroupMaskOffset) / sizeof(GROUP_AFFINITY)) {
                        result.clear();
                        break;
                    }
                    bool selected = false;
                    for (WORD g = 0; g < info->Processor.GroupCount && !selected; ++g) {
                        const GROUP_AFFINITY& group = info->Processor.GroupMask[g];
                        for (int bit = 0; bit < 64; ++bit) {
                            if ((group.Mask & (KAFFINITY(1) << bit)) != 0) {
                                result.push_back((int) group.Group * 64 + bit);
                                selected = true;
                                break;
                            }
                        }
                    }
                }
                p += info->Size;
            }
        }
    }
    if (result.empty() && !topology_query_succeeded) {
        // Topology-query fallback. Windows processor groups are at most 64
        // processors, so retain the group identity in the encoded ID. If the
        // OS returned malformed variable-size records, leave workers unpinned
        // instead of mistaking SMT siblings for distinct physical cores.
        const WORD groups = GetActiveProcessorGroupCount();
        for (WORD group = 0; group < groups; ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            if (count == 0 || count == (DWORD) -1 || count > 64) continue;
            for (DWORD bit = 0; bit < count; ++bit)
                result.push_back((int) group * 64 + (int) bit);
        }
    }
#elif defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_affinity = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    auto read_topology = [](int cpu, const char* key, int& value) {
        const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + key;
        std::ifstream in(path);
        return static_cast<bool>(in >> value);
    };
    std::vector<std::pair<int, int>> physical_ids;
    const int max_cpu = have_affinity ? CPU_SETSIZE : (int) std::thread::hardware_concurrency();
    for (int cpu = 0; cpu < max_cpu; ++cpu) {
        if (have_affinity && !CPU_ISSET(cpu, &allowed)) continue;
        int package = -1, core = -1;
        if (read_topology(cpu, "physical_package_id", package) && read_topology(cpu, "core_id", core)) {
            const std::pair<int, int> key{package, core};
            if (std::find(physical_ids.begin(), physical_ids.end(), key) != physical_ids.end()) continue;
            physical_ids.push_back(key);
        }
        // If sysfs is unavailable, treat each allowed CPU as a separate core.
        result.push_back(cpu);
    }
#else
    const unsigned n = std::thread::hardware_concurrency();
    for (unsigned i = 0; i < n; ++i) result.push_back((int) i);
#endif
    return result;
}

#if defined(_WIN32)
bool cpu_set_for_core(int core, ULONG& id) {
    if (core < 0) return false;
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    std::vector<unsigned char> buffer(bytes);
    if (bytes && !GetSystemCpuSetInformation(
            reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), bytes, &bytes,
            GetCurrentProcess(), 0)) return false;
    constexpr size_t header_size = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
    for (size_t offset = 0; offset + header_size <= bytes;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Size < header_size || info->Size > bytes - offset) break;
        if (info->Type == CpuSetInformation && info->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION) &&
            info->CpuSet.Group == core / 64 && info->CpuSet.LogicalProcessorIndex == (core & 63) &&
            (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess)) {
            id = info->CpuSet.Id;
            return true;
        }
        offset += info->Size;
    }
    SetLastError(ERROR_NOT_FOUND);
    return false;
}

bool get_thread_selected_cpu_sets(std::vector<ULONG>& ids) {
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    ids.resize(count);
    if (count == 0) return true;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), count, &count)) return false;
    ids.resize(count);
    return true;
}

#endif

bool pin_worker(int cpu) noexcept {
    if (cpu < 0) return false;
#if defined(_WIN32)
    GROUP_AFFINITY target{};
    target.Group = (WORD) (cpu / 64);
    target.Mask = KAFFINITY(1) << (cpu & 63);
    return SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr) != 0;
#elif defined(__linux__)
    if (cpu >= CPU_SETSIZE) return false;
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    return pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected) == 0;
#else
    (void) cpu;
    return false;
#endif
}

void row_range(int64_t rows, int participants, int rank, int64_t& begin, int64_t& end) noexcept {
    const int64_t base = rows / participants;
    const int64_t extra = rows % participants;
    begin = base * rank + (rank < extra ? rank : extra);
    end = begin + base + (rank < extra ? 1 : 0);
}

}  // namespace

CpuPool::CallerScope::CallerScope(bool enabled, unsigned long target_cpu_set_id) {
#if defined(_WIN32)
    if (!enabled) return;
    const ULONG target = (ULONG) target_cpu_set_id;
    if (!get_thread_selected_cpu_sets(previous_cpu_sets_)) {
        std::fprintf(stderr, "CpuPool: could not save caller CPU-set selection (%lu); caller remains unpinned\n",
                     (unsigned long) GetLastError());
        return;
    }
    // An outer forward() scope already selected this CPU. Do not issue another
    // set/restore pair for each gate/up and down phase inside that scope.
    if (previous_cpu_sets_.size() == 1 && (ULONG) previous_cpu_sets_[0] == target) return;
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), &target, 1)) {
        std::fprintf(stderr, "CpuPool: could not select caller CPU set %lu (%lu); caller remains unpinned\n",
                     (unsigned long) target, (unsigned long) GetLastError());
        return;
    }
    active_ = true;
#else
    (void) enabled;
    (void) target_cpu_set_id;
#endif
}

CpuPool::CallerScope::~CallerScope() noexcept {
#if defined(_WIN32)
    if (!active_) return;
    const ULONG* ids = previous_cpu_sets_.empty() ? nullptr : previous_cpu_sets_.data();
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), ids, (ULONG) previous_cpu_sets_.size()))
        std::fprintf(stderr, "CpuPool: could not restore caller CPU-set selection (%lu)\n",
                     (unsigned long) GetLastError());
#endif
}

CpuPool::CpuPool(int total_threads, bool pin, bool pin_caller)
    : requested_threads_(total_threads), pin_caller_(pin && pin_caller) {
    if (total_threads < 1) throw std::invalid_argument("CpuPool requires at least one total thread");

    std::vector<int> cores;
    if (pin) cores = physical_processors();
    total_threads_ = total_threads;
    if (pin && !cores.empty())
        total_threads_ = (std::min)(total_threads_, (int) cores.size());
    worker_count_ = total_threads_ - 1;
    worker_cpus_.assign((size_t) worker_count_, -1);
    workers_.reserve((size_t) worker_count_);
    if (pin) {
        // Reserve the first physical core for the run's caller, then pin workers
        // to the remaining cores. Windows applies the caller CPU set per run.
#if defined(_WIN32)
        if (pin_caller_ && !cores.empty()) {
            caller_cpu_ = cores.front();
            if (!cpu_set_for_core(caller_cpu_, caller_cpu_set_id_)) {
                std::fprintf(stderr, "CpuPool: could not map caller CPU %d to a Windows CPU set (%lu); caller remains unpinned\n",
                             caller_cpu_, (unsigned long) GetLastError());
                caller_cpu_ = -1;
            } else {
                caller_cpu_set_valid_ = true;
            }
        }
#endif
        if (!cores.empty()) cores.erase(cores.begin());
    }
    try {
        for (int i = 0; i < worker_count_; ++i) {
            const int preferred = pin && (size_t) i < cores.size() ? cores[(size_t) i] : -1;
            workers_.emplace_back([this, i, preferred] { worker_loop(i, preferred); });
        }
    } catch (...) {
        stop_and_join();
        throw;
    }

    int ready = started_.load(std::memory_order_acquire);
    while (ready < worker_count_) {
        started_.wait(ready, std::memory_order_acquire);
        ready = started_.load(std::memory_order_acquire);
    }
}

CpuPool::CallerScope CpuPool::scoped_caller() const {
#if defined(_WIN32)
    return CallerScope(pin_caller_ && caller_cpu_set_valid_, caller_cpu_set_id_);
#else
    return CallerScope(false, 0);
#endif
}

CpuPool::~CpuPool() { stop_and_join(); }

void CpuPool::stop_and_join() noexcept {
    stopping_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    epoch_.notify_all();
    for (std::thread& worker : workers_)
        if (worker.joinable()) worker.join();
}

void CpuPool::worker_loop(int worker_id, int preferred_cpu) noexcept {
    if (pin_worker(preferred_cpu)) worker_cpus_[(size_t) worker_id] = preferred_cpu;
    started_.fetch_add(1, std::memory_order_release);
    started_.notify_all();

    uint64_t seen = 0;
    for (;;) {
        uint64_t current = epoch_.load(std::memory_order_acquire);
        while (current == seen) {
            epoch_.wait(seen, std::memory_order_acquire);
            current = epoch_.load(std::memory_order_acquire);
        }
        seen = current;
        if (stopping_.load(std::memory_order_acquire)) return;

        int64_t begin = 0, end = 0;
        row_range(rows_, total_threads_, worker_id + 1, begin, end);
        if (begin < end) fn_(context_, begin, end);

        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) remaining_.notify_one();
    }
}

void CpuPool::run(int64_t rows, RowFunction fn, void* context) {
    if (rows < 0) throw std::invalid_argument("CpuPool rows must be non-negative");
    if (rows == 0) return;
    if (fn == nullptr) throw std::invalid_argument("CpuPool callback must not be null");

    std::lock_guard<std::mutex> lock(run_mutex_);
#if defined(_WIN32)
    auto caller_cpu_set = scoped_caller();
    (void) caller_cpu_set;
#endif
    rows_ = rows;
    fn_ = fn;
    context_ = context;
    remaining_.store(worker_count_, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);
    epoch_.notify_all();

    int64_t begin = 0, end = 0;
    row_range(rows, total_threads_, 0, begin, end);
    if (begin < end) fn(context, begin, end);

    int left = remaining_.load(std::memory_order_acquire);
    while (left != 0) {
        remaining_.wait(left, std::memory_order_acquire);
        left = remaining_.load(std::memory_order_acquire);
    }
}

}  // namespace dsv4
