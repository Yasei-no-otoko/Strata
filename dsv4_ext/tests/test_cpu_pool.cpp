#include "dsv4/cpu_pool.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstddef>
#endif

namespace {

struct CheckContext {
    std::atomic<int>* counts;
    int* values;
    int tag;
};

void record_rows(void* opaque, int64_t begin, int64_t end) noexcept {
    auto& ctx = *static_cast<CheckContext*>(opaque);
    for (int64_t row = begin; row < end; ++row) {
        ctx.values[(size_t) row] = ctx.tag + (int) row;
        ctx.counts[(size_t) row].fetch_add(1, std::memory_order_relaxed);
    }
}

bool check_batch(dsv4::CpuPool& pool, int64_t rows, int tag) {
    std::unique_ptr<std::atomic<int>[]> counts(new std::atomic<int>[(size_t) rows]);
    std::vector<int> values((size_t) rows, -1);
    for (int64_t row = 0; row < rows; ++row) counts[(size_t) row].store(0);
    CheckContext context{counts.get(), values.data(), tag};
    pool.run(rows, record_rows, &context);
    for (int64_t row = 0; row < rows; ++row) {
        if (counts[(size_t) row].load() != 1 || values[(size_t) row] != tag + row) {
            std::fprintf(stderr, "CpuPool row %lld was missing, duplicated, or stale (threads=%d, tag=%d)\n",
                         (long long) row, pool.threads(), tag);
            return false;
        }
    }
    return true;
}

bool check_pool(int threads, bool pin) {
    dsv4::CpuPool pool(threads, pin);
    if (pool.requested_threads() != threads || pool.threads() < 1 || pool.threads() > threads ||
        (!pin && pool.threads() != threads) || pool.worker_cpus().size() != (size_t) (pool.threads() - 1)) {
        std::fprintf(stderr, "CpuPool reported the wrong participant count for %d threads\n", threads);
        return false;
    }
    const auto cpus = pool.worker_cpus();
    for (size_t i = 0; i < cpus.size(); ++i) {
        if (!pin && cpus[i] != -1) {
            std::fprintf(stderr, "CpuPool reported a pinned worker when pinning was disabled\n");
            return false;
        }
        if (cpus[i] >= 0)
            for (size_t j = 0; j < i; ++j)
                if (cpus[i] == cpus[j]) {
                    std::fprintf(stderr, "CpuPool assigned two workers to processor %d\n", cpus[i]);
                    return false;
                }
    }

    pool.run(0, nullptr, nullptr);
    for (int repeat = 0; repeat < 5; ++repeat) {
        const int64_t row_counts[] = {1, 3, (int64_t) threads - 1, (int64_t) threads, 17, 257};
        for (int64_t rows : row_counts) {
            if (!check_batch(pool, rows, repeat * 1000 + (int) rows)) return false;
        }
    }
    return true;
}

#if defined(_WIN32)
struct WindowsThreadReadback {
    WORD group = 0;
    KAFFINITY mask = 0;
    WORD current_group = 0;
    BYTE current_number = 0;
    ULONG selected_cpu_set_id = 0;
    ULONG selected_cpu_set_count = 0;
    bool affinity_ok = false;
    bool current_ok = false;
    bool selected_cpu_sets_ok = false;
};

struct WindowsReadbackContext {
    WindowsThreadReadback* records;
    ULONG expected_caller_cpu_set_id;
};

void readback_thread_affinity(void* opaque, int64_t begin, int64_t end) noexcept {
    auto& context = *static_cast<WindowsReadbackContext*>(opaque);
    for (int64_t rank = begin; rank < end; ++rank) {
        auto& record = context.records[(size_t) rank];
        GROUP_AFFINITY affinity{};
        PROCESSOR_NUMBER current{};
        record.affinity_ok = GetThreadGroupAffinity(GetCurrentThread(), &affinity) != 0;
        if (record.affinity_ok) {
            record.group = affinity.Group;
            record.mask = affinity.Mask;
        }
        if (rank == 0) {
            ULONG selected[8]{};
            ULONG count = 0;
            record.selected_cpu_sets_ok =
                GetThreadSelectedCpuSets(GetCurrentThread(), selected, 8, &count) != 0;
            record.selected_cpu_set_count = count;
            if (record.selected_cpu_sets_ok && count == 1)
                record.selected_cpu_set_id = selected[0];
        }
        GetCurrentProcessorNumberEx(&current);
        record.current_ok = true;
        if (record.current_ok) {
            record.current_group = current.Group;
            record.current_number = current.Number;
        }
    }
}

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
    return false;
}

bool get_selected_cpu_sets(std::vector<ULONG>& ids) {
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    ids.resize(count);
    if (count == 0) return true;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), count, &count)) return false;
    ids.resize(count);
    return true;
}

bool check_windows_processor_groups() {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) {
        std::fprintf(stderr, "CpuPool topology test could not query processor records\n");
        return false;
    }
    std::vector<unsigned char> records(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(records.data()), &bytes)) {
        std::fprintf(stderr, "CpuPool topology test could not read processor records\n");
        return false;
    }

    constexpr size_t kProcessorOffset = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
    constexpr size_t kGroupCountEnd = kProcessorOffset + offsetof(PROCESSOR_RELATIONSHIP, GroupCount) + sizeof(WORD);
    constexpr size_t kGroupMaskOffset = kProcessorOffset + offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
    const unsigned char* p = records.data();
    const unsigned char* const end = p + bytes;
    std::vector<int> physical_ids;
    unsigned logical_count = 0;
    while ((size_t) (end - p) >= kProcessorOffset) {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        if (info->Size < kProcessorOffset || (size_t) info->Size > (size_t) (end - p)) return false;
        if (info->Relationship == RelationProcessorCore) {
            if (info->Size < kGroupCountEnd) return false;
            const size_t groups = (size_t) info->Processor.GroupCount;
            if (groups > ((size_t) info->Size - kGroupMaskOffset) / sizeof(GROUP_AFFINITY)) return false;
            bool selected = false;
            for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY& group = info->Processor.GroupMask[g];
                const uint64_t mask = (uint64_t) group.Mask;
                logical_count += (unsigned) std::popcount(mask);
                for (int bit = 0; bit < 64; ++bit) {
                    if ((mask & (uint64_t(1) << bit)) != 0 && !selected) {
                        physical_ids.push_back((int) group.Group * 64 + bit);
                        selected = true;
                    }
                }
            }
        }
        p += info->Size;
    }
    if (p != end || physical_ids.empty()) {
        std::fprintf(stderr, "CpuPool topology test found malformed or empty core records\n");
        return false;
    }

    unsigned active_logical_count = 0;
    for (WORD group = 0; group < GetActiveProcessorGroupCount(); ++group) {
        const DWORD count = GetActiveProcessorCount(group);
        if (count != (DWORD) -1) active_logical_count += count;
    }
    if (logical_count != active_logical_count) {
        std::fprintf(stderr, "CpuPool topology test saw %u logical processors in core masks; Windows reports %u active\n",
                     logical_count, active_logical_count);
        return false;
    }

    // A small pool suffices: a logical-processor fallback would choose an SMT
    // sibling of the reserved host core as worker 0 instead of the next core's
    // primary processor. Compare actual pins against the OS's per-core masks.
    const int wanted = (std::min)(4, (int) physical_ids.size());
    dsv4::CpuPool pool(wanted, true);
    if (pool.threads() != wanted) {
        std::fprintf(stderr, "CpuPool reported %d threads for %d expected physical-core participants\n",
                     pool.threads(), wanted);
        return false;
    }
    const auto workers = pool.worker_cpus();
    bool any_pinned = false;
    for (size_t i = 0; i < workers.size(); ++i) {
        if (workers[i] < 0) continue;  // affinity can be denied by a restricted process/job.
        any_pinned = true;
        if (workers[i] != physical_ids[i + 1]) {
            std::fprintf(stderr, "CpuPool worker %zu pinned to %d, Windows core topology expects %d\n",
                         i, workers[i], physical_ids[i + 1]);
            return false;
        }
    }
    std::printf("CpuPool Windows topology matched %zu physical cores and %u active logical processors%s\n",
                physical_ids.size(), logical_count, any_pinned ? "; worker pins verified" : "; affinity unavailable");

    // Read the OS affinity back from inside every participant, not only from
    // successful SetThreadGroupAffinity calls or the pool's requested IDs.
    // One row per participant makes the callback's range start its rank.
    const int requested_sizes[] = {32, 64};
    for (int requested : requested_sizes) {
        const int participants = (std::min)(requested, (int) physical_ids.size());
        GROUP_AFFINITY caller_before{};
        const bool caller_affinity_before_ok =
            GetThreadGroupAffinity(GetCurrentThread(), &caller_before) != 0;
        std::vector<ULONG> caller_sets_before;
        if (!get_selected_cpu_sets(caller_sets_before)) {
            std::fprintf(stderr, "CpuPool readback could not query the caller's selected CPU sets\n");
            return false;
        }
        dsv4::CpuPool large_pool(requested, true);
        if (large_pool.threads() != participants) {
            std::fprintf(stderr, "CpuPool requested %d participants but created %d (expected %d from physical topology)\n",
                         requested, large_pool.threads(), participants);
            return false;
        }
        ULONG expected_caller_cpu_set = 0;
        if (large_pool.caller_cpu() != physical_ids.front() ||
            !cpu_set_for_core(large_pool.caller_cpu(), expected_caller_cpu_set)) {
            std::fprintf(stderr, "CpuPool did not select the reserved caller physical core %d\n",
                         physical_ids.front());
            return false;
        }
        const auto reported = large_pool.worker_cpus();
        std::vector<WindowsThreadReadback> readbacks((size_t) participants);
        WindowsReadbackContext context{readbacks.data(), expected_caller_cpu_set};
        large_pool.run(participants, readback_thread_affinity, &context);
        std::vector<ULONG> caller_sets_after;
        if (!get_selected_cpu_sets(caller_sets_after) || caller_sets_before != caller_sets_after) {
            std::fprintf(stderr, "CpuPool changed the caller's explicitly selected CPU sets\n");
            return false;
        }

        const auto& caller = readbacks[0];
        const int caller_actual_id = (int) caller.current_group * 64 + (int) caller.current_number;
        if (!caller.current_ok || !caller.selected_cpu_sets_ok || caller.selected_cpu_set_count != 1 ||
            caller.selected_cpu_set_id != context.expected_caller_cpu_set_id ||
            caller_actual_id != large_pool.caller_cpu()) {
            std::fprintf(stderr, "CpuPool caller selected set/current CPU differs from reserved CPU %d (set=%lu, count=%lu, current=%d)\n",
                         large_pool.caller_cpu(), (unsigned long) caller.selected_cpu_set_id,
                         (unsigned long) caller.selected_cpu_set_count, caller_actual_id);
            return false;
        }

        if (requested == 32) {
            bool nested_ok = true;
            {
                auto outer_scope = large_pool.scoped_caller();
                std::vector<ULONG> during_outer;
                if (!get_selected_cpu_sets(during_outer) || during_outer.size() != 1 ||
                    during_outer[0] != expected_caller_cpu_set) {
                    std::fprintf(stderr, "CpuPool outer caller scope did not select its cached CPU set\n");
                    nested_ok = false;
                } else {
                    WindowsThreadReadback nested_record;
                    WindowsReadbackContext nested_context{&nested_record, expected_caller_cpu_set};
                    large_pool.run(1, readback_thread_affinity, &nested_context);
                    std::vector<ULONG> after_inner;
                    if (!get_selected_cpu_sets(after_inner) || after_inner != during_outer ||
                        !nested_record.selected_cpu_sets_ok || nested_record.selected_cpu_set_count != 1 ||
                        nested_record.selected_cpu_set_id != expected_caller_cpu_set ||
                        !nested_record.current_ok ||
                        (int) nested_record.current_group * 64 + (int) nested_record.current_number !=
                            large_pool.caller_cpu()) {
                        std::fprintf(stderr, "CpuPool inner run did not preserve the outer caller CPU-set scope\n");
                        nested_ok = false;
                    }
                }
            }
            std::vector<ULONG> after_outer;
            if (!get_selected_cpu_sets(after_outer) || after_outer != caller_sets_before) {
                std::fprintf(stderr, "CpuPool outer caller scope did not restore the prior CPU-set selection\n");
                nested_ok = false;
            }
            if (!nested_ok) return false;
        }

        std::vector<int> actual_workers;
        actual_workers.reserve((size_t) (participants - 1));
        for (int rank = 1; rank < participants; ++rank) {
            const auto& actual = readbacks[(size_t) rank];
            const size_t worker = (size_t) (rank - 1);
            if (!actual.affinity_ok || !actual.current_ok || actual.current_group != actual.group ||
                (actual.mask & (KAFFINITY(1) << actual.current_number)) == 0) {
                std::fprintf(stderr, "CpuPool worker %d has no valid OS group-affinity readback\n", rank - 1);
                return false;
            }
            const int actual_id = (int) actual.current_group * 64 + (int) actual.current_number;
            if (worker >= reported.size() || reported[worker] < 0) {
                std::fprintf(stderr, "CpuPool worker %d has no reported pinned logical CPU\n", rank - 1);
                return false;
            }
            const WORD expected_group = (WORD) (reported[worker] / 64);
            const KAFFINITY expected_mask = KAFFINITY(1) << (reported[worker] & 63);
            if (reported[worker] != actual_id) {
                std::fprintf(stderr, "CpuPool worker %d reports CPU %d but is running on group %u processor %u\n",
                             rank - 1, reported[worker],
                             (unsigned) actual.current_group, (unsigned) actual.current_number);
                return false;
            }
            if (actual.group != expected_group || actual.mask != expected_mask) {
                std::fprintf(stderr, "CpuPool worker %d OS mask %u/%016llx differs from reported CPU %d\n",
                             rank - 1, (unsigned) actual.group, (unsigned long long) actual.mask, reported[worker]);
                return false;
            }
            if (worker + 1 >= physical_ids.size() || actual_id != physical_ids[worker + 1]) {
                std::fprintf(stderr, "CpuPool worker %d read back CPU %d, expected physical-core primary %d\n",
                             rank - 1, actual_id,
                             worker + 1 < physical_ids.size() ? physical_ids[worker + 1] : -1);
                return false;
            }
            if (std::find(actual_workers.begin(), actual_workers.end(), actual_id) != actual_workers.end()) {
                std::fprintf(stderr, "CpuPool readback found duplicate worker processor %d (possible SMT sibling assignment)\n",
                             actual_id);
                return false;
            }
            actual_workers.push_back(actual_id);
        }

        std::printf("CpuPool Windows OS-affinity readback requested=%d participants=%d; caller-before=%s%u/%016llx selected-cpu-sets=%zu; caller-target=%d/set%lu; caller-during=set%lu@%u:%u; caller-after=%s%u/%016llx@%s%u:%u;",
                    requested, participants, caller_affinity_before_ok ? "" : "unavailable/",
                    (unsigned) caller_before.Group, (unsigned long long) caller_before.Mask,
                    caller_sets_after.size(), large_pool.caller_cpu(),
                    (unsigned long) context.expected_caller_cpu_set_id,
                    (unsigned long) caller.selected_cpu_set_id,
                    (unsigned) caller.current_group, (unsigned) caller.current_number,
                    caller.affinity_ok ? "" : "unavailable/",
                    (unsigned) caller.group, (unsigned long long) caller.mask,
                    caller.current_ok ? "" : "unavailable/", (unsigned) caller.current_group,
                    (unsigned) caller.current_number);
        for (int rank = 0; rank < participants; ++rank) {
            const auto& record = readbacks[(size_t) rank];
            std::printf(" r%d=%s%u/%016llx@%s%u:%u", rank,
                        record.affinity_ok ? "" : "unavailable/", (unsigned) record.group,
                        (unsigned long long) record.mask,
                        record.current_ok ? "" : "unavailable/", (unsigned) record.current_group,
                        (unsigned) record.current_number);
        }
        std::putchar('\n');
        std::printf("CpuPool caller CPU-set restore requested=%d before=[", requested);
        for (size_t i = 0; i < caller_sets_before.size(); ++i)
            std::printf("%s%lu", i ? "," : "", (unsigned long) caller_sets_before[i]);
        std::printf("] after=[");
        for (size_t i = 0; i < caller_sets_after.size(); ++i)
            std::printf("%s%lu", i ? "," : "", (unsigned long) caller_sets_after[i]);
        std::printf("]\n");
    }
    return true;
}
#endif

}  // namespace

int main() {
    try {
        if (!check_pool(1, true) || !check_pool(2, true) || !check_pool(4, true) || !check_pool(4, false)) return 1;
#if defined(_WIN32)
        if (!check_windows_processor_groups()) return 1;
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CpuPool test threw unexpectedly: %s\n", e.what());
        return 1;
    }

    bool rejected_zero = false, rejected_null = false;
    try { dsv4::CpuPool invalid(0); } catch (const std::invalid_argument&) { rejected_zero = true; }
    dsv4::CpuPool pool(2, false);
    try { pool.run(1, nullptr, nullptr); } catch (const std::invalid_argument&) { rejected_null = true; }
    if (!rejected_zero || !rejected_null) {
        std::fprintf(stderr, "CpuPool did not reject invalid construction/callback arguments\n");
        return 1;
    }
    std::puts("CpuPool repeat-batch, exact-once, context-publication, and argument checks passed");
    return 0;
}
