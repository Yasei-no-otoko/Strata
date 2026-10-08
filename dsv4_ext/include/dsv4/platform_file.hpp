#pragma once

// Read-only whole-file mapping plus offset-based reads for large GGUF shards.
// The GGUF reader uses stdio only for the small metadata section; model weights
// are accessed through this abstraction without loading the shards into RAM.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dsv4 {

#if defined(_WIN32)
inline bool utf8_to_wide(const std::string& text, std::wstring& wide) {
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, nullptr, 0);
    if (n <= 0) return false;
    wide.resize(static_cast<size_t>(n));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, &wide[0], n) == n;
}
#endif

inline FILE* open_binary_file(const std::string& path, bool write) {
#if defined(_WIN32)
    std::wstring wide;
    if (utf8_to_wide(path, wide)) {
        return _wfopen(wide.c_str(), write ? L"wb" : L"rb");
    }
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#else
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
}

inline bool atomic_replace_file(const std::string& source, const std::string& destination) {
#if defined(_WIN32)
    std::wstring source_wide, destination_wide;
    const DWORD flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    if (utf8_to_wide(source, source_wide) && utf8_to_wide(destination, destination_wide))
        return MoveFileExW(source_wide.c_str(), destination_wide.c_str(), flags) != 0;
    return MoveFileExA(source.c_str(), destination.c_str(), flags) != 0;
#else
    return std::rename(source.c_str(), destination.c_str()) == 0;
#endif
}

class ReadOnlyFile {
public:
    ReadOnlyFile() = default;
    ~ReadOnlyFile() { close(); }
    ReadOnlyFile(const ReadOnlyFile&) = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    ReadOnlyFile(ReadOnlyFile&& other) noexcept { take(other); }
    ReadOnlyFile& operator=(ReadOnlyFile&& other) noexcept {
        if (this != &other) { close(); take(other); }
        return *this;
    }

    bool open(const std::string& path, std::string& err) {
        close();
#if defined(_WIN32)
        std::wstring wide;
        HANDLE file = INVALID_HANDLE_VALUE;
        const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED;
        if (utf8_to_wide(path, wide)) {
            file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, flags, nullptr);
        } else {
            file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, flags, nullptr);
        }
        if (file == INVALID_HANDLE_VALUE) { err = "cannot open " + path; return false; }
        LARGE_INTEGER n{};
        if (!GetFileSizeEx(file, &n) || n.QuadPart <= 0) {
            CloseHandle(file); err = "cannot get file size for " + path; return false;
        }
        if (static_cast<uint64_t>(n.QuadPart) > static_cast<uint64_t>(SIZE_MAX)) {
            CloseHandle(file); err = "file is too large to map in this process: " + path; return false;
        }
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) { CloseHandle(file); err = "cannot create read-only mapping for " + path; return false; }
        void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) { CloseHandle(mapping); CloseHandle(file); err = "cannot map " + path; return false; }
        file_ = file; mapping_ = mapping; data_ = static_cast<const uint8_t*>(view); size_ = static_cast<uint64_t>(n.QuadPart);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { err = "cannot open " + path; return false; }
        struct stat st{};
        if (fstat(fd_, &st) != 0 || st.st_size <= 0) { close(); err = "cannot get file size for " + path; return false; }
        size_ = static_cast<uint64_t>(st.st_size);
        void* view = mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_SHARED, fd_, 0);
        if (view == MAP_FAILED) { close(); err = "cannot map " + path; return false; }
        data_ = static_cast<const uint8_t*>(view);
#if defined(MADV_RANDOM)
        (void) madvise(const_cast<uint8_t*>(data_), static_cast<size_t>(size_), MADV_RANDOM);
#endif
#endif
        return true;
    }

    void close() noexcept {
#if defined(_WIN32)
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ && file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE; mapping_ = nullptr;
#else
        if (data_) munmap(const_cast<uint8_t*>(data_), static_cast<size_t>(size_));
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr; size_ = 0;
    }

    const uint8_t* data() const noexcept { return data_; }
    uint64_t size() const noexcept { return size_; }

    bool read_at(uint64_t offset, uint8_t* dst, size_t bytes) const noexcept {
        if (offset > size_ || static_cast<uint64_t>(bytes) > size_ - offset) return false;
        size_t done = 0;
        while (done < bytes) {
#if defined(_WIN32)
            const size_t remaining = bytes - done;
            const DWORD chunk = static_cast<DWORD>(remaining > 0x7ffff000u ? 0x7ffff000u : remaining);
            OVERLAPPED ov{};
            const uint64_t pos = offset + done;
            ov.Offset = static_cast<DWORD>(pos & 0xffffffffu);
            ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
            ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!ov.hEvent) return false;
            DWORD got = 0;
            BOOL ok = ReadFile(file_, dst + done, chunk, &got, &ov);
            if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(file_, &ov, &got, TRUE);
            CloseHandle(ov.hEvent);
            if (!ok || got == 0) return false;
            done += got;
#else
            const uint64_t pos = offset + done;
            if (pos > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) return false;
            const ssize_t got = pread(fd_, dst + done, bytes - done, static_cast<off_t>(pos));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) return false;
            done += static_cast<size_t>(got);
#endif
        }
        return true;
    }

private:
    void take(ReadOnlyFile& other) noexcept {
        data_ = other.data_; size_ = other.size_;
        other.data_ = nullptr; other.size_ = 0;
#if defined(_WIN32)
        file_ = other.file_; mapping_ = other.mapping_;
        other.file_ = INVALID_HANDLE_VALUE; other.mapping_ = nullptr;
#else
        fd_ = other.fd_; other.fd_ = -1;
#endif
    }

    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace dsv4
