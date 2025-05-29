#ifndef RDMA_LIBRARY_HUGEPAGE_HH
#define RDMA_LIBRARY_HUGEPAGE_HH

#include <sys/mman.h>

#include <fstream>

#include "types.hh"
#include "utils.hh"

#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)

template <typename T, bool HUGE_1GB = true>
class HugePage {
public:
  HugePage() = default;
  explicit HugePage(size_t size) { allocate(size); }
  ~HugePage() { deallocate(); }

  HugePage(HugePage&) = delete;
  HugePage& operator=(HugePage&) = delete;

  void allocate(size_t size) {
    lib_assert(buffer_size == 0, "Buffer has been already allocated");
    buffer_size = size;
    buffer_length = size / sizeof(T);
    size_left_ = size;

#ifdef NOHUGEPAGES
    buffer_ = static_cast<T*>(std::aligned_alloc(64, buffer_size));
    lib_assert(reinterpret_cast<u64>(buffer_) % 64 == 0,
               "Not cache-line aligned");
    std::cerr << "allocated ALIGNED MEM (no hugepage) at "
              << reinterpret_cast<u64>(buffer_) << " with buffer size "
              << buffer_size << std::endl;
#else
    print_status("map huge page");
    void* ptr = mmap(NULL,
                     buffer_size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
                       (HUGE_1GB ? MAP_HUGE_1GB : MAP_HUGE_2MB),
                     -1,
                     0);
    lib_assert(ptr != MAP_FAILED, "Allocating huge-pages failed");
    lib_assert(reinterpret_cast<u64>(ptr) % 64 == 0, "alignment failed");
    buffer_ = static_cast<T*>(ptr);
    std::cerr << "allocated HUGEPAGE at " << reinterpret_cast<u64>(buffer_)
              << " with buffer size " << buffer_size << std::endl;
#endif

    bump_pointer_ = buffer_;
  }

  T* get_slice_unaligned(size_t size_in_bytes) {
    lib_assert(size_left_ >= size_in_bytes,
               "Pre-allocated hugepage memory exhausted");

    T* slice = static_cast<T*>(bump_pointer_);
    bump_pointer_ = static_cast<byte_t*>(bump_pointer_) + size_in_bytes;
    size_left_ -= size_in_bytes;

    return slice;
  }

  T* get_slice(size_t size_in_bytes) {
    lib_assert(size_left_ >= size_in_bytes,
               "Pre-allocated hugepage memory exhausted");

    lib_assert(
      std::align(64, size_in_bytes, bump_pointer_, size_left_) != nullptr,
      "alignment failed");

    T* slice = static_cast<T*>(bump_pointer_);
    bump_pointer_ = static_cast<byte_t*>(bump_pointer_) + size_in_bytes;
    size_left_ -= size_in_bytes;

    lib_assert(reinterpret_cast<u64>(slice) % 64 == 0, "alignment failed");

    return slice;
  }

  T* get_full_buffer() const { return buffer_; }

  void deallocate() {
    if (buffer_ != nullptr) {
#ifdef NOHUGEPAGES
      std::free(buffer_);
#else
      munmap(static_cast<void*>(buffer_), buffer_size);
#endif
      buffer_ = nullptr;
      bump_pointer_ = nullptr;
    }

    buffer_length = 0;
    buffer_size = 0;
  }

  size_t get_num_hugepages() const {
    std::ifstream is("/proc/sys/vm/nr_hugepages");
    size_t num_hugepages;

    lib_assert(is.good(), "Cannot read the number of available hugepages");
    lib_assert((is >> num_hugepages), "Cannot get the number of hugepages");

    return num_hugepages;
  }

  size_t get_memory_size() const {
#ifdef NOHUGEPAGES
    return 48UL * 1024UL * 1024UL * 1024UL;  // 48 GB
#else
    return get_num_hugepages() * 1024UL * (HUGE_1GB ? 1024UL * 1024UL : 2048UL);
#endif
  }

  T& operator[](size_t idx) { return *(buffer_ + idx); }

  void touch_memory() {
    for (size_t i = 0; i < buffer_length; ++i) {
      buffer_[i] = 0;
    }
  }

public:
  size_t buffer_size{0};
  size_t buffer_length{0};

private:
  T* buffer_{nullptr};
  void* bump_pointer_{nullptr};
  size_t size_left_{0};
};

#endif  // RDMA_LIBRARY_HUGEPAGE_HH
