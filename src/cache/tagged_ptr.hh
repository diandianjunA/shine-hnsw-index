#pragma once

#include "common/types.hh"

template <typename T>
struct tagged_ptr {
  void* raw_pointer{};

  static constexpr u64 tag_mask = ((1ULL << 16) - 1) << 48;  // (2^16 - 1) << 48
  static constexpr u64 ptr_mask = ~tag_mask;

  tagged_ptr() = default;
  explicit tagged_ptr(T* ptr) : raw_pointer(ptr) {}
  tagged_ptr(T* ptr, u16 tag) {
    raw_pointer = reinterpret_cast<void*>((static_cast<u64>(tag) << 48) | reinterpret_cast<u64>(ptr));
  }

  bool operator==(const tagged_ptr& other) const { return raw_pointer == other.raw_pointer; }

  T* get() { return reinterpret_cast<T*>(reinterpret_cast<u64>(raw_pointer) & ptr_mask); }
  u16 tag() const { return (reinterpret_cast<u64>(__atomic_load_n(&raw_pointer, __ATOMIC_ACQUIRE)) & tag_mask) >> 48; }
  void invalidate() { __atomic_store_n(&raw_pointer, nullptr, __ATOMIC_RELEASE); }

  void update_tag(u16 tag) {
    __atomic_store_n(&raw_pointer,
                     reinterpret_cast<void*>((static_cast<u64>(tag) << 48) | reinterpret_cast<u64>(get())),
                     __ATOMIC_RELEASE);
  }
};
