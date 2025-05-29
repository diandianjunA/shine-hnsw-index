#pragma once

#include <ostream>

#include "common/types.hh"

struct RemotePtr {
  static constexpr size_t SIZE = sizeof(u64);
  u64 raw_address{};  // [ memory node (16b) | byte offset (48b) ]

  RemotePtr() = default;
  explicit RemotePtr(u64 raw_address) : raw_address(raw_address) {}
  RemotePtr(u32 memory_node, u64 byte_offset) { store_address(memory_node, byte_offset); }

  u32 memory_node() const { return raw_address >> 48; }
  u64 byte_offset() const { return (raw_address << 16) >> 16; }
  bool is_null() const { return raw_address == 0; }
  void reset() { raw_address = 0; }

  void store_address(u32 memory_node, u64 byte_offset) {
    raw_address = (static_cast<u64>(memory_node) << 48) | byte_offset;
  }

  bool operator==(const RemotePtr&) const = default;  // compares raw_address

  friend std::ostream& operator<<(std::ostream& os, const RemotePtr& r) {
    return os << "[node: " << r.memory_node() << " | offset: " << r.byte_offset() << "]";
  }
};

template <>
struct std::hash<RemotePtr> {
  size_t operator()(const RemotePtr& r) const noexcept {
    u64 h = std::hash<u64>{}(r.raw_address);

    // murmur64
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;

    // murmur32
    // h ^= h >> 16;
    // h *= 0x85ebca6b;
    // h ^= h >> 13;
    // h *= 0xc2b2ae35;
    // h ^= h >> 16;

    return h;
  }
};
