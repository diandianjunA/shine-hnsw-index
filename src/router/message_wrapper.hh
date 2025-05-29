#pragma once

#include "common/types.hh"

namespace query_router {

class MessageWrapper {
public:
  explicit MessageWrapper(byte_t* ptr) : raw_ptr_(ptr) {}

  u32 destination() const { return *reinterpret_cast<u32*>(raw_ptr_); }
  node_t query_id() const { return *reinterpret_cast<node_t*>(raw_ptr_ + sizeof(u32)); }

  span<element_t> components() const {
    return {reinterpret_cast<element_t*>(raw_ptr_ + sizeof(u32) + sizeof(node_t)), Node::DIM};
  }

private:
  byte_t* raw_ptr_;
};

class AckMessageWrapper {
public:
  explicit AckMessageWrapper(byte_t* ptr) : raw_ptr_(ptr) {}
  static constexpr u32 msg_header = static_cast<u32>(-3);

  u32 destination() const { return *reinterpret_cast<u32*>(raw_ptr_); }
  u32 header() const { return *reinterpret_cast<u32*>(raw_ptr_ + sizeof(u32)); }
  u32 sender() const { return *reinterpret_cast<u32*>(raw_ptr_ + 2 * sizeof(u32)); }
  u32 progress() const { return *reinterpret_cast<u32*>(raw_ptr_ + 3 * sizeof(u32)); }

  void set(u32 destination, u32 sender, u32 progress) {
    *reinterpret_cast<u32*>(raw_ptr_) = destination;
    *reinterpret_cast<u32*>(raw_ptr_ + sizeof(u32)) = msg_header;
    *reinterpret_cast<u32*>(raw_ptr_ + 2 * sizeof(u32)) = sender;
    *reinterpret_cast<u32*>(raw_ptr_ + 3 * sizeof(u32)) = progress;
  }

private:
  byte_t* raw_ptr_;
};

}  // namespace query_router
