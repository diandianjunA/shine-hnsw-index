#pragma once

#include "compute_thread.hh"
#include "remote_pointer.hh"

class Neighborlist {
public:
  Neighborlist(u32 level, byte_t* buffer_ptr, const u_ptr<ComputeThread>& owner)  // store owner for deallocation
      : level_(level), buffer_slice_(buffer_ptr), owner_(owner.get()) {}

  ~Neighborlist() {
    const u32 thread_id = owner_->get_id();  // TODO: get caller's (not owner's) thread_id?

    // append buffer to thread_id's thread-safe freelist
    if (level_ > 0) {
      owner_->buffer_allocator.free_layer(buffer_slice_, thread_id);
    } else {
      owner_->buffer_allocator.free_layer_zero(buffer_slice_, thread_id);
    }
  }

  Neighborlist(const Neighborlist&) = delete;  // copy constructor
  Neighborlist(Neighborlist&&) noexcept = delete;  // move constructor
  Neighborlist& operator=(const Neighborlist&) = delete;  // copy assignment
  Neighborlist& operator=(Neighborlist&&) noexcept = delete;  // move assignment

  void add(RemotePtr& rptr) {
    u32* size_ptr = reinterpret_cast<u32*>(buffer_slice_);
    u64* neighbor_ptr = reinterpret_cast<u64*>(buffer_slice_ + sizeof(u32)) + *size_ptr;

    *size_ptr = *size_ptr + 1;  // increase size
    *neighbor_ptr = rptr.raw_address;
  }

  void reset() { *reinterpret_cast<u32*>(buffer_slice_) = 0; }  // reset size
  span<RemotePtr> view() const { return {reinterpret_cast<RemotePtr*>(buffer_slice_ + sizeof(u32)), num_neighbors()}; }

  u32 num_neighbors() const { return *reinterpret_cast<u32*>(buffer_slice_); }
  byte_t* buffer_ptr() const { return buffer_slice_; }
  u32 level() const { return level_; }

private:
  const u32 level_;
  byte_t* buffer_slice_{};
  ComputeThread* owner_{};
};