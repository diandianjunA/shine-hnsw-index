#pragma once

#include "common/constants.hh"
#include "node/node.hh"

/**
 * Manages the memory-registered buffer globally per compute node.
 * Each thread has its own freelist which however may be accessed concurrently.
 */
class BufferAllocator {
public:
  explicit BufferAllocator(u32 num_threads, bool use_cache = true) : use_cache_(use_cache) {
    if (use_cache) {
      local_buffer_.allocate(COMPUTE_NODE_MAX_MEMORY);
      local_buffer_.touch_memory();
      buffer_ptr_ = local_buffer_.get_full_buffer();
    } else {
      buffer_ptr_ = nullptr;
    }

    freelists_node_.resize(num_threads);
    freelists_layer_.resize(num_threads);
    freelists_layer_zero_.resize(num_threads);
  }

  HugePage<byte_t, false>& get_raw_buffer() { 
    static HugePage<byte_t, false> empty_buffer;
    return use_cache_ ? local_buffer_ : empty_buffer; 
  }

  bool use_cache() const { return use_cache_; }

  [[nodiscard]] byte_t* allocate_layer_zero(u32 thread_id) {
    return get_free_space(Node::NEIGHBORLIST_SIZE_ZERO, freelists_layer_zero_[thread_id]);
  }
  [[nodiscard]] byte_t* allocate_layer(u32 thread_id) {
    return get_free_space(Node::NEIGHBORLIST_SIZE, freelists_layer_[thread_id]);
  }
  [[nodiscard]] byte_t* allocate_node(u32 thread_id) {
    return get_free_space(Node::size_until_components(), freelists_node_[thread_id]);
  }

  [[nodiscard]] u64* allocate_pointer() { return reinterpret_cast<u64*>(allocate(sizeof(u64))); }

  void free_node(byte_t* ptr, u32 thread_id) { freelists_node_[thread_id].enqueue(ptr); }

  void free_layer_zero(byte_t* ptr, u32 thread_id) {
    std::memset(ptr, 0, sizeof(u32));  // reset size
    freelists_layer_zero_[thread_id].enqueue(ptr);
  }

  void free_layer(byte_t* ptr, u32 thread_id) {
    std::memset(ptr, 0, sizeof(u32));  // reset size
    freelists_layer_[thread_id].enqueue(ptr);
  }

  size_t allocated_memory() const { return bump_pointer_; }

private:
  static size_t align(size_t size) {
    while (size % CACHELINE_SIZE != 0) {
      ++size;
    }

    return size;
  }

  byte_t* get_free_space(size_t size, concurrent_queue<byte_t*>& freelist) {
    byte_t* ptr;

    if (!freelist.try_dequeue(ptr)) {
      ptr = allocate(size);
    }

    return ptr;
  }

  byte_t* allocate(size_t size) {
    if (!use_cache_) {
      return nullptr;
    }
    
    lib_assert(size > 0, "unable to allocate 0 bytes");

    byte_t* ptr = buffer_ptr_ + bump_pointer_.fetch_add(align(size));
    lib_assert(bump_pointer_ <= local_buffer_.buffer_size, "out of local memory");

    // do not track 8B pointers
    if (size > sizeof(u64)) {
      allocated_buffers_.push_back(ptr);
    }

    return ptr;
  }

private:
  byte_t* buffer_ptr_;
  std::atomic<idx_t> bump_pointer_{0};  // points to free space
  HugePage<byte_t, false> local_buffer_;
  bool use_cache_;

  // freelists per thread (but other threads may append to them)
  // this is significantly faster than having single global freelists
  vec<concurrent_queue<byte_t*>> freelists_layer_zero_;
  vec<concurrent_queue<byte_t*>> freelists_layer_;
  vec<concurrent_queue<byte_t*>> freelists_node_;

  concurrent_vec<byte_t*> allocated_buffers_;  // track valid pointers (for cache eviction)
};