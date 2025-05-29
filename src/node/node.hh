#pragma once

#include <ostream>

#include "remote_pointer.hh"

// forward declaration
class ComputeThread;

/**
 *  Node layout: [
 *     header: 8B                           | ... | ... | is_entry_node(1b) | ... | new_lvl_lock(1b) | ... | lock(1b) |
 *                                                  ^--------- 1B ---------^ ^--------- 1B ---------^ ^----- 1B -----^
 *     meta: 2 * 4B                         | uid(4) | level(4) |
 *     components: d * 4B                   | d_1(4) | ... | d_d(4) |
 *     base-layer: 4B + M_max_0 * 8B        | #neighbors(4) | l_0_1(8) | ... | l_0_M(8) |
 *     upper layer(s) l * (4B + M_max * 8B) | ... |                                        <- only if node's level > 0
 *   ]
 */

/**
 * @brief Provides a view of the underlying local memory buffer.
 *        Stores a pointer to the local buffer, where the node has been read from remote memory.
 *        Stores a remote pointer to remote memory, where the node is physically stored.
 */
class Node {
public:  // static storage
  static constexpr size_t HEADER_NODE_LOCK = 0b01;
  static constexpr size_t HEADER_NEW_LEVEL_LOCK = 0b100000000;
  static constexpr size_t HEADER_ENTRY_NODE = 0b10000000000000000;
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t META_SIZE = sizeof(u64);

  // header positions (little endian !!!)
  static constexpr size_t HEADER_UNTIL_LOCK = 0;
  static constexpr size_t HEADER_UNTIL_LVL_LOCK = 1;
  static constexpr size_t HEADER_UNTIL_ENTRY_NODE = 2;

  // initialized by the HNSW instance
  inline static u32 DIM;
  inline static u32 NEIGHBORLIST_SIZE_ZERO;
  inline static u32 NEIGHBORLIST_SIZE;

  static void init_static_storage(u32 dim, u32 m_max, u32 m_max_zero) {
    NEIGHBORLIST_SIZE_ZERO = sizeof(u32) + m_max_zero * sizeof(u64);
    NEIGHBORLIST_SIZE = sizeof(u32) + m_max * sizeof(u64);
    DIM = dim;
  }

  static size_t size_until_components() { return HEADER_SIZE + META_SIZE + DIM * sizeof(element_t); }
  static size_t total_size(u32 level) {
    const size_t neighborlist_size = NEIGHBORLIST_SIZE_ZERO + level * NEIGHBORLIST_SIZE;
    return size_until_components() + neighborlist_size;
  }

public:  // member storage
  Node() = default;
  Node(byte_t* buffer_ptr, const RemotePtr& rptr, ComputeThread* owner)  // store owner for deallocation
      : owner_(owner), buffer_slice_(buffer_ptr), rptr(rptr) {}

  /** Rule of 5: Intended to use Node only with a shared_ptr in containers (to not destructing the object while moving).
   *             A shared_ptr makes sense, e.g., a Node can be a `top_candidate` and a `next_candidate`. If it's no
   *             longer any kind of candidate, the object can be deconstructed (and the buffer pointer should be
   *             freed/passed back to the owner).
   *             Another solution would be to implement the copy/move constructors (and assignment operatorss) and
   *             invalidate `buffer_slice_` and the pointers in `neighborlist_ptrs_`.
   */
  Node(const Node&) = delete;  // copy constructor
  Node(Node&&) noexcept = delete;  // move constructor
  Node& operator=(const Node&) = delete;  // copy assignment
  Node& operator=(Node&&) noexcept = delete;  // move assignment

  /**
   * @brief Frees the node and its neighbor lists using `owner`'s local buffer manager.
   */
  ~Node();

  // inline methods
  bool operator==(const Node& other) const { return id() == other.id(); }

  u32 id() const { return *reinterpret_cast<u32*>(buffer_slice_ + HEADER_SIZE); }
  u32 level() const { return *reinterpret_cast<u32*>(buffer_slice_ + HEADER_SIZE + sizeof(u32)); }
  u64& header() const { return *reinterpret_cast<u64*>(buffer_slice_); }

  span<element_t> components() const {
    return {reinterpret_cast<element_t*>(buffer_slice_ + HEADER_SIZE + META_SIZE), DIM};
  }

  bool is_locked() const { return header() & HEADER_NODE_LOCK; }
  bool is_new_level_locked() const { return header() & HEADER_NEW_LEVEL_LOCK; }
  bool is_entry_node() const { return header() & HEADER_ENTRY_NODE; }

  void set_lock() { header() |= HEADER_NODE_LOCK; }  // NOLINT
  void reset_lock() { header() &= ~HEADER_NODE_LOCK; }  // NOLINT

  void set_new_level_lock() { header() |= HEADER_NEW_LEVEL_LOCK; }  // NOLINT
  void reset_new_level_lock() { header() &= ~HEADER_NEW_LEVEL_LOCK; }  // NOLINT

  void set_is_entry_node() { header() |= HEADER_ENTRY_NODE; }  // NOLINT
  void reset_is_entry_node() { header() &= ~HEADER_ENTRY_NODE; }  // NOLINT

  ComputeThread* get_owner() const { return owner_; }
  byte_t* get_underlying_buffer() const { return buffer_slice_; }

  u64 compute_remote_neighborlist_offset(u32 lvl) const;
  friend std::ostream& operator<<(std::ostream& os, const Node& n);

private:
  ComputeThread* owner_{};
  byte_t* buffer_slice_{};  // points to local buffer

public:
  RemotePtr rptr;  // points to remote memory
};
