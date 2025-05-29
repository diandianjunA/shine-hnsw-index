#include "node.hh"

#include <library/utils.hh>

#include "compute_thread.hh"

// public methods

Node::~Node() {
  // TODO: idea: could we get caller's (not owner's) thread_id? Then the freelist must no longer be thread-safe...
  const u32 thread_id = owner_->get_id();

  if (buffer_slice_ != nullptr) {
    owner_->buffer_allocator.free_node(buffer_slice_, thread_id);  // appends to thread_id's thread-safe freelist
  }
}

u64 Node::compute_remote_neighborlist_offset(u32 lvl) const {
  u64 neighborlist_offset = rptr.byte_offset() + size_until_components();

  if (lvl > 0) {
    neighborlist_offset += NEIGHBORLIST_SIZE_ZERO;
    neighborlist_offset += (lvl - 1) * NEIGHBORLIST_SIZE;
  }

  return neighborlist_offset;
}

std::ostream& operator<<(std::ostream& os, const Node& n) {
  return os << "[uid: " << n.id() << " | level: " << n.level() << " | is_entry_node: " << n.is_entry_node()
            << " | new_level_lock: " << n.is_new_level_locked() << " | lock: " << n.is_locked() << " | components ("
            << Node::DIM << "d)] (rptr: " << n.rptr << ")";
}
