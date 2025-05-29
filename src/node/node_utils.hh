#pragma once

#include "node.hh"

inline void set_header(u64& header, bool is_entry_node, bool new_level_lock, bool node_lock) {
  if (is_entry_node) {
    header |= Node::HEADER_ENTRY_NODE;
  }
  if (new_level_lock) {
    header |= Node::HEADER_NEW_LEVEL_LOCK;
  }
  if (node_lock) {
    header |= Node::HEADER_NODE_LOCK;
  }
}

inline void node_to_buffer(byte_t* buffer, u64 header, const span<element_t>& components, u32 id, u32 level) {
  *reinterpret_cast<u64*>(buffer) = header;
  buffer += Node::HEADER_SIZE;
  *reinterpret_cast<u32*>(buffer) = id;
  buffer += sizeof(u32);
  *reinterpret_cast<u32*>(buffer) = level;
  buffer += sizeof(u32);
  std::memcpy(buffer, components.data(), components.size() * sizeof(element_t));
}
