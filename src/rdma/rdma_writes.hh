#pragma once

#include "compute_thread.hh"
#include "coroutine.hh"
#include "node/neighborlist.hh"
#include "node/node_utils.hh"
#include "remote_pointer.hh"

namespace rdma {

/**
 * @brief Resets the `node_lock` by writing (RDMA WRITE) a single byte.
 */
inline auto unlock_node(const s_ptr<Node>& node, const u_ptr<ComputeThread>& thread) {
  byte_t unlock = 0;

  thread->stats.rdma_writes_in_bytes += 1;
  thread->track_post();

  const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
  qp->post_send_inlined(std::addressof(unlock),
                        1,  // write only header's last byte
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                        node->rptr.byte_offset() + Node::HEADER_UNTIL_LOCK,
                        thread->create_wr_id());

  node->reset_lock();
  return std::suspend_always{};
}

/**
 * @brief Resets the `new_level_lock` by writing (RDMA WRITE) a single byte.
 */
inline auto unlock_new_level_lock(const s_ptr<Node>& node, const u_ptr<ComputeThread>& thread) {
  byte_t unlock = 0;

  thread->stats.rdma_writes_in_bytes += 1;
  thread->track_post();

  const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
  qp->post_send_inlined(std::addressof(unlock),
                        1,  // write only single byte
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                        node->rptr.byte_offset() + Node::HEADER_UNTIL_LVL_LOCK,
                        thread->create_wr_id());

  node->reset_new_level_lock();
  return std::suspend_always{};
}

/**
 * @brief Clears the `is_entry_node` bit by writing (RDMA WRITE) a single byte.
 */
inline auto clear_entry_node_bit(const s_ptr<Node>& node, const u_ptr<ComputeThread>& thread) {
  byte_t clear = 0;

  thread->stats.rdma_writes_in_bytes += 1;
  thread->track_post();

  const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
  qp->post_send_inlined(std::addressof(clear),
                        1,  // write only single byte
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                        node->rptr.byte_offset() + Node::HEADER_UNTIL_ENTRY_NODE,
                        thread->create_wr_id());

  node->reset_is_entry_node();
  return std::suspend_always{};
}

inline auto write_node(const RemotePtr& rptr,
                       node_t id,
                       const span<element_t> components,
                       u32 level,
                       bool is_entry_node,
                       bool new_level_lock,
                       bool node_lock,
                       const u_ptr<ComputeThread>& thread) {
  u64 header = 0;
  set_header(header, is_entry_node, new_level_lock, node_lock);

  byte_t* local_buffer = thread->buffer_allocator.allocate_node(thread->get_id());
  node_to_buffer(local_buffer, header, components, id, level);

  thread->stats.rdma_writes_in_bytes += Node::size_until_components();
  thread->track_post();

  const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
  qp->post_send(reinterpret_cast<u64>(local_buffer),
                Node::size_until_components(),
                thread->ctx->get_lkey(),
                IBV_WR_RDMA_WRITE,
                true,
                false,
                thread->ctx->get_remote_mrt(rptr.memory_node()),
                rptr.byte_offset(),
                0,
                thread->create_wr_id());

  struct awaitable {
    byte_t* local_buffer;
    const RemotePtr& rptr;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}

    s_ptr<Node> await_resume() { return std::make_shared<Node>(local_buffer, rptr, thread.get()); }
  };

  return awaitable{local_buffer, rptr, thread};
}

inline auto write_header(const RemotePtr& rptr,
                         bool is_entry_node,
                         bool new_level_lock,
                         bool node_lock,
                         const u_ptr<ComputeThread>& thread) {
  u64 header = 0;
  set_header(header, is_entry_node, new_level_lock, node_lock);

  thread->stats.rdma_writes_in_bytes += Node::HEADER_SIZE;
  thread->track_post();

  const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
  qp->post_send_inlined(std::addressof(header),
                        Node::HEADER_SIZE,
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(rptr.memory_node()),
                        rptr.byte_offset(),
                        thread->create_wr_id());

  return std::suspend_always{};
}

inline auto write_neighborlist(const s_ptr<Neighborlist>& neighborlist,
                               const s_ptr<Node>& node,
                               const u_ptr<ComputeThread>& thread) {
  const size_t size = sizeof(u32) + neighborlist->num_neighbors() * RemotePtr::SIZE;

  thread->stats.rdma_writes_in_bytes += size;
  thread->track_post();

  const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
  qp->post_send(reinterpret_cast<u64>(neighborlist->buffer_ptr()),
                size,
                thread->ctx->get_lkey(),
                IBV_WR_RDMA_WRITE,
                true,
                false,
                thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                node->compute_remote_neighborlist_offset(neighborlist->level()),
                0,
                thread->create_wr_id());

  return std::suspend_always{};
}

inline auto write_last_neighbor_in_neighborlist(const s_ptr<Neighborlist>& neighborlist,
                                                const s_ptr<Node>& node,
                                                const u_ptr<ComputeThread>& thread) {
  const auto neighborlist_view = neighborlist->view();
  const u64 offset_size = node->compute_remote_neighborlist_offset(neighborlist->level());
  const u64 offset_neighbor = offset_size + sizeof(u32) + (neighborlist_view.size() - 1) * RemotePtr::SIZE;

  thread->stats.rdma_writes_in_bytes += sizeof(u32) + RemotePtr::SIZE;
  thread->track_post();

  const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
  // TODO: combine to single RDMA operation

  // update neighborlist size
  qp->post_send_inlined(neighborlist->buffer_ptr(),
                        sizeof(u32),
                        IBV_WR_RDMA_WRITE,
                        false,
                        thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                        offset_size,
                        thread->create_wr_id());

  // write last neighbor
  qp->post_send_inlined(std::addressof(neighborlist_view.back()),
                        RemotePtr::SIZE,
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                        offset_neighbor,
                        thread->create_wr_id());

  return std::suspend_always{};
}

inline auto write_entry_point_ptr(const RemotePtr& ep_ptr, const u_ptr<ComputeThread>& thread) {
  thread->stats.rdma_writes_in_bytes += RemotePtr::SIZE;
  thread->track_post();

  const QP& qp = thread->ctx->qps[0]->qp;  // ep_ptr is always on memory node 0
  qp->post_send_inlined(std::addressof(ep_ptr.raw_address),
                        RemotePtr::SIZE,
                        IBV_WR_RDMA_WRITE,
                        true,
                        thread->ctx->get_remote_mrt(0),
                        8,  // ep_ptr is stored at the very beginning after the free_ptr
                        thread->create_wr_id());

  return std::suspend_always{};
}

}  // namespace rdma
