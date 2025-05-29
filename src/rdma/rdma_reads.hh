#pragma once

#include "compute_thread.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"

namespace rdma {

inline auto read_node(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
  byte_t* node_ptr = thread->buffer_allocator.allocate_node(thread->get_id());

  thread->stats.rdma_reads_in_bytes += Node::size_until_components();
  thread->track_post();

  const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
  qp->post_send(reinterpret_cast<u64>(node_ptr),
                Node::size_until_components(),
                thread->ctx->get_lkey(),
                IBV_WR_RDMA_READ,
                true,
                false,
                thread->ctx->get_remote_mrt(rptr.memory_node()),
                rptr.byte_offset(),
                0,
                thread->create_wr_id());

  struct awaitable {
    RemotePtr rptr;
    byte_t* node_ptr;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}
    s_ptr<Node> await_resume() { return std::make_shared<Node>(node_ptr, rptr, thread.get()); }
  };

  return awaitable{rptr, node_ptr, thread};
}

inline auto read_neighborlist(RemotePtr rptr, u32 level, const u_ptr<ComputeThread>& thread) {
  const size_t size = level > 0 ? Node::NEIGHBORLIST_SIZE : Node::NEIGHBORLIST_SIZE_ZERO;

  byte_t* local_buffer = level > 0 ? thread->buffer_allocator.allocate_layer(thread->get_id())
                                   : thread->buffer_allocator.allocate_layer_zero(thread->get_id());

  thread->stats.rdma_reads_in_bytes += size;
  thread->track_post();

  const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
  qp->post_send(reinterpret_cast<u64>(local_buffer),
                size,
                thread->ctx->get_lkey(),
                IBV_WR_RDMA_READ,
                true,
                false,
                thread->ctx->get_remote_mrt(rptr.memory_node()),
                rptr.byte_offset(),
                0,
                thread->create_wr_id());

  struct awaitable {
    const u32 level;
    byte_t* local_buffer;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}
    s_ptr<Neighborlist> await_resume() const { return std::make_shared<Neighborlist>(level, local_buffer, thread); }
  };

  return awaitable{level, local_buffer, thread};
}

inline auto read_entry_point_ptr(const u_ptr<ComputeThread>& thread) {
  thread->stats.rdma_reads_in_bytes += sizeof(u64);
  thread->track_post();

  const QP& qp = thread->ctx->qps[0]->qp;  // ep_ptr is always on memory node 0
  qp->post_send(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                sizeof(u64),
                thread->ctx->get_lkey(),
                IBV_WR_RDMA_READ,
                true,
                false,
                thread->ctx->get_remote_mrt(0),
                8,  // ep_ptr is stored at the very beginning after the free_ptr
                0,
                thread->create_wr_id());

  struct awaitable {
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}
    RemotePtr await_resume() const { return RemotePtr{*thread->coros_pointer_slot()}; }
  };

  return awaitable{thread};
}

inline auto read_nodes(const span<RemotePtr> remote_ptrs, const u_ptr<ComputeThread>& thread) {
  vec<s_ptr<Node>> nodes;
  nodes.reserve(remote_ptrs.size());

  for (auto& rptr : remote_ptrs) {
    byte_t* node_ptr = thread->buffer_allocator.allocate_node(thread->get_id());
    nodes.emplace_back(std::make_shared<Node>(node_ptr, rptr, thread.get()));

    thread->stats.rdma_reads_in_bytes += Node::size_until_components();
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(node_ptr),
                  Node::size_until_components(),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());
  }

  struct awaitable {
    vec<s_ptr<Node>> nodes;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<s_ptr<Node>> await_resume() const { return nodes; }  // every node will be freed by Node's dtor
  };

  return awaitable{nodes};
}

}  // namespace rdma