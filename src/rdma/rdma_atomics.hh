#pragma once

#include "rdma_reads.hh"

namespace rdma {

/**
 * @param rptr Points to the node we try to lock.
 * @param expected_header Bits set as expected except for the lock bit (which must always be 0).
 * @param new_level_lock If `true`, `new_level_lock` is locked instead of `node_lock`.
 * @return A pair of (1) `true` on success, `false` otherwise and (2) the original value.
 */
inline auto try_lock_node(const RemotePtr& rptr,
                          u64 expected_header,
                          bool new_level_lock,
                          const u_ptr<ComputeThread>& thread) {
  const u64 lock = new_level_lock ? Node::HEADER_NEW_LEVEL_LOCK : Node::HEADER_NODE_LOCK;
  const u64 compare = expected_header & ~lock;  // make sure that lock is not set
  const u64 swap = compare | lock;

  thread->track_post();

  const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
  qp->post_CAS(reinterpret_cast<u64>(thread->coros_pointer_slot()),
               thread->ctx->get_lkey(),
               thread->ctx->get_remote_mrt(rptr.memory_node()),
               rptr.byte_offset(),
               compare,
               swap,
               true,
               thread->create_wr_id());

  struct awaitable {
    u64 compare;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}

    std::pair<bool, u64> await_resume() const {
      const u64 original_value = *thread->coros_pointer_slot();
      return {original_value == compare, original_value};
    }
  };

  return awaitable{compare, thread};
}

inline MinorCoroutine spinlock_node(const s_ptr<Node>& node, const u_ptr<ComputeThread>& thread) {
  bool success;
  u64 original_header;

  do {
    std::tie(success, original_header) = co_await try_lock_node(node->rptr, node->header(), false, thread);
    node->header() =
      original_header;  // update header, otherwise we could end up in an infinity loop because node becomes entry node
                        // (might occur when the node is concurrently inserted). Btw, try_lock_node unsets the lock bit
  } while (!success);

  node->set_lock();
}

/**
 * @brief Attempts to lock the `is_new_level` lock.
 *        On failure, `ep_ptr` and `entry_point` is updated and acquiring the lock is retried.
 */
inline MinorCoroutine lock_and_update_entry_point(RemotePtr& ep_ptr,
                                                  s_ptr<Node>& entry_point,
                                                  const u_ptr<ComputeThread>& thread) {
  bool success;
  u64 original_header;

  do {
    while (!entry_point->is_entry_node()) {
      ep_ptr = co_await read_entry_point_ptr(thread);  // also updates cache

      // TODO: read only header until locked, then finally read components (!)
      entry_point = co_await read_node(ep_ptr, thread);
    }

    std::tie(success, original_header) = co_await try_lock_node(ep_ptr, entry_point->header(), true, thread);
    entry_point->header() = original_header;  // update header
  } while (!success);

  entry_point->set_new_level_lock();  // CAS sets the header to the expected header (without lock)
}

inline auto allocate_node(u32 level, const u_ptr<ComputeThread>& thread) {
  const u32 memory_node = thread->get_random_memory_node();
  size_t node_size = Node::total_size(level);

  // make sure that the next node's address is 8B aligned (otherwise CAS the header will fail)
  while (node_size % 8 != 0) {
    node_size += 4;
  }

  thread->stats.allocation_size += node_size;
  ++thread->stats.remote_allocations;
  thread->track_post();

  const QP& qp = thread->ctx->qps[memory_node]->qp;
  qp->post_FAA(reinterpret_cast<u64>(thread->coros_pointer_slot()),
               thread->ctx->get_lkey(),
               thread->ctx->get_remote_mrt(memory_node),
               0,  // free_ptr is stored at the very beginning
               node_size,
               true,
               thread->create_wr_id());

  struct awaitable {
    const u32 memory_node;
    const size_t node_size;
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}

    RemotePtr await_resume() const {
      // the original data, before the add operation, is being written to the local buffer
      const RemotePtr rptr{memory_node, *thread->coros_pointer_slot()};

      lib_assert(rptr.byte_offset() + node_size <= MEMORY_NODE_MAX_MEMORY,
                 "memory node " << rptr.memory_node() << "out of memory");

      return rptr;
    }
  };

  return awaitable{memory_node, node_size, thread};
}

inline auto swap_entry_point_ptr(const RemotePtr& old_ep, const RemotePtr& new_ep, const u_ptr<ComputeThread>& thread) {
  thread->track_post();

  const QP& qp = thread->ctx->qps[0]->qp;  // ep_ptr is always on memory node 0
  qp->post_CAS(reinterpret_cast<u64>(thread->coros_pointer_slot()),
               thread->ctx->get_lkey(),
               thread->ctx->get_remote_mrt(0),
               8,  // ep_ptr is stored at the very beginning after the free_ptr
               old_ep.raw_address,
               new_ep.raw_address,
               true,
               thread->create_wr_id());

  struct awaitable {
    const u_ptr<ComputeThread>& thread;

    static bool await_ready() { return false; }
    static void await_suspend(std::coroutine_handle<>) {}
    RemotePtr await_resume() const { return RemotePtr{*thread->coros_pointer_slot()}; }
  };

  return awaitable{thread};
}

}  // namespace rdma