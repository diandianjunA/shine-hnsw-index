#pragma once

#include <library/connection_manager.hh>
#include <library/detached_qp.hh>

template <typename T>
class SharedContext {
public:
  SharedContext(Context& channel_context,
                ClientConnectionManager& cm,
                HugePage<byte_t>& buffer,
                const MemoryRegionTokens& remote_mrts)
      : context(channel_context.get_config()), remote_mrts(remote_mrts) {
    qps.reserve(cm.server_qps.size());

    for (QP& server_qp : cm.server_qps) {
      const auto& qp =
        qps.emplace_back(std::make_unique<DetachedQP>(context, context.get_send_cq(), context.get_receive_cq()));
      qp->connect(channel_context, context.get_lid(), server_qp);
    }

    // register full buffer
    memory_region = std::make_unique<LocalMemoryRegion>(context, buffer.get_full_buffer(), buffer.buffer_size);
  }

  void register_thread(T* thread) {
    registered_threads.push_back(thread);
    thread->ctx = this;
    thread->ctx_tid = registered_threads.size() - 1;
  }

  ibv_cq* get_cq() { return context.get_send_cq(); }
  u32 get_lkey() const { return memory_region->get_lkey(); }
  MemoryRegionToken* get_remote_mrt(u32 memory_node) { return remote_mrts[memory_node].get(); }

public:
  Context context;
  vec<u_ptr<DetachedQP>> qps;  // per memory node
  vec<T*> registered_threads;

private:
  u_ptr<LocalMemoryRegion> memory_region;
  const MemoryRegionTokens& remote_mrts;  // per memory node, owner is ComputeNode
};
