#pragma once

#include <library/latch.hh>

#include "buffer_allocator.hh"
#include "cache/cache.hh"
#include "common/configuration.hh"
#include "common/constants.hh"
#include "compute_thread.hh"
#include "hnsw/scheduler.hh"
#include "shared_context.hh"

class WorkerPool {
public:
  using Configuration = configuration::IndexConfiguration;
  using ComputeThreads = vec<u_ptr<ComputeThread>>;
  using SharedCtx = SharedContext<ComputeThread>;
  using Queue = concurrent_queue<u32>;

public:
  WorkerPool(u32 num_compute_threads,
             i32 max_send_queue_wr,
             size_t cache_size,
             size_t num_cache_buckets,
             size_t num_cooling_table_buckets,
             bool use_cache)
      : num_compute_threads_(num_compute_threads),
        max_send_queue_wr_(max_send_queue_wr),
        buffer_allocator_(num_compute_threads),
        cache_(cache_size, num_cache_buckets, num_cooling_table_buckets, num_compute_threads, use_cache) {
    reset_barriers();  // initialize latches
  }

  void allocate_worker_threads(Context& context,
                               ClientConnectionManager& cm,
                               MemoryRegionTokens& remote_mrts,
                               u32 num_coroutines) {
    // create shared contexts (and QPs)
    for (u32 i = 0; i < std::min<u32>(num_compute_threads_, MAX_QPS); ++i) {
      shared_contexts_.emplace_back(
        std::make_unique<SharedCtx>(context, cm, buffer_allocator_.get_raw_buffer(), remote_mrts));
    }

    // pre-allocate worker threads
    for (u32 id = 0; id < num_compute_threads_; ++id) {
      const u32 num_memory_nodes = remote_mrts.size();
      compute_threads_.push_back(std::make_unique<ComputeThread>(
        id, cm.client_id, max_send_queue_wr_, buffer_allocator_, cache_, num_memory_nodes, num_coroutines));
    }

    // assign the contexts (now the thread pointers can no longer change)
    for (u32 id = 0; id < num_compute_threads_; ++id) {
      const auto& ctx = shared_contexts_[id % MAX_QPS];
      ctx->register_thread(compute_threads_[id].get());
    }
  }

  ComputeThreads& get_compute_threads() { return compute_threads_; }
  BufferAllocator& get_buffer_allocator() { return buffer_allocator_; }
  void track_local_cache_statistics(statistics::Statistics& stats) const { cache_.track_cache_statistics(stats); }

  void reset_barriers() {
    start_latch_.init(static_cast<i32>(num_compute_threads_));
    end_latch_.init(static_cast<i32>(num_compute_threads_));
  }

  template <class Distance>
  void process_inserts(hnsw::HNSW<Distance>& hnsw,
                       std::atomic<idx_t>& next_insert_idx,
                       io::Database<element_t>& database,
                       u32 num_coroutines,
                       u32 thread_id) {
    start_latch_.arrive_and_wait();
    hnsw::schedule<Distance, true>(hnsw, next_insert_idx, database, num_coroutines, compute_threads_[thread_id]);
    end_latch_.arrive_and_wait();
  }

  template <class Distance>
  void process_queries(hnsw::HNSW<Distance>& hnsw,
                       std::atomic<idx_t>& next_query_idx,
                       io::Database<element_t>& queries,
                       query_router::QueryRouter<Distance>& query_router,
                       u32 num_coroutines,
                       u32 thread_id) {
    start_latch_.arrive_and_wait();
    hnsw::schedule<Distance, false>(
      hnsw, next_query_idx, queries, num_coroutines, compute_threads_[thread_id], &query_router);
    end_latch_.arrive_and_wait();
  }

private:
  const u32 num_compute_threads_;
  const i32 max_send_queue_wr_;

  ComputeThreads compute_threads_;
  vec<u_ptr<SharedCtx>> shared_contexts_;

  BufferAllocator buffer_allocator_;  // global per compute node
  cache::Cache cache_;

  Latch start_latch_{};
  Latch end_latch_{};
};
