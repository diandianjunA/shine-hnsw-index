#pragma once

#include <library/connection_manager.hh>

#include "common/configuration.hh"
#include "common/core_assignment.hh"
#include "common/statistics.hh"
#include "common/timing.hh"
#include "hnsw/hnsw.hh"
#include "router/query_router.hh"
#include "worker_pool.hh"

template <class Distance>
class ComputeNode {
private:
  using Configuration = configuration::IndexConfiguration;
  using ComputeThreads = WorkerPool::ComputeThreads;
  using Assignment = CoreAssignment<interleaved>;
  using CNStatistics = statistics::CNStatistics;

public:
  explicit ComputeNode(Configuration& config);

private:
  void run_query_router_and_queries(io::Database<element_t>& queries,
                                    WorkerPool& worker_pool,
                                    hnsw::HNSW<Distance>& hnsw,
                                    timing::Timing::IntervalPtr&& routing_timer,
                                    const Placement<Distance>& placement,
                                    const Configuration& config);
  void init_remote_tokens();
  void receive_remote_access_tokens();
  void read_dataset(const filepath_t& data_path,
                    const str& query_suffix,
                    bool load_index,
                    bool include_groundtruth,
                    bool use_cache);
  void run_inserts(hnsw::HNSW<Distance>& hnsw, WorkerPool& worker_pool, u32 num_coroutines, bool pin_threads);
  void run_queries(hnsw::HNSW<Distance>& hnsw,
                   WorkerPool& worker_pool,
                   io::Database<element_t>& queries,
                   query_router::QueryRouter<Distance>& query_router,
                   u32 num_coroutines,
                   bool pin_threads);
  void join_threads(const ComputeThreads& compute_threads);
  void wait_for_load_or_store(const Configuration& config);
  void sync_compute_nodes();
  void add_meta_statistics(const Configuration& config);
  void collect_statistics_and_timings();
  void terminate();
  f64 compute_local_recall(const ComputeThreads& compute_threads, u32 k, size_t processed_queries);

private:
  Context context_;
  ClientConnectionManager cm_;
  const u32 num_servers_;

  MemoryRegionTokens remote_access_tokens_;
  Assignment core_assignment_;

  timing::Timing timing_;
  statistics::Statistics statistics_;
  CNStatistics cn_statistics_{};

  timing::Timing::IntervalPtr t_build_{};
  timing::Timing::IntervalPtr t_query_{};

  io::Database<element_t> database_{};
  io::Database<element_t> queries_{};
  io::Database<element_t> warmup_queries_{};
  io::GroundTruth ground_truth_{};

  std::atomic<idx_t> next_insert_idx_{0};
  std::atomic<idx_t> next_query_idx_{0};
};