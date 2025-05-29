#pragma once

#include <iostream>
#include <library/types.hh>
#include <ostream>

#include "nlohmann/json.hh"

namespace statistics {

/**
 * @brief: JSON wrapper plus some convinient methods.
 */
class Statistics {
public:
  using json = nlohmann::json;

  void add_timings(const json& timings) { stats_["timings"] = timings; }

  template <typename T>
  void add_meta_stat(const str&& key, T v) {
    stats_["meta"][key] = v;
  }

  template <typename T>
  void add_meta_stats(T pair) {
    add_meta_stat(pair.first, pair.second);
  }

  template <typename T, typename... Args>
  void add_meta_stats(T pair, Args... args) {
    add_meta_stat(pair.first, pair.second);
    add_meta_stats(args...);
  }

  void output_all(const json& timings) {
    add_timings(timings);

    std::cerr << std::endl << "statistics:" << std::endl;
    std::cout << *this << std::endl;
  }

  template <typename T>
  void add_static_stat(const str& key, T v) {
    stats_[key] = v;
  }

  template <typename T>
  void add_nested_static_stat(const str& group, const str&& key, T v) {
    stats_[group][key] = v;
  }

  template <typename T>
  void add_nested_static_stat(const str& g1, const str& g2, const str&& key, T v) {
    stats_[g1][g2][key] = v;
  }

  friend std::ostream& operator<<(std::ostream& os, const Statistics& s) { return os << s.stats_.dump(2); }

private:
  json stats_;
};

/**
 * @brief Global statistics per compute node (summed up over all local threads).
 *        Must be serializable.
 */
struct CNStatistics {
  size_t build_distcomps;
  size_t build_rdma_reads;
  size_t build_rdma_writes;
  size_t remote_allocations;
  size_t total_allocation_size;
  u32 max_level;

  size_t query_distcomps;
  size_t query_rdma_reads;
  size_t query_rdma_writes;

  size_t query_cache_hits;
  size_t query_cache_misses;

  size_t query_visited_nodes;
  size_t query_visited_nodes_l0;
  size_t query_visited_neighborlists;

  size_t processed_queries;
  size_t local_allocation_size;  // bump pointer (actual usage of the local buffers)

  f64 rolling_recall;
  timespec build_time;
  timespec query_time;

  void combine(const CNStatistics& other) {
    total_allocation_size += other.total_allocation_size;
    remote_allocations += other.remote_allocations;
    rolling_recall += other.rolling_recall;  // sum over all rolling recalls
    max_level = std::max(max_level, other.max_level);

    build_distcomps += other.build_distcomps;
    build_rdma_reads += other.build_rdma_reads;
    build_rdma_writes += other.build_rdma_writes;

    processed_queries += other.processed_queries;
    local_allocation_size += other.local_allocation_size;

    query_distcomps += other.query_distcomps;
    query_rdma_reads += other.query_rdma_reads;
    query_rdma_writes += other.query_rdma_writes;
    query_cache_hits += other.query_cache_hits;
    query_cache_misses += other.query_cache_misses;
    query_visited_nodes += other.query_visited_nodes;
    query_visited_nodes_l0 += other.query_visited_nodes_l0;
    query_visited_neighborlists += other.query_visited_neighborlists;
  }

  void convert(Statistics& statistics) const {
    const str build_group = "build";
    const str query_group = "queries";
    const str cache_group = "cache";

    statistics.add_nested_static_stat(build_group, "dist_comps", build_distcomps);
    statistics.add_nested_static_stat(build_group, "rdma_reads_in_bytes", build_rdma_reads);
    statistics.add_nested_static_stat(build_group, "rdma_writes_in_bytes", build_rdma_writes);
    statistics.add_nested_static_stat(build_group, "remote_allocations", remote_allocations);
    statistics.add_nested_static_stat(build_group, "index_size", total_allocation_size);
    statistics.add_nested_static_stat(build_group, "max_level", max_level);

    statistics.add_nested_static_stat(query_group, "dist_comps", query_distcomps);
    statistics.add_nested_static_stat(query_group, "rdma_reads_in_bytes", query_rdma_reads);
    statistics.add_nested_static_stat(query_group, "rdma_writes_in_bytes", query_rdma_writes);
    statistics.add_nested_static_stat(query_group, "recall", rolling_recall);
    statistics.add_nested_static_stat(query_group, "visited_nodes", query_visited_nodes);
    statistics.add_nested_static_stat(query_group, "visited_nodes_l0", query_visited_nodes_l0);
    statistics.add_nested_static_stat(query_group, "visited_neighborlists", query_visited_neighborlists);
    statistics.add_nested_static_stat(query_group, "processed", processed_queries);

    statistics.add_nested_static_stat(cache_group, "hits_total", query_cache_hits);
    statistics.add_nested_static_stat(cache_group, "misses_total", query_cache_misses);

    statistics.add_static_stat("actual_total_local_buffer_size", local_allocation_size);
  }
};

/**
 * @brief Thread-local statistics
 */
struct ThreadStatistics {
  size_t distcomps{0};
  size_t rdma_reads_in_bytes{0};
  size_t rdma_writes_in_bytes{0};
  size_t processed{0};
  size_t remote_allocations{0};
  size_t allocation_size{0};
  size_t visited_nodes{0};
  size_t visited_nodes_l0{0};
  size_t visited_neighborlists{0};
  u32 max_level{0};

  size_t cache_hits{0};
  size_t cache_misses{0};

  void inc_visited_nodes(u32 level) {
    if (level > 0) {
      ++visited_nodes;
    } else {
      ++visited_nodes_l0;
    }
  }

  f64 cache_hit_rate() const {
    return cache_hits + cache_misses == 0 ? 0
                                          : static_cast<f64>(cache_hits) / static_cast<f64>(cache_hits + cache_misses);
  }
};

}  // namespace statistics
