#pragma once

#include "cache/kmeans.hh"
#include "common/timing.hh"
#include "coroutine.hh"
#include "node/neighborlist.hh"
#include "rdma/rdma_operations.hh"

template <class Distance>
class Placement {
public:
  using Centroids = Kmeans<Distance>::Centroids;
  using Nodes = Kmeans<Distance>::Nodes;

  struct MinHeapPlacementCompare {
    bool operator()(const auto& lhs, const auto& rhs) const { return lhs.second > rhs.second; }
  };

  using MinPlacement = heap::Heap<std::pair<idx_t, distance_t>, MinHeapPlacementCompare>;

  Placement() = default;
  Placement(u32 k, const u_ptr<ComputeThread>& thread, timing::Timing& timing) {
    const auto t_placement_fetch = timing.create_enroll("placement_fetch");
    const auto t_placement_kmeans = timing.create_enroll("placement_kmeans");
    Nodes nodes;

    t_placement_fetch->start();
    {
      const auto coro = fetch_level(nodes, 500, thread);  // TODO
      while (!coro.handle.done()) {
        thread->poll_cq();
        if (thread->is_ready(0)) {
          coro.handle.resume();
        }
      }

      thread->reset();
    }
    t_placement_fetch->stop();

    t_placement_kmeans->start();
    if constexpr (query_router::BALANCED_KMEANS_WITH_HEURISTIC) {
      std::tie(centroids_, mapping_) = Kmeans<Distance>::run_and_optimize(nodes, k);  // optimized balanced k-means

    } else {  // default k-means
      auto [c, _, cluster_sizes] = Kmeans<Distance>::run_kmeans(nodes, k);

      centroids_ = c;
      mapping_.resize(k);

      for (idx_t i = 0; i < k; ++i) {
        mapping_[i] = i;
      }

      for (idx_t i = 0; i < k; ++i) {
        std::cerr << "  cluster " << i << ": " << cluster_sizes[i] << std::endl;
      }
    }

    t_placement_kmeans->stop();
  }

  MinPlacement closest_centroids(const span<element_t> query_components) const {
    MinPlacement placement;

    for (idx_t i = 0; i < centroids_.size(); ++i) {
      const f32 distance = Distance::dist(query_components, centroids_[i], Node::DIM);
      placement.push({mapping_[i], distance});
    }

    return placement;
  }

private:
  /**
   * @brief Fetches all nodes of a level. Start with highest level, descend until we have at least k nodes.
   */
  static MinorCoroutine fetch_level(Nodes& nodes, u32 k, const u_ptr<ComputeThread>& thread) {
    RemotePtr ep_ptr = co_await rdma::read_entry_point_ptr(thread);
    nodes.emplace_back(co_await rdma::read_node(ep_ptr, thread));

    hashset_t<RemotePtr> visited_nodes;
    visited_nodes.insert(ep_ptr);
    i32 level = nodes.back()->level();

    do {
      lib_assert(level >= 0, "unable to fetch sufficiently many nodes");

      for (idx_t iter = 0; iter < nodes.size(); ++iter) {
        const auto& node = nodes[iter];

        const RemotePtr nlist_rptr{node->rptr.memory_node(), node->compute_remote_neighborlist_offset(level)};
        const s_ptr<Neighborlist> neighborlist = co_await rdma::read_neighborlist(nlist_rptr, level, thread);

        for (RemotePtr& r_ptr : neighborlist->view()) {
          if (!visited_nodes.contains(r_ptr)) {
            nodes.emplace_back(co_await rdma::read_node(r_ptr, thread));
            visited_nodes.insert(r_ptr);
          }
        }
      }

      std::cerr << "  fetched " << nodes.size() << " nodes on level " << level << std::endl;
      --level;
    } while (nodes.size() < k);
  }

private:
  Centroids centroids_;
  vec<idx_t> mapping_;
};