#pragma once

#include <common/constants.hh>
#include <random>

#include "cache/cache.hh"
#include "common/debug.hh"
#include "common/types.hh"
#include "compute_thread.hh"
#include "distance.hh"
#include "heap.hh"
#include "node/neighborlist.hh"
#include "node/node.hh"
#include "rdma/rdma_operations.hh"
#include "remote_pointer.hh"

namespace hnsw {

template <class Distance>
class HNSW {
private:
  enum Lock { with_lock = true, without_lock = false };

public:
  HNSW(u32 m, u32 ef_construction, u32 k, u32 ef_search, u32 seed, u32 dim, bool use_cache)
      : m_(m),
        m_max_(m),
        m_max_zero_(m * 2),
        ef_construction_(ef_construction),
        normalization_factor_(1. / std::log(static_cast<f64>(m))),
        k_(k),
        ef_search_(ef_search),
        use_cache_(use_cache),
        prng_(seed),
        uniform_(0., 1.) {
    lib_assert(ef_search_ >= k_, "ef_search must be >= k");
    Node::init_static_storage(dim, m_max_, m_max_zero_);
  }

  HNSWCoroutine insert(node_t id, const span<element_t> components, const u_ptr<ComputeThread>& thread) {
    dbg::print(dbg::stream{} << "T" << thread->get_id() << " inserts " << id << "\n--------------\n\n");
    ++thread->stats.processed;

    /**
     * Draw level: Note that with `m_L` = normalization_factor = 1/ln(M) (as suggested in the paper) the probability
     *             of inserting the node at level l is just 1/M^l or more generally, e^-(1/m_L * l).
     */
    u32 new_node_level = std::floor(-std::log(uniform_(prng_)) * normalization_factor_);
    bool allocated = false;

    RemotePtr new_node_ptr;
    s_ptr<Node> new_node;

    auto& cached_ep_ptr = thread->current_coroutine().cached_ep_ptr;

    if (cached_ep_ptr.is_null()) {
      cached_ep_ptr = co_await rdma::read_entry_point_ptr(thread);

      // if still null, index not yet initialized
      if (cached_ep_ptr.is_null()) {
        new_node_level = 0;
        new_node_ptr = co_await rdma::allocate_node(new_node_level, thread);
        new_node = co_await rdma::write_node(new_node_ptr, id, components, new_node_level, false, false, true, thread);

        allocated = true;
        const RemotePtr old_ep_ptr = cached_ep_ptr;

        // try to compare-and-swap ep pointer
        cached_ep_ptr = co_await rdma::swap_entry_point_ptr(old_ep_ptr, new_node_ptr, thread);
        if (cached_ep_ptr == old_ep_ptr) {
          // success: the index is now initialized, set the is_entry_node bit
          co_await rdma::write_header(new_node_ptr, true, false, false, thread);
          cached_ep_ptr = new_node_ptr;  // fix cache

          dbg::print(dbg::stream{} << "T" << thread->get_id() << " allocated " << new_node_ptr
                                   << " and set first EP\n");
          co_return;
        }

        // failure: the index has been initilized by a thread that won the race

        dbg::print(dbg::stream{} << "T" << thread->get_id() << " allocated " << new_node_ptr
                                 << " but failed CAS, another T initialized the index\n");
      }
    }

    // READ and LOCK entry point
    s_ptr<Node> entry_point = co_await rdma::read_node(cached_ep_ptr, thread);
    {
      auto coro = rdma::lock_and_update_entry_point(cached_ep_ptr, entry_point, thread);

      while (!coro.handle.done()) {
        co_await std::suspend_always{};  // gives control back to caller of insert
        coro.handle.resume();
      }
    }

    dbg::print(dbg::stream{} << "T" << thread->get_id() << " successfully locked entry point " << *entry_point << "\n");

    const u32 top_level = entry_point->level();
    const bool is_new_level = new_node_level > top_level;

    if (!is_new_level) {
      co_await rdma::unlock_new_level_lock(entry_point, thread);  // release global lock
    } else {
      new_node_level = top_level + 1;  // make sure to not overshoot
      dbg::print(dbg::stream{} << "T" << thread->get_id() << " will set new level " << new_node_level << "\n");
    }

    thread->stats.max_level = std::max(thread->stats.max_level, new_node_level);

    // we allocate at this point, because now we know the new level
    if (!allocated) {
      new_node_ptr = co_await rdma::allocate_node(new_node_level, thread);
      new_node = co_await rdma::write_node(new_node_ptr, id, components, new_node_level, false, false, true, thread);

      dbg::print(dbg::stream{} << "T" << thread->get_id() << " allocated: " << *new_node << "\n");
    }

    // at this point, we have the node allocated, locked, and the entry point available
    //                if new level, we have a global lock (i.e., the entry point is locked)

    const distance_t ep_distance = Distance::dist(components, entry_point->components(), Node::DIM);
    ++thread->stats.distcomps;

    MaxHeap& top_candidates = thread->current_coroutine().top_candidates;

    // go through upper levels and greedily determine the nearest entry point
    if (new_node_level < top_level) {
      s_ptr<Node> nn = entry_point;
      auto coro = search_for_one<with_lock>(components, nn, ep_distance, top_level, new_node_level, thread);

      while (!coro.handle.done()) {
        co_await std::suspend_always{};  // gives control back to caller of insert
        coro.handle.resume();
      }

      top_candidates.push({nn, Distance::dist(nn->components(), components, Node::DIM)});
      ++thread->stats.distcomps;

    } else {
      top_candidates.push({entry_point, ep_distance});
    }

    // start one level below because there is no entry point yet at this level
    if (is_new_level) {
      --new_node_level;
    }

    // connect node
    for (i32 current_level = static_cast<i32>(new_node_level); current_level >= 0; --current_level) {
      {
        auto coro = search_level<with_lock>(components, ef_construction_, current_level, thread);
        while (!coro.handle.done()) {
          co_await std::suspend_always{};  // gives control back to caller of insert
          coro.handle.resume();
        }
      }

      dbg::print(dbg::stream{} << "T" << thread->get_id() << " search_level done for level " << current_level << "\n");

      // picks up to M nearest neighbors by running the heuristic
      select_heuristic(top_candidates, m_, thread);

      {  // write selected neighbors of `new_node` to remote memory
        byte_t* neighborlist_ptr = current_level == 0 ? thread->buffer_allocator.allocate_layer_zero(thread->get_id())
                                                      : thread->buffer_allocator.allocate_layer(thread->get_id());
        auto neighborlist = std::make_shared<Neighborlist>(current_level, neighborlist_ptr, thread);

        for (auto& [neighbor, _] : top_candidates.heap) {
          neighborlist->add(neighbor->rptr);
        }

        co_await rdma::write_neighborlist(neighborlist, new_node, thread);
      }

      const u32 m_max = current_level == 0 ? m_max_zero_ : m_max_;

      // connect node to neighbor lists of node's neighbors
      for (auto& [neighbor, neighbor_dist] : top_candidates.heap) {
        {
          auto coro = rdma::spinlock_node(neighbor, thread);
          while (!coro.handle.done()) {
            co_await std::suspend_always{};  // gives control back to caller of insert
            coro.handle.resume();
          }
        }

        const RemotePtr nlist_rptr{neighbor->rptr.memory_node(),
                                   neighbor->compute_remote_neighborlist_offset(current_level)};
        s_ptr<Neighborlist> neighborlist = co_await rdma::read_neighborlist(nlist_rptr, current_level, thread);

        if (neighborlist->num_neighbors() < m_max) {
          neighborlist->add(new_node_ptr);
          co_await rdma::write_last_neighbor_in_neighborlist(neighborlist, neighbor, thread);

        } else {
          // read neighbor's neighbors
          vec<s_ptr<Node>> old_neighbors = co_await rdma::read_nodes(neighborlist->view(), thread);

          MaxHeap new_neighbors;
          new_neighbors.push({new_node, neighbor_dist});

          for (const auto& old_neighbor : old_neighbors) {
            // we could save the distance computation when storing the distance to remote memory
            new_neighbors.push(
              {old_neighbor, Distance::dist(neighbor->components(), old_neighbor->components(), Node::DIM)});
            ++thread->stats.distcomps;
          }

          // shrink connections
          select_heuristic(new_neighbors, m_max, thread);

          // set new neighbors
          neighborlist->reset();
          for (const auto& [new_neighbor, _] : new_neighbors.heap) {
            neighborlist->add(new_neighbor->rptr);
          }

          // write new neighbors to remote memory
          co_await rdma::write_neighborlist(neighborlist, neighbor, thread);
        }

        co_await rdma::unlock_node(neighbor, thread);
      }

      // keep only 1-NN as next entry point
      while (current_level > 0 && top_candidates.size() > 1) {
        top_candidates.pop();
      }
    }

    // unlock node; we use write_header to set multiple bits, the node is locked anyway and may not have `new_lvl_lock`
    co_await rdma::write_header(new_node_ptr, is_new_level, false, false, thread);

    if (is_new_level) {
      // TODO: combine
      co_await rdma::clear_entry_node_bit(entry_point, thread);  // invalidates caches of other threads
      co_await rdma::unlock_new_level_lock(entry_point, thread);  // releases (global) entry-point lock

      // now another thread T could read the old EP, but it's no longer entry node,
      // hence T reads the EP-pointer again, and maybe repeats

      co_await rdma::write_entry_point_ptr(new_node_ptr, thread);
      dbg::print(dbg::stream{} << "T" << thread->get_id() << " ======== NEW EP PTR SET: " << new_node_ptr << "\n");

      cached_ep_ptr = new_node_ptr;
    }

    top_candidates.clear();
  }

  HNSWCoroutine knn(node_t q_id, const span<element_t> components, const u_ptr<ComputeThread>& thread) const {
    dbg::print(std::stringstream{} << "T" << thread->get_id() << " queries " << q_id << "\n--------------\n\n");

    auto& ep_ptr = thread->current_coroutine().cached_ep_ptr;
    if (ep_ptr.is_null()) {
      ep_ptr = co_await rdma::read_entry_point_ptr(thread);
    }

    s_ptr<Node> entry_point;
    {
      auto coro = cache_lookup(ep_ptr, entry_point, thread, thread->get_id() == 0);
      while (!coro.handle.done()) {
        co_await std::suspend_always{};  // gives control back to caller of knn
        coro.handle.resume();
      }
    }

    thread->stats.inc_visited_nodes(entry_point->level());
    const distance_t ep_distance = Distance::dist(components, entry_point->components(), Node::DIM);
    ++thread->stats.distcomps;

    MaxHeap& top_candidates = thread->current_coroutine().top_candidates;

    {
      s_ptr<Node> nn = entry_point;

      auto coro = search_for_one<without_lock>(components, nn, ep_distance, entry_point->level(), 0, thread);
      while (!coro.handle.done()) {
        co_await std::suspend_always{};  // gives control back to caller of knn
        coro.handle.resume();
      }

      top_candidates.push({nn, Distance::dist(components, nn->components(), Node::DIM)});
      ++thread->stats.distcomps;
    }

    // search base layer
    auto coro = search_level<without_lock>(components, ef_search_, 0, thread);
    while (!coro.handle.done()) {
      co_await std::suspend_always{};  // gives control back to caller of knn
      coro.handle.resume();
    }

    while (top_candidates.size() > k_) {
      top_candidates.pop();
    }

    auto& results = thread->query_results[q_id];  // coroutine independent
    for (const auto& [nn, _] : top_candidates.heap) {
      results.push_back(nn->id());
    }

    top_candidates.clear();
    ++thread->stats.processed;
  }

  size_t estimate_index_size(size_t num_nodes) const {
    size_t index_size = 0;
    const u32 num_levels = std::round(std::log(num_nodes) / std::log(m_));

    for (u32 i = 0; i < num_levels; ++i) {
      const size_t size =
        i == 0 ? Node::size_until_components() + Node::NEIGHBORLIST_SIZE_ZERO : Node::NEIGHBORLIST_SIZE;
      const f64 probability = std::pow(1. / m_, i);
      index_size += std::round(probability * num_nodes) * size;
    }

    return index_size;
  }

private:
  /**
   * @brief Traverse the graph down to `target_level`.
   *        Greedily find nearest neighbor (1-NN) of `q` starting from begin_level down to `target_level`.
   *
   * @param nearest_neighbor Will contain the closest neighbor to `q`; at the beginning it's the entry point.
   * @param closest_distance The distance from `q` to the initial nearest neighbor (i.e., the entry point).
   */
  template <Lock do_lock>
  MinorCoroutine search_for_one(const span<element_t> q,
                                s_ptr<Node>& nearest_neighbor,
                                distance_t closest_distance,
                                u32 begin_level,
                                u32 target_level,
                                const u_ptr<ComputeThread>& thread) const {
    s_ptr<Node> locked_node;
    bool changed;

    for (u32 level = begin_level; level > target_level; level--) {
      do {
        changed = false;

        if constexpr (do_lock) {
          locked_node = nearest_neighbor;  // no reference, otherwise nearest_neighbor could be destructed
          auto coro = rdma::spinlock_node(locked_node, thread);

          while (!coro.handle.done()) {
            co_await std::suspend_always{};  // gives control back to caller of search_for_one
            coro.handle.resume();
          }
        }

        // READ neighbor list of nearest_neighbor w.r.t. level
        const RemotePtr nlist_rptr{nearest_neighbor->rptr.memory_node(),
                                   nearest_neighbor->compute_remote_neighborlist_offset(level)};
        const s_ptr<Neighborlist> neighborlist = co_await rdma::read_neighborlist(nlist_rptr, level, thread);
        ++thread->stats.visited_neighborlists;

        // find closest neighbor
        s_ptr<Node> best_candidate;

        for (const RemotePtr& r_ptr : neighborlist->view()) {
          thread->stats.inc_visited_nodes(level);
          s_ptr<Node> candidate;
          {
            auto coro = cache_lookup(r_ptr, candidate, thread, not do_lock);  // always admit inner nodes
            while (!coro.handle.done()) {
              co_await std::suspend_always{};  // gives control back to caller
              coro.handle.resume();
            }
          }

          const f32 distance = Distance::dist(q, candidate->components(), Node::DIM);
          ++thread->stats.distcomps;

          if (distance < closest_distance) {
            closest_distance = distance;
            best_candidate = candidate;  // use temporary here, otherwise nearest_neighbor dangles with coroutines
            changed = true;
          }
        }

        nearest_neighbor = changed ? best_candidate : nearest_neighbor;

        if constexpr (do_lock) {
          co_await rdma::unlock_node(locked_node, thread);
        }

      } while (changed);
    }
  }

  static bool admit_to_cache(f32 prob) {
    thread_local std::mt19937 gen{std::random_device{}()};
    thread_local std::uniform_real_distribution<f32> dist(0., 1.);

    return (dist(gen) < prob);
  }

  /**
   * @brief Searches for the ef-NNs on the given `level` and stores them in `top_candidates` of `thread`.
   *        In the beginning, `top_candidates` contains a single entry point.
   */
  template <Lock do_lock>
  MinorCoroutine search_level(const span<element_t> q, u32 ef, u32 level, const u_ptr<ComputeThread>& thread) const {
    hashset_t<RemotePtr>& visited_nodes = thread->current_coroutine().visited_nodes;
    MaxHeap& top_candidates = thread->current_coroutine().top_candidates;
    MinHeap& next_candidates = thread->current_coroutine().next_candidates;

    for (const auto& [node, dist] : top_candidates.heap) {
      next_candidates.push({node, dist});  // copies shared_ptr
      visited_nodes.insert(node->rptr);
    }

    while (!next_candidates.empty()) {
      const auto [candidate, candidate_dist] = next_candidates.top();  // closest node
      next_candidates.pop();

      distance_t farthest_dist = top_candidates.top().distance;

      // no valuable candidate remains
      if (candidate_dist > farthest_dist) {
        break;
      }

      if constexpr (do_lock) {
        auto coro = rdma::spinlock_node(candidate, thread);
        while (!coro.handle.done()) {
          co_await std::suspend_always{};  // gives control back to caller of search_level
          coro.handle.resume();
        }
      }

      const RemotePtr nlist_rptr{candidate->rptr.memory_node(), candidate->compute_remote_neighborlist_offset(level)};
      const s_ptr<Neighborlist> neighborlist = co_await rdma::read_neighborlist(nlist_rptr, level, thread);
      ++thread->stats.visited_neighborlists;

      for (RemotePtr& neighbor_ptr : neighborlist->view()) {
        if (!visited_nodes.contains(neighbor_ptr)) {
          thread->stats.inc_visited_nodes(level);
          visited_nodes.insert(neighbor_ptr);

          s_ptr<Node> neighbor;
          {
            const bool admit =
              do_lock ? false : (not thread->cache.is_full() ? true : admit_to_cache(cache::ADMISSION_RATIO));
            auto coro = cache_lookup(neighbor_ptr, neighbor, thread, admit);
            while (!coro.handle.done()) {
              co_await std::suspend_always{};  // gives control back to caller
              coro.handle.resume();
            }
          }

          farthest_dist = top_candidates.top().distance;

          const distance_t neighbor_dist = Distance::dist(q, neighbor->components(), Node::DIM);
          ++thread->stats.distcomps;

          if (neighbor_dist < farthest_dist || top_candidates.size() < ef) {
            // copy neighbor with shared ownership
            next_candidates.push({neighbor, neighbor_dist});
            top_candidates.push_k({neighbor, neighbor_dist}, ef);  // only keeps best ef candidates
          }
        }
      }

      if constexpr (do_lock) {
        co_await rdma::unlock_node(candidate, thread);
      }
    }

    next_candidates.clear();
    visited_nodes.clear();
  }

  /**
   * @brief Selects `m` neighbors using a heuristic to preserve the connectivity of the graph.
   *        The number of resulting neighbors in `top_candidates` is <= `m`.
   */
  static void select_heuristic(MaxHeap& top_candidates, u32 m, const u_ptr<ComputeThread>& thread) {
    if (top_candidates.size() < m) {
      return;
    }

    // to get nearest candidates (destroys the max-heap property)
    top_candidates.sort_ascending();

    const size_t initial_heap_size = top_candidates.size();

    idx_t selected = 1;  // candidates selected so far (nearest neighbor is always selected)
    idx_t consumed = 1;  // candidates consumed so far (always >= selected)

    while (selected < m && consumed < initial_heap_size) {
      bool is_selected = true;
      const auto& [c_node, c_dist_to_query] = top_candidates.heap[consumed];  // get closest node to query

      // if the distance of the closest node C to all already selected nodes is larger than the distance from C to the
      // query, the heuristics selects C
      for (idx_t i = 0; i < selected; ++i) {
        const auto& selected_node = top_candidates.heap[i].node;
        const auto c_dist_to_selected = Distance::dist(selected_node->components(), c_node->components(), Node::DIM);
        ++thread->stats.distcomps;

        if (c_dist_to_selected < c_dist_to_query) {
          is_selected = false;
          break;
        }
      }

      if (is_selected) {
        std::swap(top_candidates.heap[selected], top_candidates.heap[consumed]);
        ++selected;
      }

      ++consumed;
    }

    top_candidates.heap.resize(selected);
    top_candidates.make_heap();
  }

  template <cache::Cacheable T>
  MinorCoroutine cache_lookup(RemotePtr rptr,
                              s_ptr<T>& value,
                              const u_ptr<ComputeThread>& thread,
                              bool admit_to_cache) const {
    if (not use_cache_) {
      value = co_await rdma::read_node(rptr, thread);
      co_return;
    }

    auto cache_entry = thread->cache.get<T>(rptr);
    if (cache_entry.has_value()) {
      value = *cache_entry;
      ++thread->stats.cache_hits;

    } else {
      value = co_await rdma::read_node(rptr, thread);

      if (admit_to_cache) {
        thread->cache.insert(rptr, value, thread->get_id());
      }

      ++thread->stats.cache_misses;
    }
  }

private:
  // construction parameters
  const u32 m_;
  const u32 m_max_;
  const u32 m_max_zero_;
  const u32 ef_construction_;
  const f64 normalization_factor_;

  // search parameters
  const u32 k_;
  const u32 ef_search_;
  const bool use_cache_;

  std::mt19937 prng_;
  std::uniform_real_distribution<> uniform_;
};

}  // namespace hnsw
