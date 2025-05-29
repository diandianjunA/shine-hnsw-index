#pragma once

#include "hnsw.hh"
#include "router/query_router.hh"

namespace hnsw {

static HNSWCoroutine dummy_coroutine() {
  co_return;
}

/**
 * @brief Schedules coroutine processing (HNSW inserts or knn-queries) of a compute thread.

 * @tparam insert If true, hnsw.insert() is called, hnsw.knn() otherwise.
 * @param next_idx Next unprocessed slot. Shared across all threads. Increased via FAA. Only used for inserts.
 * @param db Either containing the (partial) vectors to insert or (partial) queries.
 */
template <class Distance, bool insert>
void schedule(HNSW<Distance>& hnsw,
              std::atomic<idx_t>& next_idx,
              io::Database<element_t>& db,
              u32 num_coroutines,
              const u_ptr<ComputeThread>& thread,
              query_router::QueryRouter<Distance>* query_router = nullptr) {
  const auto print_status = [&db](idx_t slot) {
    if (slot % (db.num_vectors_total / 10) == 0) {
      std::cerr << (insert ? "insert " : "query ") << db.get_id(slot) << "/" << db.num_vectors_total << std::endl;
    }
  };

  if constexpr (not insert) {
    lib_assert(query_router, "invalid query_router");
  }

  // initialize coroutines
  thread->coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(dummy_coroutine()));
  }

  for (;;) {
    bool all_done = true;
    for (u32 coroutine_id = 0; coroutine_id < thread->coroutines.size(); ++coroutine_id) {
      auto& coroutine = *thread->coroutines[coroutine_id];
      thread->poll_cq();

      // recycle coroutine (assign new query)
      if (coroutine.handle.done()) {
        if constexpr (insert) {
          const idx_t slot = next_idx.fetch_add(1);

          if (slot < db.num_vectors_read) {
            print_status(slot);
            all_done = false;

            coroutine.handle.destroy();
            thread->set_current_coroutine(coroutine_id);

            coroutine.handle = hnsw.insert(db.get_id(slot), db.get_components(slot), thread).handle;
          }

        } else {
          if (not query_router->done || query_router->queue_size > 0) {
            idx_t slot;
            all_done = false;

            if (query_router->query_queue.try_dequeue(slot)) {
              query_router->queue_size.fetch_sub(1);
              print_status(slot);

              coroutine.handle.destroy();
              thread->set_current_coroutine(coroutine_id);

              coroutine.handle = hnsw.knn(db.get_id(slot), db.get_components(slot), thread).handle;
            }
          }
        }

        // resume coroutine
      } else if (thread->is_ready(coroutine_id)) {
        all_done = false;

        thread->set_current_coroutine(coroutine_id);
        coroutine.handle.resume();

        // keep polling
      } else {
        all_done = false;
      }
    }

    if (all_done) {
      break;
    }
  }

  for (const auto& coroutine : thread->coroutines) {
    lib_assert(coroutine->handle.done(), "coroutine not done yet");
    coroutine->handle.destroy();
  }
}

}  // namespace hnsw
