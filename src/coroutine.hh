#pragma once

#include <coroutine>

#include "hnsw/heap.hh"
#include "remote_pointer.hh"

/**
 * Coroutines called by other coroutines.
 * Handle is destroyed by the destructor to prevent memory leaks.
 */
struct MinorCoroutine {
  struct promise_type {
    MinorCoroutine get_return_object() { return MinorCoroutine{Handle::from_promise(*this)}; }
    // std::suspend_never directly runs the coroutine (the object is created after first suspend)
    static std::suspend_never initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;

  explicit MinorCoroutine(Handle handle) : handle(handle) {}

  ~MinorCoroutine() {
    if (handle) {
      handle.destroy();
    }
  }

  MinorCoroutine(const MinorCoroutine&) = delete;
  MinorCoroutine(MinorCoroutine&&) = delete;
  MinorCoroutine& operator=(const MinorCoroutine&) = delete;
  MinorCoroutine& operator=(MinorCoroutine&&) noexcept = delete;

  Handle handle;
};

/**
 * Fixed number of HNSWCoroutines per ComputeThread.
 * Method schedule() in scheduler.hh is responsible for destroying HNSWCoroutine handles.
 */
struct HNSWCoroutine {
  struct promise_type {
    HNSWCoroutine get_return_object() { return HNSWCoroutine{Handle::from_promise(*this)}; }
    // std::suspend_always directly creates the coroutine object
    // (otherwise we cannot access our members before the first co_await)
    static std::suspend_always initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;
  Handle handle;

  // HNSW parameters
  RemotePtr cached_ep_ptr{};
  hashset_t<RemotePtr> visited_nodes{};
  MaxHeap top_candidates{};
  MinHeap next_candidates{};
};
