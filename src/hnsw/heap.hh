#pragma once

#include <algorithm>

#include "common/types.hh"

class Node;
namespace heap {

struct heap_entry_t {
  s_ptr<Node> node;
  distance_t distance;
};

struct MaxHeapCompare {
  bool operator()(const heap_entry_t& lhs, const heap_entry_t& rhs) const { return lhs.distance < rhs.distance; }
};

struct MinHeapCompare {
  bool operator()(const heap_entry_t& lhs, const heap_entry_t& rhs) const { return lhs.distance > rhs.distance; }
};

template <typename Entry, class Compare>
struct Heap {
  vec<Entry> heap;

  Heap() = default;
  explicit Heap(size_t k) { heap.reserve(k); }

  void make_heap() { std::make_heap(heap.begin(), heap.end(), Compare()); }
  void clear() { heap.clear(); }

  // push entry but keep size of Heap <= k
  void push_k(const Entry& entry, size_t k) {
    if (size() < k) {
      push(entry);
    } else if (Compare()(entry, top())) {
      pop();
      push(entry);
    }
  }

  void push(const Entry& entry) {
    heap.push_back(entry);
    std::push_heap(heap.begin(), heap.end(), Compare());
  }

  void pop() {
    std::pop_heap(heap.begin(), heap.end(), Compare());
    heap.pop_back();
  }

  void sort_ascending() {
    std::sort(heap.begin(), heap.end(), [](const auto& lhs, const auto& rhs) {  // break ties by id
      return lhs.distance == rhs.distance ? lhs.node->id() < rhs.node->id() : lhs.distance < rhs.distance;
    });
  }

  Entry top() { return heap.front(); }
  [[nodiscard]] size_t size() const { return heap.size(); }
  [[nodiscard]] bool empty() const { return heap.empty(); }
};
}  // namespace heap

using MaxHeap = heap::Heap<heap::heap_entry_t, heap::MaxHeapCompare>;
using MinHeap = heap::Heap<heap::heap_entry_t, heap::MinHeapCompare>;
