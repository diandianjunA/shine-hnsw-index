#pragma once

#include <library/types.hh>

// constexpr static u64 COMPUTE_NODE_MAX_MEMORY = 10 * 1073741824ul;  // 10 GB
constexpr static u64 COMPUTE_NODE_MAX_MEMORY = 35ul * 1073741824ul;  // 35 GB (for CSP experiments)
constexpr static u64 MEMORY_NODE_MAX_MEMORY = 44ul * 1073741824ul;  // 44 GB
constexpr static u32 MAX_QPS = 4;  // max number of QPs per compute node
constexpr static size_t CACHELINE_SIZE = 64;

namespace cache {
constexpr static bool CACHE_WARMUP = true;
constexpr static size_t MAX_LOOKUP_RESTARTS = 100;  // optimistic cache lookup restarts
constexpr static size_t COOLING_TABLE_BUCKET_ENTRIES = 6;
constexpr static f32 COOLING_TABLE_RATIO = 0.1;  // must be < 1
constexpr static f32 ADMISSION_RATIO = 0.01;  // admit nodes to cache with 1% probability
}  // namespace cache

namespace query_router {
constexpr static bool BALANCED_ROUTING = true;  // use histogram to balance routing
constexpr static bool ADAPTIVE_ROUTING = true;
constexpr static bool BALANCED_KMEANS_WITH_HEURISTIC = true;

constexpr static u32 INITIAL_RECVS = 20;
constexpr static u32 LIMIT_PER_CN = 200;
constexpr static i32 MAX_QUEUE_SIZE = 1000;
constexpr static u32 POLL_TIMEOUT = 100000;
}  // namespace query_router
