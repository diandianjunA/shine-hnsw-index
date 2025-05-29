#pragma once

#include <library/hugepage.hh>
#include <library/utils.hh>
#include <optional>
#include <type_traits>

#include "cache_entry.hh"
#include "cooling_table.hh"
#include "lock.hh"

namespace cache {
template <class T>
concept Cacheable = std::is_same_v<T, Node>;

class Cache {
  struct Bucket {
    Lock lock;
    tagged_ptr<CacheEntry> head_entry{};
    u32 count{0};
  };

public:
  explicit Cache(size_t cache_size,
                 size_t num_cache_buckets,
                 size_t num_cooling_table_buckets,
                 size_t num_threads,
                 bool use_cache)
      : num_cache_buckets_(num_cache_buckets),
        max_cache_size_(cache_size),
        cache_(num_cache_buckets),
        cooling_table_(num_cooling_table_buckets),
        tl_freelists_(num_threads) {
    if (use_cache) {
      cache_entries_.resize(num_cache_buckets_);

      std::cerr << "num cache buckets: " << num_cache_buckets_ << std::endl;
      std::cerr << "allocate cache entries hugepage of size " << num_cache_buckets_ * sizeof(CacheEntry) << std::endl;
      cache_entries_buffer_.allocate(num_cache_buckets_ * sizeof(CacheEntry));
      cache_entries_buffer_.touch_memory();

      for (auto& entry : cache_entries_) {
        // placement new operator
        entry = new (cache_entries_buffer_.get_slice_unaligned(sizeof(CacheEntry))) CacheEntry();
      }
    }
  }

  Cache(const Cache&) = delete;  // copy constructor
  Cache(Cache&&) noexcept = delete;  // move constructor
  Cache& operator=(const Cache&) = delete;  // copy assignment
  Cache& operator=(Cache&&) noexcept = delete;  // move assignment

  ~Cache() {
    std::cerr << "cache histogram\n";
    vec<size_t> histogram;
    size_t total_entries = 0;

    for (const auto& bucket : cache_) {
      const size_t len = bucket.count;

      if (len >= histogram.size()) {
        histogram.resize(len + 1);
      }
      ++histogram[len];
      total_entries += len;
    }

    for (idx_t i = 0; i < histogram.size(); ++i) {
      std::cerr << "  size: " << i << " | freq: " << histogram[i] << std::endl;
    }
    std::cerr << "total entries: " << total_entries << std::endl;
  }

  void track_cache_statistics(statistics::Statistics& stats) const {
    size_t size = 0, nodes = 0;

    for (const auto& bucket : cache_) {
      auto entry = bucket.head_entry;
      while (entry.get()) {
        if (entry.get()->value.load()) {
          size += Node::size_until_components();
          ++nodes;

        } else {
          lib_failure("empty cache entry");
        }

        entry = entry.get()->next;
      }
    }

    stats.add_nested_static_stat("cache", "local_size", size);
    stats.add_nested_static_stat("cache", "cached_nodes", nodes);
    stats.add_nested_static_stat("cache", "cache_buckets_size", num_cache_buckets_ * sizeof(CacheEntry));
  }

  /**
   * @return s_ptr<T> if the value is in the cache (cache hit), nullopt otherwise
   */
  template <Cacheable T>
  std::optional<s_ptr<T>> get(RemotePtr key) {
    const auto& bucket = cache_[hash(key)];
    size_t local_restarts = 0;

  retry:
    if (local_restarts > MAX_LOOKUP_RESTARTS) {
      return {};
    }

    auto cache_entry_ptr = bucket.head_entry;
    while (cache_entry_ptr.get()) {
      // verify pointer with entry
      if (cache_entry_ptr.tag() != cache_entry_ptr.get()->tag.load(std::memory_order_acquire)) {
        ++local_restarts;
        goto retry;
      }

      if (cache_entry_ptr.get()->key == key) {
        s_ptr<T> value = cache_entry_ptr.get()->value.load(std::memory_order_acquire);

        // validate
        if (cache_entry_ptr.tag() != cache_entry_ptr.get()->tag.load(std::memory_order_acquire)) {
          ++local_restarts;
          goto retry;
        }

        if (cache_entry_ptr.get()->cooling) {
          const bool success = cooling_table_.remove(key);

          if (success) {
            cache_entry_ptr.get()->cooling = false;
          }
        }

        lib_assert(value, "cache entry cannot be null");
        return value;
      }

      cache_entry_ptr = cache_entry_ptr.get()->next;
    }

    return {};
  }

  template <typename T>
  void insert(RemotePtr key, const s_ptr<T>& value, u32 thread_id) {
    auto& freelist = tl_freelists_[thread_id];
    tagged_ptr<CacheEntry> new_entry;

    if (not freelist.empty()) {
      new_entry = freelist.back();
      freelist.pop_back();

    } else if (is_full()) {
      new_entry = evict();

    } else {  // case before the cache is full
      const idx_t idx = cache_entry_idx_.fetch_add(1);

      if (idx < cache_entries_.size()) {
        new_entry = tagged_ptr(cache_entries_[idx]);
      } else {
        new_entry = evict();
      }
    }

    auto& bucket = cache_[hash(key)];
    bucket.lock.get_lock();

    if (bucket.count > 0) {
      CacheEntry* last_bucket{};
      auto cache_entry = bucket.head_entry.get();

      while (cache_entry) {
        if (cache_entry->key == key) {  // other thread already inserted
          freelist.push_back(new_entry);

          bucket.lock.release_lock();
          return;
        }

        last_bucket = cache_entry;
        cache_entry = cache_entry->next.get();
      }

      lib_assert(last_bucket, "last bucket cannot be null");

      new_entry.get()->key = key;
      new_entry.get()->update_value(value);

      last_bucket->next = new_entry;

    } else {
      new_entry.get()->key = key;
      new_entry.get()->update_value(value);

      bucket.head_entry = new_entry;
    }

    ++bucket.count;
    bucket.lock.release_lock();
  }

  bool is_full() {
    if (is_full_) {
      return true;
    }

    if (cache_entry_idx_ >= cache_entries_.size()) {
      is_full_ = true;
      return true;
    }

    return false;
  }

private:
  u64 hash(RemotePtr key) const { return std::hash<RemotePtr>{}(key) % num_cache_buckets_; }

  idx_t random_hash() const {
    thread_local std::mt19937 generator{std::random_device{}()};  // static storage
    thread_local std::uniform_int_distribution<idx_t> distribution(0, num_cache_buckets_ - 1);
    return distribution(generator);
  }

  /**
   * @brief Set a random cache entry to cooling and insert into cooling table.
   *        Cooling table returns an eviction candidate.
   * @return A free cache entry.
   */
  tagged_ptr<CacheEntry> evict() {
    tagged_ptr<CacheEntry> evicted_entry;
    bool evicted = false;

    while (not evicted) {
      const idx_t idx = random_hash();  // with replacement, but doesn't matter
      auto& bucket = cache_[idx];

      bucket.lock.get_lock();

      if (bucket.count == 0) {
        bucket.lock.release_lock();
        continue;
      }

      const idx_t entry_idx = random_hash() % bucket.count;
      auto entry = bucket.head_entry;
      for (idx_t i = 0; i < entry_idx; ++i) {
        entry = entry.get()->next;
      }

      lib_assert(entry.get(), "invalid eviction candidate");

      std::optional<RemotePtr> victim;
      if (not entry.get()->cooling) {
        victim = cooling_table_.insert(entry.get()->key);
        entry.get()->cooling = true;
      }

      bucket.lock.release_lock();

      if (victim.has_value()) {  // actual eviction
        auto& victim_bucket = cache_[hash(*victim)];
        victim_bucket.lock.get_lock();

        auto cache_entry_ptr = victim_bucket.head_entry;
        CacheEntry* prev_bucket{};

        // search entry in bucket and remove from linked list
        while (cache_entry_ptr.get()) {
          if (cache_entry_ptr.get()->key == *victim) {
            if (not cache_entry_ptr.get()->cooling) {
              break;
            }

            if (!prev_bucket && cache_entry_ptr.get()->next.get()) {  // first (but not only) entry in list
              victim_bucket.head_entry = cache_entry_ptr.get()->next;

            } else if (!prev_bucket) {  // only bucket in list
              victim_bucket.head_entry.invalidate();

            } else if (cache_entry_ptr.get()->next.get()) {  // bucket has predecessor and successor
              prev_bucket->next = cache_entry_ptr.get()->next;

            } else {  // last bucket in list
              prev_bucket->next.invalidate();
            }

            lib_assert(cache_entry_ptr.get()->cooling, "cannot evict non cooling cache entry");

            const u16 new_tag = cache_entry_ptr.get()->evict();  // invalidates the actual cache entry (increases tag)
            cache_entry_ptr.update_tag(new_tag);  // only for our local freelist (new valid pointer)
            evicted_entry = cache_entry_ptr;

            --victim_bucket.count;
            evicted = true;

            break;
          }

          prev_bucket = cache_entry_ptr.get();
          cache_entry_ptr = cache_entry_ptr.get()->next;
        }

        victim_bucket.lock.release_lock();
      }
    }

    return evicted_entry;
  }

private:
  const size_t num_cache_buckets_;
  const i64 max_cache_size_;

  vec<Bucket> cache_;
  CoolingTable cooling_table_;

  HugePage<byte_t> cache_entries_buffer_;
  vec<CacheEntry*> cache_entries_;
  std::atomic<idx_t> cache_entry_idx_{0};

  vec<vec<tagged_ptr<CacheEntry>>> tl_freelists_;  // thread local references to tagged cache entries
  bool is_full_ = false;
};

}  // namespace cache