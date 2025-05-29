#pragma once

#include <array>
#include <optional>

#include "common/constants.hh"
#include "lock.hh"

namespace cache {

class CoolingTable {
public:
  struct TableEntry {
    RemotePtr key{};
  };

  struct Bucket {
    Lock lock;
    u32 count{0};
    std::array<TableEntry, COOLING_TABLE_BUCKET_ENTRIES> entries;
  };

  explicit CoolingTable(size_t cooling_table_buckets)
      : cooling_table_buckets_(cooling_table_buckets), table_(cooling_table_buckets) {}

  CoolingTable(const CoolingTable&) = delete;  // copy constructor
  CoolingTable(CoolingTable&&) noexcept = delete;  // move constructor
  CoolingTable& operator=(const CoolingTable&) = delete;  // copy assignment
  CoolingTable& operator=(CoolingTable&&) noexcept = delete;  // move assignment

  ~CoolingTable() {
    std::cerr << "cooling table histogram\n";
    vec<size_t> histogram(COOLING_TABLE_BUCKET_ENTRIES + 1, 0);
    size_t total_entries = 0;

    for (const auto& bucket : table_) {
      const size_t len = bucket.count;
      ++histogram[len];
      total_entries += len;
    }

    for (idx_t i = 0; i < histogram.size(); ++i) {
      std::cerr << "  size: " << i << " | freq: " << histogram[i] << std::endl;
    }
    std::cerr << "CT total entries: " << total_entries << std::endl;
  }

  /**
   * @brief Removes an entry from the cooling table (back to hot).
   * @return Returns true on success, false otherwise.
   */
  bool remove(RemotePtr key) {
    bool success = false;

    auto& bucket = table_[hash(key)];
    bucket.lock.get_lock();

    for (idx_t i = 0; i < bucket.count; ++i) {
      if (bucket.entries[i].key == key) {
        for (idx_t j = i + 1; j < bucket.count; ++j) {
          bucket.entries[j - 1] = bucket.entries[j];
        }

        success = true;
        --bucket.count;
        break;
      }
    }

    bucket.lock.release_lock();

    return success;
  }

  /**
   * @brief Inserts a new entry into the cooling table (hot to cold).
   *        Possibly drops entry, which is returned and must be evicted.
   */
  std::optional<RemotePtr> insert(RemotePtr key) {
    auto& bucket = table_[hash(key)];
    bucket.lock.get_lock();

    std::optional<RemotePtr> victim{};
    if (bucket.count < COOLING_TABLE_BUCKET_ENTRIES) {
      ++bucket.count;
    } else {
      victim = bucket.entries.back().key;
    }

    for (i32 i = static_cast<i32>(bucket.count) - 2; i >= 0; --i) {
      bucket.entries[i + 1] = bucket.entries[i];
    }

    bucket.entries[0] = TableEntry{key};
    bucket.lock.release_lock();

    return victim;
  }

private:
  u64 hash(RemotePtr key) const {
    // SplitMix64: http://xoroshiro.di.unimi.it/splitmix64.c
    u64 x = key.raw_address;
    x += 0x9E3779B97f4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return (x ^ (x >> 31)) % cooling_table_buckets_;

    // return std::hash<RemotePtr>{}(key) % cooling_table_buckets_;
  }

private:
  const size_t cooling_table_buckets_;
  std::vector<Bucket> table_;
};

}  // namespace cache
