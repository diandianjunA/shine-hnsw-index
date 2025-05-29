#pragma once

#include "common/types.hh"

#define CAS(_p, _u, _v) (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE))

/**
 * @brief Optimistic CAS lock implementation.
 *        Credits: https://github.com/baotonglu/dex
 */
class Lock {
  constexpr static u64 lock_set = static_cast<u64>(1) << 63;
  constexpr static u64 lock_mask = (static_cast<u64>(1) << 63) - 1;
  u64 version_lock = 0;

public:
  Lock() { version_lock = 0; }
  void reset() { version_lock = 0; }

  void get_lock() {
    u64 new_value = 0;
    u64 old_value = 0;

    do {
      while (true) {
        old_value = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
        if (!(old_value & lock_set)) {
          old_value &= lock_mask;
          break;
        }
      }
      new_value = old_value | lock_set;
    } while (!CAS(&version_lock, &old_value, new_value));
  }

  void release_lock() {
    const u64 v = version_lock;
    __atomic_store_n(&version_lock, v + 1 - lock_set, __ATOMIC_RELEASE);
  }

  /**
   * @brief Returns true if lock is set.
   */
  bool test_lock_set(u64& version) const {
    version = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
    return (version & lock_set) != 0;
  }

  /**
   * @brief Returns true if the lock version has changed.
   */
  bool test_lock_version_change(u64 old_version) const {
    const u64 value = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
    return (old_version != value);
  }
};