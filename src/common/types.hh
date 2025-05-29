#pragma once

#include <filesystem>
#include <library/types.hh>
#include <unordered_map>
#include <unordered_set>

using node_t = u32;
using element_t = f32;
using distance_t = f32;

using filepath_t = std::filesystem::path;

template <typename T>
using hashset_t = std::unordered_set<T>;  // TODO: replace with faster hashset

template <typename K, typename V>
using hashmap_t = std::unordered_map<K, V>;  // TODO: replace with faster hashmap