#pragma once

#include <iostream>
#include <string>

namespace dbg {
using stream = std::stringstream;

#ifdef DEBUG_HNSW
inline void print(stream&& s) {
  std::cerr << s.str();
}
#else
inline void print(stream&&) {
}
#endif

}  // namespace dbg
