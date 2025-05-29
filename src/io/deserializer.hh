#pragma once

#include <fstream>
#include <iostream>
#include <library/utils.hh>

#include "common/types.hh"

namespace io {

class Deserializer {
public:
  explicit Deserializer(const str& binary_file) : binary_s_(binary_file, std::ios::in | std::ios::binary) {
    lib_assert(binary_s_.good(), "file \"" + binary_file + "\" does not exist");

    binary_s_.seekg(0, std::ios::end);
    file_size_ = binary_s_.tellg();
    reset_stream();
  }

  void reset_stream() { binary_s_.seekg(0, std::ios::beg); }

  template <typename T, typename ByteType>
  void read_vector(u32 dim, span<T>&& v, u32 component_size_in_file) {
    if (sizeof(T) == component_size_in_file) {
      if (!binary_s_.read(reinterpret_cast<char*>(v.data()), dim * component_size_in_file)) {
        lib_failure("cannot read file");
      }

      // read byte vector
    } else if (sizeof(byte_t) == component_size_in_file) {
      vec<ByteType> buffer(dim);
      if (!binary_s_.read(reinterpret_cast<char*>(buffer.data()), dim * component_size_in_file)) {
        lib_failure("cannot read file");
      }

      for (idx_t i = 0; i < dim; ++i) {
        v[i] = static_cast<T>(buffer[i]);
      }

    } else {
      lib_failure("unsupported file component size");
    }
  }

  u32 read_u32() {
    u32 integer;

    if (!binary_s_.read(reinterpret_cast<char*>(&integer), sizeof(u32))) {
      std::cerr << "Cannot read file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return integer;
  }

  bool bytes_left() { return binary_s_.tellg() < file_size_; }
  void jump(u32 num_bytes) { binary_s_.ignore(num_bytes); }
  i64 file_size() const { return file_size_; }

private:
  std::ifstream binary_s_;
  i64 file_size_;
};

}  // namespace io
