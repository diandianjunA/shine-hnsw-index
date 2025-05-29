#pragma once

#include "database.hh"
#include "deserializer.hh"

namespace io {

// Reads data partially if num_clients > 1 (round-robin fashion).
template <typename T>
void read_data_partially(Database<T>& db,
                         const filepath_t& input_file,
                         u32 client_id,
                         u32 num_clients,
                         bool meta_only = false,
                         u32 num_vectors_to_read = std::numeric_limits<u32>::max()) {
  Deserializer d{input_file};
  u32 component_size_in_file{};
  bool signed_byte{};

  // TODO: could be improved
  if (input_file.extension() == ".fbin") {
    component_size_in_file = sizeof(f32);
  } else if (input_file.extension() == ".u8bin") {
    component_size_in_file = sizeof(u8);
    signed_byte = false;
  } else if (input_file.extension() == ".i8bin") {
    component_size_in_file = sizeof(i8);
    signed_byte = true;
  } else if (input_file.extension() == ".bin") {
    component_size_in_file = sizeof(u32);
  } else {
    lib_failure("unsupported file extension: " + input_file.extension().string());
  }

  db.num_vectors_total = d.read_u32();
  db.dim = d.read_u32();

  if (meta_only) {
    return;
  }

  u32 to_read = db.num_vectors_total / num_clients;
  const u32 remainder = db.num_vectors_total - to_read * num_clients;

  if (client_id < remainder) {
    ++to_read;
  }

  to_read = std::min(to_read, num_vectors_to_read);

  std::cerr << "reading input data... (dim=" << db.dim << ", num vectors=" << to_read << "/" << db.num_vectors_total
            << ", filesize=" << d.file_size() << ")" << std::endl;

  db.num_vectors_read = to_read;
  db.allocate();

  for (idx_t id = 0; id < db.num_vectors_total; ++id) {
    if (id % num_clients == client_id) {
      auto& slot = db.max_slot;

      if (signed_byte) {
        d.read_vector<T, i8>(db.dim, db.get_components(slot), component_size_in_file);
      } else {
        d.read_vector<T, u8>(db.dim, db.get_components(slot), component_size_in_file);
      }

      db.set_id(slot, id);
      ++slot;

      if (slot > to_read) {
        break;
      }

    } else {
      d.jump(db.dim * component_size_in_file);
    }
  }
}

template <typename T>
void read_data(Database<T>& db, const filepath_t& input_file) {
  read_data_partially<T>(db, input_file, 0, 1);
}

}  // namespace io