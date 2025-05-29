#pragma once

#include <iomanip>
#include <iostream>
#include <library/configuration.hh>

#include "types.hh"

namespace configuration {

// struct used for sending serialized from CN to MN
struct Parameters {
  u32 num_threads{};
  bool use_cache{};
  bool routing{};
};

class IndexConfiguration : public Configuration {
public:
  filepath_t data_path{};
  str query_suffix{};
  u32 num_threads{};
  u32 num_coroutines{};
  i32 seed{};
  bool disable_thread_pinning{};
  str label{};  // for labeling benchmarks

  // HNSW parameters
  u32 ef_search{};
  u32 ef_construction{};
  u32 k{};
  u32 m{};

  bool store_index{};  // memory servers store the index; index is constructed from scratch; location is data_path
  bool load_index{};  // memory servers load index from file; cannot be used with store_index; location is data_path
  bool no_recall{};  // does not calculate the recall and thus requires no groundtruth
  bool ip_distance{};  // use the inner product distance rather than squared L2 norm

  u32 cache_size_ratio{};  // in %
  bool use_cache{};
  bool routing{};

public:
  IndexConfiguration(int argc, char** argv) {
    add_options();
    process_program_options(argc, argv);

    if (!is_server) {
      validate_compute_node_options(argv);
    }

    operator<<(std::cerr, *this);
  }

private:
  void add_options() {
    desc.add_options()("data-path,d",
                       po::value<filepath_t>(&data_path),
                       "Path to input directory containing the base vectors (\"base.fvecs\") and the \"query\" "
                       "directory (which contains the query and the groundtruth file).")(
      "threads,t", po::value<u32>(&num_threads), "Number of threads per compute node.")(
      "coroutines,C", po::value<u32>(&num_coroutines)->default_value(4), "Number of coroutines per compute thread.")(
      "disable-thread-pinning,p",
      po::bool_switch(&disable_thread_pinning)->default_value(false),
      "Disables pinning compute threads to physical cores if set.")(
      "seed", po::value<i32>(&seed)->default_value(1234), "Seed for PRNG; setting to -1 uses std::random_device.")(
      "label", po::value<str>(&label), "Optional label to identify benchmarks.")(
      "query-suffix,q", po::value<str>(&query_suffix), "Filename suffix for the query file.")(
      "store-index,s",
      po::bool_switch(&store_index),
      "Construct the index from scratch and the memory servers store the index to a file.")(
      "load-index,l",
      po::bool_switch(&load_index),
      "The index is not built, the memory servers load the index from a file.")(
      "cache", po::bool_switch(&use_cache), "Activate cache on CNs.")(
      "routing", po::bool_switch(&routing), "Activate adaptive query routing.")(
      "cache-ratio",
      po::value<u32>(&cache_size_ratio)->default_value(5),
      "Cache size ratio relative to the index size in %.")(
      "no-recall", po::bool_switch(&no_recall), "No recall computation, ground truth file can be omitted.")(
      "ip-dist", po::bool_switch(&ip_distance), "Use the inner product distance rather than the squared L2 norm.")(
      "ef-search", po::value<u32>(&ef_search), "Beam width during search.")(
      "ef-construction", po::value<u32>(&ef_construction)->default_value(200), "Beam width during construction.")(
      "k,k", po::value<u32>(&k), "Number of k nearest neighbors.")(
      "m,m", po::value<u32>(&m)->default_value(32), "Number of bidirectional connections in the HNSW graph.");
  }

  void validate_compute_node_options(char** argv) const {
    if (data_path.empty() || query_suffix.empty()) {
      std::cerr << "[ERROR]: Data path and query suffix cannot be empty" << std::endl;
      exit_with_help_message(argv);
    }

    if (num_threads == 0 || ef_search == 0 || k == 0) {
      std::cerr << "[ERROR]: Parameters threads, ef-search, and k are required" << std::endl;
      exit_with_help_message(argv);
    }

    if (store_index && load_index) {
      std::cerr << "[ERROR]: --store-index and --load-index cannot be used in conjunction" << std::endl;
      exit_with_help_message(argv);
    }

    if (use_cache && cache_size_ratio == 0) {
      std::cerr << "[ERROR]: If --cache is set, --cache-ratio must be > 0" << std::endl;
      exit_with_help_message(argv);
    }

    if (routing && not use_cache) {
      std::cerr << "[ERROR]: --routing can only be used in conjunction with --cache" << std::endl;
      exit_with_help_message(argv);
    }
  }

public:
  friend std::ostream& operator<<(std::ostream& os, const IndexConfiguration& config) {
    os << static_cast<const Configuration&>(config);

    if (config.is_initiator) {
      constexpr i32 width = 30;
      constexpr i32 max_width = width * 2;

      os << std::left << std::setfill(' ');
      os << std::setw(width) << "data path: " << config.data_path << std::endl;
      os << std::setw(width) << "query suffix: " << config.query_suffix << std::endl;
      os << std::setw(width) << "number of threads: " << config.num_threads << std::endl;
      os << std::setw(width) << "number of coroutines: " << config.num_coroutines << std::endl;
      os << std::setw(width) << "threads pinned: " << (config.disable_thread_pinning ? "false" : "true") << std::endl;
      os << std::setw(width) << "seed: " << config.seed << std::endl;
      os << std::setfill('-') << std::setw(max_width) << "" << std::endl;
      os << std::left << std::setfill(' ');
      os << std::setw(width) << "K: " << config.k << std::endl;
      os << std::setw(width) << "M: " << config.m << std::endl;
      os << std::setw(width) << "ef search: " << config.ef_search << std::endl;
      os << std::setw(width) << "ef construction: " << config.ef_construction << std::endl;
      os << std::setfill('=') << std::setw(max_width) << "" << std::endl;
    }
    return os;
  }
};

}  // namespace configuration
