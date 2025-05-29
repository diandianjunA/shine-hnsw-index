#include "common/configuration.hh"
#include "compute_node.hh"
#include "hnsw/distance.hh"
#include "memory_node.hh"

int main(int argc, char** argv) {
  configuration::IndexConfiguration config{argc, argv};
  // setenv("MLX_QP_ALLOC_TYPE", "HUGE", true);
  // setenv("MLX_CQ_ALLOC_TYPE", "HUGE", true);

  if (config.is_server) {
    MemoryNode memory_node{config};

  } else {
    if (config.ip_distance) {
      ComputeNode<IPDistance> compute_node{config};
    } else {
      ComputeNode<L2Distance> compute_node{config};
    }
  }

  return EXIT_SUCCESS;
}
