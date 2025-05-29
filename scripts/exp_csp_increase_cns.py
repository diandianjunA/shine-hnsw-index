import os
import benchmark
import vmtouch
from datasets import Datasets

# make sure that sufficiently many hugepages are available on the CN
# make sure that COMPUTE_NODE_MAX_MEMORY in constants.hh is large enough
cache_size_ratio = 5

datasets = [Datasets.DEEP_100M, Datasets.SPACEV_100M, Datasets.TURING_100M, Datasets.BIGANN_100M, Datasets.TTI_100M]
labels = ["v15-csp-inc-+cache", "v15-csp-inc-+routing"]
query_suffixes = ["a0.0-500k", "a1.0-500k"]

num_compute_threads = 32
num_memory_nodes = 3

if __name__ == "__main__":
    os.unsetenv("SSH_AUTH_SOCK")  # unset (forwarded) SSH agent for this script

    for dataset in datasets:
        benchmark.build_index_if_not_exists(dataset, num_memory_nodes, 5)

        vmtouch.unmap(num_memory_nodes)
        vmtouch.touch(dataset, num_memory_nodes)

        for num_compute_nodes in range(1, 5 + 1):
            for query_suffix in query_suffixes:
                if num_compute_nodes == 1:
                    for csr in range(2, 5 + 1):
                        benchmark.run(dataset, num_memory_nodes, num_compute_nodes, num_compute_threads, query_suffix,
                                      labels[0], int(csr * cache_size_ratio))
                else:
                    for label in labels:
                        benchmark.run(dataset, num_memory_nodes, num_compute_nodes, num_compute_threads, query_suffix,
                                      label, cache_size_ratio)
