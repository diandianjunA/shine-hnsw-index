import os
import benchmark
import vmtouch
from datasets import Datasets

datasets = [Datasets.DEEP_100M, Datasets.SPACEV_100M, Datasets.TURING_100M, Datasets.BIGANN_100M, Datasets.TTI_100M]
labels = ["v15-baseline", "v15-+cache", "v15-+adaptive-routing"]
query_suffixes = ["a0.0-500k", "a1.0-500k"]
compute_threads_range = (0, 32)  # inclusively (per compute node)

cache_size_ratio = 5  # in % relative to index size
num_compute_nodes = 5
num_memory_nodes = 3

if __name__ == "__main__":
    os.unsetenv("SSH_AUTH_SOCK")  # unset (forwarded) SSH agent for this script

    for dataset in datasets:
        benchmark.build_index_if_not_exists(dataset, num_memory_nodes, num_compute_nodes)

        vmtouch.unmap(num_memory_nodes)
        vmtouch.touch(dataset, num_memory_nodes)

        for label in labels:
            for query_suffix in query_suffixes:
                for t in range(compute_threads_range[0], compute_threads_range[1] + 1, 2):  # step=2
                    num_threads = max(1, t)
                    benchmark.run(dataset, num_memory_nodes, num_compute_nodes, num_threads, query_suffix, label,
                                  cache_size_ratio)
