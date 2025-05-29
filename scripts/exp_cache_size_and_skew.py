import os
import benchmark
import vmtouch
from datasets import Datasets

datasets = [Datasets.DEEP_100M, Datasets.SPACEV_100M, Datasets.TURING_100M, Datasets.BIGANN_100M, Datasets.TTI_100M]
labels = ["v15-csize-skew-baseline", "v15-csize-skew-+cache", "v15-csize-skew-+adaptive-routing"]
query_suffixes = ["a0.0-500k", "a0.5-500k", "a0.75-500k", "a1.0-500k", "a1.25-500k", "a1.5-500k"]
cache_size_ratios = [2, 4, 5, 6, 8, 10]  # in % relative to index size

num_compute_threads = 32
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
                for cache_size_ratio in cache_size_ratios:
                    if query_suffix not in ["a0.0-500k", "a1.0-500k"] and cache_size_ratio != 5:
                        continue
                    benchmark.run(dataset, num_memory_nodes, num_compute_nodes, num_compute_threads, query_suffix,
                                  label, cache_size_ratio)
                    if "baseline" in label:
                        break
