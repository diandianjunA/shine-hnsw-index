#!/usr/bin/python3

from datasets import Datasets
from datetime import datetime
from subprocess import Popen, DEVNULL, PIPE, call
from pymongo import MongoClient
from bson import json_util
import config
import mongodb
import sys


def exists_remote(dataset: Datasets, num_mns, m, ef_construction):
    exists = True

    for node_id in range(1, num_mns + 1):
        path = f"{config.to_path(dataset)}dump/index_m{m}_efc{ef_construction}_node{node_id}_of{num_mns}.dat"
        exit_status = call(["ssh", config.DNS[config.INITIATOR], f'test -f "{path}"'])

        if exit_status != 0:
            exists = False

    return exists


def run_memory_nodes(num_mns, num_cns):
    logfile = f"/var/log/rdma_hnsw_{datetime.now().strftime('%Y%m%d-%H%M')}.log"
    for memory_node in config.ALL_MEMORY_NODES[:num_mns]:
        Popen(["ssh", f"root@{config.DNS[memory_node]}",
               f"python3 {config.RUNNER_SCRIPT} --num-mns {num_mns} --num-cns {num_cns}", "&>", logfile],
              stdout=DEVNULL,
              stderr=DEVNULL,
              close_fds=True)


def run_compute_nodes(num_cns, num_mns, dataset: Datasets, query_suffix, threads, coroutines, label, cache_size_ratio,
                      store_index=False):
    logfile = f"/var/log/rdma_hnsw_{datetime.now().strftime('%Y%m%d-%H%M')}.log"
    command = f"python3 {config.RUNNER_SCRIPT} --label {label} --dataset {dataset.value.name} " \
              f"--query-suffix {query_suffix} --threads {threads} --coroutines {coroutines} " \
              f"-m {config.M} --ef-construction {config.EF_CONSTRUCTION} --ef-search {dataset.value.ef_search} " \
              f"--num-cns {num_cns} --num-mns {num_mns}"
    command += " --store-index" if store_index else " --load-index"

    cache_parameters = config.get_cache_parameters(label, cache_size_ratio)

    if cache_parameters:
        command += f" {cache_parameters}"
    if not config.COMPUTE_RECALL:
        command += " --no-recall"
    if dataset.value.ip_dist:
        command += " --ip-dist"

    # run compute nodes
    for compute_node in config.ALL_COMPUTE_NODES[1:num_cns]:
        Popen(["ssh", f"root@{config.DNS[compute_node]}", command, "&>", logfile], stdout=DEVNULL, stderr=DEVNULL,
              close_fds=True)

    # run initiator
    p = Popen(["ssh", f"root@{config.DNS[config.INITIATOR]}", command], stdout=PIPE, text=True)
    benchmark_output, _ = p.communicate()
    print(benchmark_output, file=sys.stderr)

    return benchmark_output


def push_benchmark(benchmark):
    if config.PUSH_BENCHMARKS:
        try:
            client = MongoClient(mongodb.connection_string)
            client["experiments"]["benchmarks"].insert_one(json_util.loads(benchmark))
            client.close()

        except Exception as e:
            sys.exit(f"error: {e}")


def build_index_if_not_exists(dataset: Datasets, num_mns, num_cns):
    # check whether binary index already exists, if not, construct it beforehand with 16 threads
    if not exists_remote(dataset, num_mns, config.M, config.EF_CONSTRUCTION):
        print(f"construct and store index for dataset {dataset.value.name}", file=sys.stderr)
        run_memory_nodes(num_mns, num_cns)
        output = run_compute_nodes(num_cns, num_mns, dataset, "a0.0-500k", 16, 4, "build", 0, store_index=True)
        push_benchmark(output)


def run(dataset: Datasets, num_mns, num_cns, num_threads, query_suffix, label, cache_size_ratio):
    run_memory_nodes(num_mns, num_cns)
    output = run_compute_nodes(num_cns, num_mns, dataset, query_suffix, num_threads, config.COROUTINES, label,
                               cache_size_ratio)
    push_benchmark(output)
