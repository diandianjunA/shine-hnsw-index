#!/usr/bin/python3

from datasets import Datasets
from subprocess import run, Popen, call
import config
import sys


def unmap(num_mns):
    for i, node in enumerate(config.ALL_MEMORY_NODES[:num_mns]):
        kill = "kill $(ps aux | grep 'vmtouch' | awk '{print $2}')"
        command = ["ssh", node, kill]
        print(" ".join(command))
        run(command)


def touch(dataset: Datasets, num_mns):
    processes = []
    for i, node in enumerate(config.ALL_MEMORY_NODES[:num_mns]):
        filename = f"{config.DATASETS_PATH}/{dataset.value.name}/dump/index_m{config.M}_efc{config.EF_CONSTRUCTION}_node{i + 1}_of{num_mns}.dat"
        exit_status = call(["ssh", node, f'test -f "{filename}"'])

        if exit_status != 0:
            sys.exit(f"file {filename} does not exist on node {node}")

        # first run vmtouch
        Popen(["ssh", node, f"vmtouch -ld {filename}"])

        # then run script that blocks until mapping is done
        command = ["ssh", node, f"python3 {config.REPOSITORY_PATH}/scripts/vmtouch_wait.py"]
        print(" ".join(command))
        process = Popen(command)
        processes.append((node, process))

    for (node, process) in processes:
        process.wait()
        print(f"{node} done")
