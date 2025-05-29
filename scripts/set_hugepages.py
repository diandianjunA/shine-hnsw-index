#!/usr/bin/python3

import sys
import config
from subprocess import run

if __name__ == "__main__":
    deallocate = len(sys.argv) > 1 and sys.argv[1] == "-d"
    nodes = config.ALL_MEMORY_NODES + config.ALL_COMPUTE_NODES

    GB_HUGEPAGES = True

    for node in nodes:
        amount_gb = 30 if node in config.ALL_COMPUTE_NODES else 40

        if GB_HUGEPAGES:
            command = ["ssh", node,
                       f"echo {0 if deallocate else amount_gb} > /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages"]
        else:
            max_num_hugepages = 0 if deallocate else int((amount_gb * 1024 ** 3) / (2 * 1024 ** 2))  # amount_gb / 2MB
            command = ["ssh", node,
                       f"echo {max_num_hugepages} > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages"]

        print(" ".join(command))
        run(command)
