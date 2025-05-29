#!/usr/bin/python3

import argparse
import config
import sys
from subprocess import Popen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--release", action="store_true", help="Fresh builds in release mode")
    parser.add_argument("-d", "--debug", action="store_true", help="Fresh builds in debug mode")
    parser.add_argument("--num-cns", help="Number of compute nodes", required=True, type=int)
    parser.add_argument("--num-mns", help="Number of memory nodes", required=True, type=int)
    args = parser.parse_args()

    if args.release and args.debug:
        sys.exit("--release and --debug are mutually exclusive")

    build_type = "release" if args.release else ("debug" if args.debug else "build")

    nodes = config.ALL_MEMORY_NODES[:args.num_mns] + config.ALL_COMPUTE_NODES[:args.num_cns]
    processes = []

    for node in nodes:
        command = f"bash remote.sh {node} {build_type}"
        print(command)
        process = Popen(command, shell=True)
        processes.append((node, process))

    for (node, process) in processes:
        process.wait()
        print(f"{node} done")
