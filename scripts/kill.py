#!/usr/bin/python3

import config
from subprocess import run

if __name__ == "__main__":
    nodes = config.ALL_MEMORY_NODES + config.ALL_COMPUTE_NODES
    executable = "shine"

    executable = "[" + executable[0] + "]" + executable[1:]  # do not pick the grep process

    for node in nodes:
        kill = "kill $(ps aux | grep '" + executable + "' | awk '{print $2}')"
        command = ["ssh", node, kill]

        print(" ".join(command))
        run(command)
