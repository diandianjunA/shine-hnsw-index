#!/usr/bin/python3

import os
import argparse
import subprocess
import sys
import config
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", default=1, help="Number of compute threads per compute node")
    parser.add_argument("-c", "--coroutines", default=2, help="Number of coroutines per compute threads")
    parser.add_argument("-d", "--dataset", help="Dataset name")
    parser.add_argument("-q", "--query-suffix", help="Query suffix")
    parser.add_argument("--label", help="Optional label to identify benchmarks")
    parser.add_argument("-e", "--ef-search", default=20, help="Beam width during search")
    parser.add_argument("--ef-construction", default=20, help="Beam width during construction")
    parser.add_argument("-m", default=16, help="Number of bidirectional connections during construction")
    parser.add_argument("-k", default=10, help="Number of nearest neighbors")
    parser.add_argument("-l", "--load-index", action="store_true", help="Memory servers load the index from file")
    parser.add_argument("-s", "--store-index", action="store_true", help="Memory servers write the index to file")
    parser.add_argument("--no-recall", action="store_true",
                        help="Avoid recall computation, ground truth file can be omitted")
    parser.add_argument("--cache", action="store_true", help="Activate CN caching")
    parser.add_argument("--routing", action="store_true", help="Activate query routing")
    parser.add_argument("--cache-ratio", default=5, help="Cache size ratio relative to index size in %")
    parser.add_argument("--ip-dist", action="store_true", help="Use inner product distance rather than squared L2 norm")
    parser.add_argument("--num-cns", help="Number of compute nodes", required=True, type=int)
    parser.add_argument("--num-mns", help="Number of memory nodes", required=True, type=int)
    args = parser.parse_args()

    host = os.uname().nodename
    numactl_args = "--preferred=1"
    command = ["numactl", numactl_args, config.EXECUTABLE]

    if host in config.ALL_COMPUTE_NODES[:args.num_cns]:
        command.extend(["--servers"] + config.ALL_MEMORY_NODES[:args.num_mns])
        command.extend(["--threads", str(args.threads)])
        command.extend(["--coroutines", str(args.coroutines)])
        command.extend(["--data-path", f"{config.DATASETS_PATH}/{args.dataset}/"])
        command.extend(["--query-suffix", args.query_suffix])
        command.extend(["--ef-search", str(args.ef_search)])
        command.extend(["--ef-construction", str(args.ef_construction)])
        command.extend(["--m", str(args.m)])
        command.extend(["--k", str(args.k)])

        if args.load_index:
            command.append("--load-index")
        if args.store_index:
            command.append("--store-index")
        if args.label:
            command.extend(["--label", str(args.label)])
        if args.no_recall:
            command.append("--no-recall")
        if args.ip_dist:
            command.append("--ip-dist")
        if args.cache:
            command.append("--cache")
            command.extend(["--cache-ratio", str(args.cache_ratio)])
        if args.routing:
            command.append("--routing")

        if host == config.INITIATOR:
            command.append("--initiator")
            if args.num_cns > 1:
                command.extend(["--clients"] + config.ALL_COMPUTE_NODES[1:args.num_cns])

    elif host in config.ALL_MEMORY_NODES[:args.num_mns]:
        command.append("--is-server")
        command.extend(["--num-clients", str(args.num_cns)])

    else:
        sys.exit(f"[ERROR]: host {host} neither specified as compute node nor as memory node")

    print(datetime.now().strftime('%Y%m%d-%H%M'), file=sys.stderr)
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command)
