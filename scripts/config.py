from datasets import Datasets

PUSH_BENCHMARKS = True
COMPUTE_RECALL = False  # can only be true if a ground truth file for the respective query file exists
COROUTINES = 4

# hnsw parameters
EF_CONSTRUCTION = 500
M = 32

# servers
ALL_COMPUTE_NODES = ["cluster11", "cluster12", "cluster13", "cluster14", "cluster15"]
ALL_MEMORY_NODES = ["cluster16", "cluster17", "cluster18", "cluster20", "cluster19"]
INITIATOR = ALL_COMPUTE_NODES[0]

# paths
DATASETS_PATH = "/mnt/dbgroup-share/mwidmoser/hnsw-data"
REPOSITORY_PATH = "/root/mw/rdma-hnsw"
EXECUTABLE = f"{REPOSITORY_PATH}/build/shine"
RUNNER_SCRIPT = "/root/mw/rdma-hnsw/scripts/run_node.py"


def to_path(dataset: Datasets) -> str:
    return f"{DATASETS_PATH}/{dataset.value.name}/"


def get_cache_parameters(label: str, cache_size_ratio: int) -> str:
    return f"--cache --cache-ratio {cache_size_ratio} --routing" if "routing" in label else f"--cache --cache-ratio {cache_size_ratio}" if "cache" in label else ""


DNS = {"cluster11": "10.60.50.60",
       "cluster12": "10.60.50.61",
       "cluster13": "10.60.50.62",
       "cluster14": "10.60.50.63",
       "cluster15": "10.60.50.64",
       "cluster16": "10.60.50.65",
       "cluster17": "10.60.50.66",
       "cluster18": "10.60.50.67",
       "cluster19": "10.60.50.68",
       "cluster20": "10.60.50.69"}
