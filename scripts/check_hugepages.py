from subprocess import run
import config


def get(node):
    result = run(["ssh", node, "hugeadm", "--pool-list", "|", "grep", "1073741824"], capture_output=True, text=True)
    num_pages_allocated = [x for x in result.stdout.strip().split(" ") if x][2]
    print(f"  - {node}: {num_pages_allocated}GB")


if __name__ == "__main__":
    print("compute nodes")
    for node in config.ALL_COMPUTE_NODES:
        get(node)

    print("\nmemory nodes")
    for node in config.ALL_MEMORY_NODES:
        get(node)
