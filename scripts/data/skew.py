#!/usr/bin/python3

import argparse
import math
import numpy as np
import pathlib
import struct
import sys

# https://en.wikipedia.org/wiki/Zipf%27s_law#Formal_definition
PLOT_DISTRIBUTION = False


def harmonic_number(n, alpha) -> float:
    H = 0
    for k in range(1, n + 1):
        H += 1 / k ** alpha

    return H


def pmf(k, h_num, alpha) -> float:
    return (1 / k ** alpha) / h_num


def read_int(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


# https://docs.python.org/3/library/struct.html#format-characters
def get_struct_format(vector_type) -> str:
    if vector_type == np.float32:
        return "<f"
    if vector_type == np.uint32:
        return "<I"
    elif vector_type == np.uint8:
        return "<B"
    elif vector_type == np.int8:
        return "<b"
    else:
        sys.exit("invalid vector type")


def read_file(filename, vector_type):
    component_size = np.dtype(vector_type).itemsize
    struct_format = get_struct_format(vector_type)
    vectors = []

    with open(filename, "rb") as f:
        num_vectors_to_read = read_int(f)
        dimension = read_int(f)

        print("num vectors:", num_vectors_to_read)
        print("dimension:", dimension)

        for _ in range(num_vectors_to_read):
            vec = []
            for _ in range(dimension):
                vec.append(struct.unpack(struct_format, f.read(component_size)))

            vectors.append(np.array(vec, dtype=vector_type))

    return vectors


def write_file(filename, vectors, vector_type):
    pathlib.Path(filename.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
    struct_format = get_struct_format(vector_type)
    dimension = len(vectors[0])

    with open(filename, "wb") as file:
        file.write(struct.pack("<I", len(vectors)))
        file.write(struct.pack("<I", dimension))

        for vec in vectors:
            for component in vec.ravel():
                file.write(struct.pack(struct_format, component))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-path", required=True,
                        help="Output path, should be the dataset path, queries are written to 'path/queries/'")
    parser.add_argument("-q", "--queries", required=True, help="Query file in [f|u8]bin format, e.g., 'query.bin'")
    parser.add_argument("-g", "--ground-truth",
                        help="Ground truth file in bin format, if omitted, no ground truth file is generated")
    parser.add_argument("-n", "--num-queries", type=int, required=True, help="Number of queries to generate")
    parser.add_argument("-a", "--alpha", type=float, required=True, help="Zipf parameter (s), 0.0 means uniform")
    parser.add_argument("-s", "--split", type=int, help="Number of queries to split for cache warmup")
    args = parser.parse_args()

    if not args.ground_truth:
        print("ground truth file omitted")

    args.out_path = args.out_path[:-1] if args.out_path[-1] == "/" else args.out_path
    file_extension = args.queries.rsplit(".", 1)[-1]

    print("reading vectors...")
    if file_extension == "u8bin":
        component_type = np.uint8
    elif file_extension == "i8bin":
        component_type = np.int8
    elif file_extension == "fbin":
        component_type = np.float32
    else:
        sys.exit("invalid file extension")

    orig_queries = read_file(args.queries, component_type)
    if args.ground_truth:
        orig_ground_truth = read_file(f"{args.ground_truth}", np.uint32)

    num_vectors = len(orig_queries)

    print("computing probabilities...")
    h_num = harmonic_number(num_vectors, args.alpha)
    probabilities: list[float] = []

    # determine the probability for each vector in the dataset
    for k in range(1, num_vectors + 1):
        probabilities.append(pmf(k, h_num, args.alpha))

    distribution: list[int] = []
    drawn = 0

    for idx in range(num_vectors):
        if drawn >= args.num_queries:
            break

        occurrences = math.ceil(args.num_queries * probabilities[idx])

        distribution.append(occurrences)
        drawn += occurrences

    print("  drawn:", drawn)
    assert drawn == args.num_queries
    # TODO: fix the case where drawn > num_queries (simply decrease from [-1])

    print("sampling new vectors...")

    # this is quite space inefficient...
    new_queries = []
    new_ground_truth = []

    for idx, frequency in enumerate(distribution):
        new_queries += [orig_queries[idx]] * frequency
        if args.ground_truth:
            new_ground_truth += [orig_ground_truth[idx]] * frequency

    print("len new queries:", len(new_queries))
    print("sum distribution:", sum(distribution))

    print("shuffling vectors...")
    p = np.random.permutation(args.num_queries)
    new_queries = np.array(new_queries)[p]
    if args.ground_truth:
        new_ground_truth = np.array(new_ground_truth)[p]

    print("writing vectors...")
    if args.split:
        num_queries = args.num_queries - args.split
        write_file(f"{args.out_path}/queries/query-a{args.alpha}-n{num_queries}.{file_extension}",
                   new_queries[:num_queries], component_type)
        write_file(f"{args.out_path}/queries/warmup-a{args.alpha}-n{args.split}.{file_extension}",
                   new_queries[num_queries:], component_type)
    else:
        write_file(f"{args.out_path}/queries/query-a{args.alpha}-n{args.num_queries}.{file_extension}",
                   new_queries, component_type)

    if args.ground_truth:
        num_queries = args.num_queries if not args.split else args.num_queries - args.split
        write_file(f"{args.out_path}/queries/groundtruth-a{args.alpha}-n{num_queries}.bin",
                   new_ground_truth[:num_queries], np.uint32)

    if PLOT_DISTRIBUTION:
        import matplotlib.pyplot as plt

        k = np.arange(0, len(distribution))
        plt.bar(k, distribution, alpha=0.5, label='sample count')
        plt.plot(k, [round(args.num_queries * pmf(i + 1, h_num, args.alpha)) for i in k], 'k.-', alpha=0.5,
                 label='expected count')
        plt.semilogy()
        plt.grid(alpha=0.4)
        plt.legend()
        plt.title(f'Zipf, alpha={args.alpha}, size={args.num_queries}')
        plt.show()
