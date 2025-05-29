#!/usr/bin/python3

# Converts from .fvecs data format to .fbin data format
# Byte order is always little endian

import argparse
import os
import struct


def read_int(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def reformat(filename, component_size):
    filesize = os.path.getsize(filename)

    with open(filename, "rb") as f:
        dimension = read_int(f)
        f.seek(0)  # reset the file pointer

        vec_size = 4 + dimension * component_size
        num_vectors = filesize // vec_size

        print("num vectors:", num_vectors)
        print("dimension:", dimension)

        filename_l, filename_r = filename.rsplit(".")
        extension = "fbin" if filename_r[0] == "f" else "bin"

        with open(f"{filename_l}.{extension}", "wb") as new_file:
            new_file.write(struct.pack("<I", num_vectors))
            new_file.write(struct.pack("<I", dimension))

            for _ in range(num_vectors):
                assert dimension == read_int(f)  # parse dimension
                new_file.write(f.read(dimension * component_size))
                # for _ in range(dimension):
                # float_value = struct.unpack(float_format, bytes_read)[0]
                # new_file.write(read_int(f, component_size).to_bytes(component_size, byteorder="little"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path", required=True,
                        help="Path to dataset containing '.fvecs' format files")
    args = parser.parse_args()

    args.data_path = args.data_path[:-1] if args.data_path[-1] == "/" else args.data_path

    reformat(f"{args.data_path}/base.fvecs", 4)
    reformat(f"{args.data_path}/query.fvecs", 4)
    reformat(f"{args.data_path}/groundtruth.ivecs", 4)
