#!/usr/bin/python3

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input-file", required=True)
    parser.add_argument("-o", "--output-file", required=True)
    parser.add_argument("-s", "--slice", required=True, type=int)
    parser.add_argument("-k", "--skip", type=int, default=0, help="Skip k many vectors (e.g., for queries)")

    args = parser.parse_args()

    if args.input_file.endswith("u8bin") or args.input_file.endswith("i8bin"):
        component_size_in_file = 1
    elif args.input_file.endswith("fbin"):
        component_size_in_file = 4
    else:
        sys.exit("invalid file extension")

    with open(args.input_file, "rb") as file_a, open(args.output_file, "wb") as file_b:
        # read header
        n = int.from_bytes(file_a.read(4), byteorder="little")
        dim = int.from_bytes(file_a.read(4), byteorder="little")

        # write new header
        file_b.write(args.slice.to_bytes(4, byteorder="little"))
        file_b.write(dim.to_bytes(4, byteorder="little"))

        if args.skip > 0:
            file_a.seek(args.skip * dim * component_size_in_file,
                        1)  # 1 means the reference point is the current file position

        # read and write components
        to_read = args.slice * dim * component_size_in_file
        max_chunk_size = 10 ** 9  # 1 GB

        while to_read > 0:
            chunk_size = max_chunk_size if to_read > max_chunk_size else to_read
            data = file_a.read(chunk_size)
            file_b.write(data)
            to_read -= chunk_size

        assert to_read == 0
