# Datasets

* Requirements: `apt install axel`

## Data Format

All datasets are in the common binary format as provided by https://big-ann-benchmarks.com/neurips21.html that starts
with 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32) followed by num_pts x num_dimensions x
sizeof(type) bytes of data stored one vector after another.
Data files will have suffixes .fbin, .u8bin, and .i8bin to represent float32, uint8 and int8 type data.

* `reformat_data.py` can be used to reformat data stored in the format (legacy format from other datasets):
    ```
    num_pts x [ num_dimensions(uint32) | num_dimensions x sizeof(type) ]
    ```
