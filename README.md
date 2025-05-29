# SHINE: A Scalable HNSW Index in Disaggregated Memory

Implementation of a distributed HNSW index for memory disaggregation. 
This is the source code of the paper "SHINE: A Scalable HNSW Index in Disaggregated Memory".

## Setup

### C++ Libraries and Unix Packages

The following C++ libraries and Unix packages are required to compile the code.
Note that `ibverbs` (the RDMA library) is Linux-only. 
The code also compiles without InfiniBand network cards.

* [ibverbs](https://github.com/linux-rdma/rdma-core/tree/master)
* [boost](https://www.boost.org/doc/libs/1_83_0/doc/html/program_options.html) (to support `boost::program_options` for
  CLI parsing)
* pthreads (for multithreading)
* [oneTBB](https://github.com/oneapi-src/oneTBB) (for concurrent data structures)
* a C++ compiler that supports C++20 (we have used `g++-12`)
* cmake
* numactl
* vmtouch (to map index files into main memory)
* axel (a download accelerator for the datasets)

For instance, to install the requirements on Debian, run the following command:
```
apt-get -y install g++ libboost-all-dev libibverbs1 libibverbs-dev numactl cmake libtbb-dev git python3-venv vmtouch axel
```

### Cluster Nodes Configuration

Adjust the IP addresses of the cluster nodes accordingly in `rdma-library/library/utils.cc`:
https://frosch.cosy.sbg.ac.at/mwidmoser/shine-hnsw-index/-/blob/main/rdma-library/library/utils.cc?ref_type=heads#L14-L23

### Compilation

After cloning the repository and installing the requirements, the code must be compiled on all cluster nodes:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Download the Data

First, install all Python requirements:
```
cd scripts
python3 -m pip install -r requirements.txt
```

Then, run the following script to download the data. 
This may take a while, we recommend to run the script within a `tmux` session.
Also make sure that `axel` (a download accelerator) is installed.
```
cd data
bash download.sh
```

Finally, create the queries (adjust the `DATASET_PATH` in `create_queries.py`):
```
python3 create_queries.py
```

Now all the data is available in `data/datasets`, move them to a location where all cluster nodes have access to (e.g., to an NFS).
Then, adjust the path in `config.py`.

## Run the Experiments

* TODO
