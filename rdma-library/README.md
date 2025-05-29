# RDMA Library

"High-level" library to connect machines, connect queue pairs, register memory regions, post RDMA verbs, etc.
The goal of this library is to conveniently wrap
the [ibverbs library](https://github.com/linux-rdma/rdma-core/tree/master/libibverbs).

[TODO: public libary interface and namespaces...]

## Required C++ Libraries

* ibverbs
* Boost (for CLI parsing)
* pthreads (for multithreading)
* oneTBB (for concurrent data structures)

## Using RDMA Library in Another Project

First add this repository as a submodule:
```
git submodule add git@frosch.cosy.sbg.ac.at:mwidmoser/rdma-library.git
```

Then in the main `CMakeLists.txt` file of the project, add
```
add_subdirectory(rdma-library)
```

Finally, link the executable with the library in `CMakeLists.txt`, e.g.,
```
add_executable(ping_pong src/ping_pong.cc)
target_link_libraries(ping_pong rdma_library)
```
