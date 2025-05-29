cmake_minimum_required(VERSION 3.1)
project(rdma-hnsw CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fno-omit-frame-pointer -fno-inline -fsanitize=address")

# set default build type to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -Wfatal-errors)
add_compile_options(-march=native -ffast-math -mavx2)
#add_compile_options(-DNOHUGEPAGES)
#add_compile_options(-DDEBUG_HNSW)
#add_compile_options(-g -fno-omit-frame-pointer -fno-inline)  # for profiling

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
include_directories(src)
include_directories(thirdparty)
add_subdirectory(rdma-library)
add_subdirectory(thirdparty)

file(GLOB_RECURSE SOURCES src/*.cc)

# executables
add_executable(shine ${SOURCES})
target_link_libraries(shine rdma_library)
target_compile_options(shine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

# status prints
get_target_property(compile_options shine COMPILE_OPTIONS)
message(STATUS "Compile options: ${compile_options}")