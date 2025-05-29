#!/bin/sh

if [ "$#" -ne 2 ]
then
  echo "Usage: $0 <node> <build | release | debug>" >&2
  exit 1
fi

PROJECT="rdma-hnsw"
COMPILER="g++"

NODE=$1
LOCAL_PATH="/Users/b1048446/Development/${PROJECT}"
REMOTE_PATH="/root/mw"

if [ $2 == "build" ]
then
  rsync -a --exclude .git --exclude .idea --exclude cmake-build-* $LOCAL_PATH $NODE:$REMOTE_PATH &&
  ssh $NODE "cd ${REMOTE_PATH}/${PROJECT} && cd build && make -j && exit"
elif [ $2 == "release" ]
then
  ssh $NODE "cd ${REMOTE_PATH} && rm -rf ${PROJECT}"
  rsync -a --exclude .git --exclude .idea --exclude cmake-build-* $LOCAL_PATH $NODE:$REMOTE_PATH &&
  ssh $NODE "cd ${REMOTE_PATH}/${PROJECT} && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=${COMPILER} .. && make -j && exit"
elif [ $2 == "debug" ]
then
  ssh $NODE "cd ${REMOTE_PATH} && rm -rf ${PROJECT}"
  rsync -a --exclude .git --exclude .idea --exclude cmake-build-* $LOCAL_PATH $NODE:$REMOTE_PATH &&
  ssh $NODE "cd ${REMOTE_PATH}/${PROJECT} && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=${COMPILER} .. && make -j && exit"
else
  echo "invalid argument"
fi
