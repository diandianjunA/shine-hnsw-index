#!/bin/bash

mkdir datasets && cd datasets
mkdir deep-1b && mkdir deep-100m && cd deep-1b

axel https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
axel https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
axel https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin
axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/deep-100M

mv base.1B.fbin base.fbin
mv query.public.10K.fbin query.fbin
mv deep_new_groundtruth.public.10K.bin groundtruth.bin
mv deep-100M ../deep-100m/groundtruth.bin

cd ../..
python3 slice.py -d datasets/deep-1b/base.fbin -o datasets/deep-100m/base.fbin -s 100000000
cd datasets

mkdir bigann-1b && mkdir bigann-100m && cd bigann-1b

axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/GT.public.1B.ibin
axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M

mv base.1B.u8bin base.u8bin
mv query.public.10K.u8bin query.u8bin
mv GT.public.1B.ibin groundtruth.bin
mv bigann-100M ../bigann-100m/groundtruth.bin

cd ../..
python3 slice.py -d datasets/bigann-1b/base.fbin -o datasets/bigann-100m/base.fbin -s 100000000
cd datasets

mkdir tti-1b && mkdir tti-100m && cd tti-1b

axel https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
axel https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
axel https://storage.yandexcloud.net/yandex-research/ann-datasets/t2i_new_groundtruth.public.100K.bin
axel https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/text2image-100M

mv base.1B.fbin base.fbin
mv query.public.100K.fbin query.fbin
mv t2i_new_groundtruth.public.100K.bin groundtruth.bin
mv text2image-100M ../tti-100m/groundtruth.bin

cd ../..
python3 slice.py -d datasets/tti-1b/base.fbin -o datasets/tti-100m/base.fbin -s 100000000
cd datasets

mkdir spacev-1b && mkdir spacev-100m && cd spacev-1b

wget https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin
wget https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/query.i8bin
wget https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/public_query_gt100.bin
wget https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/msspacev-gt-100M

mv spacev1b_base.i8bin base.i8bin
mv public_query_gt100.bin groundtruth.bin
mv msspacev-gt-100M ../spacev-100m/groundtruth-100m.bin

cd ../..
python3 slice.py -d datasets/spacev-1b/base.fbin -o datasets/spacev-100m/base.fbin -s 100000000
cd datasets

mkdir turing-1b && mkdir turing-100m && cd turing-1b

wget https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin
wget https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin
wget https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query_gt100.bin
wget https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-100M

mv base1b.fbin base.fbin
mv query100K.fbin query.fbin
mv query_gt100.bin groundtruth.bin
mv msturing-gt-100M ../turing-100m/groundtruth.bin

cd ../..
python3 slice.py -d datasets/turing-1b/base.fbin -o datasets/turing-100m/base.fbin -s 100000000
