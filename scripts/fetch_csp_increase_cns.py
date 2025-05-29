import argparse
import csv
import mongodb
import os
from pymongo import MongoClient
from datasets import Datasets

MONGO_URI = mongodb.connection_string
DATABASE_NAME = "experiments"
COLLECTION_NAME = "benchmarks"


def fetch_data(dataset: Datasets, zipf_parameter):
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    cache_label = "v15-csp-inc-+cache"
    routing_label = "v15-csp-inc-+routing"
    fixed_cache_size_ratio = 5

    cursor = collection.find({
        "meta.dataset": dataset.value.name,
        "meta.label": {"$in": [cache_label, routing_label]},
        "meta.zipf_parameter": zipf_parameter,
        "queries.queries_per_sec": {"$exists": True},
        "cache.hit_rate": {"$exists": True},
        "cache.cache_size_ratio": fixed_cache_size_ratio
    })

    cursor_csp = collection.find({
        "meta.compute_nodes": 1,
        "meta.dataset": dataset.value.name,
        "meta.label": cache_label,
        "meta.zipf_parameter": zipf_parameter,
        "cache.hit_rate": {"$exists": True},
    })

    chr_best = {}

    # store CHR_best to compute CSP
    for doc in cursor_csp:
        cache = doc.get("cache", {})
        chr = cache.get("hit_rate")
        cache_size_ratio = cache.get("cache_size_ratio")

        if chr is None or cache_size_ratio is None:
            continue

        key = int(cache_size_ratio / fixed_cache_size_ratio)
        chr_best[key] = chr

    data = {}

    for doc in cursor:
        meta = doc.get("meta", {})
        label = meta.get("label")
        num_compute_nodes = meta.get("compute_nodes")
        cache = doc.get("cache", {})
        chr = cache.get("hit_rate")

        if chr is None or label is None:
            continue

        if num_compute_nodes not in data:
            data[num_compute_nodes] = {"+cache-chr": "", "+cache-csp": "", "+routing-chr": "", "+routing-csp": ""}

        if label == cache_label:
            data[num_compute_nodes]["+cache-chr"] = chr
            data[num_compute_nodes]["+cache-csp"] = 1.0 - float(chr) / chr_best[num_compute_nodes]
        elif label == routing_label:
            data[num_compute_nodes]["+routing-chr"] = chr
            data[num_compute_nodes]["+routing-csp"] = 1.0 - float(chr) / chr_best[num_compute_nodes]

    client.close()
    return data


def write_csv(data, filename):
    header = ["cns", "+cache-chr", "+cache-csp", "+routing-chr", "+routing-csp"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for compute_nodes, values in sorted(data.items()):
            row = {
                "cns": compute_nodes,
                "+cache-chr": values.get("+cache-chr", "") * 100,
                "+cache-csp": values.get("+cache-csp", "") * 100,
                "+routing-chr": values.get("+routing-chr", "") * 100,
                "+routing-csp": values.get("+routing-csp", "") * 100,
            }
            writer.writerow(row)
    print(f"CSV file '{filename}' generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", help="Output directory for csv files)", required=True)
    args = parser.parse_args()

    if args.out_dir[-1] == "/":
        args.out_dir = args.out_dir[:-1]

    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in [Datasets.DEEP_100M, Datasets.TURING_100M, Datasets.SPACEV_100M, Datasets.TTI_100M,
                    Datasets.BIGANN_100M]:
        for zipf_parameter in ["0.0", "1.0"]:
            data = fetch_data(dataset, zipf_parameter)
            write_csv(data, f"{args.out_dir}/csp-inc-cns-a{zipf_parameter}-{dataset.value.name}.csv")
