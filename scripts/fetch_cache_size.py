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

    cache_size_ratios = [2, 4, 6, 8, 10]
    baseline_label = "v15-csize-skew-baseline"
    cache_label = "v15-csize-skew-+cache"
    routing_label = "v15-csize-skew-+adaptive-routing"

    cursor = collection.find({
        "meta.compute_threads": 160,
        "meta.dataset": dataset.value.name,
        "meta.label": {"$in": [baseline_label, cache_label, routing_label]},
        "meta.zipf_parameter": zipf_parameter,
        "queries.queries_per_sec": {"$exists": True},
        "cache.cache_size_ratio": {"$in": cache_size_ratios + [None]},
    })

    data = {}

    for doc in cursor:
        meta = doc.get("meta", {})
        label = meta.get("label")
        queries = doc.get("queries", {})
        qps = queries.get("queries_per_sec")
        cache = doc.get("cache", {})
        cache_size_ratio = cache.get("cache_size_ratio")

        if label is None or qps is None:
            continue

        if label == baseline_label:
            for csr in cache_size_ratios:
                if csr not in data:
                    data[csr] = {"baseline-tp": "", "cache-tp": "", "routing-tp": ""}
                data[csr]["baseline-tp"] = qps
        else:
            if cache_size_ratio not in data:
                data[cache_size_ratio] = {"baseline-tp": "", "cache-tp": "", "routing-tp": ""}

            if label == cache_label:
                data[cache_size_ratio]["cache-tp"] = qps
            elif label == routing_label:
                data[cache_size_ratio]["routing-tp"] = qps

    client.close()
    return data


def write_csv(data, filename):
    header = ["cache-size-ratio", "baseline-tp", "cache-tp", "routing-tp"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for cache_size_ratio, values in sorted(data.items()):
            row = {
                "cache-size-ratio": cache_size_ratio,
                "baseline-tp": values.get("baseline-tp", ""),
                "cache-tp": values.get("cache-tp", ""),
                "routing-tp": values.get("routing-tp", "")
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
            write_csv(data, f"{args.out_dir}/cache-size-a{zipf_parameter}-{dataset.value.name}.csv")
