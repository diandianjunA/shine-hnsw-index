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

    baseline_label = "v15-baseline"
    cache_label = "v15-+cache"
    routing_label = "v15-+adaptive-routing"

    cursor = collection.find({
        "meta.compute_threads": {"$exists": True},
        "meta.dataset": dataset.value.name,
        "meta.label": {"$in": [baseline_label, cache_label, routing_label]},
        "meta.zipf_parameter": zipf_parameter,
        "queries.queries_per_sec": {"$exists": True},
    })

    data = {}

    for doc in cursor:
        meta = doc.get("meta", {})
        compute_threads = meta.get("compute_threads")
        label = meta.get("label")
        queries = doc.get("queries", {})
        qps = queries.get("queries_per_sec")

        if compute_threads is None or label is None or qps is None:
            continue

        if compute_threads not in data:
            data[compute_threads] = {"baseline-tp": "", "cache-tp": "", "routing-tp": ""}

        if label == baseline_label:
            data[compute_threads]["baseline-tp"] = qps
        elif label == cache_label:
            data[compute_threads]["cache-tp"] = qps
        elif label == routing_label:
            data[compute_threads]["routing-tp"] = qps

    client.close()
    return data


def write_csv(data, filename):
    header = ["threads", "baseline-tp", "cache-tp", "routing-tp"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for compute_threads, values in sorted(data.items()):
            row = {
                "threads": compute_threads,
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
            write_csv(data, f"{args.out_dir}/scalability-a{zipf_parameter}-{dataset.value.name}.csv")
