import argparse
import csv
import mongodb
import os
from pymongo import MongoClient
from datasets import Datasets

MONGO_URI = mongodb.connection_string
DATABASE_NAME = "experiments"
COLLECTION_NAME = "benchmarks"


def fetch_data(dataset: Datasets):
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    fixed_cache_size_ratio = 5
    zipf_parameters = ["0.5", "0.75", "1.0", "1.25", "1.5"]

    baseline_label = "v15-csize-skew-baseline"
    cache_label = "v15-csize-skew-+cache"
    routing_label = "v15-csize-skew-+adaptive-routing"

    cursor = collection.find({
        "meta.compute_threads": 160,
        "meta.dataset": dataset.value.name,
        "meta.label": {"$in": [baseline_label, cache_label, routing_label]},
        "meta.zipf_parameter": {"$in": zipf_parameters},
        "queries.queries_per_sec": {"$exists": True},
        "cache.cache_size_ratio": {"$in": [fixed_cache_size_ratio, None]},
    })

    data = {}

    for doc in cursor:
        meta = doc.get("meta", {})
        zipf_parameter = meta.get("zipf_parameter")
        label = meta.get("label")
        queries = doc.get("queries", {})
        qps = queries.get("queries_per_sec")

        if zipf_parameter is None or label is None or qps is None:
            continue

        if zipf_parameter not in data:
            data[zipf_parameter] = {"baseline-tp": "", "cache-tp": "", "routing-tp": ""}

        if label == baseline_label:
            data[zipf_parameter]["baseline-tp"] = qps
        elif label == cache_label:
            data[zipf_parameter]["cache-tp"] = qps
        elif label == routing_label:
            data[zipf_parameter]["routing-tp"] = qps

    client.close()
    return data


def write_csv(data, filename):
    header = ["zipf", "baseline-tp", "cache-tp", "routing-tp"]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for zipf_parameter, values in sorted(data.items()):
            row = {
                "zipf": zipf_parameter,
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
        data = fetch_data(dataset)
        write_csv(data, f"{args.out_dir}/skew-{dataset.value.name}.csv")
