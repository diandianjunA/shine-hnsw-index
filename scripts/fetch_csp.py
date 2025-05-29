import mongodb
from pymongo import MongoClient
from datasets import Datasets

MONGO_URI = mongodb.connection_string
DATABASE_NAME = "experiments"
COLLECTION_NAME = "benchmarks"


def fetch_data(dataset: Datasets, zipf_parameter):
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    cache_label = "v15-+cache"
    routing_label = "v15-+adaptive-routing"
    cache_size_ratio = 5

    csp_cache_label = "v15-csp-inc-+cache"  # TODO: now part of exp_csp_increase experiment with 25% ratio

    cursor = collection.find({
        "meta.compute_threads": 160,
        "meta.dataset": dataset.value.name,
        "meta.label": {"$in": [cache_label, routing_label]},
        "meta.zipf_parameter": zipf_parameter,
        "queries.queries_per_sec": {"$exists": True},
        "cache.hit_rate": {"$exists": True},
        "cache.cache_size_ratio": cache_size_ratio
    })

    cursor_csp = collection.find({
        "meta.dataset": dataset.value.name,
        "meta.label": csp_cache_label,
        "meta.zipf_parameter": zipf_parameter,
        "cache.cache_size_ratio": cache_size_ratio * 5,  # 5 CNs
        "cache.hit_rate": {"$exists": True},
    })

    data = {}

    data["+cache"] = {"CHR": "", "CSP": "", "TP": ""}
    data["+routing"] = {"CHR": "", "CSP": "", "TP": ""}

    for doc in cursor:
        meta = doc.get("meta", {})
        label = meta.get("label")
        queries = doc.get("queries", {})
        qps = queries.get("queries_per_sec")
        cache = doc.get("cache", {})
        chr = cache.get("hit_rate")

        if chr is None or label is None or qps is None:
            continue

        if label == cache_label:
            data["+cache"]["CHR"] = chr
            data["+cache"]["TP"] = qps
        elif label == routing_label:
            data["+routing"]["CHR"] = chr
            data["+routing"]["TP"] = qps

    # compute CSP
    for doc in cursor_csp:
        cache = doc.get("cache", {})
        chr = cache.get("hit_rate")

        if chr is None:
            continue

        data["+cache"]["CSP"] = 1.0 - float(data["+cache"]["CHR"]) / chr
        data["+routing"]["CSP"] = 1.0 - float(data["+routing"]["CHR"]) / chr

    client.close()
    return data


if __name__ == "__main__":
    for zipf_parameter in ["0.0", "1.0"]:
        all_data = {}

        cache_row = ""
        routing_row = ""
        for dataset in [Datasets.BIGANN_100M, Datasets.DEEP_100M, Datasets.SPACEV_100M, Datasets.TTI_100M,
                        Datasets.TURING_100M]:
            data = fetch_data(dataset, zipf_parameter)
            all_data[dataset.value.name] = data
            print(f"{dataset.value.name}: zipf parameter: {zipf_parameter}")
            print(data)
            cache_row += f" & {round(100.0 * data['+cache']['CHR'])}\% & {round(100.0 * data['+cache']['CSP'])}\% & \\SI{{{data['+cache']['TP']}}}{{}}"
            routing_row += f" & {round(100.0 * data['+routing']['CHR'])}\% & {round(100.0 * data['+routing']['CSP'])}\% & \\SI{{{data['+routing']['TP']}}}{{}}"

        print(f"\nzipf parameter {zipf_parameter}:\n")
        print(cache_row + " \\\\")
        print(routing_row + " \\\\")
