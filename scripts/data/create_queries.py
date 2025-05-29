from subprocess import run

DATASETS_PATH = "/mnt/dbgroup-share/mwidmoser/hnsw-data"
datasets = [("deep", "fbin"), ("turing", "fbin"), ("spacev", "i8bin"), ("bigann", "u8bin"), ("tti", "fbin")]
zipf_parameters = ["0.0", "0.5", "0.75", "1.0", "1.25", "1.5"]

if __name__ == "__main__":
    try:
        for (dataset, extension) in datasets:
            print(f"creating queries for {dataset} dataset ({extension})...")
            run(["python3", "slice.py", "-d", f"{DATASETS_PATH}/{dataset}-1b/base.{extension}", "-o",
                 f"{DATASETS_PATH}/{dataset}-100m/query-500k-slice.{extension}", "-s", "500000", "-k", "100000000"])

            for zipf in zipf_parameters:
                run(["python3", "skew.py", "-q", f"{DATASETS_PATH}/{dataset}-100m/query-500k-slice.{extension}", "-o",
                     f"{DATASETS_PATH}/{dataset}-100m/", "-a", zipf, "-n", "500000", "-s", "100000"])

                run(["mv", f"{DATASETS_PATH}/{dataset}-100m/queries/query-a{zipf}-n400000.{extension}",
                     f"{DATASETS_PATH}/{dataset}-100m/queries/query-a{zipf}-500k.{extension}"])
                run(["mv", f"{DATASETS_PATH}/{dataset}-100m/queries/warmup-a{zipf}-n100000.{extension}",
                     f"{DATASETS_PATH}/{dataset}-100m/queries/warmup-a{zipf}-500k.{extension}"])

    except Exception as e:
        print(f"ERROR: {e}")
