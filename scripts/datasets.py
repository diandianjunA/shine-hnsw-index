from enum import Enum


class Dataset:
    def __init__(self, name, ef_search, ip_dist):
        self.name = name
        self.ef_search = ef_search
        self.ip_dist = ip_dist


class Datasets(Enum):
    # ef_search is set such that we reach 95% recall with the original queries
    SIFT_1M = Dataset("sift-1m", 100, False)
    TURING_1M = Dataset("turing-1m", 100, False)
    TTI_10M = Dataset("tti-10m", 250, True)
    DEEP_100M = Dataset("deep-100m", 100, False)
    TURING_100M = Dataset("turing-100m", 150, False)
    SPACEV_100M = Dataset("spacev-100m", 100, False)
    BIGANN_100M = Dataset("bigann-100m", 80, False)
    TTI_100M = Dataset("tti-100m", 250, True)
