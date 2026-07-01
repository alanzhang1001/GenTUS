import os
import pickle
import pandas as pd


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.environ.get("GENTUS_DATA_ROOT", os.path.join(_REPO_ROOT, "datasets"))


def _resolve_existing_rel_path(*candidates: str) -> str:
    for rel_path in candidates:
        if os.path.exists(os.path.join(DATA_ROOT, rel_path)):
            return rel_path
    return candidates[0]

DATASET_PATHS = {
    "TUS_small":    ("TUS_small/tus/datalake",  "TUS_small/tus/query",  "TUS_small/tus/benchmark.pkl"),
    "TUS_large":    ("TUS_large/tusLarge/datalake", "TUS_large/tusLarge/query", "TUS_large/tusLarge/benchmark.pkl"),
    "SANTOS_small": ("santos_small/santos_benchmark/datalake", "santos_small/santos_benchmark/query", "santos_small/santos_benchmark/santos_small_benchmark_groundtruth.csv"),
    "SANTOS_large": (
        _resolve_existing_rel_path(
            "santos_large/real_data_lake_benchmark/datalake",
            "santos_large/real_tables_benchmark/datalake",
        ),
        _resolve_existing_rel_path(
            "santos_large/real_data_lake_benchmark/query",
            "santos_large/real_tables_benchmark/query",
        ),
        _resolve_existing_rel_path(
            "santos_large/real_data_lake_benchmark/real_data_lake_benchmark_groundtruth.csv",
            "santos_large/real_tables_benchmark/real_tables_benchmark_groundtruth.csv",
        ),
    ),
    "wiki_union":   ("wiki_union/datalake", "wiki_union/query", "wiki_union/benchmark.pkl"),
    "LakeBench_webtable": ("LakeBench_webtable/datalake", "LakeBench_webtable/query", "LakeBench_webtable/benchmark.pkl"),
    "LakeBench_opendata_1k": ("LakeBench_opendata_1k/datalake", "LakeBench_opendata_1k/query", "LakeBench_opendata_1k/benchmark.pkl"),
    "LakeBench_opendata_30k": ("LakeBench_opendata_30k/datalake", "LakeBench_opendata_30k/query", "LakeBench_opendata_30k/benchmark.pkl"),
}

DATASET_DEFAULT_NROWS = {
    "TUS_small": {
        "datalake": 1000,
        "query": 1000,
    },
    "TUS_large": {
        "datalake": 1000,
        "query": 1000,
    },
    "SANTOS_small": {
        "datalake": 2000,
        "query": 2000,
    },
    "SANTOS_large": {
        "datalake": None,
        "query": None,
    },
    "wiki_union": {
        "datalake": 1000,
        "query": 1000,
    },
    "LakeBench_webtable": {
        "datalake": 1000,
        "query": 1000,
    },
    "LakeBench_opendata_1k": {
        "datalake": 1000,
        "query": 1000,
    },
    "LakeBench_opendata_30k": {
        "datalake": 30000,
        "query": 30000,
    },
}

_USE_DATASET_DEFAULT_NROWS = object()


def load_ground_truth(dataset: str) -> dict:
    """Returns {query_table_filename: [candidate_filename, ...]}"""
    _, _, gt_rel = DATASET_PATHS[dataset]
    gt_path = os.path.join(DATA_ROOT, gt_rel)

    if gt_path.endswith(".csv"):
        df = pd.read_csv(gt_path)
        grouped = df.groupby("query_table")["data_lake_table"].apply(list)
        return grouped.to_dict()

    with open(gt_path, "rb") as f:
        return pickle.load(f)


def load_table(path: str, nrows=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, on_bad_lines="skip", low_memory=False)
    except Exception:
        return pd.DataFrame()


def _resolve_nrows(dataset: str, split: str, nrows):
    if nrows is not _USE_DATASET_DEFAULT_NROWS:
        return nrows
    return DATASET_DEFAULT_NROWS.get(dataset, {}).get(split, 1000)


def load_data_lake(dataset: str, nrows=_USE_DATASET_DEFAULT_NROWS) -> dict:
    """Returns {filename: DataFrame} for all tables in the datalake.

    By default, each benchmark uses a dataset-specific row cap. Pass
    nrows=None explicitly to read full CSV files.
    """
    lake_rel, _, _ = DATASET_PATHS[dataset]
    lake_dir = os.path.join(DATA_ROOT, lake_rel)
    row_limit = _resolve_nrows(dataset, "datalake", nrows)
    tables = {}
    for fname in os.listdir(lake_dir):
        if fname.endswith(".csv"):
            df = load_table(os.path.join(lake_dir, fname), nrows=row_limit)
            if not df.empty:
                tables[fname] = df
    return tables


def load_query_tables(dataset: str, nrows=_USE_DATASET_DEFAULT_NROWS) -> dict:
    """Returns {filename: DataFrame} for query tables only.

    By default, each benchmark uses a dataset-specific row cap. Pass
    nrows=None explicitly to read full CSV files.
    """
    _, query_rel, _ = DATASET_PATHS[dataset]
    query_dir = os.path.join(DATA_ROOT, query_rel)
    row_limit = _resolve_nrows(dataset, "query", nrows)
    tables = {}
    for fname in os.listdir(query_dir):
        if fname.endswith(".csv"):
            df = load_table(os.path.join(query_dir, fname), nrows=row_limit)
            if not df.empty:
                tables[fname] = df
    return tables
