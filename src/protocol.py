from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuerySplit:
    train_queries: list[str]
    test_queries: list[str]
    train_ratio: float
    seed: int

    def to_dict(self) -> dict:
        return {
            "train_queries": self.train_queries,
            "test_queries": self.test_queries,
            "train_ratio": self.train_ratio,
            "test_ratio": round(1.0 - self.train_ratio, 10),
            "seed": self.seed,
        }


def clean_ground_truth(
    ground_truth: dict[str, list[str]],
    candidate_names: set[str],
    query_names: set[str] | None = None,
) -> dict[str, list[str]]:
    """Remove invalid positives before any train/eval filtering.

    Public GenTUS evaluates against a candidate pool defined by the datalake
    directory. Ground-truth positives that are not present in that pool are
    dataset inconsistencies, so they are fixed once here rather than masked at
    evaluation time.
    """
    cleaned: dict[str, list[str]] = {}
    for query_name, candidates in ground_truth.items():
        if query_names is not None and query_name not in query_names:
            continue
        valid = []
        seen = set()
        for candidate in candidates:
            if candidate == query_name or candidate not in candidate_names or candidate in seen:
                continue
            valid.append(candidate)
            seen.add(candidate)
        if valid:
            cleaned[query_name] = valid
    return cleaned


def make_query_split(
    query_names: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> QuerySplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")
    names = sorted(set(query_names))
    if len(names) < 2:
        raise ValueError("At least two query tables with valid ground truth are required")
    rng = random.Random(seed)
    rng.shuffle(names)
    train_size = int(round(len(names) * train_ratio))
    train_size = min(max(train_size, 1), len(names) - 1)
    train = sorted(names[:train_size])
    test = sorted(names[train_size:])
    return QuerySplit(train_queries=train, test_queries=test, train_ratio=train_ratio, seed=seed)


def filter_ground_truth_by_queries(
    ground_truth: dict[str, list[str]],
    query_names: set[str],
) -> dict[str, list[str]]:
    return {name: list(ground_truth[name]) for name in sorted(query_names) if name in ground_truth}


def write_split(path: str | Path, split: QuerySplit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")


def read_split(path: str | Path) -> QuerySplit:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return QuerySplit(
        train_queries=[str(name) for name in data["train_queries"]],
        test_queries=[str(name) for name in data["test_queries"]],
        train_ratio=float(data["train_ratio"]),
        seed=int(data["seed"]),
    )
