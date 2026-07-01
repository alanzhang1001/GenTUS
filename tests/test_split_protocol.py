#!/usr/bin/env python3
"""Tests for the public split-safe GenTUS protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from protocol import clean_ground_truth, filter_ground_truth_by_queries, make_query_split  # noqa: E402
from train import build_negative_lookup, build_records, build_unionable_pairs  # noqa: E402


class DummySIDManager:
    collision_bases = set()

    def __init__(self, table_to_sid: dict[str, tuple[int, ...]]) -> None:
        self.table_to_sid = table_to_sid

    def table_to_id(self, name: str):
        return self.table_to_sid.get(name)

    def _base_sid(self, sid: tuple[int, ...]):
        return sid


def test_make_query_split_is_seeded_and_disjoint() -> None:
    names = [f"q{i}.csv" for i in range(10)]

    split_a = make_query_split(names, train_ratio=0.8, seed=42)
    split_b = make_query_split(names, train_ratio=0.8, seed=42)

    assert split_a == split_b
    assert len(split_a.train_queries) == 8
    assert len(split_a.test_queries) == 2
    assert set(split_a.train_queries).isdisjoint(split_a.test_queries)
    assert set(split_a.train_queries) | set(split_a.test_queries) == set(names)


def test_clean_ground_truth_removes_missing_candidates_before_eval() -> None:
    ground_truth = {
        "q.csv": ["present.csv", "missing.csv", "present.csv", "q.csv"],
        "not_a_query.csv": ["present.csv"],
    }

    cleaned = clean_ground_truth(
        ground_truth,
        candidate_names={"present.csv"},
        query_names={"q.csv"},
    )

    assert cleaned == {"q.csv": ["present.csv"]}


def test_union_loss_uses_filtered_train_ground_truth_only() -> None:
    table_names = ["train_q.csv", "train_pos.csv", "test_q.csv", "test_pos.csv"]
    ground_truth = {
        "train_q.csv": ["train_pos.csv", "test_q.csv"],
        "test_q.csv": ["test_pos.csv", "train_pos.csv"],
    }
    train_gt = filter_ground_truth_by_queries(ground_truth, {"train_q.csv"})

    pairs = build_unionable_pairs(table_names, train_gt)

    assert pairs == [(0, 1), (0, 2)]
    assert (2, 3) not in pairs


def test_candidate_candidate_pairs_require_explicit_opt_in() -> None:
    table_names = ["train_q.csv", "train_pos.csv", "test_q.csv"]
    train_gt = {"train_q.csv": ["train_pos.csv", "test_q.csv"]}

    pairs = build_unionable_pairs(table_names, train_gt, add_candidate_candidate=True)

    assert pairs == [(0, 1), (0, 2), (1, 2)]


def test_negative_lookup_does_not_use_test_ground_truth() -> None:
    train_gt = {"train_q.csv": ["train_pos.csv"]}
    table = pd.DataFrame({"col": ["same"]})
    query_tables = {"train_q.csv": table}
    datalake_tables = {
        "train_pos.csv": table,
        "test_pos.csv": table,
        "negative.csv": table,
    }

    negatives = build_negative_lookup(
        train_gt,
        query_tables,
        datalake_tables,
        max_negatives_per_query=3,
    )

    assert "train_pos.csv" not in negatives["train_q.csv"]
    assert "test_pos.csv" in negatives["train_q.csv"]


def test_indexing_records_use_train_positive_candidates_only() -> None:
    table = pd.DataFrame({"col": ["value"]})
    records, stats = build_records(
        {"train_q.csv": ["train_pos.csv"]},
        {"train_q.csv": table},
        DummySIDManager({
            "train_pos.csv": (1,),
            "unused.csv": (2,),
        }),
        n_aug=0,
        datalake_tables={
            "train_pos.csv": table,
            "unused.csv": table,
        },
        use_ranking_loss=False,
        include_indexing_records=True,
        indexing_table_names=["train_pos.csv"],
    )

    target_texts = [record["target_text"] for record in records]
    assert target_texts.count("1") == 2
    assert "2" not in target_texts
    assert stats["indexing_records"] == 1
