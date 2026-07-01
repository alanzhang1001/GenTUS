#!/usr/bin/env python3
"""Tests for evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluate import _average_precision  # noqa: E402


def test_average_precision_divides_by_all_ground_truth_tables() -> None:
    ap, hits, _precision_sum, _first_gt_rank = _average_precision(
        ["a.csv", "b.csv"],
        {"a.csv", "b.csv", "c.csv", "d.csv"},
        k=2,
    )

    assert hits == 2
    assert ap == 0.5
