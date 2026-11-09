#!/usr/bin/env python3
"""Tests for evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluate import _average_precision  # noqa: E402


def test_average_precision_divides_by_cutoff_when_positives_exceed_k() -> None:
    ap, hits, _precision_sum, _first_gt_rank = _average_precision(
        ["a.csv", "b.csv"],
        {"a.csv", "b.csv", "c.csv", "d.csv"},
        k=2,
    )

    assert hits == 2
    assert ap == 1.0


def test_average_precision_divides_by_positive_count_below_cutoff() -> None:
    ap, hits, _, _ = _average_precision(["a", "x", "b"], {"a", "b"}, 5)
    assert hits == 2
    assert abs(ap - (1 + 2 / 3) / 2) < 1e-12


def test_average_precision_does_not_count_duplicate_hits() -> None:
    ap, hits, _, _ = _average_precision(["a", "a", "b"], {"a", "b", "c"}, 3)
    assert hits == 2
    assert abs(ap - (1 + 2 / 3) / 3) < 1e-12


def test_average_precision_respects_cutoff_and_empty_results() -> None:
    assert _average_precision(["x", "a"], {"a"}, 1)[0] == 0.0
    assert _average_precision([], {"a"}, 3)[0] == 0.0
    assert _average_precision(["a"], set(), 3)[0] == 0.0
