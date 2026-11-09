#!/usr/bin/env python3
"""Tests for table serialization."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data.serializer import serialize  # noqa: E402


def test_serialize_defaults_cover_all_columns_without_names() -> None:
    df = pd.DataFrame(
        {
            "a": ["v1", "v2", "v3"],
            "b": ["x1", "x2", "x3"],
        }
    )

    text = serialize(df)

    assert "[COL] a" not in text
    assert "[COL] b" not in text
    assert text.count("[COL]") == 2
    assert text.count("[VAL]") == 2


def test_serialize_honors_explicit_token_budget() -> None:
    df = pd.DataFrame(
        {
            "a": ["alpha beta gamma"],
            "b": ["delta epsilon zeta"],
        }
    )

    text = serialize(df, token_budget=5)

    assert len(text.split()) <= 5


def test_serialize_keeps_values_with_their_column() -> None:
    df = pd.DataFrame({"a": ["a1", "a2"], "b": ["b1", "b2"]})
    assert serialize(df) == "[COL] [VAL] a1 | a2 [COL] [VAL] b1 | b2"
    assert serialize(df, include_column_names=True) == (
        "[COL] a [VAL] a1 | a2 [COL] b [VAL] b1 | b2"
    )


def test_serialize_budget_preserves_column_boundaries() -> None:
    df = pd.DataFrame({"a": ["a1", "a2"], "b": ["b1", "b2"]})
    assert serialize(df, token_budget=6) == "[COL] [VAL] a1 [COL] [VAL] b1"
