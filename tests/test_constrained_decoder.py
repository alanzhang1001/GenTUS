#!/usr/bin/env python3
"""Tests for prefix-constrained Semantic-ID decoding."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model.constrained import SemanticIDConstrainedDecoder, Trie  # noqa: E402


def test_complete_prefix_allows_eos_and_suffix_continuation() -> None:
    eos = 99
    suffix = 42
    trie = Trie()
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 3, suffix])
    processor = SemanticIDConstrainedDecoder(
        trie,
        valid_token_ids={1, 2, 3, suffix},
        eos_token_id=eos,
    )

    scores = torch.zeros((1, 128))
    masked = processor(torch.tensor([[1, 2, 3]]), scores.clone())

    assert masked[0, eos].item() == 0.0
    assert masked[0, suffix].item() == 0.0
    assert torch.isneginf(masked[0, 7])


def test_complete_leaf_allows_only_eos() -> None:
    eos = 99
    suffix = 42
    trie = Trie()
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 3, suffix])
    processor = SemanticIDConstrainedDecoder(
        trie,
        valid_token_ids={1, 2, 3, suffix},
        eos_token_id=eos,
    )

    scores = torch.zeros((1, 128))
    masked = processor(torch.tensor([[1, 2, 3, suffix]]), scores.clone())

    assert masked[0, eos].item() == 0.0
    assert torch.isneginf(masked[0, suffix])
    assert torch.isneginf(masked[0, 7])
