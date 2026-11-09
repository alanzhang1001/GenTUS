import hashlib
import random

import pandas as pd


def _clean_values(series: pd.Series) -> list[str]:
    values = []
    seen = set()
    for value in series.dropna().astype(str):
        cleaned = " ".join(value.split()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        values.append(cleaned)
    return values


def serialize(
    df: pd.DataFrame,
    max_cols: int | None = None,
    max_rows: int = 24,
    include_column_names: bool = False,
    token_budget: int = 256,
    policy: str = "column_budget",
) -> str:
    """Convert a DataFrame to a text sequence for T5 input.

    The default protocol uses no column names, all columns, up to 24 values per
    column, and a 256-token text budget.
    """
    if policy == "priority":
        return serialize_priority(df, max_cols or len(df.columns), max_rows, include_column_names)
    if policy != "column_budget":
        raise ValueError(f"Unknown serialization policy: {policy}")
    if df.empty or max_rows <= 0 or token_budget <= 0:
        return ""

    selected = list(df.columns)
    if max_cols is not None and max_cols > 0:
        selected = selected[:max_cols]
    if not selected:
        return ""

    column_values = {col: _clean_values(df[col])[:max_rows] for col in selected}
    column_parts: dict[str, list[str]] = {}
    token_count = 0
    active_columns: list[str] = []
    value_counts: dict[str, int] = {}

    for col in selected:
        if include_column_names:
            prefix_tokens = f"[COL] {col} [VAL]".split()
        else:
            prefix_tokens = "[COL] [VAL]".split()
        if token_count + len(prefix_tokens) > token_budget:
            break
        column_parts[col] = list(prefix_tokens)
        token_count += len(prefix_tokens)
        active_columns.append(col)
        value_counts[col] = 0

    value_pos = 0
    while active_columns:
        added = False
        next_active = []
        for col in active_columns:
            values = column_values[col]
            if value_pos >= len(values):
                continue
            value = values[value_pos]
            value_tokens = value.split()
            needed = len(value_tokens) + (1 if value_counts[col] > 0 else 0)
            if token_count + needed > token_budget:
                continue
            if value_counts[col] > 0:
                column_parts[col].append("|")
                token_count += 1
            column_parts[col].extend(value_tokens)
            token_count += len(value_tokens)
            value_counts[col] += 1
            added = True
            if value_pos + 1 < len(values):
                next_active.append(col)
        if not added:
            break
        active_columns = next_active
        value_pos += 1

    return " ".join(token for parts in column_parts.values() for token in parts)


def _stable_rng(df: pd.DataFrame) -> random.Random:
    seed_src = f"{df.shape}|{'|'.join(map(str, df.columns))}"
    seed = int(hashlib.md5(seed_src.encode('utf-8')).hexdigest()[:8], 16)
    return random.Random(seed)


def serialize_priority(df: pd.DataFrame, max_cols: int=8, max_rows: int=8, include_column_names: bool=True) -> str:
    if df.empty or max_cols <= 0 or max_rows <= 0:
        return ''
    rng = _stable_rng(df)
    stats = []
    for (idx, col) in enumerate(df.columns):
        non_null = int(df[col].notna().sum())
        nunique = int(df[col].dropna().nunique())
        stats.append({'name': col, 'index': idx, 'non_null': non_null, 'nunique': nunique})
    selected = []
    selected_names = set()
    high_cardinality = sorted([s for s in stats if s['nunique'] > 1], key=lambda s: (-s['nunique'], -s['non_null'], s['index']))
    for stat in high_cardinality:
        if len(selected) >= max_cols:
            break
        selected.append(stat['name'])
        selected_names.add(stat['name'])
    dense_columns = sorted([s for s in stats if s['name'] not in selected_names and s['non_null'] > 0], key=lambda s: (-s['non_null'], -s['nunique'], s['index']))
    for stat in dense_columns:
        if len(selected) >= max_cols:
            break
        selected.append(stat['name'])
        selected_names.add(stat['name'])
    remaining = [s['name'] for s in stats if s['name'] not in selected_names]
    rng.shuffle(remaining)
    for col in remaining:
        if len(selected) >= max_cols:
            break
        selected.append(col)
    parts = []
    for col in selected:
        vals = _clean_values(df[col])[:max_rows]
        if include_column_names:
            parts.append(f"[COL] {col} [VAL] {' | '.join(vals)}")
        else:
            parts.append(f"[COL] [VAL] {' | '.join(vals)}")
    return ' '.join(parts)
