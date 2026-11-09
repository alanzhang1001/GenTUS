import math
from typing import List, Sequence, Tuple

import pandas as pd


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return ''
    return text


def _column_priority(df: pd.DataFrame) -> List[str]:
    ranked = []
    for (idx, col) in enumerate(df.columns):
        series = df[col].map(_normalize_text)
        non_empty = series[series != '']
        non_empty_count = int(non_empty.shape[0])
        distinct_count = int(non_empty.nunique())
        distinct_ratio = distinct_count / max(non_empty_count, 1)
        is_primary_key = int(non_empty_count > 0 and distinct_count == non_empty_count)
        ranked.append((-is_primary_key, -distinct_ratio, -distinct_count, -non_empty_count, idx, col))
    ranked.sort()
    return [col for (*_, col) in ranked]


def _iter_unique_values(series: pd.Series) -> List[str]:
    seen = set()
    values = []
    for value in series.map(_normalize_text):
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _select_columns(df: pd.DataFrame, tokenizer, max_tokens: int, include_column_names: bool=True) -> List[Tuple[str, List[int]]]:
    selected = []
    remaining_budget = max_tokens - 1
    for col in _column_priority(df):
        if include_column_names:
            name_ids = tokenizer.encode(_normalize_text(col) or str(col), add_special_tokens=False, truncation=True, max_length=8)
            if not name_ids:
                name_ids = tokenizer.encode('column', add_special_tokens=False)
        else:
            name_ids = []
        min_cost = len(name_ids) + 1
        if min_cost > remaining_budget:
            continue
        selected.append((col, name_ids))
        remaining_budget -= min_cost
        if remaining_budget <= 0:
            break
    return selected


def _encode_value_stream(values: Sequence[str], tokenizer, sep_token_id: int, budget: int) -> List[int]:
    if budget <= 0:
        return []
    encoded = []
    for value in values:
        value_ids = tokenizer.encode(value, add_special_tokens=False)
        if not value_ids:
            continue
        needed = len(value_ids) + (1 if encoded else 0)
        if needed > budget:
            remaining = budget - (1 if encoded else 0)
            if remaining <= 0:
                break
            value_ids = value_ids[:remaining]
            needed = len(value_ids) + (1 if encoded else 0)
        if encoded:
            encoded.append(sep_token_id)
        encoded.extend(value_ids)
        budget -= needed
        if budget <= 0:
            break
    return encoded


def serialize_table(df: pd.DataFrame, tokenizer, max_tokens: int, include_column_names: bool=True) -> Tuple[List[int], List[int]]:
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    if df.empty or len(df.columns) == 0:
        return ([cls_token_id], [0])
    selected_columns = _select_columns(df, tokenizer, max_tokens, include_column_names=include_column_names)
    if not selected_columns:
        return ([cls_token_id], [0])
    base_cost = sum((len(name_ids) + 1 for (_, name_ids) in selected_columns))
    value_budget = max(0, max_tokens - 1 - base_cost)
    input_ids = [cls_token_id]
    column_positions = []
    remaining_name_cost = base_cost
    remaining_value_budget = value_budget
    for (idx, (col, name_ids)) in enumerate(selected_columns):
        if len(input_ids) >= max_tokens:
            break
        available = max_tokens - len(input_ids)
        if available <= 1:
            break
        remaining_name_cost -= len(name_ids) + 1
        if name_ids:
            name_chunk = name_ids[:max(1, available - 1)]
            input_ids.extend(name_chunk)
            if len(input_ids) >= max_tokens:
                break
        input_ids.append(sep_token_id)
        column_positions.append(len(input_ids) - 1)
        max_value_tokens = max_tokens - len(input_ids) - remaining_name_cost
        if max_value_tokens <= 0:
            continue
        values = _iter_unique_values(df[col])
        remaining_cols = max(1, len(selected_columns) - idx)
        target_budget = max(1, math.ceil(remaining_value_budget / remaining_cols)) if remaining_value_budget > 0 else 0
        value_ids = _encode_value_stream(values, tokenizer, sep_token_id, min(max_value_tokens, target_budget))
        input_ids.extend(value_ids[:max_value_tokens])
        remaining_value_budget = max(0, remaining_value_budget - len(value_ids))
    if not column_positions:
        return ([cls_token_id], [0])
    return (input_ids[:max_tokens], column_positions)
