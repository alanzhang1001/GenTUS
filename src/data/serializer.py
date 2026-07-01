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
) -> str:
    """Convert a DataFrame to a text sequence for T5 input.

    The default protocol uses no column names, all columns, up to 24 values per
    column, and a 256-token text budget.
    """
    if df.empty or max_rows <= 0 or token_budget <= 0:
        return ""

    selected = list(df.columns)
    if max_cols is not None and max_cols > 0:
        selected = selected[:max_cols]
    if not selected:
        return ""

    column_values = {col: _clean_values(df[col])[:max_rows] for col in selected}
    parts: list[str] = []
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
        parts.extend(prefix_tokens)
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
                parts.append("|")
                token_count += 1
            parts.extend(value_tokens)
            token_count += len(value_tokens)
            value_counts[col] += 1
            added = True
            if value_pos + 1 < len(values):
                next_active.append(col)
        if not added:
            break
        active_columns = next_active
        value_pos += 1

    return " ".join(parts)
