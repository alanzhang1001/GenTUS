import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import load_data_lake, load_ground_truth, load_query_tables
from data.serializer import serialize
from model.seq2seq import RankingTrainer, RankingTUSDataset, TUSDataset, build_model
from protocol import (
    clean_ground_truth,
    filter_ground_truth_by_queries,
    make_query_split,
    write_split,
)
from semantic_id.embedder import embed
from semantic_id.manager import SemanticIDManager
from semantic_id.task_aware_rqvae import (
    quantize_task_aware,
    save_task_aware_rqvae_artifact,
    train_task_aware_rqvae,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tf32_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def ensure_parent_dir(path: str | None) -> None:
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def write_json(path: str, data) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def names_match(expected_names: list[str], cached_names: list[str]) -> bool:
    return list(expected_names) == list(cached_names)


def name_sets_match(expected_names: list[str], cached_names: list[str]) -> bool:
    return set(expected_names) == set(cached_names) and len(expected_names) == len(cached_names)


def format_name_mismatch(expected_names: list[str], cached_names: list[str]) -> str:
    expected = set(expected_names)
    cached = set(cached_names)
    missing = sorted(expected - cached)[:10]
    extra = sorted(cached - expected)[:10]
    return (
        f"expected {len(expected_names)} datalake tables, cached SID has {len(cached_names)} tables; "
        f"missing_from_sid={missing}, extra_in_sid={extra}"
    )


def sid_metadata_path(sid_path: str) -> str:
    return sid_path.replace(".json", "_metadata.json")


def train_protocol_metadata(args, table_names: list[str], split, train_ground_truth: dict[str, list[str]]) -> dict:
    return {
        "dataset": args.dataset,
        "table_names": list(table_names),
        "train_queries": list(split.train_queries),
        "test_queries": list(split.test_queries),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "rqvae_mode": args.rqvae_mode,
        "embed_model": args.embed_model,
        "max_cols": args.max_cols,
        "max_rows": args.max_rows,
        "serializer_token_budget": args.serializer_token_budget,
        "include_column_names": args.include_column_names,
        "num_quantizers": args.num_quantizers,
        "codebook_size": args.codebook_size,
        "rqvae_epochs": args.rqvae_epochs,
        "rqvae_lr": args.rqvae_lr,
        "rqvae_batch_size": args.rqvae_batch_size,
        "rqvae_commitment_weight": args.rqvae_commitment_weight,
        "rqvae_kmeans_iters": args.rqvae_kmeans_iters,
        "task_aware_latent_dim": args.task_aware_latent_dim,
        "task_aware_hidden_dims": args.task_aware_hidden_dims,
        "task_aware_union_loss_weight": args.task_aware_union_loss_weight,
        "task_aware_temperature": args.task_aware_temperature,
        "task_aware_dropout": args.task_aware_dropout,
        "add_candidate_candidate_pairs": args.add_candidate_candidate_pairs,
        "mask_false_negatives": args.mask_false_negatives,
        "fn_jaccard_threshold": args.fn_jaccard_threshold,
        "train_ground_truth": {
            query: list(candidates)
            for query, candidates in sorted(train_ground_truth.items())
        },
    }


def sid_cache_metadata_matches(sid_path: str, expected: dict) -> tuple[bool, str]:
    metadata_path = sid_metadata_path(sid_path)
    if not os.path.exists(metadata_path):
        return False, "metadata file is missing"
    with open(metadata_path, encoding="utf-8") as f:
        cached = json.load(f)
    mismatched = []
    for key, expected_value in expected.items():
        if cached.get(key) != expected_value:
            mismatched.append(key)
    if mismatched:
        return False, "metadata mismatch: " + ", ".join(mismatched[:10])
    return True, ""


def parse_hidden_dims(spec: str) -> str:
    # Keep the original comma-separated API consumed by task_aware_rqvae.
    return spec


def augment(df: pd.DataFrame, n: int = 3) -> list[pd.DataFrame]:
    views = []
    for _ in range(n):
        rows = df.sample(frac=0.7, replace=False) if len(df) > 2 else df
        cols = random.sample(list(df.columns), max(1, int(len(df.columns) * 0.7)))
        views.append(rows[cols])
    return views


def _sid_to_text(sid: tuple) -> str:
    return " ".join(f"[SFX{-(t + 1)}]" if t < 0 else str(t) for t in sid)


def _table_profile(df: pd.DataFrame, include_column_names: bool = False) -> set[str]:
    tokens = {str(col).lower() for col in df.columns} if include_column_names else set()
    sample = df.head(5)
    for value in sample.astype(str).to_numpy().flatten().tolist():
        if value and value.lower() != "nan":
            tokens.add(value.lower()[:32])
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_jaccard_adjacency(tables, names, threshold=0.5, max_values_per_table=2000):
    import re

    idx = {n: i for i, n in enumerate(names)}
    sets = [None] * len(names)
    for n in names:
        df = tables.get(n)
        toks = set()
        if df is not None:
            for col in df.columns:
                for v in df[col].dropna().astype(str):
                    for t in re.findall(r"[a-z0-9]+", v.lower()):
                        toks.add(t)
                        if len(toks) >= max_values_per_table:
                            break
                    if len(toks) >= max_values_per_table:
                        break
        sets[idx[n]] = toks
    adj = [set() for _ in range(len(names))]
    for i in range(len(names)):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, len(names)):
            sj = sets[j]
            if not sj:
                continue
            inter = len(si & sj)
            if inter == 0:
                continue
            jac = inter / (len(si) + len(sj) - inter)
            if jac >= threshold:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def build_unionable_pairs(
    table_names: list[str],
    train_ground_truth: dict[str, list[str]],
    add_candidate_candidate: bool = False,
) -> list[tuple[int, int]]:
    """Build task-aware positives from training-query ground truth only."""
    name_to_idx = {name: idx for idx, name in enumerate(table_names)}
    pairs = set()
    for query_name, candidates in train_ground_truth.items():
        i = name_to_idx.get(query_name)
        if i is not None:
            for candidate_name in candidates:
                j = name_to_idx.get(candidate_name)
                if j is None or i == j:
                    continue
                pairs.add((i, j) if i < j else (j, i))
        if add_candidate_candidate:
            idxs = sorted({name_to_idx[c] for c in candidates if c in name_to_idx})
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.add((idxs[a], idxs[b]))
    return sorted(pairs)


def build_negative_lookup(
    train_ground_truth,
    train_query_tables,
    datalake_tables,
    *,
    max_negatives_per_query=5,
    include_column_names=False,
):
    if max_negatives_per_query <= 0:
        return {}

    candidate_profiles = {
        name: _table_profile(df, include_column_names=include_column_names)
        for name, df in datalake_tables.items()
    }
    negative_lookup = {}
    for idx, (query_name, query_df) in enumerate(train_query_tables.items(), start=1):
        positives = set(train_ground_truth.get(query_name, [])) - {query_name}
        q_profile = _table_profile(query_df, include_column_names=include_column_names)
        scored = []
        for candidate_name, candidate_profile in candidate_profiles.items():
            if candidate_name == query_name or candidate_name in positives:
                continue
            scored.append((_jaccard(q_profile, candidate_profile), candidate_name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        negative_lookup[query_name] = [name for _score, name in scored[:max_negatives_per_query]]
        if idx % 50 == 0:
            print(f"  negative mining progress: {idx}/{len(train_query_tables)} queries")
    return negative_lookup


def build_records(
    train_ground_truth,
    train_query_tables,
    sid_manager,
    n_aug=3,
    datalake_tables=None,
    max_cols=0,
    max_rows=24,
    serializer_token_budget=256,
    use_ranking_loss=True,
    max_negatives_per_query=5,
    include_indexing_records=True,
    include_column_names=False,
    indexing_table_names=None,
):
    records = []
    negative_lookup = {}
    if use_ranking_loss and datalake_tables:
        negative_lookup = build_negative_lookup(
            train_ground_truth,
            train_query_tables,
            datalake_tables,
            max_negatives_per_query=max_negatives_per_query,
            include_column_names=include_column_names,
        )

    stats = {
        "train_ground_truth_queries": len(train_ground_truth),
        "train_query_tables": len(train_query_tables),
        "queries_with_positive_pairs": 0,
        "positive_pairs": 0,
        "ranking_records": 0,
        "ranking_negative_queries": 0,
        "ranking_max_negatives_per_query": max_negatives_per_query if use_ranking_loss else 0,
        "skipped_self_pairs": 0,
        "missing_sid_candidates": 0,
        "views_per_pair": n_aug + 1,
        "indexing_records": 0,
    }

    for query_name, candidates in train_ground_truth.items():
        if query_name not in train_query_tables:
            continue
        query_df = train_query_tables[query_name]
        views = [query_df] + augment(query_df, n_aug)
        serialized_views = []
        for view in views:
            serialized_views.append(
                serialize(
                    view,
                    max_cols=max_cols,
                    max_rows=max_rows,
                    include_column_names=include_column_names,
                    token_budget=serializer_token_budget,
                )
            )
        query_positive_pairs = 0
        for candidate_name in candidates:
            if candidate_name == query_name:
                stats["skipped_self_pairs"] += 1
                continue
            sid = sid_manager.table_to_id(candidate_name)
            if sid is None:
                stats["missing_sid_candidates"] += 1
                continue
            query_positive_pairs += 1
            stats["positive_pairs"] += 1
            target_text = _sid_to_text(sid)
            negative_text = None
            query_negatives = negative_lookup.get(query_name, [])
            if query_negatives:
                offset = (query_positive_pairs - 1) % len(query_negatives)
                ordered_negatives = query_negatives[offset:] + query_negatives[:offset]
            else:
                ordered_negatives = []
            for negative_name in ordered_negatives:
                neg_sid = sid_manager.table_to_id(negative_name)
                if neg_sid is not None:
                    negative_text = _sid_to_text(neg_sid)
                    break
            if negative_text:
                stats["ranking_records"] += len(serialized_views)
            for input_text in serialized_views:
                row = {"input_text": input_text, "target_text": target_text}
                if negative_text:
                    row["negative_text"] = negative_text
                records.append(row)
        if query_positive_pairs > 0:
            stats["queries_with_positive_pairs"] += 1
            if negative_lookup.get(query_name):
                stats["ranking_negative_queries"] += 1

    if datalake_tables and include_indexing_records:
        collision_bases = getattr(sid_manager, "collision_bases", set())
        allowed_indexing_names = set(indexing_table_names) if indexing_table_names is not None else None
        for table_name, df in datalake_tables.items():
            if allowed_indexing_names is not None and table_name not in allowed_indexing_names:
                continue
            sid = sid_manager.table_to_id(table_name)
            if sid is None:
                continue
            target_text = _sid_to_text(sid)
            base = sid_manager._base_sid(sid)
            views = [df] + augment(df, n_aug) if base in collision_bases else [df]
            for view in views:
                records.append({
                    "input_text": serialize(
                        view,
                        max_cols=max_cols,
                        max_rows=max_rows,
                        include_column_names=include_column_names,
                        token_budget=serializer_token_budget,
                    ),
                    "target_text": target_text,
                })
            stats["indexing_records"] += len(views)

    stats["total_records"] = len(records)
    return records, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train GenTUS with a split-safe query protocol.")
    parser.add_argument("--dataset", default="TUS_small")
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--embed_model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output_dir", default="results/checkpoints/tus_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_quantizers", type=int, default=6)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--rqvae_epochs", type=int, default=200)
    parser.add_argument("--rqvae_lr", type=float, default=1e-3)
    parser.add_argument("--rqvae_batch_size", type=int, default=256)
    parser.add_argument("--rqvae_commitment_weight", type=float, default=0.25)
    parser.add_argument("--rqvae_kmeans_iters", type=int, default=100)
    parser.add_argument("--n_aug", type=int, default=3)
    parser.add_argument("--max_cols", type=int, default=0,
                        help="Maximum columns to serialize; 0 uses all columns.")
    parser.add_argument("--max_rows", type=int, default=24)
    parser.add_argument("--serializer_token_budget", type=int, default=256)
    parser.add_argument("--include-column-names", dest="include_column_names",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sid_path", default=None)
    parser.add_argument("--split_output", default=None)
    parser.add_argument("--records_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--log_history_output", default=None)
    parser.add_argument("--rqvae_state_output", default=None)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--rqvae_mode", choices=["task_aware"], default="task_aware")
    parser.add_argument("--task_aware_latent_dim", type=int, default=128)
    parser.add_argument("--task_aware_hidden_dims", default="512,256")
    parser.add_argument("--task_aware_union_loss_weight", type=float, default=0.1)
    parser.add_argument("--add_candidate_candidate_pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask_false_negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fn_jaccard_threshold", type=float, default=0.0)
    parser.add_argument("--task_aware_temperature", type=float, default=0.1)
    parser.add_argument("--task_aware_dropout", type=float, default=0.0)
    parser.add_argument("--include-indexing-records", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_ranking_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ranking_weight", type=float, default=0.2)
    parser.add_argument("--ranking_margin", type=float, default=0.2)
    parser.add_argument("--max_negatives_per_query", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    started_at = datetime.now().isoformat()
    stage_timings: dict[str, float] = {}

    def mark_stage(name: str, started: float) -> None:
        stage_timings[name] = round(time.perf_counter() - started, 6)

    print(f"[1/5] Loading data: {args.dataset}")
    stage_started = time.perf_counter()
    tables = load_data_lake(args.dataset)
    query_tables_all = load_query_tables(args.dataset)
    raw_ground_truth = load_ground_truth(args.dataset)
    candidate_names = set(tables)
    cleaned_ground_truth = clean_ground_truth(raw_ground_truth, candidate_names, set(query_tables_all))
    split = make_query_split(sorted(cleaned_ground_truth), train_ratio=args.train_ratio, seed=args.seed)
    train_query_names = set(split.train_queries)
    test_query_names = set(split.test_queries)
    train_ground_truth = filter_ground_truth_by_queries(cleaned_ground_truth, train_query_names)
    test_ground_truth = filter_ground_truth_by_queries(cleaned_ground_truth, test_query_names)
    train_query_tables = {name: query_tables_all[name] for name in split.train_queries if name in query_tables_all}
    if not train_ground_truth:
        raise RuntimeError("No training queries have valid ground truth after cleaning")
    if not test_ground_truth:
        raise RuntimeError("No test queries have valid ground truth after cleaning")
    print(
        f"  datalake={len(tables)} query={len(query_tables_all)} "
        f"train_queries={len(train_ground_truth)} test_queries={len(test_ground_truth)}"
    )
    mark_stage("data_loading_seconds", stage_started)

    sid_path = args.sid_path or os.path.join(args.output_dir, "semantic_ids.json")
    split_output = args.split_output or os.path.join(args.output_dir, "split.json")
    summary_output = args.summary_output or os.path.join(args.output_dir, "train_summary.json")
    log_history_output = args.log_history_output or os.path.join(args.output_dir, "trainer_log_history.json")
    write_split(split_output, split)

    names = list(tables.keys())
    cache_flags = {"semantic_ids_cache_hit": False, "embedding_cache_hit": False, "resumed_seq2seq_checkpoint": False}
    expected_sid_metadata = train_protocol_metadata(args, names, split, train_ground_truth)

    print("[2/5] Building Semantic IDs")
    stage_started = time.perf_counter()
    sid_manager = None
    if os.path.exists(sid_path):
        loaded_sid_manager = SemanticIDManager.load(sid_path)
        cached_sid_names = list(loaded_sid_manager.table2id)
        metadata_ok, metadata_reason = sid_cache_metadata_matches(sid_path, expected_sid_metadata)
        if name_sets_match(names, cached_sid_names) and metadata_ok:
            sid_manager = loaded_sid_manager
            cache_flags["semantic_ids_cache_hit"] = True
        elif name_sets_match(names, cached_sid_names):
            print(f"  cached Semantic IDs ignored: {metadata_reason}")
        else:
            print("  cached Semantic IDs ignored: " + format_name_mismatch(names, cached_sid_names))

    task_aware_stats = None
    if sid_manager is None:
        embed_cache = sid_path.replace(".json", "_embeddings.npy")
        names_cache = sid_path.replace(".json", "_names.json")
        embeddings = None
        if os.path.exists(embed_cache) and os.path.exists(names_cache):
            with open(names_cache, encoding="utf-8") as f:
                cached_names = json.load(f)
            if names_match(names, cached_names):
                embeddings = np.load(embed_cache)
                cache_flags["embedding_cache_hit"] = True
            else:
                print("  cached embeddings ignored: datalake table order changed")
        if embeddings is None:
            texts = [
                serialize(tables[n], max_cols=args.max_cols, max_rows=args.max_rows,
                          include_column_names=args.include_column_names,
                          token_budget=args.serializer_token_budget)
                for n in names
            ]
            embeddings = embed(
                texts,
                model_name=args.embed_model,
                cache_dir=args.model_cache_dir,
                local_files_only=args.local_files_only,
            )
            ensure_parent_dir(embed_cache)
            np.save(embed_cache, embeddings)
            ensure_parent_dir(names_cache)
            with open(names_cache, "w", encoding="utf-8") as f:
                json.dump(names, f)

        positive_pairs = build_unionable_pairs(
            names,
            train_ground_truth,
            add_candidate_candidate=args.add_candidate_candidate_pairs,
        )
        jaccard_adj = None
        if args.fn_jaccard_threshold > 0:
            jaccard_adj = build_jaccard_adjacency(tables, names, threshold=args.fn_jaccard_threshold)
        task_aware_model, task_aware_stats = train_task_aware_rqvae(
            embeddings,
            positive_pairs=positive_pairs,
            num_quantizers=args.num_quantizers,
            codebook_size=args.codebook_size,
            epochs=args.rqvae_epochs,
            lr=args.rqvae_lr,
            commitment_weight=args.rqvae_commitment_weight,
            batch_size=args.rqvae_batch_size,
            latent_dim=args.task_aware_latent_dim,
            hidden_dims=parse_hidden_dims(args.task_aware_hidden_dims),
            union_loss_weight=args.task_aware_union_loss_weight,
            temperature=args.task_aware_temperature,
            dropout=args.task_aware_dropout,
            seed=args.seed,
            kmeans_iters=args.rqvae_kmeans_iters,
            mask_false_negatives=args.mask_false_negatives,
            extra_exclude_adjacency=jaccard_adj,
        )
        indices = quantize_task_aware(task_aware_model, embeddings, batch_size=args.rqvae_batch_size)
        sid_manager = SemanticIDManager()
        sid_manager.build(names, indices, embeddings=embeddings)
        ensure_parent_dir(sid_path)
        sid_manager.save(sid_path)
        write_json(sid_metadata_path(sid_path), expected_sid_metadata)
        if args.rqvae_state_output:
            save_task_aware_rqvae_artifact(
                task_aware_model,
                args.rqvae_state_output,
                extra_metadata={
                    "dataset": args.dataset,
                    "sid_path": sid_path,
                    "train_ratio": args.train_ratio,
                    "seed": args.seed,
                    "rqvae_kmeans_iters": args.rqvae_kmeans_iters,
                    "task_aware_stats": task_aware_stats,
                },
            )
    mark_stage("semantic_id_build_seconds", stage_started)

    print("[3/5] Building training records")
    stage_started = time.perf_counter()
    indexing_table_names = sorted({
        candidate
        for candidates in train_ground_truth.values()
        for candidate in candidates
        if candidate in tables
    })
    records, record_stats = build_records(
        train_ground_truth,
        train_query_tables,
        sid_manager,
        args.n_aug,
        datalake_tables=tables,
        max_cols=args.max_cols,
        max_rows=args.max_rows,
        serializer_token_budget=args.serializer_token_budget,
        use_ranking_loss=args.use_ranking_loss,
        max_negatives_per_query=args.max_negatives_per_query,
        include_indexing_records=args.include_indexing_records,
        include_column_names=args.include_column_names,
        indexing_table_names=indexing_table_names,
    )
    if not records:
        raise RuntimeError("No training records were generated")
    if args.records_output:
        write_jsonl(args.records_output, records)
    print(
        f"  {len(records)} samples "
        f"({record_stats['indexing_records']} indexing + "
        f"{record_stats['total_records'] - record_stats['indexing_records']} retrieval)"
    )
    mark_stage("record_build_seconds", stage_started)

    print(f"[4/5] Loading model: {args.model}")
    stage_started = time.perf_counter()
    model, tokenizer = build_model(
        args.model,
        args.codebook_size,
        cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    dataset_cls = RankingTUSDataset if args.use_ranking_loss and args.ranking_weight > 0 else TUSDataset
    dataset = dataset_cls(records, tokenizer)
    val_n = max(1, int(0.1 * len(dataset)))
    split_generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n], generator=split_generator)
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=torch.cuda.is_available(),
        tf32=tf32_supported(),
        max_grad_norm=1.0,
        logging_steps=100,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        disable_tqdm=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    mark_stage("model_load_seconds", stage_started)

    print("[5/5] Training")
    stage_started = time.perf_counter()
    if args.use_ranking_loss and args.ranking_weight > 0:
        trainer = RankingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
        )
    else:
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
    ckpt_dirs = sorted([
        d for d in (os.scandir(args.output_dir) if os.path.isdir(args.output_dir) else [])
        if d.is_dir() and d.name.startswith("checkpoint-")
    ], key=lambda d: int(d.name.split("-")[-1]))
    resume_ckpt = ckpt_dirs[-1].path if ckpt_dirs else None
    if resume_ckpt:
        cache_flags["resumed_seq2seq_checkpoint"] = True
        print(f"  resuming from {resume_ckpt}")
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    mark_stage("seq2seq_training_seconds", stage_started)

    finished_at = datetime.now().isoformat()
    write_json(log_history_output, trainer.state.log_history)
    summary = {
        "dataset": args.dataset,
        "status": "completed",
        "model": args.model,
        "embed_model": args.embed_model,
        "output_dir": args.output_dir,
        "sid_path": sid_path,
        "split_path": split_output,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "args": vars(args),
        "data_summary": {
            "datalake_tables": len(tables),
            "query_tables": len(query_tables_all),
            "raw_ground_truth_queries": len(raw_ground_truth),
            "clean_ground_truth_queries": len(cleaned_ground_truth),
            "train_queries": len(train_ground_truth),
            "test_queries": len(test_ground_truth),
            "sid_collision_rate": sid_manager.collision_rate(),
            "rqvae_mode": args.rqvae_mode,
            "task_aware_stats": task_aware_stats,
            "indexing_table_names": indexing_table_names,
            "serializer_token_budget": args.serializer_token_budget,
            **record_stats,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        },
        "train_metrics": train_result.metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "stage_timings": stage_timings,
        **cache_flags,
    }
    write_json(summary_output, summary)
    print(f"Done. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
