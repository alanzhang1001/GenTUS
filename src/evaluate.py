import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(__file__))

from data.loader import load_data_lake, load_ground_truth, load_query_tables
from model.constrained import build_trie_from_ids
from protocol import clean_ground_truth, filter_ground_truth_by_queries, read_split
from retrieve import load_model, retrieve
from semantic_id.manager import SemanticIDManager


def synchronize_cuda(model=None) -> None:
    if not torch.cuda.is_available():
        return
    device = getattr(model, "device", None)
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    else:
        torch.cuda.synchronize()


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


def _average_precision(ranked: list[str], gt_set: set[str], k: int) -> tuple[float, int, float, int | None]:
    hits = 0
    precision_sum = 0.0
    first_gt_rank = None
    seen_hits = set()
    for rank, table_name in enumerate(ranked[:k], 1):
        if table_name in gt_set and table_name not in seen_hits:
            seen_hits.add(table_name)
            hits += 1
            precision_at_rank = hits / rank
            precision_sum += precision_at_rank
            if first_gt_rank is None:
                first_gt_rank = rank
    denominator = min(k, len(gt_set))
    ap = precision_sum / denominator if denominator > 0 else 0.0
    return ap, hits, precision_sum, first_gt_rank


def _ndcg(ranked: list[str], gt_set: set[str], k: int) -> float:
    dcg = 0.0
    seen_hits = set()
    for rank, table_name in enumerate(ranked[:k], 1):
        if table_name in gt_set and table_name not in seen_hits:
            seen_hits.add(table_name)
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gt_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def compute_metrics(
    ground_truth,
    query_tables,
    model,
    tokenizer,
    sid_manager,
    k,
    trie=None,
    valid_toks=None,
    beam_size=None,
    num_beam_groups=1,
    diversity_penalty=0.5,
    exclude_self=True,
    max_cols=0,
    max_rows=24,
    serializer_token_budget=256,
    save_ranked_results=False,
    decoding_mode="constrained",
    include_column_names=False,
    serialization_policy="column_budget",
):
    aps = []
    precisions = []
    recalls = []
    reciprocal_ranks = []
    hit_rates = []
    ndcgs = []
    query_latencies = []
    per_query = []
    ranked_results = {} if save_ranked_results else None
    generation_totals = {
        "generated_sequences": 0,
        "parseable_sid_sequences": 0,
        "mapped_sid_sequences": 0,
        "malformed_sid_sequences": 0,
        "unmapped_sid_sequences": 0,
        "excluded_sid_sequences": 0,
        "duplicate_sid_sequences": 0,
        "unique_ranked_tables": 0,
    }
    query_names = list(query_tables.keys())
    stats = {
        "ground_truth_queries": len(ground_truth),
        "requested_queries": len(query_names),
        "query_tables": len(query_names),
        "evaluated_queries": 0,
        "skipped_missing_ground_truth": 0,
        "skipped_empty_ground_truth": 0,
    }
    if beam_size is None:
        beam_size = max(k * 2, 100)

    for i, query_name in enumerate(query_names):
        gt_tables = ground_truth.get(query_name)
        if gt_tables is None:
            stats["skipped_missing_ground_truth"] += 1
            continue
        gt_set = set(gt_tables) - {query_name}
        if not gt_set:
            stats["skipped_empty_ground_truth"] += 1
            continue

        stats["evaluated_queries"] += 1
        exclude_tables = {query_name} if exclude_self else None
        synchronize_cuda(model)
        started = time.perf_counter()
        ranked, generation_stats = retrieve(
            query_tables[query_name],
            model,
            tokenizer,
            sid_manager,
            beam_size=beam_size,
            trie=trie,
            valid_toks=valid_toks,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            exclude_tables=exclude_tables,
            max_cols=max_cols,
            max_rows=max_rows,
            serializer_token_budget=serializer_token_budget,
            include_column_names=include_column_names,
            serialization_policy=serialization_policy,
            decoding_mode=decoding_mode,
            return_generation_stats=True,
        )
        ranked = ranked[:k]
        synchronize_cuda(model)
        latency_seconds = time.perf_counter() - started
        query_latencies.append(latency_seconds)
        for key in generation_totals:
            generation_totals[key] += int(generation_stats.get(key) or 0)

        ap, hits, _prec_sum, first_gt_rank = _average_precision(ranked, gt_set, k)
        precision = hits / k
        recall = hits / len(gt_set)
        reciprocal_rank = 1.0 / first_gt_rank if first_gt_rank else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0
        ndcg = _ndcg(ranked, gt_set, k)
        aps.append(ap)
        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        hit_rates.append(hit_rate)
        ndcgs.append(ndcg)

        per_query.append({
            "query": query_name,
            "gt_count": len(gt_set),
            "ranked_count": len(ranked),
            "hits@k": hits,
            "AP@k": round(ap, 6),
            "Precision@k": round(precision, 6),
            "Recall@k": round(recall, 6),
            "MRR@k": round(reciprocal_rank, 6),
            "HitRate@k": hit_rate,
            "nDCG@k": round(ndcg, 6),
            "first_gt_rank": first_gt_rank,
            "latency_seconds": round(latency_seconds, 6),
            "generated_sequences": generation_stats.get("generated_sequences"),
            "parseable_sid_rate": round(generation_stats.get("parseable_sid_rate", 0.0), 6),
            "valid_sid_rate": round(generation_stats.get("valid_sid_rate", 0.0), 6),
            "mapped_sid_sequences": generation_stats.get("mapped_sid_sequences"),
            "malformed_sid_sequences": generation_stats.get("malformed_sid_sequences"),
            "unmapped_sid_sequences": generation_stats.get("unmapped_sid_sequences"),
            "duplicate_sid_sequences": generation_stats.get("duplicate_sid_sequences"),
        })
        if ranked_results is not None:
            ranked_results[query_name] = ranked
        if (i + 1) % 10 == 0 and aps:
            print(f"  [{i + 1}/{len(query_names)}] MAP@{k}={sum(aps) / len(aps):.4f}")

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "MAP@k": mean(aps),
        "Precision@k": mean(precisions),
        "Recall@k": mean(recalls),
        "MRR@k": mean(reciprocal_ranks),
        "HitRate@k": mean(hit_rates),
        "nDCG@k": mean(ndcgs),
        "mean_query_latency_seconds": mean(query_latencies),
        "total_query_latency_seconds": sum(query_latencies),
    }
    generated = generation_totals["generated_sequences"]
    metrics.update({
        **generation_totals,
        "parseable_sid_rate": generation_totals["parseable_sid_sequences"] / generated if generated else 0.0,
        "valid_sid_rate": generation_totals["mapped_sid_sequences"] / generated if generated else 0.0,
    })
    return metrics, stats, per_query, ranked_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GenTUS on the held-out query split.")
    parser.add_argument("--dataset", default="TUS_small")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sid_path", required=True)
    parser.add_argument("--split_path", default=None,
                        help="Path to split.json saved by src/train.py. Defaults to model_dir/split.json.")
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--diversity_penalty", type=float, default=0.5)
    parser.add_argument("--max_cols", type=int, default=0,
                        help="Maximum columns to serialize; 0 uses all columns.")
    parser.add_argument("--max_rows", type=int, default=24)
    parser.add_argument("--serializer_token_budget", type=int, default=256)
    parser.add_argument("--serialization_policy", choices=["column_budget", "priority"], default="column_budget")
    parser.add_argument("--include-column-names", dest="include_column_names",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_self", action="store_true")
    parser.add_argument("--restrict_gt_to_sid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--per_query_output", default=None)
    parser.add_argument("--ranked_results_output", default=None)
    parser.add_argument("--model_cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--decoding_mode", "--decoding-mode", dest="decoding_mode",
                        choices=["constrained", "unconstrained"], default="constrained")
    return parser.parse_args()


def main():
    args = parse_args()
    split_path = args.split_path or os.path.join(args.model_dir, "split.json")
    split = read_split(split_path)

    print(f"Loading model from {args.model_dir}")
    model, tokenizer = load_model(args.model_dir, cache_dir=args.model_cache_dir,
                                  local_files_only=args.local_files_only)
    print(f"Loading Semantic IDs from {args.sid_path}")
    sid_manager = SemanticIDManager.load(args.sid_path)

    print(f"Loading held-out query split: {split_path}")
    datalake_tables = load_data_lake(args.dataset)
    query_tables_all = load_query_tables(args.dataset)
    raw_ground_truth = load_ground_truth(args.dataset)
    candidates = set(sid_manager.table2id) if args.restrict_gt_to_sid else None
    cleaned_ground_truth = clean_ground_truth(raw_ground_truth, candidates, set(query_tables_all))
    test_ground_truth = filter_ground_truth_by_queries(cleaned_ground_truth, set(split.test_queries))
    test_query_tables = {name: query_tables_all[name] for name in split.test_queries if name in query_tables_all}
    if not test_query_tables:
        raise RuntimeError("No test query tables were found for the saved split")
    if not test_ground_truth:
        raise RuntimeError("No test queries have valid ground truth after cleaning")

    print(f"Evaluating MAP@{args.k} on {len(test_query_tables)} held-out queries")
    trie, valid_toks = build_trie_from_ids(sid_manager.all_ids(), tokenizer)
    metrics, eval_stats, per_query, ranked_results = compute_metrics(
        test_ground_truth,
        test_query_tables,
        model,
        tokenizer,
        sid_manager,
        args.k,
        trie=trie,
        valid_toks=valid_toks,
        beam_size=args.beam_size,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        exclude_self=not args.include_self,
        max_cols=args.max_cols,
        max_rows=args.max_rows,
        serializer_token_budget=args.serializer_token_budget,
        save_ranked_results=bool(args.ranked_results_output),
        decoding_mode=args.decoding_mode,
        include_column_names=args.include_column_names,
        serialization_policy=args.serialization_policy,
    )

    result = {
        "dataset": args.dataset,
        "model_dir": args.model_dir,
        "sid_path": args.sid_path,
        "split_path": split_path,
        "train_ratio": split.train_ratio,
        "seed": split.seed,
        "MAP@k": round(metrics["MAP@k"], 4),
        "Precision@k": round(metrics["Precision@k"], 4),
        "Recall@k": round(metrics["Recall@k"], 4),
        "MRR@k": round(metrics["MRR@k"], 4),
        "HitRate@k": round(metrics["HitRate@k"], 4),
        "nDCG@k": round(metrics["nDCG@k"], 4),
        "mean_query_latency_seconds": round(metrics["mean_query_latency_seconds"], 6),
        "total_query_latency_seconds": round(metrics["total_query_latency_seconds"], 6),
        "generated_sequences": metrics["generated_sequences"],
        "parseable_sid_sequences": metrics["parseable_sid_sequences"],
        "mapped_sid_sequences": metrics["mapped_sid_sequences"],
        "malformed_sid_sequences": metrics["malformed_sid_sequences"],
        "unmapped_sid_sequences": metrics["unmapped_sid_sequences"],
        "excluded_sid_sequences": metrics["excluded_sid_sequences"],
        "duplicate_sid_sequences": metrics["duplicate_sid_sequences"],
        "parseable_sid_rate": round(metrics["parseable_sid_rate"], 6),
        "valid_sid_rate": round(metrics["valid_sid_rate"], 6),
        "k": args.k,
        "n_queries": len(test_query_tables),
        "exclude_self": not args.include_self,
        "decoding_mode": args.decoding_mode,
        "max_cols": args.max_cols,
        "max_rows": args.max_rows,
        "serializer_token_budget": args.serializer_token_budget,
        "include_column_names": args.include_column_names,
        "serialization_policy": args.serialization_policy,
        "restrict_gt_to_sid": args.restrict_gt_to_sid,
        **eval_stats,
        "timestamp": datetime.now().isoformat(),
    }

    print(
        f"MAP@{args.k}: {result['MAP@k']:.4f} | "
        f"P@{args.k}: {result['Precision@k']:.4f} | "
        f"R@{args.k}: {result['Recall@k']:.4f}"
    )
    if args.output:
        write_json(args.output, result)
        print(f"Result saved to: {args.output}")
    if args.per_query_output:
        write_jsonl(args.per_query_output, per_query)
        print(f"Per-query metrics saved to: {args.per_query_output}")
    if args.ranked_results_output and ranked_results is not None:
        write_json(args.ranked_results_output, ranked_results)
        print(f"Ranked results saved to: {args.ranked_results_output}")


if __name__ == "__main__":
    main()
