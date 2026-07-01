#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-TUS_small}"
MODEL="${MODEL:-t5-base}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}/seed42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-42}"
K="${K:-60}"

COMMON_ARGS=()
if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
  COMMON_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  COMMON_ARGS+=(--local_files_only)
fi

python src/train.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --embed_model "$EMBED_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --train_ratio "$TRAIN_RATIO" \
  --seed "$SEED" \
  "${COMMON_ARGS[@]}"

python src/evaluate.py \
  --dataset "$DATASET" \
  --model_dir "$OUTPUT_DIR" \
  --sid_path "$OUTPUT_DIR/semantic_ids.json" \
  --split_path "$OUTPUT_DIR/split.json" \
  --k "$K" \
  --output "$OUTPUT_DIR/eval_summary.json" \
  --per_query_output "$OUTPUT_DIR/per_query_metrics.jsonl" \
  "${COMMON_ARGS[@]}"
