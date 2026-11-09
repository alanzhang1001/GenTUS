#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--dataset "${DATASET:-TUS_small}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${ENCODER_CHECKPOINT:-}" ]]; then
  ARGS+=(--encoder-checkpoint "$ENCODER_CHECKPOINT")
fi
if [[ -n "${MODEL:-}" ]]; then
  ARGS+=(--model "$MODEL")
fi
if [[ -n "${EMBED_MODEL:-}" ]]; then
  ARGS+=(--embed-model "$EMBED_MODEL")
fi
if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
  ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  ARGS+=(--local-files-only)
fi
exec python "$SCRIPT_DIR/run_gentus.py" "${ARGS[@]}" "$@"
