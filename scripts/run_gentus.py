import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_config(dataset):
    path = ROOT / "configs" / f"{dataset}.json"
    config = json.loads(path.read_text())
    if config["dataset"] != dataset:
        raise ValueError(f"Dataset mismatch in {path}")
    return config


def cli_options(values):
    result = []
    for name, value in values.items():
        if value is None:
            continue
        option = name.replace("_", "-") if name in {"include_column_names", "include_indexing_records"} else name
        if isinstance(value, bool):
            result.append("--" + ("" if value else "no-") + option)
        else:
            result.extend(["--" + option, str(value)])
    return result


def build_commands(config, output_dir, encoder_checkpoint=None, model=None,
                   embed_model=None, model_cache_dir=None, local_files_only=False):
    training = dict(config["training"])
    evaluation = dict(config["evaluation"])
    if model:
        training["model"] = model
    if embed_model:
        training["embed_model"] = embed_model
    checkpoint = Path(encoder_checkpoint).resolve() if encoder_checkpoint else ROOT / config["encoder"]["checkpoint"]
    training.update(dataset=config["dataset"], output_dir=str(output_dir),
                    encoder_checkpoint=str(checkpoint), split_path=str(ROOT / config["split_path"]),
                    rqvae_state_output=str(output_dir / "rqvae.pt"))
    evaluation.update(dataset=config["dataset"], model_dir=str(output_dir),
                      sid_path=str(output_dir / "semantic_ids.json"),
                      split_path=str(ROOT / config["split_path"]),
                      output=str(output_dir / "eval_summary.json"),
                      per_query_output=str(output_dir / "per_query_metrics.jsonl"),
                      ranked_results_output=str(output_dir / "ranked_results.json"))
    shared = []
    if model_cache_dir:
        shared.extend(["--model_cache_dir", str(model_cache_dir)])
    if local_files_only:
        shared.append("--local_files_only")
    return {
        "train": [sys.executable, str(ROOT / "src/train.py")] + cli_options(training) + shared,
        "evaluate": [sys.executable, str(ROOT / "src/evaluate.py")] + cli_options(evaluation) + shared,
    }


def verify_file(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TUS_small", choices=sorted(p.stem for p in (ROOT / "configs").glob("*.json")))
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--embed-model")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.dataset)
    output = args.output_dir.resolve() if args.output_dir else ROOT / "results" / args.dataset / f"seed{config['training']['seed']}"
    commands = build_commands(config, output, args.encoder_checkpoint, args.model,
                              args.embed_model, args.model_cache_dir, args.local_files_only)
    stages = ["train", "evaluate"] if args.stage == "all" else [args.stage]
    if not args.dry_run:
        data_root = Path(os.environ.get("GENTUS_DATA_ROOT", ROOT / "datasets"))
        verify_file(data_root / config["ground_truth"]["path"], config["ground_truth"]["sha256"])
        if "train" in stages:
            checkpoint = args.encoder_checkpoint or ROOT / config["encoder"]["checkpoint"]
            verify_file(checkpoint, config["encoder"]["sha256"])
    for stage in stages:
        print(shlex.join(commands[stage]), flush=True)
        if not args.dry_run:
            subprocess.run(commands[stage], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
