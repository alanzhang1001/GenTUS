import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protocol import clean_ground_truth, read_split
import evaluate
import train

spec = importlib.util.spec_from_file_location("run_gentus", ROOT / "scripts/run_gentus.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("dataset,k,levels,size,epochs", [
    ("TUS_small", 60, 4, 256, 30),
    ("TUS_large", 60, 6, 512, 30),
    ("SANTOS_small", 10, 4, 256, 80),
    ("SANTOS_large", 20, 8, 1024, 80),
    ("wiki_union", 40, 6, 512, 30),
    ("LakeBench_opendata_1k", 60, 6, 512, 30),
    ("LakeBench_opendata_30k", 60, 6, 512, 30),
])
def test_retained_dataset_config(dataset, k, levels, size, epochs, tmp_path, monkeypatch):
    config = runner.load_config(dataset)
    split = read_split(ROOT / config["split_path"])
    assert not set(split.train_queries) & set(split.test_queries)
    assert config["training"]["num_quantizers"] == levels
    assert config["training"]["codebook_size"] == size
    assert config["training"]["epochs"] == epochs
    assert config["training"]["seed"] == 42
    assert len(config["ground_truth"]["sha256"]) == 64
    assert len(config["encoder"]["sha256"]) == 64
    commands = runner.build_commands(config, tmp_path)
    evaluation = commands["evaluate"]
    assert evaluation[evaluation.index("--k") + 1] == str(k)
    training = commands["train"]
    assert training[training.index("--split_path") + 1] == str(ROOT / config["split_path"])
    assert "--ranked_results_output" in evaluation
    for stage, command in commands.items():
        monkeypatch.setattr(sys, "argv", command[1:])
        (train if stage == "train" else evaluate).parse_args()


def test_saved_split_rejects_overlap(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(dict(train_queries=["q"], test_queries=["q"], train_ratio=.8, seed=42)))
    with pytest.raises(ValueError, match="overlap"):
        read_split(path)


def test_data_checksum_rejects_wrong_version(tmp_path):
    path = tmp_path / "benchmark.pkl"
    path.write_bytes(b"wrong ground truth")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.verify_file(path, "0" * 64)


def test_ground_truth_pool_restriction_is_explicit():
    truth = {"q": ["q", "a", "a", "outside_pool"]}
    assert clean_ground_truth(truth, None) == {"q": ["a", "outside_pool"]}
    assert clean_ground_truth(truth, {"a"}) == {"q": ["a"]}
