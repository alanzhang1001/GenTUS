from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.encoder_serializer import serialize_table
from model.table_encoder import AttentiveTableEncoder
from semantic_id.embedder import embed, load_encoder


@pytest.fixture
def encoder_files(tmp_path):
    torch.manual_seed(42)
    model_dir = tmp_path / "bert"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "city", "year", "alpha", "beta", "2020", "2021"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"))
    tokenizer.save_pretrained(model_dir)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(model_dir)
    model = AttentiveTableEncoder(str(model_dir), attn_heads=2, mlp_dim=8, local_files_only=True)
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"model": model.state_dict(), "config": {
        "lm": str(model_dir), "attn_heads": 2, "mlp_dim": 8,
        "max_tokens": 32, "include_column_names": False,
    }}, checkpoint)
    return model_dir, checkpoint, model.eval()


def test_encoder_loads_retained_state_and_returns_normalized_vectors(encoder_files):
    model_dir, checkpoint, original = encoder_files
    loaded, _ = load_encoder(checkpoint, str(model_dir), local_files_only=True, device="cpu")
    df = pd.DataFrame({"city": ["alpha", "beta"], "year": [2020, 2021]})
    ids, cols = serialize_table(df, loaded.tokenizer, 32, include_column_names=False)
    inputs, positions = torch.tensor([ids]), torch.tensor([cols])
    with torch.no_grad():
        expected = original(inputs, positions)
        actual = loaded(inputs, positions)
    torch.testing.assert_close(actual, expected)
    result = embed(tables={"a": df, "b": df}, encoder_checkpoint=checkpoint,
                   model_name=str(model_dir), local_files_only=True, device="cpu", batch_size=2)
    np.testing.assert_allclose(result[0], expected[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=1e-6)


def test_column_padding_does_not_change_table_vector(encoder_files):
    _, _, model = encoder_files
    tokenizer = model.tokenizer
    ids, cols = serialize_table(pd.DataFrame({"city": ["alpha"]}), tokenizer, 32, False)
    with torch.no_grad():
        one = model(torch.tensor([ids]), torch.tensor([cols]))
        padded = model(torch.tensor([ids + [tokenizer.pad_token_id] * 3]), torch.tensor([cols + [-1, -1]]))
    torch.testing.assert_close(one, padded, atol=1e-6, rtol=1e-5)


def test_encoder_requires_trained_checkpoint():
    with pytest.raises(ValueError, match="encoder_checkpoint"):
        embed(["table text"])


def test_encoder_rejects_missing_attention_weights(encoder_files, tmp_path):
    model_dir, checkpoint, _ = encoder_files
    state = torch.load(checkpoint, weights_only=True)
    del state["model"]["query_token"]
    broken = tmp_path / "broken.pt"
    torch.save(state, broken)
    with pytest.raises(RuntimeError, match="query_token"):
        load_encoder(broken, str(model_dir), local_files_only=True, device="cpu")
