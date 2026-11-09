import numpy as np
import torch
from tqdm import tqdm

from data.encoder_serializer import serialize_table
from model.table_encoder import AttentiveTableEncoder


def load_encoder(checkpoint_path, model_name="bert-base-uncased", cache_dir=None,
                 local_files_only=None, device=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = AttentiveTableEncoder(
        lm=model_name or config.get("lm", "bert-base-uncased"),
        attn_heads=config.get("attn_heads") or 8,
        mlp_dim=config.get("mlp_dim") or 256,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), config


def embed(texts=None, batch_size=64, model_name="bert-base-uncased", cache_dir=None,
          local_files_only=None, device=None, tables=None, encoder_checkpoint=None,
          include_column_names=None):
    if tables is None or encoder_checkpoint is None:
        raise ValueError("BERT table encoding requires tables and --encoder_checkpoint")
    model, config = load_encoder(encoder_checkpoint, model_name, cache_dir,
                                 local_files_only, device)
    max_tokens = config.get("max_tokens") or 256
    if include_column_names is None:
        include_column_names = config.get("include_column_names", True)
    names = list(tables)
    vectors = []
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in tqdm(range(0, len(names), batch_size), desc="BERT table encoding"):
            batch = [serialize_table(tables[name], tokenizer, max_tokens,
                                     include_column_names=include_column_names)
                     for name in names[start:start + batch_size]]
            max_length = max(len(ids) for ids, _ in batch)
            max_columns = max(len(cols) for _, cols in batch)
            input_ids = torch.full((len(batch), max_length), model.pad_token_id, dtype=torch.long)
            positions = torch.full((len(batch), max_columns), -1, dtype=torch.long)
            for i, (ids, cols) in enumerate(batch):
                input_ids[i, :len(ids)] = torch.tensor(ids)
                positions[i, :len(cols)] = torch.tensor(cols)
            vectors.append(model(input_ids.to(device), positions.to(device)).cpu().numpy())
    dimension = model.mlp[-1].out_features
    return np.vstack(vectors).astype("float32") if vectors else np.empty((0, dimension), dtype="float32")
