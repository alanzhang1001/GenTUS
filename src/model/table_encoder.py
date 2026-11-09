import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from hf_utils import resolve_model_source, transformers_load_kwargs


class AttentiveTableEncoder(nn.Module):
    def __init__(self, lm="bert-base-uncased", attn_heads=8, mlp_dim=256,
                 cache_dir=None, local_files_only=None):
        super().__init__()
        source = resolve_model_source(lm, cache_dir)
        kwargs = transformers_load_kwargs(cache_dir, local_files_only)
        self.bert = AutoModel.from_pretrained(source, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        hidden = self.bert.config.hidden_size
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, attn_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_dim), nn.LayerNorm(mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )

    def _gather_column_vectors(self, hidden, column_positions):
        valid_mask = column_positions >= 0
        safe_positions = column_positions.clamp(min=0)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device).unsqueeze(1)
        batch_indices = batch_indices.expand_as(safe_positions)
        column_vectors = hidden[batch_indices, safe_positions]
        column_vectors = column_vectors * valid_mask.unsqueeze(-1)
        return column_vectors, ~valid_mask

    def forward(self, input_ids, column_positions):
        attention_mask = (input_ids != self.pad_token_id).long()
        hidden = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        columns, padding = self._gather_column_vectors(hidden, column_positions)
        query = self.query_token.expand(input_ids.size(0), -1, -1)
        pooled, _ = self.attn(query, columns, columns, key_padding_mask=padding)
        return F.normalize(self.mlp(pooled.squeeze(1)), dim=-1)
