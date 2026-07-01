import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput

try:
    from hf_utils import load_t5_components
except ModuleNotFoundError:
    from src.hf_utils import load_t5_components

SPECIAL_TOKENS = ["[COL]", "[VAL]"]
# Dedicated suffix tokens to avoid collision with codebook tokens 0..codebook_size-1.
# SANTOS_large already needs suffixes > 63, so keep a larger safety margin.
MAX_COLLISION_SUFFIX = 256
SUFFIX_TOKENS = [f"[SFX{i}]" for i in range(MAX_COLLISION_SUFFIX)]


def build_model(model_name: str, codebook_size: int, cache_dir: str | None = None,
                local_files_only: bool | None = None):
    """Load T5 model and tokenizer, extending the vocabulary with Semantic ID tokens."""
    model, tokenizer = load_t5_components(model_name, cache_dir, local_files_only)

    new_tokens = [str(i) for i in range(codebook_size)] + SPECIAL_TOKENS + SUFFIX_TOKENS
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


class TUSDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_input: int = 512, max_target: int = 20):
        """records: list of {"input_text": str, "target_text": str}"""
        self.records = records
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["input_text"],
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            rec["target_text"],
            max_length=self.max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
        }


class RankingTUSDataset(TUSDataset):
    """TUS dataset with optional negative Semantic-ID targets for pairwise loss."""

    def _target_labels(self, text: str):
        target = self.tokenizer(
            text,
            max_length=self.max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze()
        labels = target.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        rec = self.records[idx]
        neg_text = rec.get("negative_text")
        if neg_text:
            item["negative_labels"] = self._target_labels(neg_text)
            item["has_negative"] = torch.tensor(1, dtype=torch.long)
        else:
            item["negative_labels"] = torch.full_like(item["labels"], -100)
            item["has_negative"] = torch.tensor(0, dtype=torch.long)
        return item


def sequence_log_prob_from_logits(logits, labels):
    vocab_size = logits.size(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.size())
    mask = (labels != -100).float()
    token_count = mask.sum(dim=1).clamp_min(1.0)
    return -(token_loss * mask).sum(dim=1) / token_count


def sequence_log_prob(model, input_ids, attention_mask, labels):
    decoder_input_ids = model._shift_right(labels)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    return sequence_log_prob_from_logits(outputs.logits, labels)


def sequence_log_prob_from_encoder(model, encoder_hidden_state, attention_mask, labels):
    decoder_input_ids = model._shift_right(labels)
    outputs = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_state),
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    )
    return sequence_log_prob_from_logits(outputs.logits, labels)


class RankingTrainer(Trainer):
    """Trainer that adds pairwise sequence log-prob ranking loss.

    The base CE loss still trains query -> positive Semantic ID. When a batch
    contains negative labels, the ranking term enforces:
      log P(pos_sid | query) >= log P(neg_sid | query) + margin
    """

    def __init__(self, *args, ranking_weight: float = 0.2, ranking_margin: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        negative_labels = inputs.pop("negative_labels", None)
        has_negative = inputs.pop("has_negative", None)
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss

        if self.ranking_weight > 0 and negative_labels is not None and has_negative is not None:
            active = has_negative.bool()
            if active.any():
                input_ids = inputs["input_ids"][active]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask[active]
                pos_labels = inputs["labels"][active]
                neg_labels = negative_labels[active]
                pos_score = sequence_log_prob_from_logits(outputs.logits[active], pos_labels)
                encoder_hidden_state = getattr(outputs, "encoder_last_hidden_state", None)
                if encoder_hidden_state is None:
                    neg_score = sequence_log_prob(model, input_ids, attention_mask, neg_labels)
                else:
                    neg_score = sequence_log_prob_from_encoder(
                        model,
                        encoder_hidden_state[active],
                        attention_mask,
                        neg_labels,
                    )
                rank_loss = F.relu(self.ranking_margin - pos_score + neg_score).mean()
                loss = loss + self.ranking_weight * rank_loss

        return (loss, outputs) if return_outputs else loss
