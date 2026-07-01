import os
import sys
import torch
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from data.serializer import serialize
try:
    from hf_utils import load_t5_components
except ModuleNotFoundError:
    from src.hf_utils import load_t5_components
from semantic_id.manager import SemanticIDManager
from model.constrained import build_trie_from_ids, SemanticIDConstrainedDecoder


_DIVERSE_BEAM_WARNED = False


def load_model(model_dir: str, cache_dir: str | None = None, local_files_only: bool | None = None):
    model, tokenizer = load_t5_components(model_dir, cache_dir, local_files_only)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def _supports_local_group_beam_search(model) -> bool:
    """Return whether this transformers install can run diverse beam search locally."""
    try:
        from transformers.generation.configuration_utils import GenerationMode
        from transformers.generation.utils import GENERATION_MODES_MAPPING
    except Exception:
        return hasattr(type(model), "group_beam_search") or hasattr(type(model), "_group_beam_search")

    method_name = GENERATION_MODES_MAPPING.get(GenerationMode.GROUP_BEAM_SEARCH)
    if not method_name or "/" in method_name:
        return False
    return hasattr(type(model), method_name)


def _remote_group_beam_enabled() -> bool:
    return os.environ.get("GEN_TUS_REMOTE_GROUP_BEAM", "0").lower() in {"1", "true", "yes", "on"}


def _resolve_beam_groups(model, beam_size: int, num_beam_groups: int) -> int:
    """Choose a generation-compatible beam group count."""
    global _DIVERSE_BEAM_WARNED

    if num_beam_groups <= 1:
        return 1
    if beam_size % num_beam_groups != 0:
        if not _DIVERSE_BEAM_WARNED:
            warnings.warn(
                f"num_beam_groups={num_beam_groups} must divide beam_size={beam_size}; "
                "falling back to regular beam search.",
                RuntimeWarning,
            )
            _DIVERSE_BEAM_WARNED = True
        return 1
    if _supports_local_group_beam_search(model):
        return num_beam_groups

    if not _DIVERSE_BEAM_WARNED:
        warnings.warn(
            "Diverse beam search is not available in this transformers install; "
            "falling back to regular beam search.",
            RuntimeWarning,
        )
        _DIVERSE_BEAM_WARNED = True
    return 1


def retrieve(df, model, tokenizer, sid_manager: SemanticIDManager,
             beam_size: int = 100, max_new_tokens: int = 20,
             trie=None, valid_toks=None,
             num_beam_groups: int = 1, diversity_penalty: float = 0.5,
             exclude_tables: set[str] | None = None,
             max_cols: int = 0, max_rows: int = 24,
             serializer_token_budget: int = 256,
             include_column_names: bool = False,
             decoding_mode: str = "constrained",
             return_generation_stats: bool = False) -> list[str] | tuple[list[str], dict]:
    """Return ranked list of candidate table names for a query DataFrame."""
    text = serialize(
        df,
        max_cols=max_cols,
        max_rows=max_rows,
        include_column_names=include_column_names,
        token_budget=serializer_token_budget,
    )
    inputs = tokenizer(text, return_tensors="pt", max_length=512,
                       truncation=True).to(model.device)

    if decoding_mode not in {"constrained", "unconstrained"}:
        raise ValueError(f"unknown decoding_mode: {decoding_mode}")
    if trie is None or valid_toks is None:
        trie, valid_toks = build_trie_from_ids(sid_manager.all_ids(), tokenizer)
    processor = None
    if decoding_mode == "constrained":
        processor = SemanticIDConstrainedDecoder(trie, valid_toks, tokenizer.eos_token_id)

    groups = _resolve_beam_groups(model, beam_size, num_beam_groups)

    gen_kwargs = dict(
        num_beams=beam_size,
        num_return_sequences=beam_size,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
    )
    if processor is not None:
        gen_kwargs["logits_processor"] = [processor]
    if groups > 1:
        gen_kwargs["num_beam_groups"] = groups
        gen_kwargs["diversity_penalty"] = diversity_penalty

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    import re
    skip_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id} - {None}
    exclude_tables = exclude_tables or set()
    generation_stats = {
        "generated_sequences": 0,
        "parseable_sid_sequences": 0,
        "mapped_sid_sequences": 0,
        "malformed_sid_sequences": 0,
        "unmapped_sid_sequences": 0,
        "excluded_sid_sequences": 0,
        "duplicate_sid_sequences": 0,
        "unique_ranked_tables": 0,
    }
    seen, ranked = set(), []
    for out in outputs:
        generation_stats["generated_sequences"] += 1
        try:
            sid_parts = []
            for tok_id in out.tolist():
                if tok_id in skip_ids:
                    continue
                s = tokenizer.decode([tok_id], skip_special_tokens=False)
                m = re.fullmatch(r'\[SFX(\d+)\]', s)
                if m:
                    sid_parts.append(-(int(m.group(1)) + 1))
                else:
                    sid_parts.append(int(s))
            sid = tuple(sid_parts)
        except ValueError:
            generation_stats["malformed_sid_sequences"] += 1
            continue
        generation_stats["parseable_sid_sequences"] += 1
        table = sid_manager.id_to_table(sid)
        if not table:
            generation_stats["unmapped_sid_sequences"] += 1
            continue
        generation_stats["mapped_sid_sequences"] += 1
        if table in exclude_tables:
            generation_stats["excluded_sid_sequences"] += 1
            continue
        if table in seen:
            generation_stats["duplicate_sid_sequences"] += 1
            continue
        seen.add(table)
        ranked.append(table)
    generation_stats["unique_ranked_tables"] = len(ranked)
    generated = generation_stats["generated_sequences"]
    generation_stats["parseable_sid_rate"] = (
        generation_stats["parseable_sid_sequences"] / generated if generated else 0.0
    )
    generation_stats["valid_sid_rate"] = (
        generation_stats["mapped_sid_sequences"] / generated if generated else 0.0
    )
    if return_generation_stats:
        return ranked, generation_stats
    return ranked
