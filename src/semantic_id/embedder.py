import numpy as np

try:
    from hf_utils import configured_model_cache_dir, format_load_error, local_files_only_enabled, resolve_model_source
except ModuleNotFoundError:
    from src.hf_utils import configured_model_cache_dir, format_load_error, local_files_only_enabled, resolve_model_source

_MODELS = {}


def _get_sentence_transformer(model_name, cache_dir=None, local_files_only=None, device=None):
    from sentence_transformers import SentenceTransformer
    model_source = resolve_model_source(model_name, cache_dir)
    configured_cache = configured_model_cache_dir(cache_dir)
    cache_root = str(configured_cache) if configured_cache is not None else None
    local_only = local_files_only_enabled(local_files_only)
    cache_key = (model_source, cache_root, local_only, device)
    if cache_key not in _MODELS:
        try:
            load_kwargs = {"local_files_only": local_only}
            if cache_root is not None:
                load_kwargs["cache_folder"] = cache_root
            if device is not None:
                load_kwargs["device"] = device
            _MODELS[cache_key] = SentenceTransformer(model_source, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(format_load_error(model_name, model_source, cache_dir, exc)) from exc
    return _MODELS[cache_key]


def embed(texts: list, batch_size: int = 64, model_name: str = "sentence-transformers/all-mpnet-base-v2",
          cache_dir: str = None, local_files_only: bool = None,
          device: str = None) -> np.ndarray:
    model = _get_sentence_transformer(model_name, cache_dir, local_files_only, device=device)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
