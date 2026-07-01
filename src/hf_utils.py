import os
from pathlib import Path

from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CACHE_DIR = REPO_ROOT / "models"
PROXY_ENV_VARS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]


def sanitize_model_id(model_name_or_path: str) -> str:
    return model_name_or_path.strip("/").replace("/", "--")


def configured_model_cache_dir(cache_dir: str | None = None) -> Path | None:
    if cache_dir:
        return Path(cache_dir).expanduser()
    for env_name in ("GEN_TUS_MODEL_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env_dir = os.environ.get(env_name)
        if env_dir:
            return Path(env_dir).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return None


def default_model_cache_dir(cache_dir: str | None = None) -> Path:
    configured = configured_model_cache_dir(cache_dir)
    if configured is not None:
        return configured
    return DEFAULT_MODEL_CACHE_DIR


def local_files_only_enabled(local_files_only: bool | None = None) -> bool:
    if local_files_only is not None:
        return local_files_only
    return os.environ.get("GEN_TUS_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes", "on"}


def default_download_dir(model_name_or_path: str, cache_dir: str | None = None) -> Path:
    return default_model_cache_dir(cache_dir) / sanitize_model_id(model_name_or_path)


def resolve_model_source(model_name_or_path: str, cache_dir: str | None = None) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)

    repo_local_path = DEFAULT_MODEL_CACHE_DIR / sanitize_model_id(model_name_or_path)
    if repo_local_path.exists():
        return str(repo_local_path)

    if os.path.sep in model_name_or_path or model_name_or_path.startswith("."):
        return model_name_or_path

    cached_path = default_download_dir(model_name_or_path, cache_dir)
    if cached_path.exists():
        return str(cached_path)

    return model_name_or_path


def transformers_load_kwargs(cache_dir: str | None = None, local_files_only: bool | None = None) -> dict:
    kwargs = {}
    configured = configured_model_cache_dir(cache_dir)
    if configured is not None:
        kwargs["cache_dir"] = str(configured)
    if local_files_only_enabled(local_files_only):
        kwargs["local_files_only"] = True
    return kwargs


def clear_proxy_env() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)


def format_load_error(model_name_or_path: str, resolved_source: str, cache_dir: str | None, exc: Exception) -> str:
    model_path = Path(model_name_or_path).expanduser()
    lines = [f"Failed to load model '{model_name_or_path}' (resolved to '{resolved_source}')."]

    if model_path.exists():
        lines.append("The local model directory exists, but required weight files are missing or incomplete.")
    else:
        target_dir = default_download_dir(model_name_or_path, cache_dir)
        lines.extend([
            f"Recommended local cache directory: {target_dir}",
            "Download first with: python scripts/download_hf_model.py --model "
            f"{model_name_or_path}",
            f"Then retry with: --model {target_dir} --local_files_only",
        ])

    if any(os.environ.get(key) for key in PROXY_ENV_VARS):
        lines.append(
            "Proxy variables are set in the current shell. If downloads fail, unset "
            "http_proxy/https_proxy/all_proxy or run the download script with --proxy off."
        )

    for env_name in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            lines.append(f"Current {env_name}={value}")

    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        lines.append(f"Current HF_ENDPOINT={endpoint}")

    lines.append(f"Original error: {exc}")
    return "\n".join(lines)


def load_t5_components(model_name_or_path: str, cache_dir: str | None = None,
                       local_files_only: bool | None = None):
    model_source = resolve_model_source(model_name_or_path, cache_dir)
    load_kwargs = transformers_load_kwargs(cache_dir, local_files_only)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(format_load_error(model_name_or_path, model_source, cache_dir, exc)) from exc

    try:
        model = T5ForConditionalGeneration.from_pretrained(model_source, **load_kwargs)
        return model, tokenizer
    except Exception as exc:
        flax_weights = Path(model_source) / "flax_model.msgpack"
        if Path(model_source).is_dir() and flax_weights.exists():
            try:
                from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                config = T5Config.from_pretrained(model_source, **load_kwargs)
                model = T5ForConditionalGeneration(config)
                model = load_flax_checkpoint_in_pytorch_model(model, flax_weights)
                return model, tokenizer
            except ModuleNotFoundError as flax_exc:
                message = format_load_error(model_name_or_path, model_source, cache_dir, exc)
                message += "\nDetected flax_model.msgpack in the local directory."
                message += "\nInstall `jax` and `flax`, then retry to auto-convert Flax weights."
                raise RuntimeError(message) from flax_exc
            except Exception as flax_exc:
                message = format_load_error(model_name_or_path, model_source, cache_dir, exc)
                message += "\nDetected flax_model.msgpack in the local directory."
                message += f"\nAutomatic Flax -> PyTorch conversion failed: {flax_exc}"
                raise RuntimeError(message) from flax_exc

        raise RuntimeError(format_load_error(model_name_or_path, model_source, cache_dir, exc)) from exc
