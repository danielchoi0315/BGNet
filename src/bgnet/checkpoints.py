from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile

import torch
from safetensors.torch import load_file, save_file

from .config import BGNetConfig

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
METADATA_NAME = "metadata.json"
CHANNELS_NAME = "channels.json"
REGISTRY_NAME = "checkpoints.json"


def _registry_path() -> Path:
    from importlib import resources

    return resources.files("bgnet.data").joinpath(REGISTRY_NAME)


def available_checkpoints() -> dict[str, Any]:
    text = _registry_path().read_text(encoding="utf-8")
    return json.loads(text)


def checkpoint_cache_dir() -> Path:
    root = os.environ.get("BGNET_HOME")
    if root:
        return Path(root).expanduser().resolve() / "checkpoints"
    return Path.home() / ".cache" / "bgnet" / "checkpoints"


def _resolve_ref(ref: str | Path) -> Path:
    path = Path(ref)
    if path.exists():
        return path
    registry = available_checkpoints()
    if str(ref) in registry:
        entry = registry[str(ref)]
        local = entry.get("local_path")
        if local and Path(local).exists():
            return Path(local)
        return download_pretrained(str(ref))
    raise FileNotFoundError(f"Unknown checkpoint ref: {ref}")


def structural_signature(config: BGNetConfig) -> dict[str, Any]:
    resolved = config.resolved()
    return {
        "preset": resolved.preset,
        "n_outputs": resolved.n_outputs,
        "ch_names": list(resolved.ch_names),
        "source_names": list(resolved.source_names),
        "sfreq": resolved.sfreq,
        "time_window_size": resolved.time_window_size,
        "time_window_stride": resolved.time_window_stride,
        "d_model": resolved.d_model,
        "osc_depth": resolved.osc_depth,
        "n_heads": resolved.n_heads,
        "dropout": resolved.dropout,
        "low_rank": resolved.low_rank,
        "source_graph_sigma": resolved.source_graph_sigma,
        "source_graph_self_weight": resolved.source_graph_self_weight,
        "attention_sigma": resolved.attention_sigma,
        "use_pair_expert": resolved.use_pair_expert,
        "use_event_expert": resolved.use_event_expert,
        "use_artifact_expert": resolved.use_artifact_expert,
    }


def load_pretrained_bundle(ref: str | Path) -> tuple[BGNetConfig, dict[str, torch.Tensor], dict[str, Any]]:
    root = _resolve_ref(ref)
    config = BGNetConfig.from_dict(json.loads((root / CONFIG_NAME).read_text(encoding="utf-8")))
    state_dict = load_file(root / WEIGHTS_NAME)
    metadata_path = root / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    expected = metadata.get("structural_signature")
    if expected is not None and expected != structural_signature(config):
        raise ValueError("Checkpoint metadata structural signature does not match config.json.")
    return config, state_dict, metadata


def save_pretrained_bundle(
    path: str | Path,
    *,
    model,
    config: BGNetConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().clone().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state_dict, root / WEIGHTS_NAME)
    (root / CONFIG_NAME).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (root / CHANNELS_NAME).write_text(json.dumps(list(config.ch_names), indent=2), encoding="utf-8")
    payload = dict(metadata or {})
    payload["structural_signature"] = structural_signature(config)
    (root / METADATA_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def download_pretrained(ref: str, *, force: bool = False) -> Path:
    registry = available_checkpoints()
    if ref not in registry:
        raise FileNotFoundError(f"Unknown checkpoint ref: {ref}")
    entry = registry[ref]
    cache_root = checkpoint_cache_dir() / ref
    if not force and _bundle_complete(cache_root):
        return cache_root
    if "hf_repo_id" in entry:
        return _download_from_huggingface(ref, entry, cache_root, force=force)
    if "url" in entry:
        return _download_from_url(ref, entry, cache_root, force=force)
    local = entry.get("local_path")
    if local and Path(local).exists():
        return Path(local)
    raise FileNotFoundError(f"Checkpoint '{ref}' has no resolvable local_path, url, or hf_repo_id.")


def convert_research_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: BGNetConfig,
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "mil_head_state" in ckpt:
            raise ValueError(
                "This checkpoint appears to be a record-level MIL checkpoint with a separate mil_head_state. "
                "BGNet public bundles currently support encoder/window checkpoints only."
            )
        for key in ("model_state", "encoder_state", "state_dict"):
            maybe = ckpt.get(key)
            if isinstance(maybe, dict) and maybe:
                state_dict = maybe
                break
        else:
            if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
                state_dict = ckpt
            else:
                raise ValueError("Unsupported research checkpoint format.")
    else:
        raise ValueError("Unsupported research checkpoint format.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, root / WEIGHTS_NAME)
    (root / CONFIG_NAME).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (root / CHANNELS_NAME).write_text(json.dumps(list(config.ch_names), indent=2), encoding="utf-8")
    (root / METADATA_NAME).write_text(
        json.dumps(
            {
                "converted_from": str(checkpoint_path),
                "format": "research-encoder-checkpoint",
                "structural_signature": structural_signature(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _bundle_complete(path: Path) -> bool:
    required = (CONFIG_NAME, WEIGHTS_NAME, METADATA_NAME, CHANNELS_NAME)
    return all((path / name).exists() for name in required)


def _download_from_huggingface(ref: str, entry: dict[str, Any], cache_root: Path, *, force: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"Checkpoint '{ref}' requires huggingface_hub. Install with `pip install bgnet-eeg[hub]`."
        ) from exc

    cache_root.mkdir(parents=True, exist_ok=True)
    repo_id = str(entry["hf_repo_id"])
    revision = entry.get("revision")
    for filename in (CONFIG_NAME, WEIGHTS_NAME, METADATA_NAME, CHANNELS_NAME):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=cache_root,
            force_download=force,
            local_dir_use_symlinks=False,
        )
    return cache_root


def _download_from_url(ref: str, entry: dict[str, Any], cache_root: Path, *, force: bool) -> Path:
    url = str(entry["url"])
    archive_name = entry.get("archive_name") or Path(url).name or f"{ref}.zip"
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root.parent / archive_name
    if force or not archive_path.exists():
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    expected_sha256 = entry.get("sha256")
    if expected_sha256 is not None:
        actual = _sha256_file(archive_path)
        if actual != expected_sha256:
            raise ValueError(f"Checkpoint archive hash mismatch for '{ref}': expected {expected_sha256}, got {actual}")
    if force and cache_root.exists():
        shutil.rmtree(cache_root)
    _extract_bundle_archive(archive_path, cache_root)
    return cache_root


def _extract_bundle_archive(archive_path: Path, cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bgnet_bundle_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_path)
        extracted_root = _single_directory_or_self(tmp_path)
        for item in extracted_root.iterdir():
            destination = cache_root / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
    if not _bundle_complete(cache_root):
        raise ValueError(f"Downloaded archive {archive_path} did not contain a valid BGNet bundle.")


def _single_directory_or_self(path: Path) -> Path:
    items = list(path.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
