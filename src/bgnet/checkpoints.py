from __future__ import annotations

from pathlib import Path
from typing import Any
import json

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
        raise FileNotFoundError(
            f"Checkpoint '{ref}' is registered but no local artifact is configured yet."
        )
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


def convert_research_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: BGNetConfig,
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
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
                "format": "research-best-model",
                "structural_signature": structural_signature(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root
