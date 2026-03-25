from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .config import BGNetConfig
from .geometry import canonicalize_channel_name, get_channel_positions


def to_tensor(x: np.ndarray | Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def ensure_bct(x: np.ndarray | Tensor) -> Tensor:
    tensor = to_tensor(x)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected shape [B, C, T] or [C, T], got {tuple(tensor.shape)}")
    return tensor


def canonical_channel_list(ch_names: Sequence[str]) -> list[str]:
    return [canonicalize_channel_name(name) for name in ch_names]


@lru_cache(maxsize=128)
def _cached_layout(ch_names: tuple[str, ...], montage_name: str) -> tuple[np.ndarray, np.ndarray]:
    pos_np, mask_np, _ = get_channel_positions(ch_names, montage_name=montage_name)
    return pos_np, mask_np


def reorder_channels(x: Tensor, input_ch_names: Sequence[str], target_ch_names: Sequence[str]) -> Tensor:
    input_names = canonical_channel_list(input_ch_names)
    target_names = canonical_channel_list(target_ch_names)
    index = {name: i for i, name in enumerate(input_names)}
    missing = [name for name in target_names if name not in index]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")
    order = [index[name] for name in target_names]
    return x[:, order, :]


def resolve_input_array(
    raw_or_x,
    *,
    config: BGNetConfig,
    ch_names: Sequence[str] | None = None,
) -> Tensor:
    if hasattr(raw_or_x, "get_data") and hasattr(raw_or_x, "ch_names"):
        x = ensure_bct(raw_or_x.get_data())
        raw_ch_names = list(raw_or_x.ch_names)
        return reorder_channels(x, raw_ch_names, config.ch_names)

    x = ensure_bct(raw_or_x)
    if ch_names is None:
        if x.shape[1] != len(config.ch_names):
            raise ValueError(
                f"Input has {x.shape[1]} channels but config expects {len(config.ch_names)}. "
                "Provide ch_names to reorder explicitly."
            )
        return x
    return reorder_channels(x, ch_names, config.ch_names)


def sensor_geometry_tensors(
    config: BGNetConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    pos_np, mask_np = _cached_layout(tuple(config.ch_names), config.montage_name)
    sensor_pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    sensor_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1)
    return sensor_pos, sensor_mask
