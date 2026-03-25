from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import List, Sequence, Tuple
import json

import numpy as np

DEFAULT_SOURCE_NAMES: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "CP5", "CP3", "CP1", "CP2", "CP4", "CP6",
    "P3", "Pz", "P4", "Oz",
)


@dataclass(frozen=True)
class SourceLayout:
    names: Tuple[str, ...]
    positions: np.ndarray


def canonicalize_channel_name(name: str) -> str:
    n = name.strip()
    for prefix in ("EEG ", "eeg ", "EEG", "eeg"):
        if n.startswith(prefix):
            n = n[len(prefix):].strip()
    n = n.replace("-REF", "").replace("-LE", "").replace("-A1", "").replace("-A2", "")
    n = n.replace(" ", "")
    aliases = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "CZ": "Cz",
        "PZ": "Pz",
        "FZ": "Fz",
        "OZ": "Oz",
        "FP1": "Fp1",
        "FP2": "Fp2",
    }
    if n.upper() in aliases:
        return aliases[n.upper()]
    if len(n) >= 2 and n[0].isalpha():
        return n[0].upper() + n[1:]
    return n


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, list[float]]:
    text = resources.files("bgnet.data").joinpath("standard_1005_positions.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    return {str(k): list(v) for k, v in payload["positions"].items()}


def get_channel_positions(
    ch_names: Sequence[str],
    montage_name: str = "standard_1005",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if montage_name != "standard_1005":
        raise ValueError("BGNet currently ships a frozen standard_1005 geometry only.")
    lookup = _standard_positions()
    positions: List[np.ndarray] = []
    valid: List[bool] = []
    normalized: List[str] = []
    for name in ch_names:
        clean = canonicalize_channel_name(name)
        normalized.append(clean)
        if clean in lookup:
            positions.append(np.asarray(lookup[clean], dtype=np.float32))
            valid.append(True)
        else:
            positions.append(np.zeros(3, dtype=np.float32))
            valid.append(False)
    return np.stack(positions, axis=0), np.asarray(valid, dtype=np.bool_), normalized


def build_default_source_layout(
    source_names: Sequence[str] = DEFAULT_SOURCE_NAMES,
    montage_name: str = "standard_1005",
    inward_scale: float = 0.85,
) -> SourceLayout:
    positions, valid, names = get_channel_positions(source_names, montage_name=montage_name)
    if not valid.all():
        missing = [name for name, ok in zip(names, valid) if not ok]
        raise ValueError(f"Missing source coordinates for names: {missing}")
    return SourceLayout(
        names=tuple(names),
        positions=(positions * float(inward_scale)).astype(np.float32),
    )


def gaussian_distance_bias(
    query_pos: np.ndarray,
    key_pos: np.ndarray,
    sigma: float = 0.35,
) -> np.ndarray:
    q = np.asarray(query_pos, dtype=np.float32)
    k = np.asarray(key_pos, dtype=np.float32)
    dist2 = ((q[:, None, :] - k[None, :, :]) ** 2).sum(axis=-1)
    sigma2 = max(float(sigma) ** 2, 1e-6)
    return -dist2 / (2.0 * sigma2)


def normalized_source_adjacency(
    source_pos: np.ndarray,
    sigma: float = 0.25,
    self_weight: float = 1.0,
) -> np.ndarray:
    bias = gaussian_distance_bias(source_pos, source_pos, sigma=sigma)
    adj = np.exp(bias).astype(np.float32)
    np.fill_diagonal(adj, np.diag(adj) + self_weight)
    deg = adj.sum(axis=-1, keepdims=True)
    deg = np.maximum(deg, 1e-6)
    adj = adj / deg
    return adj.astype(np.float32)

