from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RawPredictionResult:
    probabilities: np.ndarray
    prediction: int
    window_probabilities: np.ndarray
    window_start_seconds: np.ndarray
    window_stop_seconds: np.ndarray


def mne_info_channel_names(info: Any) -> list[str]:
    if hasattr(info, "ch_names"):
        return [str(name) for name in info.ch_names]
    if isinstance(info, dict) and "chs" in info:
        return [str(ch["ch_name"]) for ch in info["chs"]]
    raise ValueError("Could not extract channel names from info; provide an MNE info-like object.")


def mne_info_sfreq(info: Any) -> float:
    if hasattr(info, "__getitem__"):
        try:
            return float(info["sfreq"])
        except Exception:
            pass
    if hasattr(info, "sfreq"):
        return float(info.sfreq)
    raise ValueError("Could not extract sampling rate from info; provide an MNE info-like object.")


def prepare_raw_windows(
    raw: Any,
    *,
    target_sfreq: float,
    window_seconds: float,
    stride_seconds: float | None = None,
) -> tuple[np.ndarray, Sequence[str], np.ndarray, np.ndarray]:
    if not hasattr(raw, "get_data") or not hasattr(raw, "ch_names") or not hasattr(raw, "info"):
        raise TypeError("predict_raw_* expects an MNE Raw-like object with get_data(), ch_names, and info.")

    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if stride_seconds is not None and stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")

    raw_obj = raw.copy() if hasattr(raw, "copy") else raw
    if hasattr(raw_obj, "load_data"):
        loaded = raw_obj.load_data()
        if loaded is not None:
            raw_obj = loaded

    current_sfreq = mne_info_sfreq(raw_obj.info)
    if abs(current_sfreq - target_sfreq) > 1e-6:
        if not hasattr(raw_obj, "resample"):
            raise ValueError(
                f"Raw object sampling rate {current_sfreq} does not match BGNet config {target_sfreq}, "
                "and the object does not support resample()."
            )
        resampled = raw_obj.resample(target_sfreq, npad="auto")
        if resampled is not None:
            raw_obj = resampled
        current_sfreq = mne_info_sfreq(raw_obj.info)

    x = np.asarray(raw_obj.get_data(), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected raw.get_data() to return [C, T], got {tuple(x.shape)}")

    ch_names = list(raw_obj.ch_names)
    window_size = max(1, int(round(window_seconds * current_sfreq)))
    stride = max(1, int(round((stride_seconds or window_seconds) * current_sfreq)))

    starts = _window_starts(n_samples=x.shape[1], window_size=window_size, stride=stride)
    windows = np.stack([_slice_or_pad(x, start, window_size) for start in starts], axis=0)
    start_seconds = starts.astype(np.float32) / float(current_sfreq)
    stop_seconds = (starts + window_size).astype(np.float32) / float(current_sfreq)
    return windows, ch_names, start_seconds, stop_seconds


def aggregate_probabilities(window_probabilities: np.ndarray, *, method: str = "mean") -> np.ndarray:
    probs = np.asarray(window_probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"Expected [W, K] probabilities, got {tuple(probs.shape)}")
    if method == "mean":
        return probs.mean(axis=0)
    if method == "max":
        return probs.max(axis=0)
    raise ValueError(f"Unsupported aggregation method: {method}")


def _window_starts(n_samples: int, window_size: int, stride: int) -> np.ndarray:
    if n_samples <= window_size:
        return np.asarray([0], dtype=np.int64)
    starts = list(range(0, max(1, n_samples - window_size + 1), stride))
    last = n_samples - window_size
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def _slice_or_pad(x: np.ndarray, start: int, window_size: int) -> np.ndarray:
    stop = start + window_size
    window = x[:, start:stop]
    if window.shape[1] == window_size:
        return window
    out = np.zeros((x.shape[0], window_size), dtype=np.float32)
    out[:, : window.shape[1]] = window
    return out
