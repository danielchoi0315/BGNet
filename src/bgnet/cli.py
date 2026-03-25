from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import available_checkpoints, download_pretrained
from .model import BGNet


def env_check_main() -> None:
    import torch

    payload = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "available_checkpoints": list(available_checkpoints().keys()),
    }
    print(json.dumps(payload, indent=2))


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download a named BGNet checkpoint bundle into the local cache.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = download_pretrained(args.checkpoint, force=args.force)
    print(json.dumps({"checkpoint": args.checkpoint, "path": str(root)}, indent=2))


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Run BGNet prediction from a checkpoint and .npy input.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to a .npy array shaped [C, T] or [B, C, T].")
    parser.add_argument("--ch-names", nargs="+", default=None)
    args = parser.parse_args()

    model = BGNet.from_pretrained(args.checkpoint)
    x = np.load(Path(args.input))
    probs = model.predict_proba(x, ch_names=args.ch_names)
    print(json.dumps({"probabilities": probs.tolist(), "predictions": probs.argmax(axis=-1).tolist()}, indent=2))


def predict_raw_main() -> None:
    parser = argparse.ArgumentParser(description="Run BGNet prediction from an EDF file using sliding-window aggregation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--edf", required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Raw EDF prediction requires `mne`. Install with `pip install bgnet-eeg[braindecode]`.") from exc

    model = BGNet.from_pretrained(args.checkpoint)
    raw = mne.io.read_raw_edf(args.edf, preload=True)
    result = model.predict_raw_full(
        raw,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        aggregation=args.aggregation,
    )
    print(
        json.dumps(
            {
                "prediction": result.prediction,
                "probabilities": result.probabilities.tolist(),
                "window_probabilities": result.window_probabilities.tolist(),
                "window_start_seconds": result.window_start_seconds.tolist(),
                "window_stop_seconds": result.window_stop_seconds.tolist(),
            },
            indent=2,
        )
    )
