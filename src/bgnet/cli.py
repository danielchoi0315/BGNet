from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints import available_checkpoints
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

