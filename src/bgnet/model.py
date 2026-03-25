from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .checkpoints import load_pretrained_bundle, save_pretrained_bundle
from .config import BGNetConfig
from .factory import build_bgnet_core
from .runtime import resolve_input_array, sensor_geometry_tensors
from ._core_adaptive import AdaptiveModelOutput as BGNetOutput


@dataclass(frozen=True)
class PredictionResult:
    labels: np.ndarray
    probabilities: np.ndarray


class BGNet(nn.Module):
    def __init__(self, config: BGNetConfig) -> None:
        super().__init__()
        self.config = config.resolved()
        self.core = build_bgnet_core(self.config)

    @classmethod
    def from_config(cls, config: BGNetConfig) -> "BGNet":
        return cls(config)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        n_outputs: int,
        ch_names: Sequence[str],
        sfreq: float,
        **overrides: Any,
    ) -> "BGNet":
        return cls(BGNetConfig.from_preset(preset, n_outputs=n_outputs, ch_names=ch_names, sfreq=sfreq, **overrides))

    @classmethod
    def from_pretrained(cls, ref: str | Path, *, strict: bool = True) -> "BGNet":
        config, state_dict, _ = load_pretrained_bundle(ref)
        model = cls(config)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def forward_full(
        self,
        x,
        *,
        ch_names: Sequence[str] | None = None,
        mask_ratio: float = 0.0,
    ) -> BGNetOutput:
        batch = resolve_input_array(x, config=self.config, ch_names=ch_names)
        batch = batch.to(next(self.parameters()).device)
        sensor_pos, sensor_mask = sensor_geometry_tensors(
            self.config,
            batch_size=batch.shape[0],
            device=batch.device,
        )
        return self.core(batch, sensor_pos=sensor_pos, sensor_mask=sensor_mask, mask_ratio=mask_ratio)

    def forward(self, x, *, ch_names: Sequence[str] | None = None, mask_ratio: float = 0.0) -> Tensor:
        return self.forward_full(x, ch_names=ch_names, mask_ratio=mask_ratio).logits

    @torch.inference_mode()
    def predict_proba(self, raw_or_x, *, ch_names: Sequence[str] | None = None) -> np.ndarray:
        self.eval()
        logits = self.forward(raw_or_x, ch_names=ch_names)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    @torch.inference_mode()
    def predict(self, raw_or_x, *, ch_names: Sequence[str] | None = None) -> np.ndarray:
        probs = self.predict_proba(raw_or_x, ch_names=ch_names)
        return probs.argmax(axis=-1)

    def save_pretrained(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        return save_pretrained_bundle(path, model=self, config=self.config, metadata=metadata)

