from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .checkpoints import (
    load_mil_pretrained_bundle,
    normalize_label_map,
    save_mil_pretrained_bundle,
)
from .config import BGNetConfig
from .inference import prepare_raw_windows
from .model import BGNet


@dataclass(frozen=True)
class MILPredictionResult:
    probabilities: np.ndarray
    prediction: int
    attention_weights: np.ndarray
    window_start_seconds: np.ndarray | None = None
    window_stop_seconds: np.ndarray | None = None
    label_map: dict[str, int] | None = None


class BGNetMILHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, window_embeddings: Tensor, bag_slices: Tensor) -> tuple[Tensor, Tensor]:
        logits, bag_embeddings, _ = self.forward_with_attention(window_embeddings, bag_slices)
        return logits, bag_embeddings

    def forward_with_attention(
        self,
        window_embeddings: Tensor,
        bag_slices: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        bag_embeddings = []
        bag_attn = []
        for start, end in bag_slices.tolist():
            chunk = window_embeddings[int(start):int(end)]
            if chunk.numel() == 0:
                raise ValueError("Encountered an empty MIL bag.")
            scores = self.attn(chunk).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            bag_embeddings.append(torch.sum(chunk * weights[:, None], dim=0))
            bag_attn.append(weights)
        bag_tensor = torch.stack(bag_embeddings, dim=0)
        logits = self.classifier(bag_tensor)
        return logits, bag_tensor, bag_attn


class BGNetMILModel(nn.Module):
    def __init__(
        self,
        encoder: BGNet,
        mil_head: BGNetMILHead,
        *,
        label_map: Mapping[str, int] | Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mil_head = mil_head
        self.label_map = normalize_label_map(label_map)
        self.metadata = dict(metadata or {})

    @property
    def config(self) -> BGNetConfig:
        return self.encoder.config

    @classmethod
    def from_config(
        cls,
        config: BGNetConfig,
        *,
        mil_dropout: float | None = None,
        label_map: Mapping[str, int] | Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BGNetMILModel":
        encoder = BGNet.from_config(config)
        mil_head = BGNetMILHead(
            d_model=int(config.d_model),
            n_classes=int(config.n_outputs),
            dropout=float(config.dropout if mil_dropout is None else mil_dropout),
        )
        return cls(encoder, mil_head, label_map=label_map, metadata=metadata)

    @classmethod
    def from_pretrained(cls, ref: str | Path, *, strict: bool = True) -> "BGNetMILModel":
        config, encoder_state, mil_state, metadata, label_map = load_mil_pretrained_bundle(ref)
        model = cls.from_config(
            config,
            mil_dropout=float(metadata.get("mil_dropout", config.dropout)),
            label_map=label_map,
            metadata=metadata,
        )
        model.encoder.load_state_dict(encoder_state, strict=strict)
        model.mil_head.load_state_dict(mil_state, strict=strict)
        return model

    def forward_embeddings(
        self,
        window_embeddings: Tensor,
        bag_slices: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        return self.mil_head.forward_with_attention(window_embeddings, bag_slices)

    def forward_windows(
        self,
        x,
        *,
        bag_slices: Tensor,
        ch_names: Sequence[str] | None = None,
        on_missing: str = "zero",
    ) -> tuple[Tensor, list[Tensor]]:
        window_embeddings = self.encoder.forward_full(
            x,
            ch_names=ch_names,
            on_missing=on_missing,
        ).pooled
        logits, _, attention = self.mil_head.forward_with_attention(window_embeddings, bag_slices)
        return logits, attention

    @torch.inference_mode()
    def predict_bag_proba(
        self,
        x,
        *,
        bag_slices: Tensor,
        ch_names: Sequence[str] | None = None,
        on_missing: str = "zero",
    ) -> np.ndarray:
        self.eval()
        logits, _ = self.forward_windows(x, bag_slices=bag_slices, ch_names=ch_names, on_missing=on_missing)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    @torch.inference_mode()
    def predict_bag(
        self,
        x,
        *,
        bag_slices: Tensor,
        ch_names: Sequence[str] | None = None,
        on_missing: str = "zero",
    ) -> np.ndarray:
        probs = self.predict_bag_proba(x, bag_slices=bag_slices, ch_names=ch_names, on_missing=on_missing)
        return probs.argmax(axis=-1)

    @torch.inference_mode()
    def predict_raw_full(
        self,
        raw,
        *,
        window_seconds: float = 10.0,
        stride_seconds: float | None = None,
        on_missing: str = "zero",
    ) -> MILPredictionResult:
        self.eval()
        windows, ch_names, start_seconds, stop_seconds = prepare_raw_windows(
            raw,
            target_sfreq=self.config.sfreq,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        bag_slices = torch.tensor([[0, int(windows.shape[0])]], dtype=torch.long)
        logits, attention = self.forward_windows(
            windows,
            bag_slices=bag_slices,
            ch_names=ch_names,
            on_missing=on_missing,
        )
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        attn_np = attention[0].detach().cpu().numpy()
        return MILPredictionResult(
            probabilities=probs,
            prediction=int(probs.argmax(axis=-1)),
            attention_weights=attn_np,
            window_start_seconds=start_seconds,
            window_stop_seconds=stop_seconds,
            label_map=dict(self.label_map),
        )

    @torch.inference_mode()
    def predict_raw(self, raw, *, window_seconds: float = 10.0, stride_seconds: float | None = None, on_missing: str = "zero") -> int:
        return self.predict_raw_full(
            raw,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            on_missing=on_missing,
        ).prediction

    def save_pretrained(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        label_map: Mapping[str, int] | Sequence[str] | None = None,
        split_manifest_path: str | Path | None = None,
    ) -> Path:
        return save_mil_pretrained_bundle(
            path,
            encoder=self.encoder,
            mil_head=self.mil_head,
            config=self.config,
            metadata={**self.metadata, **dict(metadata or {})},
            label_map=self.label_map if label_map is None else label_map,
            split_manifest_path=split_manifest_path,
        )
