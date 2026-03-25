from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import BGNet
from .runtime import resolve_input_array


@dataclass
class FitHistory:
    train_loss: list[float]
    val_loss: list[float]


class BGNetClassifier:
    def __init__(self, model: BGNet, *, device: str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.history = FitHistory(train_loss=[], val_loss=[])

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        n_outputs: int,
        ch_names: Sequence[str],
        sfreq: float,
        **overrides: Any,
    ) -> "BGNetClassifier":
        return cls(BGNet.from_preset(preset, n_outputs=n_outputs, ch_names=ch_names, sfreq=sfreq, **overrides))

    @classmethod
    def from_pretrained(cls, ref, *, strict: bool = True, device: str | None = None) -> "BGNetClassifier":
        return cls(BGNet.from_pretrained(ref, strict=strict), device=device)

    def fit(
        self,
        X_train,
        y_train,
        *,
        X_val=None,
        y_val=None,
        ch_names: Sequence[str] | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> FitHistory:
        train_x = resolve_input_array(X_train, config=self.model.config, ch_names=ch_names)
        train_y = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_x = resolve_input_array(X_val, config=self.model.config, ch_names=ch_names)
            val_y = torch.as_tensor(np.asarray(y_val), dtype=torch.long)
            val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

        optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(int(epochs)):
            self.model.train()
            total = 0.0
            n = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                total += float(loss.detach().cpu()) * int(xb.shape[0])
                n += int(xb.shape[0])
            self.history.train_loss.append(total / max(n, 1))

            if val_loader is None:
                continue

            self.model.eval()
            total = 0.0
            n = 0
            with torch.inference_mode():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    total += float(loss.detach().cpu()) * int(xb.shape[0])
                    n += int(xb.shape[0])
            self.history.val_loss.append(total / max(n, 1))
        return self.history

    def evaluate(self, X, y, *, ch_names: Sequence[str] | None = None) -> dict[str, float]:
        x = resolve_input_array(X, config=self.model.config, ch_names=ch_names)
        y_true = np.asarray(y)
        probs = self.predict_proba(x)
        pred = probs.argmax(axis=-1)
        metrics = {
            "accuracy": float((pred == y_true).mean()),
        }
        try:
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score

            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
            if probs.shape[1] == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            pass
        return metrics

    def predict(self, X, *, ch_names: Sequence[str] | None = None) -> np.ndarray:
        return self.model.predict(X, ch_names=ch_names)

    def predict_proba(self, X, *, ch_names: Sequence[str] | None = None) -> np.ndarray:
        self.model.to(self.device)
        return self.model.predict_proba(X, ch_names=ch_names)

    def save_pretrained(self, path, *, metadata: dict[str, Any] | None = None):
        return self.model.save_pretrained(path, metadata=metadata)
