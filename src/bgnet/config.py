from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from .geometry import DEFAULT_SOURCE_NAMES

_PRESET_DEFAULTS = {
    "rest": dict(use_pair_expert=False, use_event_expert=False, use_artifact_expert=True),
    "resting": dict(use_pair_expert=False, use_event_expert=False, use_artifact_expert=True),
    "mi": dict(use_pair_expert=True, use_event_expert=False, use_artifact_expert=False),
    "clinical": dict(use_pair_expert=False, use_event_expert=True, use_artifact_expert=True),
    "abnormal": dict(use_pair_expert=False, use_event_expert=True, use_artifact_expert=True),
}

_ALIASES = {
    "hidden_dim": "d_model",
    "depth": "osc_depth",
    "sample_rate_hz": "sfreq",
    "n_classes": "n_outputs",
}


@dataclass(frozen=True)
class BGNetConfig:
    n_outputs: int
    ch_names: tuple[str, ...]
    sfreq: float
    preset: str = "clinical"
    time_window_size: int = 250
    time_window_stride: int = 125
    d_model: int = 128
    osc_depth: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    low_rank: int = 8
    source_graph_sigma: float = 0.10
    source_graph_self_weight: float = 1.5
    attention_sigma: float = 0.05
    montage_name: str = "standard_1005"
    inward_scale: float = 0.85
    use_pair_expert: bool | None = None
    use_event_expert: bool | None = None
    use_artifact_expert: bool | None = None
    source_names: tuple[str, ...] = DEFAULT_SOURCE_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "preset", str(self.preset).lower())
        object.__setattr__(self, "ch_names", tuple(str(x) for x in self.ch_names))
        object.__setattr__(self, "source_names", tuple(str(x) for x in self.source_names))
        if self.n_outputs < 1:
            raise ValueError("n_outputs must be >= 1")
        if len(self.ch_names) < 1:
            raise ValueError("ch_names must be non-empty")
        if self.time_window_stride < 1:
            raise ValueError("time_window_stride must be >= 1")
        if self.time_window_size < 2:
            raise ValueError("time_window_size must be >= 2")

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        n_outputs: int,
        ch_names: Sequence[str],
        sfreq: float,
        **overrides: Any,
    ) -> "BGNetConfig":
        config = cls(
            n_outputs=int(n_outputs),
            ch_names=tuple(ch_names),
            sfreq=float(sfreq),
            preset=str(preset).lower(),
            **overrides,
        )
        return config.resolved()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BGNetConfig":
        normalized = dict(payload)
        for old_key, new_key in _ALIASES.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)
        if "ch_names" in normalized and not isinstance(normalized["ch_names"], tuple):
            normalized["ch_names"] = tuple(normalized["ch_names"])
        if "source_names" in normalized and not isinstance(normalized["source_names"], tuple):
            normalized["source_names"] = tuple(normalized["source_names"])
        return cls(**normalized).resolved()

    def resolved(self) -> "BGNetConfig":
        defaults = _PRESET_DEFAULTS.get(self.preset)
        if defaults is None:
            raise ValueError(f"Unsupported preset: {self.preset}")
        updated = self
        for key, value in defaults.items():
            if getattr(updated, key) is None:
                updated = replace(updated, **{key: value})
        return updated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.resolved())

