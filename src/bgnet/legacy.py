from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import BGNetConfig
from .geometry import DEFAULT_SOURCE_NAMES


def from_research_cfg(cfg: Mapping[str, Any], *, ch_names: Sequence[str], sfreq: float | None = None) -> BGNetConfig:
    model_cfg = dict(cfg.get("model", {}))
    arch = str(model_cfg.get("arch", "background_first")).lower()
    if arch != "background_first":
        raise ValueError(f"BGNet public package only supports background_first, got: {arch}")
    preprocessing = dict(cfg.get("preprocessing", {}))
    resolved_sfreq = float(model_cfg.get("sample_rate_hz", sfreq or preprocessing.get("resample", 250.0)))
    return BGNetConfig.from_preset(
        str(model_cfg.get("preset", "clinical")),
        n_outputs=int(model_cfg["n_classes"]),
        ch_names=tuple(ch_names),
        sfreq=resolved_sfreq,
        time_window_size=int(model_cfg.get("time_window_size", 250)),
        time_window_stride=int(model_cfg.get("time_window_stride", 125)),
        d_model=int(model_cfg.get("d_model", 128)),
        osc_depth=int(model_cfg.get("osc_depth", 3)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        low_rank=int(model_cfg.get("low_rank", 8)),
        source_graph_sigma=float(model_cfg.get("source_graph_sigma", 0.10)),
        source_graph_self_weight=float(model_cfg.get("source_graph_self_weight", 1.5)),
        attention_sigma=float(model_cfg.get("attention_sigma", 0.05)),
        use_pair_expert=model_cfg.get("use_pair_expert"),
        use_event_expert=model_cfg.get("use_event_expert"),
        use_artifact_expert=model_cfg.get("use_artifact_expert"),
        source_names=tuple(model_cfg.get("source_names", DEFAULT_SOURCE_NAMES)),
    )
