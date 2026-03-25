from __future__ import annotations

from .config import BGNetConfig
from .geometry import build_default_source_layout, normalized_source_adjacency
from ._core_background_first import BackgroundFirstSourceFieldEEG


def build_bgnet_core(config: BGNetConfig) -> BackgroundFirstSourceFieldEEG:
    resolved = config.resolved()
    source_layout = build_default_source_layout(
        source_names=resolved.source_names,
        montage_name=resolved.montage_name,
        inward_scale=resolved.inward_scale,
    )
    adjacency = normalized_source_adjacency(
        source_pos=source_layout.positions,
        sigma=resolved.source_graph_sigma,
        self_weight=resolved.source_graph_self_weight,
    )
    return BackgroundFirstSourceFieldEEG(
        source_positions=source_layout.positions,
        source_names=source_layout.names,
        graph_adjacency=adjacency,
        n_sensor_channels=len(resolved.ch_names),
        n_classes=resolved.n_outputs,
        time_window_size=resolved.time_window_size,
        time_window_stride=resolved.time_window_stride,
        sample_rate_hz=resolved.sfreq,
        d_model=resolved.d_model,
        osc_depth=resolved.osc_depth,
        n_heads=resolved.n_heads,
        dropout=resolved.dropout,
        sigma=resolved.attention_sigma,
        low_rank=resolved.low_rank,
        use_pair_expert=bool(resolved.use_pair_expert),
        use_event_expert=bool(resolved.use_event_expert),
        use_artifact_expert=bool(resolved.use_artifact_expert),
    )

