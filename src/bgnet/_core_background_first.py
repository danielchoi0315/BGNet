from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ._core_model import (
    FeedForward,
    GradientReversal,
    PositionMLP,
    SourceGraphMix,
    SourceStateBlock,
    SourceToSensorProjector,
)
from ._core_adaptive import AdaptiveModelOutput, EventTemporalExpert, HemisphericPairExpert


def _band_defs() -> Tuple[Tuple[str, float, float], ...]:
    return (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    )


def _safe_mask(values: Tensor, fallback: bool = False) -> Tensor:
    if bool(values.any()):
        return values
    return torch.full_like(values, bool(fallback), dtype=torch.bool)


class MultibandSpectroTopographicTokenizer(nn.Module):
    def __init__(
        self,
        window_size: int,
        window_stride: int,
        n_channels: int,
        d_model: int,
        sample_rate_hz: float = 250.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.n_channels = int(n_channels)
        self.sample_rate_hz = float(sample_rate_hz)
        self.eps = float(eps)
        self.bands = _band_defs()
        self.n_bands = len(self.bands)
        freqs = torch.fft.rfftfreq(self.window_size, d=1.0 / self.sample_rate_hz).to(torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        band_masks = []
        for _, f_lo, f_hi in self.bands:
            band_mask = (freqs >= f_lo) & (freqs < f_hi)
            if not bool(band_mask.any()):
                band_mask = freqs >= f_lo
            if not bool(band_mask.any()):
                band_mask = torch.ones_like(freqs, dtype=torch.bool)
            band_masks.append(band_mask)
        self.register_buffer("band_masks", torch.stack(band_masks, dim=0), persistent=False)
        self.register_buffer("alpha_mask", ((freqs >= 7.0) & (freqs <= 13.5)), persistent=False)

        in_features = self.n_bands * 3 + 3
        desc_dim = self.n_bands * 2 + 3
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _frames(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.shape[1]}")
        if x.shape[-1] < self.window_size:
            raise ValueError(f"Input time dimension {x.shape[-1]} shorter than window_size {self.window_size}")
        frames = x.unfold(dimension=-1, size=self.window_size, step=self.window_stride)
        return frames.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: Tensor, sensor_pos: Tensor, sensor_mask: Tensor) -> Tuple[Tensor, Tensor]:
        frames = self._frames(x)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        valid = sensor_mask.to(frames.dtype)[:, None, :, None]
        frames = frames * valid

        spec = torch.fft.rfft(frames.to(torch.float32), dim=-1)
        power = spec.abs().square() + self.eps

        band_power = []
        for band_mask in self.band_masks:
            band_power.append(power[..., band_mask].mean(dim=-1))
        band_power_t = torch.stack(band_power, dim=-1)
        total_power = torch.clamp(band_power_t.sum(dim=-1, keepdim=True), min=self.eps)
        rel_power = band_power_t / total_power

        valid_ch = sensor_mask.to(band_power_t.dtype)[:, None, :, None]
        chan_mean = (band_power_t * valid_ch).sum(dim=2, keepdim=True) / torch.clamp(valid_ch.sum(dim=2, keepdim=True), min=1.0)
        topo_dev = torch.log(torch.clamp(band_power_t, min=self.eps)) - torch.log(torch.clamp(chan_mean, min=self.eps))

        pos_x = sensor_pos[..., 0]
        pos_y = sensor_pos[..., 1]
        left = _safe_mask((pos_x < -0.02) & sensor_mask, fallback=False)
        right = _safe_mask((pos_x > 0.02) & sensor_mask, fallback=False)
        posterior = _safe_mask((pos_y < -0.02) & sensor_mask, fallback=True)
        anterior = _safe_mask((pos_y > 0.02) & sensor_mask, fallback=False)

        alpha_idx = 2
        alpha_power = band_power_t[..., alpha_idx]
        left_alpha = (alpha_power * left[:, None, :].to(alpha_power.dtype)).sum(dim=2) / torch.clamp(left[:, None, :].sum(dim=2), min=1.0)
        right_alpha = (alpha_power * right[:, None, :].to(alpha_power.dtype)).sum(dim=2) / torch.clamp(right[:, None, :].sum(dim=2), min=1.0)
        posterior_alpha = (alpha_power * posterior[:, None, :].to(alpha_power.dtype)).sum(dim=2) / torch.clamp(posterior[:, None, :].sum(dim=2), min=1.0)
        anterior_alpha = (alpha_power * anterior[:, None, :].to(alpha_power.dtype)).sum(dim=2) / torch.clamp(anterior[:, None, :].sum(dim=2), min=1.0)
        alpha_asym = (left_alpha - right_alpha) / torch.clamp(left_alpha + right_alpha + self.eps, min=self.eps)
        alpha_ap = (posterior_alpha - anterior_alpha) / torch.clamp(posterior_alpha + anterior_alpha + self.eps, min=self.eps)
        anterior_support = anterior[:, None, :].sum(dim=2) > 0
        posterior_support = posterior[:, None, :].sum(dim=2) > 0
        alpha_ap = torch.where(anterior_support & posterior_support, alpha_ap, torch.zeros_like(alpha_ap))

        freqs_view = self.freqs.view(1, 1, 1, -1)
        alpha_mask = self.alpha_mask.view(1, 1, 1, -1)
        alpha_spec = power * alpha_mask.to(power.dtype)
        alpha_sum = torch.clamp(alpha_spec.sum(dim=-1), min=self.eps)
        alpha_peak_freq = (alpha_spec * freqs_view).sum(dim=-1) / alpha_sum
        alpha_peak_score = alpha_spec.amax(dim=-1) / torch.clamp(power.mean(dim=-1), min=self.eps)

        log_power = torch.log(torch.clamp(band_power_t, min=self.eps))
        channel_ap = sensor_pos[:, None, :, 1].expand_as(alpha_peak_freq)
        features = torch.cat(
            [
                log_power,
                rel_power,
                topo_dev,
                alpha_peak_freq.unsqueeze(-1),
                alpha_peak_score.unsqueeze(-1),
                channel_ap.unsqueeze(-1),
            ],
            dim=-1,
        )

        band_mean = (band_power_t * valid_ch).sum(dim=2) / torch.clamp(valid_ch.sum(dim=2), min=1.0)
        band_rel_mean = (rel_power * valid_ch).sum(dim=2) / torch.clamp(valid_ch.sum(dim=2), min=1.0)
        band_log_mean = torch.log(torch.clamp(band_mean, min=self.eps))
        delta_theta = band_rel_mean[..., 0] + band_rel_mean[..., 1]
        desc = torch.cat(
            [
                band_log_mean,
                band_rel_mean,
                torch.stack([alpha_asym, alpha_ap, delta_theta], dim=-1),
            ],
            dim=-1,
        )
        tokens = self.norm(self.feature_proj(features.to(x.dtype)))
        return tokens, self.desc_proj(desc.to(x.dtype))


class MultiScaleTransientTokenizer(nn.Module):
    def __init__(
        self,
        window_size: int,
        window_stride: int,
        n_channels: int,
        d_model: int,
        scales: Sequence[int] = (32, 64, 128),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.n_channels = int(n_channels)
        self.scales = tuple(int(s) for s in scales)
        self.eps = float(eps)
        if len(self.scales) == 0:
            raise ValueError("At least one transient scale is required")
        if max(self.scales) > self.window_size:
            raise ValueError(f"Transient scale {max(self.scales)} exceeds window_size {self.window_size}")

        feat_dim = len(self.scales) * 4
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(len(self.scales) * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _frames(self, x: Tensor) -> Tensor:
        frames = x.unfold(dimension=-1, size=self.window_size, step=self.window_stride)
        return frames.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: Tensor, sensor_mask: Tensor) -> Tuple[Tensor, Tensor]:
        frames = self._frames(x)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        valid = sensor_mask.to(frames.dtype)[:, None, :, None]
        frames = frames * valid

        feats = []
        desc = []
        for scale in self.scales:
            patch = frames.unfold(dimension=-1, size=scale, step=max(scale // 2, 1))
            diff = patch[..., 1:] - patch[..., :-1]
            line_len = diff.abs().mean(dim=-1)
            peak = patch.abs().amax(dim=-1)
            rms = patch.square().mean(dim=-1).sqrt()
            sharp = diff.abs().amax(dim=-1)
            feats.append(torch.stack([line_len.mean(dim=-1), peak.mean(dim=-1), rms.mean(dim=-1), sharp.mean(dim=-1)], dim=-1))
            patch_mask = sensor_mask[:, None, :, None].to(line_len.dtype)
            denom = torch.clamp(patch_mask.expand_as(line_len).sum(dim=(2, 3)), min=1.0)
            desc.append(torch.stack([(line_len * patch_mask).sum(dim=(2, 3)) / denom, (peak * patch_mask).sum(dim=(2, 3)) / denom], dim=-1))

        feat_t = torch.cat(feats, dim=-1)
        desc_t = torch.cat(desc, dim=-1)
        return self.norm(self.proj(feat_t.to(x.dtype))), self.desc_proj(desc_t.to(x.dtype))


class ArtifactStatTokenizer(nn.Module):
    def __init__(
        self,
        window_size: int,
        window_stride: int,
        n_channels: int,
        d_model: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _frames(self, x: Tensor) -> Tensor:
        frames = x.unfold(dimension=-1, size=self.window_size, step=self.window_stride)
        return frames.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: Tensor, sensor_mask: Tensor) -> Tuple[Tensor, Tensor]:
        frames = self._frames(x)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        valid = sensor_mask.to(frames.dtype)[:, None, :, None]
        frames = frames * valid
        diff = frames[..., 1:] - frames[..., :-1]
        rms = frames.square().mean(dim=-1).sqrt()
        line_len = diff.abs().mean(dim=-1)
        peak = frames.abs().amax(dim=-1)
        flat = 1.0 / torch.clamp(frames.std(dim=-1), min=self.eps)
        feat = torch.stack([rms, line_len, peak, flat], dim=-1)
        channel_mask = sensor_mask[:, None, :].to(rms.dtype)
        denom = torch.clamp(channel_mask.sum(dim=2), min=1.0)
        desc = torch.stack(
            [
                (rms * channel_mask).sum(dim=2) / denom,
                (line_len * channel_mask).sum(dim=2) / denom,
                (peak * channel_mask).sum(dim=2) / denom,
                (flat * channel_mask).sum(dim=2) / denom,
            ],
            dim=-1,
        )
        return self.norm(self.proj(feat.to(x.dtype))), self.desc_proj(desc.to(x.dtype))


class LeadfieldAdapterCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        sigma: float = 0.05,
        low_rank: int = 8,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.sigma = float(sigma)
        self.low_rank = int(low_rank)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_bias_proj = nn.Linear(3, self.low_rank, bias=False)
        self.k_bias_proj = nn.Linear(3, self.low_rank, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _distance_bias(self, query_pos: Tensor, key_pos: Tensor) -> Tensor:
        if query_pos.shape[0] == 1 and key_pos.shape[0] > 1:
            query_pos = query_pos.expand(key_pos.shape[0], -1, -1)
        dist2 = (query_pos[:, :, None, :] - key_pos[:, None, :, :]).pow(2).sum(dim=-1)
        sigma2 = max(self.sigma * self.sigma, 1e-6)
        return -dist2 / (2.0 * sigma2)

    def _leadfield_bias(self, query_pos: Tensor, key_pos: Tensor) -> Tensor:
        if query_pos.shape[0] == 1 and key_pos.shape[0] > 1:
            query_pos = query_pos.expand(key_pos.shape[0], -1, -1)
        q = self.q_bias_proj(query_pos)
        k = self.k_bias_proj(key_pos)
        return torch.einsum("bqr,bkr->bqk", q, k) / max(float(self.low_rank) ** 0.5, 1.0)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, n_steps, n_query, _ = query.shape
        _, _, n_key, _ = key_value.shape

        q = self.q_proj(query).view(bsz, n_steps, n_query, self.n_heads, self.head_dim)
        k = self.k_proj(key_value).view(bsz, n_steps, n_key, self.n_heads, self.head_dim)
        v = self.v_proj(key_value).view(bsz, n_steps, n_key, self.n_heads, self.head_dim)

        logits = torch.einsum("bnqhd,bnkhd->bnhqk", q, k) * self.scale
        bias = self._distance_bias(query_pos, key_pos) + self._leadfield_bias(query_pos, key_pos)
        logits = logits + bias[:, None, None, :, :]
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask[:, None, None, None, :].bool(), -1e4)
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = torch.einsum("bnhqk,bnkhd->bnqhd", attn, v).reshape(bsz, n_steps, n_query, self.d_model)
        return self.out_proj(out)


class ArtifactResidualExpert(nn.Module):
    def __init__(self, adjacency: np.ndarray, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_bg = nn.LayerNorm(d_model)
        self.norm_art = nn.LayerNorm(d_model)
        self.graph_mix = SourceGraphMix(adjacency=adjacency, d_model=d_model, dropout=dropout)
        self.mixer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ff = FeedForward(d_model=d_model, mlp_ratio=2.0, dropout=dropout)

    def forward(self, background_source: Tensor, artifact_source: Tensor) -> Tensor:
        bg = self.norm_bg(background_source)
        art = self.norm_art(artifact_source)
        delta = self.mixer(torch.cat([bg, art], dim=-1))
        delta = delta + self.graph_mix(delta)
        return self.ff(delta)


class BackgroundFirstSourceFieldEEG(nn.Module):
    def __init__(
        self,
        source_positions: np.ndarray,
        n_sensor_channels: int,
        n_classes: int,
        time_window_size: int = 250,
        time_window_stride: int = 125,
        sample_rate_hz: float = 250.0,
        d_model: int = 128,
        osc_depth: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        graph_adjacency: Optional[np.ndarray] = None,
        source_names: Optional[Tuple[str, ...]] = None,
        n_domains: int = 0,
        domain_lambda: float = 0.1,
        sigma: float = 0.05,
        low_rank: int = 8,
        use_pair_expert: bool = True,
        use_event_expert: bool = False,
        use_artifact_expert: bool = True,
    ) -> None:
        super().__init__()
        source_pos = torch.as_tensor(source_positions, dtype=torch.float32)
        self.register_buffer("source_pos", source_pos)
        self.source_names = source_names
        self.n_sources = int(source_pos.shape[0])
        self.n_sensor_channels = int(n_sensor_channels)
        self.n_classes = int(n_classes)
        self.d_model = int(d_model)
        self.use_pair_expert = bool(use_pair_expert)
        self.use_event_expert = bool(use_event_expert)
        self.use_artifact_expert = bool(use_artifact_expert)

        if graph_adjacency is None:
            graph_adjacency = np.eye(self.n_sources, dtype=np.float32)

        self.background_tokenizer = MultibandSpectroTopographicTokenizer(
            window_size=time_window_size,
            window_stride=time_window_stride,
            n_channels=n_sensor_channels,
            d_model=d_model,
            sample_rate_hz=sample_rate_hz,
        )
        self.transient_tokenizer = MultiScaleTransientTokenizer(
            window_size=time_window_size,
            window_stride=time_window_stride,
            n_channels=n_sensor_channels,
            d_model=d_model,
        )
        self.artifact_tokenizer = ArtifactStatTokenizer(
            window_size=time_window_size,
            window_stride=time_window_stride,
            n_channels=n_sensor_channels,
            d_model=d_model,
        )

        self.sensor_pos_embed = PositionMLP(d_model)
        self.source_pos_embed = PositionMLP(d_model)
        self.sensor_mask_token = nn.Parameter(torch.zeros(d_model))
        self.source_queries = nn.Parameter(torch.randn(self.n_sources, d_model) * 0.02)

        self.background_to_source = LeadfieldAdapterCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, sigma=sigma, low_rank=low_rank)
        self.event_to_source = LeadfieldAdapterCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, sigma=sigma, low_rank=low_rank)
        self.artifact_to_source = LeadfieldAdapterCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, sigma=sigma, low_rank=low_rank)

        self.background_blocks = nn.ModuleList(
            [SourceStateBlock(adjacency=graph_adjacency, d_model=d_model, dropout=dropout, mlp_ratio=4.0) for _ in range(int(osc_depth))]
        )
        self.pair_expert = HemisphericPairExpert(d_model=d_model, source_names=source_names, dropout=dropout)
        self.event_expert = EventTemporalExpert(adjacency=graph_adjacency, d_model=d_model, dropout=dropout)
        self.artifact_expert = ArtifactResidualExpert(adjacency=graph_adjacency, d_model=d_model, dropout=dropout)

        gate_in_dim = 4 * d_model
        self.regime_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.regime_embed = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.register_buffer(
            "active_regime_mask",
            torch.tensor([True, self.use_pair_expert, self.use_event_expert, self.use_artifact_expert], dtype=torch.bool),
            persistent=False,
        )

        self.source_norm = nn.LayerNorm(d_model)
        self.pool_head = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.reconstructor = SourceToSensorProjector(d_model=d_model, n_heads=n_heads, dropout=dropout, sigma=sigma)

        if n_domains > 1:
            self.domain_grl = GradientReversal(domain_lambda)
            self.domain_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_domains),
            )
        else:
            self.domain_grl = None
            self.domain_head = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.source_queries, std=0.02)
        nn.init.normal_(self.sensor_mask_token, std=0.02)
        nn.init.trunc_normal_(self.regime_embed, std=0.02)

    def _mask_sensor_tokens(self, sensor_tokens: Tensor, sensor_mask: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        bsz, _, n_channels, d_model = sensor_tokens.shape
        if mask_ratio <= 0.0:
            return sensor_tokens, torch.zeros(bsz, n_channels, dtype=torch.bool, device=sensor_tokens.device)

        valid = sensor_mask.bool()
        rand = torch.rand(bsz, n_channels, device=sensor_tokens.device)
        rand = rand.masked_fill(~valid, 1.0)
        n_valid = valid.sum(dim=1)
        n_to_mask = torch.clamp((n_valid.float() * mask_ratio).round().long(), min=1)
        mask = torch.zeros_like(valid)
        for b in range(bsz):
            if n_valid[b] == 0:
                continue
            idx = torch.argsort(rand[b])[: n_to_mask[b]]
            mask[b, idx] = True

        masked_tokens = sensor_tokens.clone()
        masked_tokens[mask[:, None, :, None].expand_as(masked_tokens)] = 0.0
        masked_tokens = masked_tokens + mask[:, None, :, None].to(masked_tokens.dtype) * self.sensor_mask_token.view(1, 1, 1, d_model)
        return masked_tokens, mask

    def _sensor_to_source(
        self,
        sensor_tokens: Tensor,
        sensor_pos: Tensor,
        sensor_mask: Tensor,
        projector: LeadfieldAdapterCrossAttention,
        source_query: Tensor | None = None,
    ) -> Tensor:
        bsz, n_steps, _, _ = sensor_tokens.shape
        if source_query is None:
            source_query = self.source_queries + self.source_pos_embed(self.source_pos)
            source_query = source_query[None, None, :, :].expand(bsz, n_steps, -1, -1)
        return projector(
            query=source_query,
            key_value=sensor_tokens,
            query_pos=self.source_pos[None, :, :],
            key_pos=sensor_pos,
            key_mask=sensor_mask,
        )

    def encode(self, x: Tensor, sensor_pos: Tensor, sensor_mask: Tensor, mask_ratio: float = 0.0):
        if not bool(sensor_mask.any(dim=1).all()):
            raise ValueError("Each sample must contain at least one valid sensor")
        pos_bias = self.sensor_pos_embed(sensor_pos)[:, None, :, :]
        background_sensor, background_desc = self.background_tokenizer(x=x, sensor_pos=sensor_pos, sensor_mask=sensor_mask)
        background_sensor = background_sensor + pos_bias
        original_sensor_tokens = background_sensor.clone()
        background_masked, channel_mask = self._mask_sensor_tokens(background_sensor, sensor_mask, mask_ratio)
        source_query = self.source_queries + self.source_pos_embed(self.source_pos)
        source_query = source_query[None, None, :, :].expand(x.shape[0], background_sensor.shape[1], -1, -1)

        background_source = self._sensor_to_source(
            background_masked,
            sensor_pos=sensor_pos,
            sensor_mask=sensor_mask,
            projector=self.background_to_source,
            source_query=source_query,
        )
        for block in self.background_blocks:
            background_source = block(background_source)

        if self.use_artifact_expert:
            artifact_sensor, artifact_desc = self.artifact_tokenizer(x=x, sensor_mask=sensor_mask)
            artifact_sensor = artifact_sensor + pos_bias
            artifact_source = self._sensor_to_source(
                artifact_sensor,
                sensor_pos=sensor_pos,
                sensor_mask=sensor_mask,
                projector=self.artifact_to_source,
                source_query=source_query,
            )
            artifact_delta = self.artifact_expert(background_source, artifact_source)
        else:
            artifact_desc = torch.zeros_like(background_desc)
            artifact_delta = torch.zeros_like(background_source)
        if self.use_event_expert:
            event_sensor, event_desc = self.transient_tokenizer(x=x, sensor_mask=sensor_mask)
            event_sensor = event_sensor + pos_bias
            event_source = self._sensor_to_source(
                event_sensor,
                sensor_pos=sensor_pos,
                sensor_mask=sensor_mask,
                projector=self.event_to_source,
                source_query=source_query,
            )
            event_delta = self.event_expert(event_source) - event_source
        else:
            event_desc = torch.zeros_like(background_desc)
            event_delta = torch.zeros_like(background_source)

        pair_delta = self.pair_expert(background_source) - background_source if self.use_pair_expert else torch.zeros_like(background_source)

        gate_in = torch.cat([background_source.mean(dim=2), background_desc, event_desc, artifact_desc], dim=-1)
        regime_logits = self.regime_gate(gate_in)
        regime_logits = regime_logits.masked_fill(~self.active_regime_mask.view(1, 1, -1), -1e4)
        regime_probs = torch.softmax(regime_logits, dim=-1)

        fused = background_source
        fused = fused + regime_probs[:, :, 1, None, None] * pair_delta
        fused = fused + regime_probs[:, :, 2, None, None] * event_delta
        fused = fused + regime_probs[:, :, 3, None, None] * artifact_delta
        fused = self.source_norm(fused)

        aux_losses: Dict[str, Tensor] = {}
        return fused, original_sensor_tokens, channel_mask, regime_probs, aux_losses

    def forward(self, x: Tensor, sensor_pos: Tensor, sensor_mask: Tensor, mask_ratio: float = 0.0) -> AdaptiveModelOutput:
        source_tokens, sensor_tokens, channel_mask, regime_probs, aux_losses = self.encode(
            x=x,
            sensor_pos=sensor_pos,
            sensor_mask=sensor_mask,
            mask_ratio=mask_ratio,
        )
        recon_tokens = self.reconstructor(
            source_tokens=source_tokens,
            source_pos=self.source_pos,
            sensor_pos=sensor_pos,
            sensor_mask=sensor_mask,
        )
        pool_logits = self.pool_head(source_tokens).squeeze(-1)
        pool_weights = torch.softmax(pool_logits.flatten(start_dim=1), dim=-1).view_as(pool_logits)
        pooled = (pool_weights.unsqueeze(-1) * source_tokens).sum(dim=(1, 2))
        regime_context = torch.einsum("bnr,rd->bnd", regime_probs, self.regime_embed).mean(dim=1)
        pooled = pooled + regime_context
        logits = self.classifier(pooled)

        domain_logits = None
        if self.domain_head is not None:
            domain_logits = self.domain_head(self.domain_grl(pooled))

        return AdaptiveModelOutput(
            logits=logits,
            pooled=pooled,
            source_tokens=source_tokens,
            sensor_tokens=sensor_tokens,
            recon_tokens=recon_tokens,
            channel_mask=channel_mask,
            domain_logits=domain_logits,
            router_weights=regime_probs,
            aux_losses=aux_losses,
        )
