
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ._core_model import (
    FeedForward,
    GradientReversal,
    MultiHeadCrossAttention,
    PositionMLP,
    SourceGraphMix,
    SourceStateBlock,
    SourceToSensorProjector,
)


def _build_left_right_pairs(names: Optional[Sequence[str]]) -> Tuple[Tuple[int, int], ...]:
    if names is None:
        return tuple()
    index = {str(name): i for i, name in enumerate(names)}
    pairs = []
    for i, name in enumerate(names):
        s = str(name)
        if s.startswith('L_') and ('R_' + s[2:]) in index:
            pairs.append((i, index['R_' + s[2:]]))
            continue
        digits = ''.join(ch for ch in s if ch.isdigit())
        if not digits:
            continue
        try:
            num = int(digits)
        except ValueError:
            continue
        if num % 2 == 0:
            continue
        partner = s[:-len(digits)] + str(num + 1)
        if partner in index:
            pairs.append((i, index[partner]))
    return tuple(pairs)


@dataclass
class AdaptiveModelOutput:
    logits: Tensor
    pooled: Tensor
    source_tokens: Tensor
    sensor_tokens: Tensor
    recon_tokens: Tensor
    channel_mask: Tensor
    domain_logits: Optional[Tensor] = None
    router_weights: Optional[Tensor] = None
    aux_losses: Optional[Dict[str, Tensor]] = None


class CovarianceRowTokenizer(nn.Module):
    """
    EEG-native oscillatory tokenizer.

    Each time window is represented by a shrinkage covariance matrix in sensor space,
    then mapped to per-sensor tokens using each covariance row plus local log-variance.
    This captures the fact that much of EEG information is expressed in topographic
    covariance and band-limited amplitude structure rather than raw pointwise tokens.
    """

    def __init__(
        self,
        window_size: int,
        window_stride: int,
        n_channels: int,
        d_model: int,
        shrinkage: float = 0.10,
        eps: float = 1e-4,
        use_log_euclidean: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.n_channels = int(n_channels)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.use_log_euclidean = bool(use_log_euclidean)

        in_features = self.n_channels + 1
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _window_covariance(self, x: Tensor, sensor_mask: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected x with shape [B, C, T], got {tuple(x.shape)}')
        if x.shape[1] != self.n_channels:
            raise ValueError(f'Expected {self.n_channels} channels, got {x.shape[1]}')
        if x.shape[-1] < self.window_size:
            raise ValueError(f'Input time dimension {x.shape[-1]} shorter than window_size {self.window_size}')

        frames = x.unfold(dimension=-1, size=self.window_size, step=self.window_stride)
        frames = frames.permute(0, 2, 1, 3).contiguous()  # [B, N, C, W]
        frames = frames - frames.mean(dim=-1, keepdim=True)

        valid = sensor_mask.to(frames.dtype)[:, None, :, None]
        frames = frames * valid
        denom = max(self.window_size - 1, 1)
        cov = torch.einsum('bncw,bndw->bncd', frames, frames) / float(denom)

        valid_channels = sensor_mask.to(cov.dtype)
        valid_mat = valid_channels[:, None, :, None] * valid_channels[:, None, None, :]
        cov = cov * valid_mat

        trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        cov = cov / torch.clamp(trace[..., None], min=self.eps)

        eye = torch.eye(self.n_channels, dtype=cov.dtype, device=cov.device)
        mean_diag = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * mean_diag[..., None] * eye
        cov = cov + self.eps * eye[None, None, :, :]
        return cov

    def _matrix_log(self, cov: Tensor) -> Tensor:
        cov32 = cov.to(torch.float32)
        eigvals, eigvecs = torch.linalg.eigh(cov32)
        eigvals = torch.clamp(eigvals, min=self.eps)
        log_cov = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2)
        return log_cov.to(cov.dtype)

    def forward(self, x: Tensor, sensor_mask: Tensor) -> Tensor:
        cov = self._window_covariance(x=x, sensor_mask=sensor_mask)
        cov_features = self._matrix_log(cov) if self.use_log_euclidean else cov
        logvar = torch.log(torch.clamp(cov.diagonal(dim1=-2, dim2=-1), min=self.eps))
        features = torch.cat([cov_features, logvar.unsqueeze(-1)], dim=-1)
        return self.norm(self.feature_proj(features))


class TransientPatchTokenizer(nn.Module):
    """
    Event/transient tokenizer for sharp or phase-reset-like EEG structure.

    Instead of modelling text-like token dependencies, this branch summarises each
    short patch by raw shape, first derivative, line length, RMS energy and peak
    magnitude. It is meant to preserve clinically important transients that are often
    diluted by covariance-only processing.
    """

    def __init__(self, patch_size: int, patch_stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.raw_proj = nn.Linear(self.patch_size, d_model)
        self.diff_proj = nn.Linear(max(self.patch_size - 1, 1), d_model)
        self.stat_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected x with shape [B, C, T], got {tuple(x.shape)}')
        if x.shape[-1] < self.patch_size:
            raise ValueError(f'Input time dimension {x.shape[-1]} shorter than patch_size {self.patch_size}')
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()  # [B, N, C, P]
        patches = patches - patches.mean(dim=-1, keepdim=True)
        diff = patches[..., 1:] - patches[..., :-1]
        abs_mean = patches.abs().mean(dim=-1)
        rms = patches.square().mean(dim=-1).sqrt()
        line_len = diff.abs().mean(dim=-1)
        peak = patches.abs().amax(dim=-1)
        stats = torch.stack([abs_mean, rms, line_len, peak], dim=-1)

        raw_tok = self.raw_proj(patches)
        diff_tok = self.diff_proj(diff)
        stat_tok = self.stat_proj(stats)
        return self.norm(raw_tok + diff_tok + stat_tok)


class HemisphericPairExpert(nn.Module):
    def __init__(self, d_model: int, source_names: Optional[Sequence[str]], dropout: float = 0.1) -> None:
        super().__init__()
        self.pairs = _build_left_right_pairs(source_names)
        if len(self.pairs) == 0:
            self.left_idx = None
            self.right_idx = None
        else:
            left, right = zip(*self.pairs)
            self.register_buffer('left_idx', torch.tensor(left, dtype=torch.long), persistent=False)
            self.register_buffer('right_idx', torch.tensor(right, dtype=torch.long), persistent=False)
        self.norm = nn.LayerNorm(d_model)
        self.common_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.diff_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ff = FeedForward(d_model=d_model, mlp_ratio=2.0, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.left_idx is None or self.right_idx is None:
            return x
        h = self.norm(x)
        left = h[:, :, self.left_idx, :]
        right = h[:, :, self.right_idx, :]
        common = 0.5 * (left + right)
        diff = 0.5 * (left - right)
        common_u = self.common_mlp(common)
        diff_u = self.diff_mlp(diff)
        out = x.clone()
        out[:, :, self.left_idx, :] = x[:, :, self.left_idx, :] + common_u + diff_u
        out[:, :, self.right_idx, :] = x[:, :, self.right_idx, :] + common_u - diff_u
        out = out + self.ff(self.norm(out))
        return self.dropout(out)


class EventTemporalExpert(nn.Module):
    def __init__(self, adjacency: np.ndarray, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_time = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.graph_mix = SourceGraphMix(adjacency=adjacency, d_model=d_model, dropout=dropout)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.ff = FeedForward(d_model=d_model, mlp_ratio=2.0, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.graph_mix(self.norm_graph(x))
        h = self.norm_time(x)
        bsz, n_steps, n_sources, d_model = h.shape
        h = h.permute(0, 2, 3, 1).reshape(bsz * n_sources, d_model, n_steps)
        h = self.pw_conv(F.gelu(self.dw_conv(h)))
        h = h.reshape(bsz, n_sources, d_model, n_steps).permute(0, 3, 1, 2)
        x = x + self.dropout(h)
        x = x + self.ff(self.norm_ff(x))
        return x


class AdaptiveSourceFieldEEG(nn.Module):
    """
    EEG-native mixture-of-physiology model.

    Design goals:
    - oscillations/covariance are handled explicitly rather than hidden inside raw token attention
    - transients are modelled with a separate branch
    - scalp channels are treated as observations of latent source fields
    - different EEG regimes can route to different experts instead of forcing one universal block
    """

    def __init__(
        self,
        source_positions: np.ndarray,
        n_sensor_channels: int,
        n_classes: int,
        time_window_size: int = 250,
        time_window_stride: int = 125,
        cov_shrinkage: float = 0.10,
        use_log_euclidean: bool = True,
        d_model: int = 128,
        osc_depth: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        graph_adjacency: Optional[np.ndarray] = None,
        source_names: Optional[Tuple[str, ...]] = None,
        n_domains: int = 0,
        domain_lambda: float = 0.1,
        sigma: float = 0.05,
        router_temperature: float = 1.0,
        router_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        source_pos = torch.as_tensor(source_positions, dtype=torch.float32)
        self.register_buffer('source_pos', source_pos)
        self.source_names = source_names
        self.n_sources = int(source_pos.shape[0])
        self.n_sensor_channels = int(n_sensor_channels)
        self.n_classes = int(n_classes)
        self.d_model = int(d_model)
        self.router_temperature = float(router_temperature)
        self.router_balance_weight = float(router_balance_weight)

        if graph_adjacency is None:
            graph_adjacency = np.eye(self.n_sources, dtype=np.float32)

        self.osc_tokenizer = CovarianceRowTokenizer(
            window_size=time_window_size,
            window_stride=time_window_stride,
            n_channels=n_sensor_channels,
            d_model=d_model,
            shrinkage=cov_shrinkage,
            use_log_euclidean=use_log_euclidean,
        )
        self.transient_tokenizer = TransientPatchTokenizer(
            patch_size=time_window_size,
            patch_stride=time_window_stride,
            d_model=d_model,
        )

        self.sensor_pos_embed = PositionMLP(d_model)
        self.source_pos_embed = PositionMLP(d_model)
        self.sensor_mask_token = nn.Parameter(torch.zeros(d_model))
        self.source_queries = nn.Parameter(torch.randn(self.n_sources, d_model) * 0.02)

        self.osc_sensor_to_source = MultiHeadCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sigma=sigma,
        )
        self.transient_sensor_to_source = MultiHeadCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sigma=sigma,
        )

        self.osc_blocks = nn.ModuleList(
            [
                SourceStateBlock(
                    adjacency=graph_adjacency,
                    d_model=d_model,
                    dropout=dropout,
                    mlp_ratio=4.0,
                )
                for _ in range(int(osc_depth))
            ]
        )
        self.pair_expert = HemisphericPairExpert(d_model=d_model, source_names=source_names, dropout=dropout)
        self.event_expert = EventTemporalExpert(adjacency=graph_adjacency, d_model=d_model, dropout=dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.router = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
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
        self.reconstructor = SourceToSensorProjector(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sigma=sigma,
        )

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
        masked_tokens = masked_tokens + (
            mask[:, None, :, None].to(masked_tokens.dtype) * self.sensor_mask_token.view(1, 1, 1, d_model)
        )
        return masked_tokens, mask

    def _sensor_to_source(self, sensor_tokens: Tensor, sensor_pos: Tensor, sensor_mask: Tensor, attn: MultiHeadCrossAttention) -> Tensor:
        bsz, n_steps, _, _ = sensor_tokens.shape
        source_query = self.source_queries + self.source_pos_embed(self.source_pos)
        source_query = source_query[None, None, :, :].expand(bsz, n_steps, -1, -1)
        return attn(
            query=source_query,
            key_value=sensor_tokens,
            query_pos=self.source_pos[None, :, :],
            key_pos=sensor_pos,
            key_mask=sensor_mask,
        )

    def encode(self, x: Tensor, sensor_pos: Tensor, sensor_mask: Tensor, mask_ratio: float = 0.0):
        pos_bias = self.sensor_pos_embed(sensor_pos)[:, None, :, :]
        osc_sensor = self.osc_tokenizer(x=x, sensor_mask=sensor_mask) + pos_bias
        evt_sensor = self.transient_tokenizer(x=x) + pos_bias

        original_sensor_tokens = osc_sensor.clone()
        osc_sensor_masked, channel_mask = self._mask_sensor_tokens(osc_sensor, sensor_mask, mask_ratio)

        osc_source = self._sensor_to_source(osc_sensor_masked, sensor_pos=sensor_pos, sensor_mask=sensor_mask, attn=self.osc_sensor_to_source)
        evt_source = self._sensor_to_source(evt_sensor, sensor_pos=sensor_pos, sensor_mask=sensor_mask, attn=self.transient_sensor_to_source)

        for block in self.osc_blocks:
            osc_source = block(osc_source)

        osc_expert = osc_source
        pair_expert = self.pair_expert(osc_source)
        event_expert = self.event_expert(evt_source)

        router_in = torch.cat([osc_source.mean(dim=2), evt_source.mean(dim=2)], dim=-1)
        router_logits = self.router(router_in) / max(self.router_temperature, 1e-6)
        router_weights = torch.softmax(router_logits, dim=-1)

        experts = torch.stack([osc_expert, pair_expert, event_expert], dim=2)  # [B, N, E, S, D]
        fused = (router_weights[:, :, :, None, None] * experts).sum(dim=2)
        shared_residual = self.fusion_proj(torch.cat([osc_source, evt_source], dim=-1))
        fused = self.source_norm(fused + shared_residual)

        mean_gate = router_weights.mean(dim=(0, 1))
        uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
        router_balance = F.kl_div(torch.log(torch.clamp(mean_gate, min=1e-6)), uniform, reduction='sum')
        return fused, original_sensor_tokens, channel_mask, router_weights, {'router_balance': router_balance * self.router_balance_weight}

    def forward(self, x: Tensor, sensor_pos: Tensor, sensor_mask: Tensor, mask_ratio: float = 0.0) -> AdaptiveModelOutput:
        source_tokens, sensor_tokens, channel_mask, router_weights, aux_losses = self.encode(
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
            router_weights=router_weights,
            aux_losses=aux_losses,
        )
