from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambd)


class TemporalPatchTokenizer(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.proj = nn.Linear(self.patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape [B, C, T]

        Returns
        -------
        Tensor
            Shape [B, N, C, D]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        # [B, C, N, P] -> [B, N, C, P]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        tokens = self.proj(patches)
        return self.norm(tokens)


class PositionMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos: Tensor) -> Tensor:
        return self.net(pos)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        sigma: float = 0.35,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.sigma = float(sigma)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _distance_bias(self, query_pos: Tensor, key_pos: Tensor) -> Tensor:
        # query_pos: [B, Q, 3] or [1, Q, 3], key_pos: [B, K, 3]
        if query_pos.shape[0] == 1 and key_pos.shape[0] > 1:
            query_pos = query_pos.expand(key_pos.shape[0], -1, -1)
        dist2 = (query_pos[:, :, None, :] - key_pos[:, None, :, :]).pow(2).sum(dim=-1)
        sigma2 = max(self.sigma * self.sigma, 1e-6)
        return -dist2 / (2.0 * sigma2)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        query: [B, N, Q, D]
        key_value: [B, N, K, D]
        query_pos: [B|1, Q, 3]
        key_pos: [B, K, 3]
        key_mask: [B, K] bool where True means valid
        """
        bsz, n_steps, n_query, _ = query.shape
        _, _, n_key, _ = key_value.shape

        q = self.q_proj(query).view(bsz, n_steps, n_query, self.n_heads, self.head_dim)
        k = self.k_proj(key_value).view(bsz, n_steps, n_key, self.n_heads, self.head_dim)
        v = self.v_proj(key_value).view(bsz, n_steps, n_key, self.n_heads, self.head_dim)

        # [B, N, H, Q, K]
        logits = torch.einsum("bnqhd,bnkhd->bnhqk", q, k) * self.scale
        bias = self._distance_bias(query_pos, key_pos)  # [B, Q, K]
        logits = logits + bias[:, None, None, :, :]

        if key_mask is not None:
            mask = ~key_mask[:, None, None, None, :].bool()
            logits = logits.masked_fill(mask, -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bnhqk,bnkhd->bnqhd", attn, v).reshape(
            bsz, n_steps, n_query, self.d_model
        )
        return self.out_proj(out)


class SourceGraphMix(nn.Module):
    def __init__(self, adjacency: np.ndarray, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        adj = torch.as_tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adj)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        mixed = torch.einsum("ij,btjd->btid", self.adjacency, x)
        return self.dropout(self.proj(mixed))


class ExponentialStateMixer(nn.Module):
    """
    Linear-time temporal state update with learned interpolation.

    This is not a vanilla Transformer attention block; it behaves more like a
    physiology-inspired leaky integrator over source states.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, S, D]
        gate, cand = self.in_proj(x).chunk(2, dim=-1)
        alpha = torch.sigmoid(gate)
        cand = torch.tanh(cand)

        bsz, n_steps, n_sources, d_model = x.shape
        state = torch.zeros(bsz, n_sources, d_model, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(n_steps):
            state = (1.0 - alpha[:, t]) * state + alpha[:, t] * cand[:, t]
            outputs.append(state)
        y = torch.stack(outputs, dim=1)
        return self.dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SourceStateBlock(nn.Module):
    def __init__(
        self,
        adjacency: np.ndarray,
        d_model: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_time = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.graph_mix = SourceGraphMix(adjacency=adjacency, d_model=d_model, dropout=dropout)
        self.time_mix = ExponentialStateMixer(d_model=d_model, dropout=dropout)
        self.ff = FeedForward(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.graph_mix(self.norm_graph(x))
        x = x + self.time_mix(self.norm_time(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class SourceToSensorProjector(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        sigma: float = 0.35,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sigma=sigma,
        )
        self.sensor_pos_embed = PositionMLP(d_model)
        self.sensor_query_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        source_tokens: Tensor,
        source_pos: Tensor,
        sensor_pos: Tensor,
        sensor_mask: Tensor,
    ) -> Tensor:
        bsz, n_steps, _, d_model = source_tokens.shape
        sensor_queries = self.sensor_query_scale * self.sensor_pos_embed(sensor_pos)
        sensor_queries = sensor_queries[:, None, :, :].expand(bsz, n_steps, -1, -1)
        return self.attn(
            query=sensor_queries,
            key_value=source_tokens,
            query_pos=sensor_pos,
            key_pos=source_pos[None, :, :],
            key_mask=torch.ones(
                bsz,
                source_tokens.shape[2],
                dtype=torch.bool,
                device=source_tokens.device,
            ),
        ) * sensor_mask[:, None, :, None].to(source_tokens.dtype)


@dataclass
class ModelOutput:
    logits: Tensor
    pooled: Tensor
    source_tokens: Tensor
    sensor_tokens: Tensor
    recon_tokens: Tensor
    channel_mask: Tensor
    domain_logits: Optional[Tensor] = None


class SourceStateTransformer(nn.Module):
    def __init__(
        self,
        source_positions: np.ndarray,
        n_classes: int,
        d_model: int = 128,
        patch_size: int = 25,
        patch_stride: int = 10,
        depth: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
        graph_adjacency: Optional[np.ndarray] = None,
        source_names: Optional[Tuple[str, ...]] = None,
        n_domains: int = 0,
        domain_lambda: float = 0.1,
        sigma: float = 0.35,
    ) -> None:
        super().__init__()
        source_pos = torch.as_tensor(source_positions, dtype=torch.float32)
        self.register_buffer("source_pos", source_pos)
        self.source_names = source_names
        self.n_sources = int(source_pos.shape[0])
        self.d_model = int(d_model)
        self.n_classes = int(n_classes)

        if graph_adjacency is None:
            graph_adjacency = np.eye(self.n_sources, dtype=np.float32)

        self.tokenizer = TemporalPatchTokenizer(
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
        )
        self.sensor_pos_embed = PositionMLP(d_model)
        self.source_pos_embed = PositionMLP(d_model)
        self.sensor_mask_token = nn.Parameter(torch.zeros(d_model))
        self.source_queries = nn.Parameter(torch.randn(self.n_sources, d_model) * 0.02)
        self.sensor_to_source = MultiHeadCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            sigma=sigma,
        )
        self.blocks = nn.ModuleList(
            [
                SourceStateBlock(
                    adjacency=graph_adjacency,
                    d_model=d_model,
                    dropout=dropout,
                    mlp_ratio=4.0,
                )
                for _ in range(depth)
            ]
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

    def _mask_sensor_tokens(
        self,
        sensor_tokens: Tensor,
        sensor_mask: Tensor,
        mask_ratio: float,
    ) -> Tuple[Tensor, Tensor]:
        bsz, n_steps, n_channels, d_model = sensor_tokens.shape
        if mask_ratio <= 0.0:
            return sensor_tokens, torch.zeros(
                bsz, n_channels, dtype=torch.bool, device=sensor_tokens.device
            )

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

    def encode(
        self,
        x: Tensor,
        sensor_pos: Tensor,
        sensor_mask: Tensor,
        mask_ratio: float = 0.15,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sensor_tokens = self.tokenizer(x)
        sensor_tokens = sensor_tokens + self.sensor_pos_embed(sensor_pos)[:, None, :, :]
        original_sensor_tokens = sensor_tokens.clone()
        sensor_tokens, channel_mask = self._mask_sensor_tokens(
            sensor_tokens=sensor_tokens,
            sensor_mask=sensor_mask,
            mask_ratio=mask_ratio,
        )

        bsz, n_steps, _, _ = sensor_tokens.shape
        source_query = self.source_queries + self.source_pos_embed(self.source_pos)
        source_query = source_query[None, None, :, :].expand(bsz, n_steps, -1, -1)
        source_tokens = self.sensor_to_source(
            query=source_query,
            key_value=sensor_tokens,
            query_pos=self.source_pos[None, :, :],
            key_pos=sensor_pos,
            key_mask=sensor_mask,
        )
        for block in self.blocks:
            source_tokens = block(source_tokens)
        source_tokens = self.source_norm(source_tokens)
        return source_tokens, original_sensor_tokens, sensor_tokens, channel_mask

    def forward(
        self,
        x: Tensor,
        sensor_pos: Tensor,
        sensor_mask: Tensor,
        mask_ratio: float = 0.15,
    ) -> ModelOutput:
        source_tokens, original_sensor_tokens, masked_sensor_tokens, channel_mask = self.encode(
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

        pool_logits = self.pool_head(source_tokens).squeeze(-1)  # [B, N, S]
        pool_weights = torch.softmax(pool_logits.flatten(start_dim=1), dim=-1).view_as(pool_logits)
        pooled = (pool_weights.unsqueeze(-1) * source_tokens).sum(dim=(1, 2))
        logits = self.classifier(pooled)

        domain_logits = None
        if self.domain_head is not None:
            domain_logits = self.domain_head(self.domain_grl(pooled))

        return ModelOutput(
            logits=logits,
            pooled=pooled,
            source_tokens=source_tokens,
            sensor_tokens=original_sensor_tokens,
            recon_tokens=recon_tokens,
            channel_mask=channel_mask,
            domain_logits=domain_logits,
        )
