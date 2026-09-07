"""Learned multi-view conditioning for TRELLIS.2 image-to-3D pipelines.

The original TRELLIS.2 image conditioner accepts ``[B, N, C]`` DINO tokens.
It has no view axis: passing several images to the DINO extractor creates a
batch of independent conditions. This module explicitly keeps views as
``[B, V, N, C]`` and learns to reduce them to one object-level condition.

The weights in this file are intentionally *not* part of the published
single-view checkpoint. They must be trained (and loaded separately) before
they are suitable for multi-view reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiViewConditioningConfig:
    """Serializable architecture parameters for multi-view checkpoints."""

    feature_dim: int = 1024
    max_views: int = 4
    num_fused_tokens: int = 256
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    camera_metadata_dim: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class _FusionBlock(nn.Module):
    """Perceiver-style cross-view fusion followed by latent self-attention."""

    def __init__(self, feature_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.source_norm = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(feature_dim)
        self.self_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, int(feature_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(int(feature_dim * mlp_ratio), feature_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The cross-attention is the learned operation that relates every
        # fused object token to DINO tokens from all valid views. Flattening
        # source tokens only forms an attention sequence; it is not fusion.
        cross, _ = self.cross_attention(
            self.cross_norm(fused_tokens),
            self.source_norm(source_tokens),
            self.source_norm(source_tokens),
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        fused_tokens = fused_tokens + cross

        refined, _ = self.self_attention(
            self.self_norm(fused_tokens),
            self.self_norm(fused_tokens),
            self.self_norm(fused_tokens),
            need_weights=False,
        )
        fused_tokens = fused_tokens + refined
        return fused_tokens + self.mlp(self.mlp_norm(fused_tokens))


class MultiViewFeatureFusion(nn.Module):
    """Fuse DINO features from up to four views into one object condition.

    Inputs have a distinct object batch and view dimension:

    * ``features``: ``[B, V, N, C]`` DINO tokens.
    * ``view_mask``: optional boolean ``[B, V]`` mask, where ``True`` means a
      valid input view. Padded views cannot participate in attention.
    * ``camera_metadata``: optional ``[B, V, D]`` metadata. It is only
      accepted when ``camera_metadata_dim=D`` was configured; this module
      never invents camera poses from image-only input.

    Returns one unified object condition ``[B, num_fused_tokens, C]``. The
    transformer is deliberately learned, rather than an average or a raw
    token concatenation.
    """

    def __init__(
        self,
        config: Optional[MultiViewConditioningConfig] = None,
        **config_kwargs,
    ):
        super().__init__()
        # Accept keyword architecture arguments so the repository's JSON model
        # registry can instantiate this module during fine-tuning.
        if config is None:
            config = MultiViewConditioningConfig(**config_kwargs)
        elif isinstance(config, dict):
            config = MultiViewConditioningConfig(**config)
        elif config_kwargs:
            raise ValueError("Pass either config or keyword architecture arguments, not both")
        if config.max_views < 1:
            raise ValueError("max_views must be at least 1")
        if config.num_fused_tokens < 1:
            raise ValueError("num_fused_tokens must be at least 1")
        if config.feature_dim % config.num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        if config.camera_metadata_dim < 0:
            raise ValueError("camera_metadata_dim cannot be negative")

        self.config = config
        self.view_embedding = nn.Embedding(config.max_views, config.feature_dim)
        self.fused_token_queries = nn.Parameter(
            torch.empty(1, config.num_fused_tokens, config.feature_dim)
        )
        nn.init.trunc_normal_(self.fused_token_queries, std=0.02)
        self.camera_projection = (
            nn.Linear(config.camera_metadata_dim, config.feature_dim)
            if config.camera_metadata_dim
            else None
        )
        self.blocks = nn.ModuleList([
            _FusionBlock(
                config.feature_dim,
                config.num_heads,
                config.mlp_ratio,
                config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(config.feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        camera_metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "Multi-view features must have shape [B, V, N, C], "
                f"got {tuple(features.shape)}"
            )
        batch_size, num_views, num_tokens, feature_dim = features.shape
        if not 1 <= num_views <= self.config.max_views:
            raise ValueError(
                f"Expected 1-{self.config.max_views} views, got {num_views}"
            )
        if feature_dim != self.config.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.config.feature_dim}, got {feature_dim}"
            )

        if view_mask is None:
            view_mask = torch.ones(
                batch_size, num_views, dtype=torch.bool, device=features.device,
            )
        else:
            if view_mask.shape != (batch_size, num_views):
                raise ValueError(
                    "view_mask must have shape [B, V] matching features; "
                    f"got {tuple(view_mask.shape)}"
                )
            view_mask = view_mask.to(device=features.device, dtype=torch.bool)
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("Every object must have at least one valid view")

        view_ids = torch.arange(num_views, device=features.device)
        source = features + self.view_embedding(view_ids).view(1, num_views, 1, feature_dim)

        if camera_metadata is not None:
            if self.camera_projection is None:
                raise ValueError(
                    "camera_metadata was provided but this checkpoint was "
                    "configured without camera_metadata_dim"
                )
            expected = (batch_size, num_views, self.config.camera_metadata_dim)
            if tuple(camera_metadata.shape) != expected:
                raise ValueError(
                    "camera_metadata must have shape [B, V, D] matching the "
                    f"configured metadata size; expected {expected}, got "
                    f"{tuple(camera_metadata.shape)}"
                )
            source = source + self.camera_projection(
                camera_metadata.to(device=features.device, dtype=features.dtype)
            ).unsqueeze(2)

        # Keep the view mask aligned with *every* DINO token belonging to the
        # view. ``True`` in MHA's key_padding_mask means "ignore".
        source = source.reshape(batch_size, num_views * num_tokens, feature_dim)
        source_padding_mask = (~view_mask).unsqueeze(-1).expand(
            batch_size, num_views, num_tokens
        ).reshape(batch_size, num_views * num_tokens)

        fused = self.fused_token_queries.expand(batch_size, -1, -1)
        for block in self.blocks:
            fused = block(fused, source, source_padding_mask)
        return self.output_norm(fused)


class MultiViewConditioningAdapter(nn.Module):
    """Trainable adapter from fused tokens to TRELLIS' 1024-D condition space."""

    def __init__(self, feature_dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(feature_dim * mlp_ratio)
        self.norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim != 3:
            raise ValueError(
                "Fused features must have shape [B, N_fused, C], "
                f"got {tuple(fused_features.shape)}"
            )
        return fused_features + self.projection(self.norm(fused_features))
