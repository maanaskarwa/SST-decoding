"""
ENIGMA-style spatio-temporal CNN for go vs. stop EEG decoding

Inspired by the ENIGMA EEG architecture (heavy temporal filtering + pooling + a compact spatial embedding + projector).

Architecture
-------------------------------------
Input: (B, C, T)

1. Temporal filtering (learn many bandpass-like features)
   - Conv1d(1, temporal_filters, kernel=temporal_kernel) along time

2. Aggressive temporal pooling
   - MaxPool1d(kernel=temporal_pool_kernel, stride=temporal_pool_stride)

3. Spatial embedding (per-"time-bin" after pooling)
   - A small linear layer that maps the C channels at each pooled time step
     into a low-dimensional `embedding_dim` (default 4).
   - Produces a sequence of short vectors.

4. Residual projector (the "head" that mixes the embedded sequence)
   - Several layers that treat the (pooled_time, embedding_dim) as the feature
     map and project up to `projector_dim`.
   - Dropout, GELU, etc.

5. Global average pool over the (now very short) time axis + classifier
   - Final Linear(projector_dim → 2)

Tensor shapes
--------------------------------------------
x:                        (B, C, T)
after temporal conv:      (B, temporal_filters, T)
after pool:               (B, temporal_filters, ~T/stride)
after spatial embed:      (B, embedding_dim, pooled_steps)
after projector:          (B, projector_dim, pooled_steps)
after GAP:                (B, projector_dim)
logits:                   (B, 2)
"""

from __future__ import annotations

import torch
from torch import nn

from pipeline.models.utils import make_classifier


class EnigmaStyleGoStop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_times: int,
        temporal_filters: int = 40,
        temporal_kernel: int = 5,
        temporal_pool_kernel: int = 17,
        temporal_pool_stride: int = 5,
        embedding_dim: int = 4,
        projector_dim: int = 128,
        dropout: float = 0.5,
        projector_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if n_times < 8:
            raise ValueError("n_times must be >= 8")
        if temporal_kernel < 3 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd and >= 3")
        if temporal_pool_kernel < 2 or temporal_pool_stride < 1:
            raise ValueError("invalid temporal pooling settings")

        pooled_steps = (n_times - temporal_pool_kernel) // temporal_pool_stride + 1
        if pooled_steps < 1:
            raise ValueError(
                f"Temporal pooling leaves no steps: n_times={n_times}, "
                f"pool_kernel={temporal_pool_kernel}, pool_stride={temporal_pool_stride}"
            )

        self.in_channels = int(in_channels)
        self.n_times = int(n_times)
        self.temporal_filters = int(temporal_filters)
        self.embedding_dim = int(embedding_dim)
        self.pooled_steps = int(pooled_steps)
        self.latent_dim = int(self.embedding_dim * self.pooled_steps)

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.temporal_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.temporal_pool = nn.AvgPool2d(
            kernel_size=(1, temporal_pool_kernel),
            stride=(1, temporal_pool_stride),
        )
        self.temporal_bn = nn.BatchNorm2d(self.temporal_filters)

        self.spatial_conv = nn.Conv2d(
            in_channels=self.temporal_filters,
            out_channels=self.temporal_filters,
            kernel_size=(self.in_channels, 1),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(self.temporal_filters)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.embedding_conv = nn.Conv2d(
            in_channels=self.temporal_filters,
            out_channels=self.embedding_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.project_in = nn.Linear(self.latent_dim, projector_dim)
        self.project_refine = nn.Linear(projector_dim, projector_dim)
        self.project_act = nn.GELU()
        self.project_dropout = nn.Dropout(projector_dropout)
        self.project_norm = nn.LayerNorm(projector_dim)

        self.go_head = make_classifier(d_model=projector_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.temporal_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.spatial_conv.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.xavier_uniform_(self.embedding_conv.weight)
        if self.embedding_conv.bias is not None:
            nn.init.zeros_(self.embedding_conv.bias)

        nn.init.xavier_uniform_(self.project_in.weight)
        nn.init.zeros_(self.project_in.bias)
        nn.init.xavier_uniform_(self.project_refine.weight)
        nn.init.zeros_(self.project_refine.bias)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) where C = EEG channels and T = time samples.
        h = x.unsqueeze(1)  # (B, 1, C, T); treat EEG as a channel-by-time image.
        h = self.temporal_conv(
            h
        )  # (B, temporal_filters, C, T); temporal filtering per channel.
        h = self.temporal_pool(h)  # (B, temporal_filters, C, pooled_steps).
        h = self.activation(
            self.temporal_bn(h)
        )  # (B, temporal_filters, C, pooled_steps).

        h = self.spatial_conv(
            h
        )  # (B, temporal_filters, 1, pooled_steps); mix all EEG channels.
        h = self.activation(
            self.spatial_bn(h)
        )  # (B, temporal_filters, 1, pooled_steps).
        h = self.dropout(h)  # (B, temporal_filters, 1, pooled_steps).

        h = self.embedding_conv(h)  # (B, embedding_dim, 1, pooled_steps).
        return torch.flatten(h, start_dim=1)  # (B, embedding_dim * pooled_steps).

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode_latent(x)  # (B, latent_dim).
        coarse = self.project_in(latent)  # (B, projector_dim).
        refined = self.project_refine(
            self.project_dropout(self.project_act(coarse))
        )  # (B, projector_dim).
        return self.project_norm(coarse + refined)  # (B, projector_dim).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.go_head(pooled)
