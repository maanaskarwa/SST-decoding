"""
CNN+Transformer architecture for go vs. stop EEG decoding

Main model

Architecture
-------------------------------------
Input:  (B, C, T)   raw epoched EEG  (C=channels, T=time samples)

1. Temporal CNN frontend (local feature extraction + mild downsampling)
   - Conv1d(k=9, pad=4) → BN → GELU → Dropout
   - Depthwise Conv1d(k=5, groups=CNN_WIDTH) → BN → GELU → Dropout
   - MaxPool1d(k=2, s=2)  →  halves the time dimension
   - Conv1d(1x1) to d_model

2. Tokenization, positional encoding
   - Transpose to (B, T//2, d_model)  → one token per pooled time step
   - Prepend learnable [CLS] token
   - Add sinusoidal positional encoding

3. Transformer encoder
   - TransformerEncoder
   - n_layers, n_heads, ff_dim, dropout

4. Classification head
   - Take only the CLS token representation
   - LayerNorm
   - Linear(d_model → 2)  → logits for (go, stop)

Tensor shapes
-----------------------
x:                  (B, C, T)
after frontend:     (B, d_model, T//2)
tokens (+CLS):      (B, 1 + T//2, d_model)
after transformer:  (B, 1 + T//2, d_model)
CLS pooled:         (B, d_model)
logits:             (B, 2)
"""

from __future__ import annotations

from typing import override

import numpy as np
import torch
from torch import nn

from pipeline.models.utils import (
    make_classifier,
    make_transformer_encoder,
    reduce_attention_times,
    sinusoidal_positional_encoding,
)


class CNNTransformerGoStop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        cnn_width: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
        record_attention: bool = False,
    ) -> None:
        self.time_downsample = 2
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.record_attention = bool(record_attention)

        self.temporal_frontend = nn.Sequential(
            nn.Conv1d(in_channels, cnn_width, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(cnn_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                cnn_width,
                cnn_width,
                kernel_size=5,
                padding=2,
                groups=cnn_width,
                bias=False,
            ),
            nn.BatchNorm1d(cnn_width),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=self.time_downsample, stride=self.time_downsample),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_width, d_model, kernel_size=1, bias=True),
        )

        self.encoder = make_transformer_encoder(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            n_layers=n_layers,
            record_attention=self.record_attention,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pre_head_norm = nn.LayerNorm(d_model)
        self.go_head = make_classifier(d_model=d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.temporal_frontend(x)
        # (B, d_model, floor(T / 2))
        x = x.transpose(1, 2)
        # (B, floor(T / 2), d_model); one token per pooled time step.
        batch_size, _, d_model = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        # (B, 1, d_model)
        tokens = torch.cat([cls, x], dim=1)  # (B, 1 + floor(T / 2), d_model).

        pe = sinusoidal_positional_encoding(
            length=tokens.shape[1],
            d_model=d_model,
            device=tokens.device,
            dtype=tokens.dtype,
        )  # (1 + floor(T / 2), d_model).
        tokens = tokens + pe.unsqueeze(0)  # (B, 1 + floor(T / 2), d_model).
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(x)  # (B, tokens, d_model).
        if self.record_attention:
            encoded = self.encoder(
                tokens, record_attention=False
            )  # (B, tokens, d_model).
        else:
            encoded = self.encoder(tokens)  # (B, tokens, d_model).
        return self.pre_head_norm(encoded[:, 0, :])  # CLS token: (B, d_model).

    def encode_with_attention(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.record_attention:
            raise RuntimeError(
                "Set record_attention=True before calling encode_with_attention"
            )
        tokens = self._tokens(x)  # (B, tokens, d_model).
        encoded, attn_per_layer = self.encoder(tokens, record_attention=True)
        pooled = self.pre_head_norm(encoded[:, 0, :])  # (B, d_model).
        return pooled, torch.stack(
            attn_per_layer, dim=0
        )  # attention: (layers, B, heads, tokens, tokens).

    def get_attention_times(self, original_times_s: np.ndarray) -> np.ndarray:
        """Since we downsample and want to be able to find which timestamps from the original input contribute to the most signal in learning the answer (in analyze_attention.py)"""
        return reduce_attention_times(original_times_s, pool_size=self.time_downsample)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.go_head(pooled)
