"""
Transformer-only architecture for go vs. stop EEG decoding

Architecture
-------------------------------------
Input:  (B, C, T)

1. Pointwise channel projection
   - Conv1d(kernel_size=1) from C channels → d_model
     mixes information across channels at each individual time step,
     no temporal mixing or downsampling.
   - Dropout

2. Tokenization + positional encoding
   - Transpose to (B, T, d_model)  → one token per original time sample
   - Prepend learnable [CLS] token
   - Add sinusoidal positional encoding

3. Transformer encoder
   - Same custom AttentionRecordingTransformerEncoder as CNN+Transformer

4. Classification head
   - CLS token only → LayerNorm → Linear(d_model → 2)

Tensor shapes
-----------------------
x:                  (B, C, T)
after pointwise:    (B, d_model, T)
tokens (+CLS):      (B, 1 + T, d_model)
after transformer:  (B, 1 + T, d_model)
CLS pooled:         (B, d_model)
logits:             (B, 2)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pipeline.models.utils import (
    make_classifier,
    make_transformer_encoder,
    reduce_attention_times,
    sinusoidal_positional_encoding,
)


class TransformerOnlyGoStop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
        record_attention: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.record_attention = bool(record_attention)
        self.input_projection = nn.Conv1d(
            in_channels, d_model, kernel_size=1, bias=True
        )
        self.input_dropout = nn.Dropout(dropout)

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
        nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) where C = EEG channels and T = time samples.
        x = self.input_projection(
            x
        )  # (B, d_model, T); pointwise projection across channels.
        x = x.transpose(1, 2)  # (B, T, d_model); one Transformer token per time sample.
        x = self.input_dropout(x)  # (B, T, d_model).

        batch_size, _, d_model = x.shape
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model).
        tokens = torch.cat([cls, x], dim=1)  # (B, 1 + T, d_model).
        pe = sinusoidal_positional_encoding(
            length=tokens.shape[1],
            d_model=d_model,
            device=tokens.device,
            dtype=tokens.dtype,
        )  # (1 + T, d_model).
        return tokens + pe.unsqueeze(0)  # (B, 1 + T, d_model).

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
        """No-op for transformer-only model. No temporal downsampling."""
        return original_times_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)  # (B, d_model).
        return self.go_head(pooled)  # logits: (B, 2) for go/stop.
