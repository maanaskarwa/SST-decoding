"""
Temporal CNN architecture for go vs. stop EEG decoding

Architecture
-------------------------------------
Input: (B, C, T)

1. Initial temporal convolution (aggressive local feature extraction)
   - Conv1d(k=9) → BN → GELU → Dropout

2. Stack of `n_blocks-1` residual-style depthwise blocks
   - Each block:
       Depthwise Conv1d (groups=in_ch, k=kernel_size) → BN → GELU → Dropout
       1x1 Conv1d to next channel width
       (Optional final transition to d_model)
   - Progressive channel increase: cnn_width → cnn_width → d_model

3. Global average pooling (GAP) over time
   - Produces a single vector per example (B, d_model)

4. Classifier
   - Linear(d_model → 2)

Tensor shapes
-----------------------
x:                  (B, C, T)
after conv stack:   (B, d_model, T')
after GAP:          (B, d_model)
logits:             (B, 2)
"""

from __future__ import annotations

import torch
from torch import nn

from pipeline.models.utils import make_classifier


class PureCNNGoStop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        cnn_width: int = 128,
        dropout: float = 0.2,
        n_blocks: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")

        padding = kernel_size // 2
        channels = [in_channels, cnn_width, cnn_width, d_model]

        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, cnn_width, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(cnn_width),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for block_idx in range(n_blocks - 1):
            in_ch = channels[min(block_idx + 1, len(channels) - 2)]
            out_ch = channels[min(block_idx + 2, len(channels) - 1)]
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        in_ch,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=in_ch,
                        bias=False,
                    ),
                    nn.BatchNorm1d(in_ch),
                    nn.GELU(),
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout),
                ]
            )

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.pre_head_norm = nn.LayerNorm(d_model)
        self.go_head = make_classifier(d_model=d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)  # (B, d_model, 1); average over remaining time.
        x = x.squeeze(-1)  # (B, d_model).
        return self.pre_head_norm(x)  # (B, d_model).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.go_head(pooled)
