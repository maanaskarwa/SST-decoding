"""
1. sinusoidal_positional_encoding
   - The exact positional encoding added to the token sequence in both
     CNN+Transformer and TransformerOnly.

2. make_classifier
   - The standard final Linear(d_model → 2) used by every model for
     consistency of initialization and checkpoint compatibility.

3. The AttentionRecording* classes
   - These are the reason `record_attention=True` works on the transformer
     models.
   - They are drop-in replacements for nn.TransformerEncoderLayer /
     TransformerEncoder that can return per-head, per-layer attention
     weights when asked.
   - Used only for post-hoc interpretability (analyze_attention.py etc.).
   - When record_attention=False they behave exactly like the stock PyTorch
     versions (except for the fastpath toggle hack when attention is requested).

4. make_transformer_encoder
   - Factory that builds the right encoder (plain or attention-recording)
     given the record_attention flag.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn


def sinusoidal_positional_encoding(
    *,
    length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding (used by both transformer models).

    Matches the original "Attention is All You Need" formulation.
    """
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-np.log(10000.0) / d_model)
    )
    pe = torch.zeros((length, d_model), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def make_classifier(d_model: int) -> nn.Linear:
    """
    The single classifier head factory used by every SST model.
    Always Linear(d_model, 2) with xavier weight init and zero bias.
    """
    classifier = nn.Linear(d_model, 2)

    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    return classifier


class AttentionRecordingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Drop-in replacement for nn.TransformerEncoderLayer that can capture
    per-head attention weights when record_attention=True.

    Main change made in _sa_block: it passes need_weights=... and
    average_attn_weights=False to self_attn, then stores result.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.record_attention = False
        self.last_attn_weights: torch.Tensor | None = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.record_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights if self.record_attention else None
        return self.dropout1(attn_out)


class AttentionRecordingTransformerEncoder(nn.Module):
    """
    Drop-in replacement for nn.TransformerEncoder that supports the
    record_attention path.

    When record_attention=True it forces the slow path (disables the
    fastpath) and collects per-layer attention weights.
    The returned tuple from forward is what encode_with_attention
    in the model files turns into the final (pooled, attn_stack) result.
    """

    def __init__(
        self, encoder_layer: AttentionRecordingTransformerEncoderLayer, num_layers: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer]
            + [_copy_attention_layer(encoder_layer) for _ in range(1, num_layers)]
        )
        self.num_layers = int(num_layers)

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        record_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        output = src
        attn_weights: list[torch.Tensor] = []
        fastpath_was_enabled = torch.backends.mha.get_fastpath_enabled()
        if record_attention and fastpath_was_enabled:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            for layer in self.layers:
                attention_layer = cast(AttentionRecordingTransformerEncoderLayer, layer)
                attention_layer.record_attention = bool(record_attention)
                attention_layer.last_attn_weights = None
                output = attention_layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
                if record_attention:
                    if attention_layer.last_attn_weights is None:
                        raise RuntimeError("Attention weights were not recorded")
                    attn_weights.append(attention_layer.last_attn_weights)
        finally:
            if record_attention and fastpath_was_enabled:
                torch.backends.mha.set_fastpath_enabled(True)
        if record_attention:
            return output, attn_weights
        return output


def make_transformer_encoder(
    *,
    d_model: int,
    n_heads: int,
    ff_dim: int,
    dropout: float,
    n_layers: int,
    record_attention: bool,
) -> nn.Module:
    """
    When record_attention=False it returns a plain nn.TransformerEncoder
    (the fast path is left alone).

    When record_attention=True it returns an
    AttentionRecordingTransformerEncoder used in the paper's
    interpretability figures.
    """
    if record_attention:
        return AttentionRecordingTransformerEncoder(
            AttentionRecordingTransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layers,
        )
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ),
        num_layers=n_layers,
    )


def _copy_attention_layer(
    encoder_layer: AttentionRecordingTransformerEncoderLayer,
) -> AttentionRecordingTransformerEncoderLayer:
    return type(encoder_layer)(
        d_model=encoder_layer.self_attn.embed_dim,
        nhead=encoder_layer.self_attn.num_heads,
        dim_feedforward=encoder_layer.linear1.out_features,
        dropout=encoder_layer.dropout.p,
        activation="gelu",
        batch_first=True,
        norm_first=encoder_layer.norm_first,
    )


def reduce_attention_times(times_s: np.ndarray, *, pool_size: int) -> np.ndarray:
    """Downsample a time axis by averaging.

    Need this when analyzing attention rollouts. Refer sst_campaign/analyze_attention.py
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    if len(times_s) < 2:
        return times_s.copy()
    usable = times_s[: len(times_s) - (len(times_s) % pool_size)]
    if len(usable) == 0:
        return times_s.copy()
    return usable.reshape(-1, pool_size).mean(axis=1)
