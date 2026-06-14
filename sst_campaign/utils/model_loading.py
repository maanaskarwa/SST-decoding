"""Model reconstruction helpers for saved SST campaign runs and training configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline.models.cnn_transformer import CNNTransformerGoStop
from pipeline.models.pure_cnn import PureCNNGoStop
from pipeline.models.spatio_temporal_cnn import EnigmaStyleGoStop
from pipeline.models.transformer_only import TransformerOnlyGoStop
from sst_campaign.utils.model_specs import metadata_filename, model_name_from_cfg


def load_run_cfg_meta(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_cfg_path = run_dir / "run_config.json"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_cfg_path}")
    with run_cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = model_name_from_cfg(cfg)
    meta_path = run_dir / metadata_filename(model_name)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return cfg, meta


def transformer_model_kwargs(
    meta: dict[str, Any],
    *,
    in_channels: int,
) -> dict[str, Any]:
    """Resolve shared CNN+Transformer constructor kwargs from saved metadata."""
    model_meta = meta["model"]
    return {
        "in_channels": in_channels,
        "d_model": int(model_meta["d_model"]),
        "cnn_width": int(model_meta["cnn_width"]),
        "n_heads": int(model_meta["n_heads"]),
        "n_layers": int(model_meta["n_layers"]),
        "ff_dim": int(model_meta["ff_dim"]),
        "dropout": float(model_meta["dropout"]),
    }


def transformer_only_model_kwargs(
    meta: dict[str, Any],
    *,
    in_channels: int,
) -> dict[str, Any]:
    """Resolve shared Transformer-only constructor kwargs from saved metadata."""
    model_meta = meta["model"]
    return {
        "in_channels": in_channels,
        "d_model": int(model_meta["d_model"]),
        "n_heads": int(model_meta["n_heads"]),
        "n_layers": int(model_meta["n_layers"]),
        "ff_dim": int(model_meta["ff_dim"]),
        "dropout": float(model_meta["dropout"]),
    }


def pure_cnn_model_kwargs(
    meta: dict[str, Any],
    *,
    in_channels: int,
) -> dict[str, Any]:
    """Resolve shared Pure-CNN constructor kwargs from saved metadata."""
    model_meta = meta["model"]
    return {
        "in_channels": in_channels,
        "d_model": int(model_meta["d_model"]),
        "cnn_width": int(model_meta["cnn_width"]),
        "dropout": float(model_meta["dropout"]),
        "n_blocks": int(model_meta["cnn_only_blocks"]),
        "kernel_size": int(model_meta["cnn_only_kernel"]),
    }


def enigma_model_kwargs(
    meta: dict[str, Any],
    *,
    in_channels: int,
    n_times: int,
) -> dict[str, Any]:
    """Resolve shared ENIGMA constructor kwargs from saved metadata."""
    model_meta = meta["model"]
    return {
        "in_channels": in_channels,
        "n_times": n_times,
        "temporal_filters": int(model_meta["temporal_filters"]),
        "temporal_kernel": int(model_meta["temporal_kernel"]),
        "temporal_pool_kernel": int(model_meta["temporal_pool_kernel"]),
        "temporal_pool_stride": int(model_meta["temporal_pool_stride"]),
        "embedding_dim": int(model_meta["embedding_dim"]),
        "projector_dim": int(model_meta["projector_dim"]),
        "dropout": float(model_meta["dropout"]),
        "projector_dropout": float(model_meta["projector_dropout"]),
    }


_MetadataField = tuple[str, str, Any]

_TRANSFORMER_FIELDS: tuple[_MetadataField, ...] = (
    ("d_model", "d_model", int),
    ("cnn_width", "cnn_width", int),
    ("n_heads", "n_heads", int),
    ("n_layers", "n_layers", int),
    ("ff_dim", "ff_dim", int),
    ("dropout", "dropout", float),
)
_TRANSFORMER_ONLY_FIELDS: tuple[_MetadataField, ...] = tuple(
    field for field in _TRANSFORMER_FIELDS if field[0] != "cnn_width"
)
_CNN_ONLY_FIELDS: tuple[_MetadataField, ...] = (
    ("d_model", "d_model", int),
    ("cnn_width", "cnn_width", int),
    ("cnn_only_blocks", "cnn_only_blocks", int),
    ("cnn_only_kernel", "cnn_only_kernel", int),
    ("dropout", "dropout", float),
)
_ENIGMA_FIELDS: tuple[_MetadataField, ...] = (
    ("temporal_filters", "enigma_temporal_filters", int),
    ("temporal_kernel", "enigma_temporal_kernel", int),
    ("temporal_pool_kernel", "enigma_temporal_pool_kernel", int),
    ("temporal_pool_stride", "enigma_temporal_pool_stride", int),
    ("embedding_dim", "enigma_embedding_dim", int),
    ("projector_dim", "enigma_projector_dim", int),
    ("dropout", "dropout", float),
    ("projector_dropout", "enigma_projector_dropout", float),
)

_TRAINING_METADATA_SPECS: dict[str, tuple[str, tuple[_MetadataField, ...]]] = {
    "cnn_transformer": ("CNNTransformerGoStop", _TRANSFORMER_FIELDS),
    "transformer_only": ("TransformerOnlyGoStop", _TRANSFORMER_ONLY_FIELDS),
    "pure_cnn": ("PureCNNGoStop", _CNN_ONLY_FIELDS),
    "enigma": ("EnigmaStyleGoStop", _ENIGMA_FIELDS),
}


def model_metadata_from_training_args(args: Any) -> dict[str, Any]:
    """Return the saved-run model metadata for a baseline training namespace."""
    model_name = model_name_from_cfg({"model": str(args.model)})
    type_name, fields = _TRAINING_METADATA_SPECS[model_name]
    metadata: dict[str, Any] = {"type": type_name}
    for output_key, attr_name, caster in fields:
        metadata[output_key] = caster(getattr(args, attr_name))
    return metadata


def _kwargs_from_training_args(
    args: Any, fields: tuple[_MetadataField, ...]
) -> dict[str, Any]:
    return {
        out_key: caster(getattr(args, attr_name))
        for out_key, attr_name, caster in fields
    }


@dataclass(frozen=True)
class _TrainingModelSpec:
    model_cls: type[torch.nn.Module]
    fields: tuple[_MetadataField, ...]
    needs_n_times: bool = False


_PURE_CNN_CONSTRUCTOR_FIELDS: tuple[_MetadataField, ...] = (
    ("d_model", "d_model", int),
    ("cnn_width", "cnn_width", int),
    ("dropout", "dropout", float),
    ("n_blocks", "cnn_only_blocks", int),
    ("kernel_size", "cnn_only_kernel", int),
)
_TRAINING_MODEL_SPECS: dict[str, _TrainingModelSpec] = {
    "cnn_transformer": _TrainingModelSpec(CNNTransformerGoStop, _TRANSFORMER_FIELDS),
    "transformer_only": _TrainingModelSpec(
        TransformerOnlyGoStop, _TRANSFORMER_ONLY_FIELDS
    ),
    "pure_cnn": _TrainingModelSpec(PureCNNGoStop, _PURE_CNN_CONSTRUCTOR_FIELDS),
    "enigma": _TrainingModelSpec(EnigmaStyleGoStop, _ENIGMA_FIELDS, needs_n_times=True),
}


def build_model_from_training_args(
    args: Any,
    *,
    in_channels: int,
    n_times: int,
) -> torch.nn.Module:
    """Build a model from the training CLI/config namespace."""
    model_name = model_name_from_cfg({"model": str(args.model)})
    spec = _TRAINING_MODEL_SPECS[model_name]
    kwargs = {
        "in_channels": in_channels,
        **_kwargs_from_training_args(args, spec.fields),
    }
    if spec.needs_n_times:
        kwargs["n_times"] = n_times
    return spec.model_cls(**kwargs)


def build_model_from_config(
    cfg: dict[str, Any],
    meta: dict[str, Any],
    *,
    in_channels: int,
    n_times: int,
    attention_enabled: bool = False,
) -> torch.nn.Module:
    model_name = model_name_from_cfg(cfg)
    if model_name == "transformer_only":
        return TransformerOnlyGoStop(
            record_attention=attention_enabled,
            **transformer_only_model_kwargs(meta=meta, in_channels=in_channels),
        )
    if model_name == "pure_cnn":
        return PureCNNGoStop(
            **pure_cnn_model_kwargs(meta=meta, in_channels=in_channels),
        )
    if model_name == "enigma":
        return EnigmaStyleGoStop(
            **enigma_model_kwargs(meta=meta, in_channels=in_channels, n_times=n_times),
        )
    return CNNTransformerGoStop(
        record_attention=attention_enabled,
        **transformer_model_kwargs(meta=meta, in_channels=in_channels),
    )


def subject_mapping_info_for_fold(
    fold_dir: Path, subject_col: np.ndarray
) -> tuple[np.ndarray, int]:
    map_path = fold_dir / "subject_index_map.json"
    with map_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    subj_map = {int(k): int(v) for k, v in payload["subject_to_index"].items()}
    unknown_idx = int(payload["unknown_subject_index"])
    mapped = np.asarray(
        [subj_map.get(int(s), unknown_idx) for s in subject_col], dtype=np.int64
    )
    return mapped, unknown_idx
