"""Central model-name metadata for SST campaign entrypoints and analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for one model name accepted by campaign scripts."""

    name: str
    display_name: str
    base_family: str
    metadata_filename: str
    supports_attention: bool = False
    aliases: tuple[str, ...] = ()


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="cnn_transformer",
        display_name="CNN+Transformer",
        base_family="cnn_transformer",
        metadata_filename="cnn_transformer_run_metadata.json",
        supports_attention=True,
    ),
    ModelSpec(
        name="transformer_only",
        display_name="Transformer only",
        base_family="transformer_only",
        metadata_filename="transformer_only_run_metadata.json",
        supports_attention=True,
        aliases=("transformer", "transformer-only"),
    ),
    ModelSpec(
        name="pure_cnn",
        display_name="Pure CNN",
        base_family="pure_cnn",
        metadata_filename="pure_cnn_run_metadata.json",
        aliases=("cnn_only", "cnn-only"),
    ),
    ModelSpec(
        name="enigma",
        display_name="Spatio-temporal CNN",
        base_family="enigma",
        metadata_filename="enigma_style_run_metadata.json",
        aliases=("enigma_style",),
    ),
)

_SPEC_BY_NAME: dict[str, ModelSpec] = {
    key: spec for spec in MODEL_SPECS for key in (spec.name, *spec.aliases)
}


def canonical_model_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _SPEC_BY_NAME:
        raise ValueError(f"Unsupported model: {name}")
    return _SPEC_BY_NAME[key].name


def model_spec(name: str) -> ModelSpec:
    return _SPEC_BY_NAME[canonical_model_name(name)]


def model_name_from_cfg(cfg: dict[str, Any]) -> str:
    if "model" not in cfg:
        raise ValueError("Missing model in run_config.json")
    return canonical_model_name(str(cfg["model"]))


def base_family(name: str) -> str:
    return model_spec(name).base_family


def metadata_filename(name: str) -> str:
    return model_spec(name).metadata_filename


def metadata_filenames() -> tuple[str, ...]:
    seen: list[str] = []
    for spec in MODEL_SPECS:
        if spec.metadata_filename not in seen:
            seen.append(spec.metadata_filename)
    return tuple(seen)


def supports_attention(name: str) -> bool:
    return model_spec(name).supports_attention
