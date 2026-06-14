from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _normalize_obj(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize_obj(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_hash(
    config: dict[str, Any], ignore_keys: set[str] | None = None, n_chars: int = 10
) -> str:
    ignore_keys = ignore_keys or set()
    filtered = {k: v for k, v in config.items() if k not in ignore_keys}
    normalized = _normalize_obj(filtered)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n_chars]


def _next_run_index(cfg_dir: Path) -> int:
    max_idx = 0
    for p in cfg_dir.glob("run_*"):
        if not p.is_dir():
            continue
        match = re.match(r"^run_(\d+)(?:__.*)?$", p.name)
        if match is not None:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _sanitize_label(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:48]


def _append_run_registry(
    cfg_dir: Path,
    *,
    run_index: int,
    run_dir_name: str,
    experiment_name: str,
    config_hash_value: str,
    created_at_utc: str,
    created_at_local: str,
    run_label: str,
) -> None:
    registry_path = cfg_dir / "run_registry.csv"
    header = [
        "run_index",
        "run_dir_name",
        "experiment_name",
        "config_hash",
        "created_at_utc",
        "created_at_local",
        "run_label",
    ]
    write_header = not registry_path.exists()
    with registry_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_index": int(run_index),
                "run_dir_name": run_dir_name,
                "experiment_name": experiment_name,
                "config_hash": config_hash_value,
                "created_at_utc": created_at_utc,
                "created_at_local": created_at_local,
                "run_label": run_label,
            }
        )


def prepare_versioned_output_dir(
    base_output_dir: Path,
    experiment_name: str,
    config: dict[str, Any],
    disable_versioning: bool = False,
) -> tuple[Path, dict[str, Any]]:
    base_output_dir = base_output_dir.resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    now_utc_s = now_utc.isoformat()
    now_local_s = now_local.isoformat()

    if disable_versioning:
        run_dir = base_output_dir
        meta = {
            "versioning_enabled": False,
            "experiment_name": experiment_name,
            "created_at_utc": now_utc_s,
            "created_at_local": now_local_s,
        }
        with (run_dir / "run_version.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return run_dir, meta

    cfg_h = config_hash(
        config=config,
        ignore_keys={"output_dir", "disable_versioning", "versioning"},
    )

    cfg_dir = base_output_dir / experiment_name / f"cfg_{cfg_h}"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = cfg_dir / "config_snapshot.json"
    if not snapshot_path.exists():
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(_normalize_obj(config), f, indent=2, default=str)

    run_idx = _next_run_index(cfg_dir)
    run_label = _sanitize_label(str(config.get("run_label", "")))
    time_slug = now_local.strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{run_idx:04d}__{time_slug}"
    if run_label:
        run_dir_name = f"{run_dir_name}__{run_label}"
    run_dir = cfg_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)

    _append_run_registry(
        cfg_dir=cfg_dir,
        run_index=run_idx,
        run_dir_name=run_dir_name,
        experiment_name=experiment_name,
        config_hash_value=cfg_h,
        created_at_utc=now_utc_s,
        created_at_local=now_local_s,
        run_label=run_label,
    )

    meta = {
        "versioning_enabled": True,
        "experiment_name": experiment_name,
        "config_hash": cfg_h,
        "config_dir": str(cfg_dir),
        "run_index": int(run_idx),
        "run_dir_name": run_dir_name,
        "created_at_utc": now_utc_s,
        "created_at_local": now_local_s,
        "run_label": run_label,
    }
    with (run_dir / "run_version.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return run_dir, meta
