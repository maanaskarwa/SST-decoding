from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from pipeline.misc import save_json
from sst_campaign.utils.common import timestamp_slug
from sst_campaign.utils.config import load_toml_config
from sst_campaign.utils.model_specs import base_family, supports_attention
from sst_campaign.utils.run_tasks import (
    TaskRecord,
    emit_progress,
    manifest_status_frame,
    parallel_settings,
    run_task_specs,
    write_campaign_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SST deep-model campaign.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-label", default="")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Repo/data root passed to child scripts; defaults to campaign.data_root or this checkout",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _task_record(
    task_id: int,
    experiment_family: str,
    model_name: str,
    base_family: str,
    section_name: str,
    args: list[str],
) -> TaskRecord:
    return TaskRecord(
        task_id=int(task_id),
        experiment_family=experiment_family,
        model_name=model_name,
        base_model_family=base_family,
        section_name=section_name,
        args=args,
    )


def _extend_cli_args(
    task_args: list[str],
    cfg_block: dict[str, Any],
    *,
    value_keys: list[str] | tuple[str, ...] = (),
    bool_keys: list[str] | tuple[str, ...] = (),
) -> None:
    for key in value_keys:
        if key in cfg_block:
            task_args.extend(["--" + key.replace("_", "-"), str(cfg_block[key])])
    for key in bool_keys:
        if bool(cfg_block.get(key, False)):
            task_args.append("--" + key.replace("_", "-"))


_RUNTIME_VALUE_KEYS = ["matmul_precision"]
_RUNTIME_BOOL_KEYS = ["cudnn_benchmark"]
_LOADER_VALUE_KEYS = ["num_workers", "prefetch_factor"]
_LOADER_BOOL_KEYS = ["persistent_workers"]
_TRAIN_VALUE_KEYS = ["amp_dtype"]
_TRAIN_BOOL_KEYS = ["amp"]
_TRAINING_SCRIPT_MAP = {
    "baseline": "sst_campaign/experiments/train_repro_baseline.py",
    "loso": "sst_campaign/experiments/train_zero_shot_loso.py",
}


def _campaign_root(common: dict[str, Any], campaign_label: str) -> Path:
    campaign_root = Path(common.get("output_root", "sst_campaign_runs"))
    if not campaign_root.is_absolute():
        campaign_root = ROOT / campaign_root
    return (
        campaign_root
        / f"{campaign_label or common.get('name', 'campaign')}__{timestamp_slug()}"
    )


def _data_root(common: dict[str, Any], cli_data_root: str | None) -> Path:
    data_root = Path(cli_data_root or common.get("data_root", common.get("root", ".")))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    return data_root.resolve()


def _append_section_values(
    task_args: list[str], section: dict[str, Any], keys: list[str] | tuple[str, ...]
) -> None:
    for key in keys:
        if key in section:
            task_args.extend(["--" + key.replace("_", "-"), str(section[key])])


def _extend_performance_args(
    task_args: list[str], performance: dict[str, Any], *, include_train: bool
) -> None:
    value_keys = _RUNTIME_VALUE_KEYS + _LOADER_VALUE_KEYS
    bool_keys = _RUNTIME_BOOL_KEYS + _LOADER_BOOL_KEYS
    if include_train:
        value_keys += _TRAIN_VALUE_KEYS
        bool_keys += _TRAIN_BOOL_KEYS
    _extend_cli_args(task_args, performance, value_keys=value_keys, bool_keys=bool_keys)


def _build_training_tasks(
    *,
    cfg: dict[str, Any],
    campaign_root: Path,
    device: str,
    random_state: int,
    train_subjects: list[Any],
    models: list[str],
    performance: dict[str, Any],
    data_root: Path,
    start_task_id: int = 1,
) -> tuple[list[TaskRecord], int]:
    tasks: list[TaskRecord] = []
    task_id = start_task_id
    artifacts_dir = campaign_root / "artifacts"

    for experiment_family, section_keys in (
        ("baseline", ["epochs", "patience", "batch_size", "n_splits", "val_fraction"]),
        ("loso", ["epochs", "patience", "batch_size", "val_fraction"]),
    ):
        if not cfg.get(experiment_family, {}).get("enabled", True):
            continue
        section = cfg[experiment_family]
        for model_name in models:
            task_args = [
                "--root",
                str(data_root),
                "--subjects",
                *[str(s) for s in train_subjects],
                "--model",
                model_name,
                "--device",
                device,
                "--random-state",
                str(random_state),
                "--output-dir",
                str(artifacts_dir),
                "--run-label",
                f"{experiment_family}_{model_name}",
            ]
            _append_section_values(task_args, section, section_keys)
            _extend_performance_args(task_args, performance, include_train=True)
            tasks.append(
                _task_record(
                    task_id,
                    experiment_family,
                    model_name,
                    base_family(model_name),
                    experiment_family,
                    task_args,
                )
            )  # type: ignore[reportArgumentType]
            task_id += 1

    return tasks, task_id


def _select_best_training_runs(
    successful_runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    train_candidates = [
        row
        for row in successful_runs
        if row["experiment_family"] in {"baseline", "loso"}
    ]
    best_by_family: dict[str, dict[str, Any]] = {}
    for row in train_candidates:
        family = row["base_model_family"]
        score = float(row.get("mean_test_balanced_accuracy", float("-inf")))
        auc = float(row.get("mean_test_auc", float("-inf")))
        prev = best_by_family.get(family)
        if (
            prev is None
            or score > float(prev.get("mean_test_balanced_accuracy", float("-inf")))
            or (
                score == float(prev.get("mean_test_balanced_accuracy", float("-inf")))
                and auc > float(prev.get("mean_test_auc", float("-inf")))
            )
        ):
            best_by_family[family] = row
    return best_by_family


def _run_interpretability_tasks(
    *,
    cfg: dict[str, Any],
    campaign_root: Path,
    device: str,
    random_state: int,
    performance: dict[str, Any],
    data_root: Path,
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    best_by_family: dict[str, dict[str, Any]],
    task_id: int,
    parallel: dict[str, Any] | None = None,
) -> int:
    if cfg.get("interpretability", {}).get("enabled", True):
        task_specs: list[tuple[TaskRecord, str]] = []
        interp_cfg = cfg["interpretability"]
        for family, source_row in best_by_family.items():
            task_args = [
                "--root",
                str(data_root),
                "--run-dir",
                source_row["run_dir"],
                "--device",
                device,
                "--random-state",
                str(random_state),
                "--output-dir",
                str(campaign_root / "artifacts"),
                "--run-label",
                f"interpretability_{family}",
            ]
            _append_section_values(
                task_args,
                interp_cfg,
                [
                    "ig_steps",
                    "ig_batch_size",
                    "max_stop_samples",
                    "max_go_samples",
                    "batch_size",
                    "p3_tmin",
                    "p3_tmax",
                ],
            )
            _extend_cli_args(
                task_args,
                performance,
                value_keys=_RUNTIME_VALUE_KEYS,
                bool_keys=_RUNTIME_BOOL_KEYS,
            )
            task = _task_record(
                task_id,
                "interpretability",
                source_row["model_name"],
                family,
                "interpretability",
                task_args,
            )
            task_id += 1
            task_specs.append(
                (
                    task,
                    "sst_campaign/interp.py",
                )
            )
        posthoc_runs: list[dict[str, Any]] = []
        run_task_specs(
            task_specs=task_specs,
            logs_dir=logs_dir,
            manifest_rows=manifest_rows,
            successful_runs=posthoc_runs,
            stop_on_error=False,
            phase="interpretability",
            cwd=ROOT,
            parallel=parallel,
        )
    return task_id


def _run_explainer_tasks(
    *,
    cfg: dict[str, Any],
    campaign_root: Path,
    device: str,
    random_state: int,
    performance: dict[str, Any],
    data_root: Path,
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    best_by_family: dict[str, dict[str, Any]],
    task_id: int,
    parallel: dict[str, Any] | None = None,
) -> int:
    if cfg.get("explainers", {}).get("enabled", True):
        task_specs: list[tuple[TaskRecord, str]] = []
        exp_cfg = cfg["explainers"]
        for family, source_row in best_by_family.items():
            task_args = [
                "--root",
                str(data_root),
                "--run-dir",
                source_row["run_dir"],
                "--device",
                device,
                "--random-state",
                str(random_state),
                "--output-dir",
                str(campaign_root / "artifacts"),
                "--run-label",
                f"explainers_{family}",
            ]
            if "methods" in exp_cfg:
                task_args.extend(["--methods", *[str(x) for x in exp_cfg["methods"]]])
            _append_section_values(
                task_args,
                exp_cfg,
                [
                    "max_samples",
                    "max_surrogate_samples",
                    "lime_samples",
                    "shap_samples",
                    "am_steps",
                    "explainer_batch_size",
                    "batch_size",
                    "p3_tmin",
                    "p3_tmax",
                ],
            )
            _extend_cli_args(
                task_args,
                performance,
                value_keys=_RUNTIME_VALUE_KEYS,
                bool_keys=_RUNTIME_BOOL_KEYS,
            )
            task = _task_record(
                task_id,
                "explainer_suite",
                source_row["model_name"],
                family,
                "explainers",
                task_args,
            )
            task_id += 1
            task_specs.append(
                (
                    task,
                    "sst_campaign/explainer.py",
                )
            )
        posthoc_runs: list[dict[str, Any]] = []
        run_task_specs(
            task_specs=task_specs,
            logs_dir=logs_dir,
            manifest_rows=manifest_rows,
            successful_runs=posthoc_runs,
            stop_on_error=False,
            phase="explainers",
            cwd=ROOT,
            parallel=parallel,
        )
    return task_id


def _run_attention_task(
    *,
    cfg: dict[str, Any],
    campaign_root: Path,
    device: str,
    random_state: int,
    performance: dict[str, Any],
    data_root: Path,
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    best_by_family: dict[str, dict[str, Any]],
    task_id: int,
) -> int:
    if cfg.get("attention", {}).get("enabled", True):
        att_cfg = cfg["attention"]
        for family, source_row in best_by_family.items():
            if not supports_attention(source_row["model_name"]):
                continue
            task_args = [
                "--root",
                str(data_root),
                "--run-dir",
                source_row["run_dir"],
                "--device",
                device,
                "--random-state",
                str(random_state),
                "--output-dir",
                str(campaign_root / "artifacts"),
                "--run-label",
                f"attention_{family}",
            ]
            _append_section_values(
                task_args,
                att_cfg,
                [
                    "max_samples",
                    "batch_size",
                    "p3_tmin",
                    "p3_tmax",
                    "rollout_permutations",
                ],
            )
            _extend_cli_args(
                task_args,
                performance,
                value_keys=_RUNTIME_VALUE_KEYS,
                bool_keys=_RUNTIME_BOOL_KEYS,
            )
            task = _task_record(
                task_id,
                "attention",
                source_row["model_name"],
                family,
                "attention",
                task_args,
            )
            task_id += 1
            run_task_specs(
                task_specs=[(task, "sst_campaign/analyze_attention.py")],
                logs_dir=logs_dir,
                manifest_rows=manifest_rows,
                successful_runs=[],
                stop_on_error=False,
                phase="attention",
                cwd=ROOT,
            )
    return task_id


def main() -> None:
    args = parse_args()
    cfg = load_toml_config(Path(args.config).resolve())
    common = cfg.get("campaign", {})
    campaign_root = _campaign_root(common, args.campaign_label)
    data_root = _data_root(common, args.data_root)
    campaign_root.mkdir(parents=True, exist_ok=False)
    logs_dir = campaign_root / "logs"
    save_json(campaign_root / "campaign_config_snapshot.json", cfg)

    device = str(common.get("device", "auto"))
    random_state = int(common.get("random_state", 9))
    train_subjects = common.get("train_subjects", ["all"])
    models = common.get("models", ["cnn_transformer", "pure_cnn", "enigma"])
    performance = cfg.get("performance", {})
    parallel = parallel_settings(cfg)

    emit_progress(
        "CAMPAIGN_STATUS",
        status="started",
        campaign_root=str(campaign_root),
        data_root=str(data_root),
        device=device,
        model_count=len(models),
    )

    tasks, task_id = _build_training_tasks(
        cfg=cfg,
        campaign_root=campaign_root,
        device=device,
        random_state=random_state,
        train_subjects=train_subjects,
        models=models,
        performance=performance,
        data_root=data_root,
    )
    manifest_rows: list[dict[str, Any]] = []
    successful_runs: list[dict[str, Any]] = []

    run_task_specs(
        task_specs=[
            (task, _TRAINING_SCRIPT_MAP[task["experiment_family"]]) for task in tasks
        ],
        logs_dir=logs_dir,
        manifest_rows=manifest_rows,
        successful_runs=successful_runs,
        stop_on_error=args.stop_on_error,
        phase="training",
        cwd=ROOT,
        parallel=parallel,
    )

    emit_progress(
        "CAMPAIGN_PHASE",
        phase="training_complete",
        campaign_root=str(campaign_root),
        completed_tasks=len(manifest_rows),
    )

    best_by_family = _select_best_training_runs(successful_runs)
    task_id = _run_interpretability_tasks(
        cfg=cfg,
        campaign_root=campaign_root,
        device=device,
        random_state=random_state,
        performance=performance,
        data_root=data_root,
        logs_dir=logs_dir,
        manifest_rows=manifest_rows,
        best_by_family=best_by_family,
        task_id=task_id,
        parallel=parallel,
    )
    task_id = _run_explainer_tasks(
        cfg=cfg,
        campaign_root=campaign_root,
        device=device,
        random_state=random_state,
        performance=performance,
        data_root=data_root,
        logs_dir=logs_dir,
        manifest_rows=manifest_rows,
        best_by_family=best_by_family,
        task_id=task_id,
        parallel=parallel,
    )
    _run_attention_task(
        cfg=cfg,
        campaign_root=campaign_root,
        device=device,
        random_state=random_state,
        performance=performance,
        data_root=data_root,
        logs_dir=logs_dir,
        manifest_rows=manifest_rows,
        best_by_family=best_by_family,
        task_id=task_id,
    )

    report_paths = write_campaign_reports(
        campaign_root=campaign_root,
        manifest_rows=manifest_rows,
        best_by_family=best_by_family,
    )
    manifest_df = manifest_status_frame(manifest_rows)

    emit_progress(
        "CAMPAIGN_STATUS",
        status="completed",
        campaign_root=str(campaign_root),
        task_count=len(manifest_rows),
        failure_count=int((manifest_df["status"] == "failed").sum())
        if not manifest_df.empty
        else 0,
    )
    print(f"Campaign root: {campaign_root}")
    print(f"Saved: {report_paths['manifest_json']}")
    print(f"Saved: {report_paths['manifest_csv']}")
    print(f"Saved: {report_paths['failures_csv']}")
    print(f"Saved: {report_paths['aggregate_csv']}")
    print(f"Saved: {report_paths['paper_md']}")


if __name__ == "__main__":
    main()
