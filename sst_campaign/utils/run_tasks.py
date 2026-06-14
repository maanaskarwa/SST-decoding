from __future__ import annotations

import csv
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd

from pipeline.misc import save_json
from sst_campaign.utils.common import run_python_script
from sst_campaign.utils.model_specs import metadata_filenames


class TaskRecord(TypedDict):
    """Structured dict for a campaign task"""

    task_id: int
    experiment_family: str
    model_name: str
    base_model_family: str
    section_name: str
    args: list[str]


def load_best_training_run(campaign_root: Path, expected_family: str) -> dict[str, Any]:
    metadata_path = campaign_root / "campaign_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing manifest: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    best = payload.get("best_training_runs_by_family", {}).get(expected_family)
    if not best:
        raise RuntimeError(
            f"No best training run found for family {expected_family} in {campaign_root}"
        )
    return best


def emit_progress(event: str, **fields: Any) -> None:
    parts = [event, f"timestamp_utc={datetime.now(timezone.utc).isoformat()}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", " ").strip()
        parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)


def parallel_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("parallel", {})
    max_tasks = max(1, int(block.get("max_tasks", 1)))
    max_gpu_slots = max(1, int(block.get("max_gpu_slots", max_tasks)))
    experiment_max_tasks = {
        str(experiment_family): max(1, int(limit))
        for experiment_family, limit in block.get("experiment_max_tasks", {}).items()
    }
    worker_count = (
        max([max_tasks, *experiment_max_tasks.values()])
        if experiment_max_tasks
        else max_tasks
    )
    return {
        "enabled": bool(block.get("enabled", False)) and worker_count > 1,
        "max_tasks": max_tasks,
        "worker_count": worker_count,
        "max_gpu_slots": max_gpu_slots,
        "default_gpu_slots": max(1, int(block.get("default_gpu_slots", 1))),
        "model_gpu_slots": block.get("model_gpu_slots", {}),
        "experiment_gpu_slots": block.get("experiment_gpu_slots", {}),
        "experiment_model_gpu_slots": {
            str(experiment_family): model_slots
            for experiment_family, model_slots in block.get(
                "experiment_model_gpu_slots", {}
            ).items()
        },
        "experiment_max_tasks": experiment_max_tasks,
    }


def run_task_specs(
    *,
    task_specs: list[tuple[TaskRecord, str]],
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    successful_runs: list[dict[str, Any]],
    stop_on_error: bool,
    phase: str,
    cwd: Path,
    parallel: dict[str, Any] | None = None,
) -> None:
    parallel = parallel or parallel_settings({})
    if not parallel["enabled"] or len(task_specs) <= 1:
        _run_task_specs_serial(
            task_specs=task_specs,
            logs_dir=logs_dir,
            manifest_rows=manifest_rows,
            successful_runs=successful_runs,
            stop_on_error=stop_on_error,
            cwd=cwd,
        )
        return

    emit_progress(
        "CAMPAIGN_PARALLEL",
        phase=phase,
        enabled=True,
        max_tasks=int(parallel["max_tasks"]),
        worker_count=min(int(parallel["worker_count"]), len(task_specs)),
        max_gpu_slots=int(parallel["max_gpu_slots"]),
    )
    _run_task_specs_parallel(
        task_specs=task_specs,
        logs_dir=logs_dir,
        manifest_rows=manifest_rows,
        successful_runs=successful_runs,
        stop_on_error=stop_on_error,
        cwd=cwd,
        parallel=parallel,
    )


def _run_task_specs_serial(
    *,
    task_specs: list[tuple[TaskRecord, str]],
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    successful_runs: list[dict[str, Any]],
    stop_on_error: bool,
    cwd: Path,
) -> None:
    for task, script in task_specs:
        row = _execute_campaign_task(
            task=task, script=script, logs_dir=logs_dir, cwd=cwd
        )
        _record_task_row(
            row=row, manifest_rows=manifest_rows, successful_runs=successful_runs
        )
        if row["returncode"] != 0 and stop_on_error:
            break


def _run_task_specs_parallel(
    *,
    task_specs: list[tuple[TaskRecord, str]],
    logs_dir: Path,
    manifest_rows: list[dict[str, Any]],
    successful_runs: list[dict[str, Any]],
    stop_on_error: bool,
    cwd: Path,
    parallel: dict[str, Any],
) -> None:
    next_task_idx = 0
    stop_submitting = False
    running: dict[Future[dict[str, Any]], tuple[TaskRecord, int]] = {}
    worker_count = min(int(parallel["worker_count"]), len(task_specs))
    max_gpu_slots = int(parallel["max_gpu_slots"])

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while next_task_idx < len(task_specs) or running:
            while not stop_submitting and next_task_idx < len(task_specs):
                task, script = task_specs[next_task_idx]
                if len(running) >= _task_max_tasks(task, parallel):
                    break
                slots = _task_gpu_slots(task, parallel)
                used_slots = sum(task_slots for _, task_slots in running.values())
                if running and used_slots + slots > max_gpu_slots:
                    break
                future = executor.submit(
                    _execute_campaign_task,
                    task=task,
                    script=script,
                    logs_dir=logs_dir,
                    cwd=cwd,
                )
                running[future] = (task, slots)
                next_task_idx += 1

            if not running:
                break

            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task, _slots = running.pop(future)
                try:
                    row = future.result()
                except BaseException as exc:
                    row = _task_exception_row(task, exc)
                _record_task_row(
                    row=row,
                    manifest_rows=manifest_rows,
                    successful_runs=successful_runs,
                )
                if row["returncode"] != 0 and stop_on_error:
                    stop_submitting = True


def _task_gpu_slots(task: TaskRecord, parallel: dict[str, Any]) -> int:
    experiment_family = str(task["experiment_family"])
    model_name = str(task["model_name"])
    requested_slots = int(
        parallel.get("experiment_model_gpu_slots", {})
        .get(experiment_family, {})
        .get(
            model_name,
            parallel.get("experiment_gpu_slots", {}).get(
                experiment_family,
                parallel.get("model_gpu_slots", {}).get(
                    model_name, parallel.get("default_gpu_slots", 1)
                ),
            ),
        )
    )
    return max(1, min(requested_slots, int(parallel["max_gpu_slots"])))


def _task_max_tasks(
    task: TaskRecord, parallel: dict[str, Any]
) -> int:  # shape: TaskRecord (see types above)
    experiment_limits = parallel.get("experiment_max_tasks", {})
    return max(
        1,
        int(
            experiment_limits.get(str(task["experiment_family"]), parallel["max_tasks"])
        ),
    )


def _execute_campaign_task(
    *, task: TaskRecord, script: str, cwd: Path, logs_dir: Path
) -> dict[str, Any]:  # shape: TaskRecord (see types above)
    _emit_task_start(task, script)  # type: ignore[arg-type] if strict
    row = _run_task(task=task, script=script, cwd=cwd, logs_dir=logs_dir)
    if row["status"] == "completed" and row["run_dir"]:
        row.update(
            _extract_metrics(
                run_dir=Path(str(row["run_dir"])),
                experiment_family=task["experiment_family"],
            )
        )
    _emit_task_end(row)
    return row


def _run_task(
    *, task: TaskRecord, script: str, cwd: Path, logs_dir: Path
) -> dict[str, Any]:  # TaskRecord shape expected at call sites
    result = run_python_script(script=script, args=task["args"], cwd=cwd)
    stdout_path, stderr_path = _task_log_paths(
        logs_dir=logs_dir, task_id=int(task["task_id"])
    )
    _write_task_log(stdout_path, result["stdout"])
    _write_task_log(stderr_path, result["stderr"])
    return {
        **task,
        "started_at_utc": result["started_at_utc"],
        "ended_at_utc": result["ended_at_utc"],
        "reproducible_command": result["reproducible_command"],
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "run_dir": result["run_dir"] or "",
        "status": "completed" if result["returncode"] == 0 else "failed",
        "returncode": int(result["returncode"]),
        "error_summary": (result["stderr"] or result["stdout"])[-800:],
    }


def _record_task_row(
    *,
    row: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    successful_runs: list[dict[str, Any]],
) -> None:
    if row["status"] == "completed" and row["run_dir"]:
        successful_runs.append(row.copy())
    manifest_rows.append(row)


def _task_exception_row(
    task: TaskRecord, exc: BaseException
) -> dict[str, Any]:  # shape: TaskRecord (see types above)
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        **task,
        "started_at_utc": timestamp,
        "ended_at_utc": timestamp,
        "reproducible_command": "",
        "stdout_log": "",
        "stderr_log": "",
        "run_dir": "",
        "status": "failed",
        "returncode": -1,
        "error_summary": repr(exc),
    }


def _emit_task_start(
    task: dict[str, Any], script: str
) -> None:  # TaskRecord shape expected at call sites
    emit_progress(
        "TASK_START",
        task_id=f"{task['task_id']:04d}",
        family=task["experiment_family"],
        model=task["model_name"],
        base_family=task["base_model_family"],
        section=task["section_name"],
        script=script,
    )


def _emit_task_end(row: dict[str, Any]) -> None:
    emit_progress(
        "TASK_END",
        task_id=f"{row['task_id']:04d}",
        family=row["experiment_family"],
        model=row["model_name"],
        status=row["status"],
        returncode=row["returncode"],
        run_dir=str(row["run_dir"]),
        stdout_log=row["stdout_log"],
    )


def _task_log_paths(logs_dir: Path, task_id: int) -> tuple[Path, Path]:
    task_stem = f"task_{task_id:04d}"
    return logs_dir / f"{task_stem}.stdout.log", logs_dir / f"{task_stem}.stderr.log"


def _write_task_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _extract_metrics(run_dir: Path, experiment_family: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    meta_path = next(
        (
            path
            for name in (
                *metadata_filenames(),
                "interpretability_summary.json",
                "explainer_suite_summary.json",
                "attention_summary.json",
            )
            if (path := run_dir / name).exists()
        ),
        None,
    )
    if meta_path is None:
        return metrics

    payload = json.loads(meta_path.read_text())
    block = payload.get("overall_metrics") or payload.get("overall")
    if isinstance(block, dict):
        for key in [
            "mean_test_auc",
            "mean_test_average_precision",
            "mean_test_accuracy",
            "mean_test_balanced_accuracy",
        ]:
            if key in block:
                metrics[key] = block[key]
    elif experiment_family in {"baseline", "loso"}:
        raise RuntimeError(
            f"Run metadata at {meta_path} has neither 'overall_metrics' nor legacy 'overall' block. "
            "Cannot extract test metrics for this training run."
        )
    if experiment_family in {"interpretability", "explainer_suite", "attention"}:
        for key in [
            "model_name",
            "model_type",
            "stop_ig_peak_time_s",
            "rollout_peak_time_s",
            "rollout_p3_share",
        ]:
            if key in payload:
                metrics[key] = payload[key]
    return metrics


def write_campaign_reports(
    *,
    campaign_root: Path,
    manifest_rows: list[dict[str, Any]],
    best_by_family: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Write metadata, aggregate CSVs, failures, and a compact Markdown summary."""
    manifest_rows = sorted(manifest_rows, key=lambda row: int(row.get("task_id", 0)))
    manifest_json = campaign_root / "campaign_metadata.json"
    manifest_csv = campaign_root / "campaign_run_index.csv"
    failures_csv = campaign_root / "campaign_failures.csv"
    aggregate_csv = campaign_root / "aggregate_metrics.csv"
    paper_md = campaign_root / "paper_summary.md"

    save_json(
        manifest_json,
        {
            "campaign_root": str(campaign_root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": manifest_rows,
            "best_training_runs_by_family": best_by_family,
        },
    )
    manifest_df = manifest_status_frame(
        manifest_rows,
        empty_columns=[
            "status",
            "experiment_family",
            "model_name",
            "base_model_family",
            "error_summary",
        ],
    )
    _write_manifest_csv(manifest_csv, manifest_rows, manifest_df)

    failure_df = cast(
        pd.DataFrame, manifest_df.loc[manifest_df["status"] == "failed"].copy()
    )
    failure_df.to_csv(failures_csv, index=False)
    manifest_df.copy().to_csv(aggregate_csv, index=False)

    paper_md.write_text(
        "# SST Campaign Paper Summary\n\n"
        + "## Training Comparison\n\n"
        + _markdown_table(_training_summary_frame(manifest_df))
        + "\n\n## Failed Runs\n\n"
        + _markdown_table(_failure_summary_frame(manifest_df))
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_json": manifest_json,
        "manifest_csv": manifest_csv,
        "failures_csv": failures_csv,
        "aggregate_csv": aggregate_csv,
        "paper_md": paper_md,
    }


def manifest_status_frame(
    manifest_rows: list[dict[str, Any]],
    *,
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a manifest dataframe with guaranteed columns for empty campaigns."""
    if manifest_rows:
        return pd.DataFrame(manifest_rows)
    return pd.DataFrame(
        {column: pd.Series(dtype=object) for column in (empty_columns or ["status"])}
    )


def _write_manifest_csv(
    path: Path, manifest_rows: list[dict[str, Any]], manifest_df: pd.DataFrame
) -> None:
    if not manifest_rows:
        manifest_df.to_csv(path, index=False)
        return
    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def _training_summary_frame(manifest_df: pd.DataFrame) -> pd.DataFrame:
    train_mask = (manifest_df["status"] == "completed") & (
        manifest_df["experiment_family"].isin(["baseline", "loso"])
    )
    train_df = cast(pd.DataFrame, manifest_df.loc[train_mask].copy())
    train_cols = [
        column
        for column in [
            "experiment_family",
            "model_name",
            "base_model_family",
            "mean_test_balanced_accuracy",
            "mean_test_auc",
            "mean_test_average_precision",
            "mean_test_accuracy",
            "run_dir",
        ]
        if column in train_df.columns
    ]
    if train_cols:
        train_df = cast(pd.DataFrame, train_df.loc[:, train_cols].copy())
    if not train_df.empty:
        train_df = train_df.sort_values(
            by=["experiment_family", "base_model_family", "model_name"],
            ascending=[True, True, True],
        )
    return _clean_table(train_df)


def _failure_summary_frame(manifest_df: pd.DataFrame) -> pd.DataFrame:
    fail_df = cast(
        pd.DataFrame, manifest_df.loc[manifest_df["status"] == "failed"].copy()
    )
    fail_cols = [
        column
        for column in [
            "experiment_family",
            "model_name",
            "base_model_family",
            "error_summary",
        ]
        if column in fail_df.columns
    ]
    if fail_cols:
        fail_df = cast(pd.DataFrame, fail_df.loc[:, fail_cols].copy())
    return _clean_table(fail_df)


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, df.astype(object).where(pd.notnull(df), ""))


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in cols) + " |")
    return "\n".join(lines)
