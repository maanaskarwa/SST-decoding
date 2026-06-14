from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

from pipeline.misc import save_json
from pipeline.perf import resolve_device
from sst_campaign.run_causal_go_stop_cycle import (
    _run_post_hoc_ablation,
    _write_ablation_outputs,
)
from sst_campaign.utils.behavioral_sst import (
    inhibition_function_table,
    load_behavioral_trials,
    summarize_subject_behavior,
    trial_table,
)
from sst_campaign.utils.common import timestamp_slug
from sst_campaign.utils.manuscript_stats import (
    benjamini_hochberg,
    comparison_record,
    exact_sign_test_greater,
    finite_column,
    paired_permutation_p_greater,
    paired_sign_test_greater,
    spearman_correlation,
    spearman_permutation_p_two_sided,
    summarize_metric,
)
from sst_campaign.utils.model_specs import model_spec
from sst_campaign.utils.run_tasks import load_best_training_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
MODEL_COLORS = {
    "cnn_transformer": "#4f81bd",
    "transformer_only": "#8064a2",
    "pure_cnn": "#c0504d",
    "enigma": "#9bbb59",
    "patch_mixer": "#7f7f7f",
}
METRIC_COLUMNS = {
    "balanced_accuracy": "test_balanced_accuracy",
    "roc_auc": "test_auc",
    "average_precision": "test_average_precision",
    "accuracy": "test_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready SST decoding artifacts from a fixed config."
    )
    parser.add_argument("--config", required=True, help="Manuscript TOML config")
    parser.add_argument(
        "--output-dir", default=None, help="Override output directory from config"
    )
    parser.add_argument(
        "--source-campaign-root",
        default=None,
        help="Override manuscript.source_campaign_root from the config",
    )
    parser.add_argument(
        "--model-order",
        nargs="+",
        default=None,
        help="Override manuscript.model_order from the config",
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Override manuscript.primary_model from the config",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip configured heavy validation steps such as P3 ablation",
    )
    return parser.parse_args()


def _resolve_path(path_text: str | Path, *, root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _test_subject_for_fold(run_dir: Path, fold: int) -> int | None:
    split_path = run_dir / f"fold_{int(fold):02d}" / "split_indices.npz"
    if not split_path.exists():
        return None
    split = np.load(split_path)
    test_idx = split["test_idx"].astype(np.int64)
    subjects = split["subject"].astype(np.int64)[test_idx]
    unique = np.unique(subjects)
    return int(unique[0]) if unique.size == 1 else None


def _load_model_subject_metrics(
    campaign_root: Path, model_order: list[str]
) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[pd.DataFrame] = []
    run_dirs: dict[str, Path] = {}
    for model_name in model_order:
        best = load_best_training_run(
            campaign_root=campaign_root, expected_family=model_name
        )
        run_dir = Path(str(best["run_dir"])).resolve()
        run_dirs[model_name] = run_dir
        df = pd.read_csv(sorted(run_dir.glob("*_fold_metrics.csv"))[0])
        df.insert(0, "model", model_name)
        df.insert(1, "model_display", model_spec(model_name).display_name)
        df["subject"] = [
            _test_subject_for_fold(run_dir, int(fold)) or int(fold)
            for fold in df["fold"].to_numpy(dtype=int)
        ]
        df["source_run_dir"] = str(run_dir)
        rows.append(df)
    return pd.concat(rows, axis=0, ignore_index=True), run_dirs


def _trial_counts_from_predictions(pred_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    if not {"subject", "y_true"}.issubset(pred.columns):
        raise ValueError(f"Prediction CSV lacks subject/y_true columns: {pred_csv}")
    rows = []
    for subject, group in pred.groupby("subject"):
        subject_id = int(str(subject))
        y = group["y_true"].to_numpy(dtype=int)
        rows.append(
            {
                "subject": subject_id,
                "n_trials": int(len(group)),
                "n_go": int((y == 0).sum()),
                "n_stop": int((y == 1).sum()),
                "stop_prevalence": float((y == 1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["subject"]).reset_index(drop=True)


def _model_metric_values(
    subject_metrics: pd.DataFrame, model_name: str, column: str
) -> tuple[np.ndarray, np.ndarray]:
    model_mask = np.asarray(subject_metrics["model"], dtype=str) == str(model_name)
    subjects = np.asarray(subject_metrics.loc[model_mask, "subject"], dtype=np.int64)
    values = np.asarray(subject_metrics.loc[model_mask, column], dtype=np.float64)
    order = np.argsort(subjects)
    return subjects[order], values[order]


def _paired_metric_values(
    subject_metrics: pd.DataFrame, left_model: str, right_model: str, column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_subjects, left_values = _model_metric_values(
        subject_metrics, left_model, column
    )
    right_subjects, right_values = _model_metric_values(
        subject_metrics, right_model, column
    )
    right_lookup = {
        int(subject): float(value)
        for subject, value in zip(right_subjects, right_values)
    }
    subjects: list[int] = []
    paired_left: list[float] = []
    paired_right: list[float] = []
    for subject, value in zip(left_subjects, left_values):
        subject_id = int(subject)
        if subject_id not in right_lookup:
            continue
        subjects.append(subject_id)
        paired_left.append(float(value))
        paired_right.append(float(right_lookup[subject_id]))
    return (
        np.asarray(subjects, dtype=np.int64),
        np.asarray(paired_left, dtype=np.float64),
        np.asarray(paired_right, dtype=np.float64),
    )


def _write_performance_outputs(
    *,
    subject_metrics: pd.DataFrame,
    run_dirs: dict[str, Path],
    model_order: list[str],
    primary_model: str,
    out_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_metrics_csv = out_dir / "model_subject_metrics.csv"
    summary_csv = out_dir / "table1_model_performance.csv"
    stats_csv = out_dir / "model_comparison_stats.csv"
    pr_points_csv = out_dir / "figure1_pr_curve_points.csv"
    subject_metrics.to_csv(subject_metrics_csv, index=False)

    rng = np.random.default_rng(int(random_state))
    prediction_paths = {
        model: sorted(run_dirs[model].glob("*_predictions.csv"))[0]
        for model in model_order
    }
    trial_counts = _trial_counts_from_predictions(prediction_paths[model_order[0]])
    trial_counts.to_csv(out_dir / "trial_counts_by_subject.csv", index=False)
    stop_prevalence = float(
        trial_counts["n_stop"].sum() / trial_counts["n_trials"].sum()
    )

    summary_rows: list[dict[str, Any]] = []
    for model_name in model_order:
        subjects, _ = _model_metric_values(
            subject_metrics, model_name, METRIC_COLUMNS["balanced_accuracy"]
        )
        pred_df = pd.read_csv(prediction_paths[model_name])
        y_true = pred_df["y_true"].to_numpy(dtype=int)
        p_stop = pred_df["p_stop"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "model": model_name,
            "model_display": model_spec(model_name).display_name,
            "global_average_precision": float(average_precision_score(y_true, p_stop)),
            "n_subjects": int(np.unique(subjects).size),
        }
        for metric_name, column in METRIC_COLUMNS.items():
            _, values = _model_metric_values(subject_metrics, model_name, column)
            metric_summary = summarize_metric(values, rng=rng)
            row.update(metric_summary.as_dict(metric_name))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    stat_records: list[dict[str, Any]] = []
    chance_by_metric = {
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5,
        "average_precision": stop_prevalence,
    }
    for model_name in model_order:
        for metric_name, chance in chance_by_metric.items():
            _, values = _model_metric_values(
                subject_metrics, model_name, METRIC_COLUMNS[metric_name]
            )
            stat_records.append(
                comparison_record(
                    comparison=f"{model_name} > chance",
                    metric=metric_name,
                    estimate=float(values.mean() - chance),
                    p_value=exact_sign_test_greater(values, null_value=chance),
                    n=int(values.size),
                    test="one-sided exact sign test across subjects",
                )
            )
    for model_name in model_order:
        if model_name == primary_model:
            continue
        for metric_name, column in METRIC_COLUMNS.items():
            subjects, primary_values, other_values = _paired_metric_values(
                subject_metrics, primary_model, model_name, column
            )
            stat_records.append(
                comparison_record(
                    comparison=f"{primary_model} > {model_name}",
                    metric=metric_name,
                    estimate=float((primary_values - other_values).mean()),
                    p_value=paired_permutation_p_greater(
                        primary_values, other_values, rng=rng
                    ),
                    n=int(subjects.size),
                    test="one-sided paired sign-flip permutation across subjects",
                )
            )
            stat_records.append(
                comparison_record(
                    comparison=f"{primary_model} > {model_name}",
                    metric=metric_name,
                    estimate=float((primary_values - other_values).mean()),
                    p_value=paired_sign_test_greater(primary_values, other_values),
                    n=int(subjects.size),
                    test="one-sided exact paired sign test across subjects",
                )
            )
    adjusted = benjamini_hochberg([row["p_value"] for row in stat_records])
    for row, p_adjusted in zip(stat_records, adjusted):
        row["p_bh_fdr"] = p_adjusted
    pd.DataFrame(stat_records).to_csv(stats_csv, index=False)

    pr_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for model_name in model_order:
        pred_df = pd.read_csv(prediction_paths[model_name])
        y_true = pred_df["y_true"].to_numpy(dtype=int)
        p_stop = pred_df["p_stop"].to_numpy(dtype=float)
        precision, recall, thresholds = precision_recall_curve(y_true, p_stop)
        label = model_spec(model_name).display_name
        ax.plot(recall, precision, label=label, color=MODEL_COLORS.get(model_name))
        for idx, (recall_value, precision_value) in enumerate(zip(recall, precision)):
            pr_rows.append(
                {
                    "model": model_name,
                    "model_display": label,
                    "point_index": int(idx),
                    "recall": float(recall_value),
                    "precision": float(precision_value),
                    "threshold": float(thresholds[idx])
                    if idx < len(thresholds)
                    else float("nan"),
                }
            )
    ax.axhline(
        stop_prevalence,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Stop prevalence ({stop_prevalence:.3f})",
    )
    ax.set_xlabel("Recall for stop trials")
    ax.set_ylabel("Precision for stop trials")
    ax.set_title("Leave-one-subject-out stop-trial detection")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"figure1_precision_recall_loso.{suffix}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)
    pd.DataFrame(pr_rows).to_csv(pr_points_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    x = np.arange(len(model_order))
    values = [
        float(
            summary_df.loc[summary_df["model"] == model, "mean_balanced_accuracy"].iloc[
                0
            ]
        )
        for model in model_order
    ]
    errors = [
        float(
            summary_df.loc[summary_df["model"] == model, "se_balanced_accuracy"].iloc[0]
        )
        for model in model_order
    ]
    labels = [model_spec(model).display_name for model in model_order]
    colors = [MODEL_COLORS.get(model, "#4f81bd") for model in model_order]
    ax.bar(x, values, yerr=errors, color=colors, capsize=4)
    ax.axhline(
        0.5,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Balanced-accuracy chance",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.45, min(1.0, max(values) + max(errors) + 0.08))
    ax.set_title("LOSO balanced accuracy by model family")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"figure1b_loso_balanced_accuracy_bar.{suffix}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)

    return {
        "stop_prevalence": stop_prevalence,
        "subject_metrics_csv": str(subject_metrics_csv),
        "summary_csv": str(summary_csv),
        "stats_csv": str(stats_csv),
        "pr_points_csv": str(pr_points_csv),
        "prediction_paths": {
            model: str(path) for model, path in prediction_paths.items()
        },
    }


def _write_behavioral_outputs(
    *,
    root: Path,
    subjects: list[int],
    subject_metrics: pd.DataFrame,
    primary_model: str,
    out_dir: Path,
    random_state: int,
    n_correlation_permutations: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_behavioral_trials(root, subjects)
    trials_df = trial_table(trials)
    behavior_df = summarize_subject_behavior(trials)
    inhibition_df = inhibition_function_table(trials)

    trial_csv = out_dir / "behavioral_trials.csv"
    subject_csv = out_dir / "behavioral_subject_metrics.csv"
    inhibition_csv = out_dir / "inhibition_function_by_subject_ssd.csv"
    inhibition_aggregate_csv = out_dir / "inhibition_function_aggregate.csv"
    correlation_csv = out_dir / "model_behavior_correlations.csv"
    summary_json = out_dir / "behavioral_summary.json"
    trials_df.to_csv(trial_csv, index=False)
    behavior_df.to_csv(subject_csv, index=False)
    inhibition_df.to_csv(inhibition_csv, index=False)

    aggregate_rows: list[dict[str, Any]] = []
    if not inhibition_df.empty:
        ssd_ms = np.asarray(inhibition_df["ssd_ms"], dtype=np.int64)
        n_stop = np.asarray(inhibition_df["n_stop"], dtype=np.int64)
        n_response = np.asarray(inhibition_df["n_response"], dtype=np.int64)
        for value in np.unique(ssd_ms):
            mask = ssd_ms == int(value)
            total_stop = int(n_stop[mask].sum())
            total_response = int(n_response[mask].sum())
            aggregate_rows.append(
                {
                    "ssd_ms": int(value),
                    "n_stop": total_stop,
                    "n_response": total_response,
                    "p_response": float(total_response / total_stop)
                    if total_stop
                    else float("nan"),
                }
            )
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(inhibition_aggregate_csv, index=False)

    rng = np.random.default_rng(int(random_state))
    behavior_summaries: dict[str, Any] = {}
    for column in [
        "mean_go_rt_s",
        "mean_stop_failure_rt_s",
        "mean_ssd_s",
        "stop_success_rate",
        "ssrt_integration_s",
    ]:
        values = finite_column(behavior_df, column)
        metric_summary = summarize_metric(values, rng=rng)
        behavior_summaries[column] = {
            "mean": metric_summary.mean,
            "sd": metric_summary.sd,
            "se": metric_summary.se,
            "ci95_low": metric_summary.ci95_low,
            "ci95_high": metric_summary.ci95_high,
            "n_subjects": metric_summary.n,
        }

    behavior_subjects = np.asarray(behavior_df["subject"], dtype=np.int64)
    correlation_rows: list[dict[str, Any]] = []
    behavior_columns = [
        "ssrt_integration_s",
        "stop_success_rate",
        "mean_go_rt_s",
        "mean_ssd_s",
        "mean_stop_failure_rt_s",
    ]
    for performance_label, performance_column in {
        "balanced_accuracy": "test_balanced_accuracy",
        "average_precision": "test_average_precision",
    }.items():
        perf_subjects, perf_values = _model_metric_values(
            subject_metrics, primary_model, performance_column
        )
        perf_lookup = {
            int(subject): float(value)
            for subject, value in zip(perf_subjects, perf_values)
        }
        for behavior_column in behavior_columns:
            behavior_values = np.asarray(behavior_df[behavior_column], dtype=np.float64)
            paired_perf: list[float] = []
            paired_behavior: list[float] = []
            for subject, behavior_value in zip(behavior_subjects, behavior_values):
                subject_id = int(subject)
                if subject_id in perf_lookup and np.isfinite(float(behavior_value)):
                    paired_perf.append(perf_lookup[subject_id])
                    paired_behavior.append(float(behavior_value))
            x = np.asarray(paired_perf, dtype=np.float64)
            y = np.asarray(paired_behavior, dtype=np.float64)
            rho, n_pairs = spearman_correlation(x, y)
            p_value = spearman_permutation_p_two_sided(
                x,
                y,
                rng=rng,
                n_permutations=int(n_correlation_permutations),
            )
            correlation_rows.append(
                {
                    "model": primary_model,
                    "performance_metric": performance_label,
                    "behavior_metric": behavior_column,
                    "spearman_rho": rho,
                    "p_permutation_two_sided": p_value,
                    "n_subjects": int(n_pairs),
                    "test": "two-sided subject-label permutation Spearman correlation",
                }
            )
    pd.DataFrame(correlation_rows).to_csv(correlation_csv, index=False)

    ssrt_ms = finite_column(behavior_df, "ssrt_integration_s") * 1000.0
    if ssrt_ms.size:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.hist(
            ssrt_ms,
            bins=min(10, max(4, int(np.sqrt(ssrt_ms.size)))),
            color="#4f81bd",
            edgecolor="white",
        )
        ax.set_xlabel("Integration SSRT (ms)")
        ax.set_ylabel("Subjects")
        ax.set_title("Subject-level behavioral SSRT")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                out_dir / f"figure_behavior_ssrt_distribution.{suffix}",
                dpi=250,
                bbox_inches="tight",
            )
        plt.close(fig)

    if aggregate_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ssd = np.asarray([row["ssd_ms"] for row in aggregate_rows], dtype=np.float64)
        p_response = np.asarray(
            [row["p_response"] for row in aggregate_rows], dtype=np.float64
        )
        ax.plot(ssd, p_response, marker="o", color="#c0504d")
        ax.set_xlabel("Stop-signal delay (ms)")
        ax.set_ylabel("P(response | stop)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Behavioral inhibition function")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                out_dir / f"figure_behavior_inhibition_function.{suffix}",
                dpi=250,
                bbox_inches="tight",
            )
        plt.close(fig)

    summary = {
        "n_trials": int(len(trials_df)),
        "n_subjects": int(len(behavior_df)),
        "trial_csv": str(trial_csv),
        "subject_csv": str(subject_csv),
        "inhibition_csv": str(inhibition_csv),
        "inhibition_aggregate_csv": str(inhibition_aggregate_csv),
        "correlation_csv": str(correlation_csv),
        "metric_summaries": behavior_summaries,
        "ssrt_method": "integration method: nth go RT at subject stop-response probability minus mean SSD",
    }
    save_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary


def _copy_optional_artifact(path_text: str, *, root: Path, out_dir: Path) -> str:
    source = _resolve_path(path_text, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / source.name
    if source.is_file():
        shutil.copy2(source, destination)
    elif source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        raise FileNotFoundError(source)
    return str(destination)


def _run_optional_p3_ablation(
    *,
    cfg: dict[str, Any],
    root: Path,
    run_dirs: dict[str, Path],
    out_dir: Path,
    skip_heavy: bool,
) -> dict[str, Any] | None:
    block = cfg.get("p3_ablation", {})
    if not bool(block.get("enabled", False)) or skip_heavy:
        return None
    model_name = str(
        block.get(
            "model", cfg.get("manuscript", {}).get("primary_model", "cnn_transformer")
        )
    )
    device = resolve_device(str(block.get("device", "auto")))
    ablation_dir = out_dir / "p3_roi_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    df, summary = _run_post_hoc_ablation(
        run_dir=run_dirs[model_name],
        root=root,
        device=device,
        random_state=int(
            block.get("random_state", cfg.get("manuscript", {}).get("random_state", 9))
        ),
        cudnn_benchmark=bool(block.get("cudnn_benchmark", False)),
        matmul_precision=block.get("matmul_precision"),
        p3_tmin=float(block.get("p3_tmin", 0.25)),
        p3_tmax=float(block.get("p3_tmax", 0.45)),
    )
    _write_ablation_outputs(ablation_dir, df, summary)
    return {"output_dir": str(ablation_dir), "summary": summary}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("rb") as file:
        cfg = tomllib.load(file)
    manuscript_cfg = cfg.get("manuscript", {})
    root = _resolve_path(manuscript_cfg.get("root", "."), root=ROOT)
    source_campaign_root = args.source_campaign_root or manuscript_cfg.get(
        "source_campaign_root"
    )
    if not source_campaign_root:
        raise SystemExit(
            "Missing source campaign root. Pass --source-campaign-root or set "
            "manuscript.source_campaign_root in the config."
        )
    campaign_root = _resolve_path(source_campaign_root, root=root)
    output_dir = _resolve_path(
        args.output_dir
        or manuscript_cfg.get("output_dir")
        or f"sst_campaign_runs/manuscript_rebuild__{timestamp_slug()}",
        root=root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_order = [
        str(model)
        for model in (
            args.model_order
            or manuscript_cfg.get(
                "model_order",
                ["cnn_transformer", "transformer_only", "pure_cnn", "enigma"],
            )
        )
    ]
    primary_model = str(
        args.primary_model or manuscript_cfg.get("primary_model", model_order[0])
    )
    random_state = int(manuscript_cfg.get("random_state", 9))

    subject_metrics, run_dirs = _load_model_subject_metrics(campaign_root, model_order)
    perf_summary = _write_performance_outputs(
        subject_metrics=subject_metrics,
        run_dirs=run_dirs,
        model_order=model_order,
        primary_model=primary_model,
        out_dir=output_dir / "performance",
        random_state=random_state,
    )

    behavior_summary = None
    behavior_cfg = cfg.get("behavior", {})
    if bool(behavior_cfg.get("enabled", True)):
        subjects = sorted(
            int(subject)
            for subject in np.unique(
                np.asarray(subject_metrics["subject"], dtype=np.int64)
            )
        )
        behavior_summary = _write_behavioral_outputs(
            root=root,
            subjects=subjects,
            subject_metrics=subject_metrics,
            primary_model=primary_model,
            out_dir=output_dir / "behavior",
            random_state=random_state,
            n_correlation_permutations=int(
                behavior_cfg.get("n_correlation_permutations", 10000)
            ),
        )

    copied_artifacts: dict[str, str] = {}
    for key, value in cfg.get("copy_artifacts", {}).items():
        copied_artifacts[str(key)] = _copy_optional_artifact(
            str(value), root=root, out_dir=output_dir / "source_artifacts"
        )

    p3_ablation = _run_optional_p3_ablation(
        cfg=cfg,
        root=root,
        run_dirs=run_dirs,
        out_dir=output_dir / "interpretability_validation",
        skip_heavy=bool(args.skip_heavy),
    )

    result_summary = {
        "campaign_root": str(campaign_root),
        "model_order": model_order,
        "primary_model": primary_model,
        "performance": perf_summary,
        "behavior": behavior_summary,
        "copied_artifacts": copied_artifacts,
        "p3_ablation": p3_ablation,
    }
    save_json(output_dir / "manuscript_pipeline_summary.json", result_summary)
    print(f"Saved manuscript artifacts: {output_dir}")
    print(f"Saved: {output_dir / 'manuscript_pipeline_summary.json'}")


if __name__ == "__main__":
    main()
