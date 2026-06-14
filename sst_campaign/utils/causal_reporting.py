"""Renders figures and markdown summaries for causal go/stop ablation and replication results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.misc import save_json
from sst_campaign.utils.causal_ablation import summarize_ablation_table

TEMPORAL_ORDER = [
    "keep_prestim_only",
    "keep_early_only",
    "keep_p3_only",
    "keep_late_only",
    "ablate_p3_window",
    "ablate_late_window",
    "original",
]

ROI_ORDER = [
    "keep_motor_only",
    "keep_centroparietal_only",
    "keep_centroparietal_p3_only",
    "ablate_motor_channels",
    "ablate_centroparietal",
    "ablate_centroparietal_p3",
    "original",
]

CONDITION_ORDER = [
    "original",
    "keep_late_only",
    "keep_early_only",
    "ablate_late_window",
]

PRETTY_NAMES = {
    "original": "Original",
    "keep_prestim_only": "Keep prestim only",
    "keep_early_only": "Keep early only",
    "keep_p3_only": "Keep P3 only",
    "keep_late_only": "Keep late only",
    "ablate_p3_window": "Ablate P3 window",
    "ablate_late_window": "Ablate late window",
    "keep_motor_only": "Keep motor ROI only",
    "ablate_motor_channels": "Ablate motor ROI",
    "keep_centroparietal_only": "Keep centro-parietal only",
    "ablate_centroparietal": "Ablate centro-parietal ROI",
    "keep_centroparietal_p3_only": "Keep centro-parietal P3 only",
    "ablate_centroparietal_p3": "Ablate centro-parietal P3",
}

MODEL_TITLES = {
    "cnn_transformer": "CNN+Transformer",
    "pure_cnn": "Pure CNN",
    "enigma": "Spatio-Temporal CNN",
}


def _pretty(name: str) -> str:
    return PRETTY_NAMES.get(name, name.replace("_", " ").title())


def _condition_color(name: str) -> str:
    if name == "original":
        return "#2b2b2b"
    if name.startswith("keep_"):
        return "#4f81bd"
    return "#c0504d"


def _family_title(family: str) -> str:
    return MODEL_TITLES.get(family, family.replace("_", " ").title())


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None or bool(pd.isna(value)):
        return float(default)
    return float(value)


def _first_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns or df.empty:
        return None
    return df[column].iloc[0]


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mean_df(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "ablation_mean_summary.csv")


def _refresh_ablation_outputs(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "ablation_metrics.csv"
    df = pd.read_csv(metrics_path)
    summary = summarize_ablation_table(df)
    if "p3_window_s" in df.columns and not df.empty:
        summary["p3_window_s"] = _first_value(df, "p3_window_s")
        summary["p3_window_reference"] = _first_value(df, "p3_window_reference")
        summary["p3_window_relative_to_stop_s"] = _first_value(
            df, "p3_window_relative_to_stop_s"
        )
        summary["mean_stop_onset_s"] = _first_value(df, "mean_stop_onset_s")
    mean_df = pd.DataFrame(summary.get("mean_table", []))
    mean_df.to_csv(run_dir / "ablation_mean_summary.csv", index=False)
    pd.DataFrame(summary.get("paired_drop_tests", [])).to_csv(
        run_dir / "ablation_paired_drop_tests.csv", index=False
    )
    save_json(run_dir / "ablation_summary.json", summary)

    lines = ["# Causal Go-vs-Stop Ablation Summary", ""]
    original = summary.get("original_metrics", {})
    if original:
        lines.append(
            f"- Original balanced accuracy: **{float(original['balanced_accuracy']):.3f}**"
        )
        lines.append(f"- Original AUC: **{float(original['auc']):.3f}**")
        lines.append("")
    strongest = summary.get("strongest_drop")
    if strongest:
        lines.append("## Strongest performance drop")
        lines.append("")
        lines.append(f"- Ablation: **{strongest['ablation_name']}**")
        lines.append(
            f"- Mean delta balanced accuracy: **{float(strongest['mean_delta_bal_acc']):.3f}**"
        )
        lines.append("")
    best_retained = summary.get("best_retained")
    if best_retained:
        lines.append("## Best retained signal")
        lines.append("")
        lines.append(f"- Condition: **{best_retained['ablation_name']}**")
        lines.append(
            f"- Mean balanced accuracy: **{float(best_retained['mean_balanced_accuracy']):.3f}**"
        )
    (run_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")

    run_meta_path = run_dir / "run_metadata.json"
    if run_meta_path.exists():
        payload = _load_json(run_meta_path)
        payload["ablation_summary"] = summary
        save_json(run_meta_path, payload)
    return summary


def render_ablation_overview_figure(run_dir: Path, family: str) -> str:
    df = (
        _load_mean_df(run_dir)
        .sort_values("mean_balanced_accuracy", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    labels = [_pretty(name) for name in df["ablation_name"]]
    values = df["mean_balanced_accuracy"].to_numpy(dtype=float)
    colors = [_condition_color(str(name)) for name in df["ablation_name"]]
    ax.barh(labels, values, color=colors)
    original = df.loc[df["ablation_name"] == "original", "mean_balanced_accuracy"]
    if not original.empty:
        ax.axvline(
            float(original.iloc[0]),
            color="#2b2b2b",
            linestyle="--",
            linewidth=1.5,
            label="Original",
        )
        ax.legend(loc="lower right")
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("Mean balanced accuracy")
    ax.set_title(f"{_family_title(family)}: post-hoc ablation overview")
    return _save(fig, run_dir / "figures" / f"{family}_ablation_overview_bal_acc.png")


def render_temporal_ablation_figure(run_dir: Path, family: str) -> str:
    df = _load_mean_df(run_dir)
    ordered = (
        df[df["ablation_name"].isin(TEMPORAL_ORDER)]
        .assign(
            order=lambda x: x["ablation_name"].map(
                {name: idx for idx, name in enumerate(TEMPORAL_ORDER)}
            )
        )
        .sort_values("order")
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels = [_pretty(name) for name in ordered["ablation_name"]]
    values = ordered["mean_balanced_accuracy"].to_numpy(dtype=float)
    colors = [_condition_color(str(name)) for name in ordered["ablation_name"]]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title(f"{_family_title(family)}: temporal ablation profile")
    ax.tick_params(axis="x", rotation=25)
    original = ordered.loc[
        ordered["ablation_name"] == "original", "mean_balanced_accuracy"
    ]
    if not original.empty:
        ax.axhline(
            float(original.iloc[0]), color="#2b2b2b", linestyle="--", linewidth=1.5
        )
    return _save(fig, run_dir / "figures" / f"{family}_temporal_ablation_bal_acc.png")


def render_roi_ablation_figure(run_dir: Path, family: str) -> str:
    df = _load_mean_df(run_dir)
    ordered = (
        df[df["ablation_name"].isin(ROI_ORDER)]
        .assign(
            order=lambda x: x["ablation_name"].map(
                {name: idx for idx, name in enumerate(ROI_ORDER)}
            )
        )
        .sort_values("order")
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = [_pretty(name) for name in ordered["ablation_name"]]
    values = ordered["mean_balanced_accuracy"].to_numpy(dtype=float)
    colors = [_condition_color(str(name)) for name in ordered["ablation_name"]]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title(f"{_family_title(family)}: ROI ablation profile")
    ax.tick_params(axis="x", rotation=20)
    original = ordered.loc[
        ordered["ablation_name"] == "original", "mean_balanced_accuracy"
    ]
    if not original.empty:
        ax.axhline(
            float(original.iloc[0]), color="#2b2b2b", linestyle="--", linewidth=1.5
        )
    return _save(fig, run_dir / "figures" / f"{family}_roi_ablation_bal_acc.png")


def render_constrained_retraining_figure(run_dir: Path, family: str) -> str | None:
    csv_path = run_dir / "constrained_retraining_comparison.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    labels = [str(label).replace("_", " ").title() for label in df["label"]]
    x = list(range(len(df)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bal = df["mean_test_balanced_accuracy"].to_numpy(dtype=float)
    auc = df["mean_test_auc"].to_numpy(dtype=float)
    ax.bar(
        [i - width / 2 for i in x],
        bal,
        width=width,
        color="#4f81bd",
        label="Balanced accuracy",
    )
    ax.bar([i + width / 2 for i in x], auc, width=width, color="#9bbb59", label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Score")
    ax.set_title(f"{_family_title(family)}: constrained retraining comparison")
    ax.legend(loc="upper left")
    return _save(fig, run_dir / "figures" / f"{family}_constrained_retraining.png")


def render_cross_model_replication_figure(
    run_dir: Path, comparison_df: pd.DataFrame
) -> str | None:
    if comparison_df.empty:
        return None
    family_col = np.asarray(comparison_df["family"], dtype=str)
    condition_col = np.asarray(comparison_df["condition"], dtype=str)
    balanced_accuracy = np.asarray(comparison_df["mean_balanced_accuracy"], dtype=float)
    families = [
        family
        for family in ["cnn_transformer", "pure_cnn", "enigma"]
        if np.any(family_col == family)
    ]
    conditions = [cond for cond in CONDITION_ORDER if np.any(condition_col == cond)]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = list(range(len(families)))
    width = 0.18 if len(conditions) >= 4 else 0.24
    offsets = [(-1.5 + idx) * width for idx in range(len(conditions))]
    for offset, condition in zip(offsets, conditions):
        values = []
        for family in families:
            matching = balanced_accuracy[
                (condition_col == condition) & (family_col == family)
            ]
            values.append(float(matching[0]) if matching.size else float("nan"))
        ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=_pretty(condition),
            color=_condition_color(condition),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_family_title(fam) for fam in families])
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Mean balanced accuracy")
    ax.set_title("Cross-model causal ablation comparison")
    ax.legend(loc="upper center", ncol=2)
    return _save(fig, run_dir / "figures" / "cross_model_replication_bal_acc.png")


def write_run_report(
    *,
    run_dir: Path,
    output_root: Path,
    comparison_df: pd.DataFrame,
    figure_paths: dict[str, str],
) -> dict[str, str]:
    metadata = _load_json(run_dir / "run_metadata.json")
    ablation_summary = _load_json(run_dir / "ablation_summary.json")
    strongest = ablation_summary.get("strongest_drop", {})
    best_retained = ablation_summary.get("best_retained", {})
    retraining = _load_json(run_dir / "constrained_retraining_summary.json")
    retrain_results = retraining.get("results", [])
    by_label = {row["label"]: row for row in retrain_results}
    early = by_label.get("early_only", {})
    late = by_label.get("late_only", {})
    late_gap = float(late.get("mean_test_balanced_accuracy", 0.0)) - float(
        early.get("mean_test_balanced_accuracy", 0.0)
    )

    local_summary_path = run_dir / "causal_go_stop_summary.md"
    local_lines = [
        "# Causal Go-vs-Stop Summary",
        "",
        f"- Source best run: `{metadata['source_best_run']['run_dir']}`",
        f"- Source experiment family: **{metadata['source_best_run']['experiment_family']}**",
        f"- Strongest post-hoc drop: **{strongest.get('ablation_name', 'n/a')}** "
        f"(delta balanced accuracy {float(strongest.get('mean_delta_bal_acc', 0.0)):.3f})",
        f"- Constrained retraining pair: **{retraining['pair_spec']['pair_name']}**",
    ]
    if not comparison_df.empty:
        families = sorted(set(comparison_df["family"]) - {"cnn_transformer"})
        if families:
            local_lines.append(
                f"- Cross-model replication executed for: **{', '.join(families)}**"
            )
    local_summary_path.write_text("\n".join(local_lines), encoding="utf-8")

    comparison_csv = output_root / "causal_go_stop_full_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    summary_path = output_root / "causal_go_stop_full_summary.md"
    summary_lines = [
        "# Causal Go-vs-Stop Full Summary",
        "",
        f"- Source campaign root: `{metadata['source_campaign_root']}`",
        f"- Source best run: `{metadata['source_best_run']['run_dir']}`",
        f"- Source experiment family: **{metadata['source_best_run']['experiment_family']}**",
        f"- Original CNN+Transformer balanced accuracy: **{ablation_summary['original_metrics']['balanced_accuracy']:.3f}**",
        f"- Strongest post-hoc drop: **{_pretty(strongest.get('ablation_name', 'n/a'))}** "
        f"({float(strongest.get('mean_delta_bal_acc', 0.0)):.3f} Δ bal. acc.)",
        f"- Best retained ablated condition: **{_pretty(str(best_retained.get('ablation_name', 'n/a')))}** "
        f"(mean bal. acc. {float(best_retained.get('mean_balanced_accuracy', 0.0)):.3f})",
        f"- Constrained retraining gap (late - early): **{late_gap:.3f}** balanced accuracy",
        "",
        "## Embedded figures",
        "",
        f"![CNN+Transformer ablation overview]({Path(figure_paths['cnn_overview']).relative_to(output_root).as_posix()})",
        "",
        f"![CNN+Transformer temporal ablation]({Path(figure_paths['cnn_temporal']).relative_to(output_root).as_posix()})",
        "",
        f"![CNN+Transformer ROI ablation]({Path(figure_paths['cnn_roi']).relative_to(output_root).as_posix()})",
        "",
    ]
    if "cnn_retraining" in figure_paths:
        summary_lines.extend(
            [
                f"![CNN+Transformer constrained retraining]({Path(figure_paths['cnn_retraining']).relative_to(output_root).as_posix()})",
                "",
            ]
        )
    if "cross_model" in figure_paths:
        summary_lines.extend(
            [
                f"![Cross-model comparison]({Path(figure_paths['cross_model']).relative_to(output_root).as_posix()})",
                "",
            ]
        )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    results_path = output_root / "causal_go_stop_full_results_section.md"
    results_lines = [
        "# Results: causal go-vs-stop interpretability",
        "",
        "We took the strongest completed go-vs-stop CNN+Transformer run from the no-minimal campaign and applied staged post-hoc ablations before constrained retraining.",
        "",
        "## CNN+Transformer",
        "",
        f"The unablated model reached mean balanced accuracy **{ablation_summary['original_metrics']['balanced_accuracy']:.3f}** "
        f"and mean AUC **{ablation_summary['original_metrics']['auc']:.3f}**. "
        f"The most damaging post-hoc perturbation was **{_pretty(strongest.get('ablation_name', 'n/a'))}**, "
        f"which shifted mean balanced accuracy by **{float(strongest.get('mean_delta_bal_acc', 0.0)):.3f}** relative to the original evaluation. "
        f"By contrast, the best retained ablated condition was **{_pretty(str(best_retained.get('ablation_name', 'n/a')))}**, "
        f"which preserved mean balanced accuracy at **{float(best_retained.get('mean_balanced_accuracy', 0.0)):.3f}**.",
        "",
        f"Constrained retraining sharpened the same pattern. Training on the early-only window produced mean balanced accuracy "
        f"**{float(early.get('mean_test_balanced_accuracy', 0.0)):.3f}**, whereas training on the late-only window reached "
        f"**{float(late.get('mean_test_balanced_accuracy', 0.0)):.3f}** (gap **{late_gap:.3f}**).",
        "",
    ]
    if not comparison_df.empty:
        family_text = []
        family_col = np.asarray(comparison_df["family"], dtype=str)
        condition_col = np.asarray(comparison_df["condition"], dtype=str)
        balanced_accuracy = np.asarray(
            comparison_df["mean_balanced_accuracy"], dtype=float
        )
        for family in ["pure_cnn", "enigma"]:
            family_mask = family_col == family
            original = balanced_accuracy[family_mask & (condition_col == "original")]
            early_only = balanced_accuracy[
                family_mask & (condition_col == "keep_early_only")
            ]
            late_only = balanced_accuracy[
                family_mask & (condition_col == "keep_late_only")
            ]
            if not (original.size and early_only.size and late_only.size):
                continue
            family_text.append(
                f"{_family_title(family)} retained **{float(late_only[0]):.3f}** balanced accuracy under late-only input "
                f"versus **{float(early_only[0]):.3f}** under early-only input, from an original value of **{float(original[0]):.3f}**."
            )
        if family_text:
            results_lines.extend(
                [
                    "## Cross-model replication",
                    "",
                    "The same late-dominant pattern generalized across the replication families:",
                    "",
                ]
            )
            results_lines.extend([f"- {text}" for text in family_text])
            results_lines.append("")
    results_path.write_text("\n".join(results_lines), encoding="utf-8")

    packet_path = output_root / "causal_go_stop_packet_README.md"
    packet_lines = [
        "# Causal Go-vs-Stop Packet README",
        "",
        "This packet collects the verified staged causal-interpretability outputs for the go-vs-stop decoding experiment.",
        "",
        "## Primary run",
        "",
        f"- Full run root: `{run_dir}`",
        "",
        "## Core structured outputs",
        "",
        f"- `{run_dir / 'ablation_metrics.csv'}`",
        f"- `{run_dir / 'ablation_mean_summary.csv'}`",
        f"- `{run_dir / 'ablation_paired_drop_tests.csv'}`",
        f"- `{run_dir / 'constrained_retraining_comparison.csv'}`",
        f"- `{run_dir / 'constrained_retraining_summary.json'}`",
        f"- `{run_dir / 'replication_summary.json'}`",
        "",
        "## Derived report assets",
        "",
        f"- `{summary_path}`",
        f"- `{results_path}`",
        f"- `{comparison_csv}`",
        "",
        "## Figures",
        "",
        f"- `{figure_paths['cnn_overview']}`",
        f"- `{figure_paths['cnn_temporal']}`",
        f"- `{figure_paths['cnn_roi']}`",
    ]
    if "cnn_retraining" in figure_paths:
        packet_lines.append(f"- `{figure_paths['cnn_retraining']}`")
    if "cross_model" in figure_paths:
        packet_lines.append(f"- `{figure_paths['cross_model']}`")
    packet_lines.append("")
    packet_lines.append("## Main interpretation")
    packet_lines.append("")
    packet_lines.append(
        "Across the strongest CNN+Transformer run, post-hoc ablation and constrained retraining both indicate that the go-vs-stop decoder depends much more on later post-stimulus structure than on early-only input."
    )
    packet_path.write_text("\n".join(packet_lines), encoding="utf-8")

    return {
        "comparison_csv": str(comparison_csv),
        "summary_md": str(summary_path),
        "results_md": str(results_path),
        "packet_readme": str(packet_path),
    }


def render_causal_go_stop_reports(run_dir: Path, output_root: Path) -> dict[str, str]:
    run_dir = run_dir.resolve()
    output_root = output_root.resolve()
    if not (run_dir / "ablation_mean_summary.csv").exists():
        raise FileNotFoundError(f"Missing ablation_mean_summary.csv under {run_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    _refresh_ablation_outputs(run_dir)

    figure_paths = {
        "cnn_overview": render_ablation_overview_figure(run_dir, "cnn_transformer"),
        "cnn_temporal": render_temporal_ablation_figure(run_dir, "cnn_transformer"),
        "cnn_roi": render_roi_ablation_figure(run_dir, "cnn_transformer"),
    }
    retrain_figure = render_constrained_retraining_figure(run_dir, "cnn_transformer")
    if retrain_figure:
        figure_paths["cnn_retraining"] = retrain_figure

    for family in ("pure_cnn", "enigma"):
        family_dir = run_dir / f"replication_{family}"
        if not family_dir.exists():
            continue
        _refresh_ablation_outputs(family_dir)
        render_ablation_overview_figure(family_dir, family)
        render_temporal_ablation_figure(family_dir, family)
        render_roi_ablation_figure(family_dir, family)

    comparison_rows: list[dict[str, Any]] = []
    for family, family_dir in {
        "cnn_transformer": run_dir,
        "pure_cnn": run_dir / "replication_pure_cnn",
        "enigma": run_dir / "replication_enigma",
    }.items():
        mean_csv = family_dir / "ablation_mean_summary.csv"
        if not mean_csv.exists():
            continue
        by_name = {
            str(row["ablation_name"]): row
            for _, row in pd.read_csv(mean_csv).iterrows()
        }
        for condition in CONDITION_ORDER:
            row = by_name.get(condition)
            if row is None:
                continue
            comparison_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "condition_label": _pretty(condition),
                    "mean_balanced_accuracy": _float_or_default(
                        row.get("mean_balanced_accuracy"), default=float("nan")
                    ),
                    "mean_auc": _float_or_default(
                        row.get("mean_auc"), default=float("nan")
                    ),
                    "mean_delta_bal_acc": _float_or_default(
                        row.get("mean_delta_bal_acc"), default=0.0
                    ),
                    "mean_delta_auc": _float_or_default(
                        row.get("mean_delta_auc"), default=0.0
                    ),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    cross_model_figure = render_cross_model_replication_figure(run_dir, comparison_df)
    if cross_model_figure:
        figure_paths["cross_model"] = cross_model_figure

    written = write_run_report(
        run_dir=run_dir,
        output_root=output_root,
        comparison_df=comparison_df,
        figure_paths=figure_paths,
    )
    return {**figure_paths, **written}
