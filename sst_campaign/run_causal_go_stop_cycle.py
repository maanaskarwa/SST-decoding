from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.misc import save_json
from pipeline.perf import configure_torch_runtime, resolve_device
from pipeline.run_versioning import prepare_versioned_output_dir
from pipeline.train.driver import set_random_seed
from sst_campaign.utils.causal_ablation import (
    choose_constrained_retrain_pair,
    default_ablation_specs,
    evaluate_ablation_specs,
    summarize_ablation_table,
)
from sst_campaign.utils.causal_reporting import render_causal_go_stop_reports
from sst_campaign.utils.common import emit_run_output_dir, run_python_script
from sst_campaign.utils.model_loading import load_run_cfg_meta
from sst_campaign.utils.model_specs import base_family, metadata_filenames
from sst_campaign.utils.run_tasks import load_best_training_run
from sst_campaign.utils.saved_run_context import (
    iter_saved_run_folds,
    load_saved_run_data,
)
from sst_campaign.utils.stop_windows import resolve_p3_window

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SST campaign causal go-vs-stop interpretability cycle: post-hoc ablations on the best completed run, then constrained retraining under the most diagnostic time-window contrast."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source-campaign-root",
        required=True,
        help="Completed CNN+Transformer campaign root containing campaign_metadata.json",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=9)
    parser.add_argument(
        "--p3-tmin",
        type=float,
        default=0.25,
        help="P3 window start after stop signal, in seconds.",
    )
    parser.add_argument(
        "--p3-tmax",
        type=float,
        default=0.45,
        help="P3 window end after stop signal, in seconds.",
    )
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default=None
    )
    parser.add_argument("--output-dir", default="sst_campaign_runs")
    parser.add_argument("--run-label", default="causal_go_stop")
    parser.add_argument("--disable-versioning", action="store_true")
    parser.add_argument("--replicate-on", nargs="+", default=["pure_cnn", "enigma"])
    parser.add_argument("--replication-threshold", type=float, default=0.03)
    parser.add_argument("--replication-root-pure-cnn", default=None)
    parser.add_argument("--replication-root-enigma", default=None)
    parser.add_argument("--retrain-epochs", type=int, default=40)
    parser.add_argument("--retrain-patience", type=int, default=8)
    parser.add_argument("--retrain-batch-size", type=int, default=64)
    return parser.parse_args()


def _run_post_hoc_ablation(
    run_dir: Path,
    *,
    root: Path,
    device: Any,
    random_state: int,
    cudnn_benchmark: bool,
    matmul_precision: str | None,
    p3_tmin: float = 0.25,
    p3_tmax: float = 0.45,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_random_seed(int(random_state))
    run_data = load_saved_run_data(root=root, run_dir=run_dir)
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(cudnn_benchmark),
        matmul_precision=matmul_precision,
    )

    p3_window = resolve_p3_window(
        y=run_data.y,
        stop_signal_onset_s=run_data.stop_signal_onset_s,
        p3_tmin=float(p3_tmin),
        p3_tmax=float(p3_tmax),
    )

    rows: list[pd.DataFrame] = []
    specs = default_ablation_specs(p3_window_s=p3_window.window_s)

    for fold in iter_saved_run_folds(run_data, device=device):
        df = evaluate_ablation_specs(
            model=fold.model,
            X=fold.X_te,
            y=fold.y_te,
            times_s=run_data.times_s,
            channel_names=run_data.channel_names,
            device=device,
            specs=specs,
            metadata={
                "fold": int(fold.fold_idx),
                "source_run_dir": str(run_dir),
                "source_model_name": run_data.model_name,
                "base_model_family": base_family(run_data.model_name),
                **p3_window.summary(),
            },
        )
        rows.append(df)

    full_df = pd.concat(rows, axis=0, ignore_index=True)
    summary = summarize_ablation_table(full_df)
    summary.update(p3_window.summary())
    return full_df, summary


def _write_ablation_outputs(
    out_dir: Path, df: pd.DataFrame, summary: dict[str, Any]
) -> None:
    ablation_csv = out_dir / "ablation_metrics.csv"
    mean_csv = out_dir / "ablation_mean_summary.csv"
    drop_tests_csv = out_dir / "ablation_paired_drop_tests.csv"
    summary_json = out_dir / "ablation_summary.json"
    summary_md = out_dir / "ablation_summary.md"

    df.to_csv(ablation_csv, index=False)
    mean_df = pd.DataFrame(summary.get("mean_table", []))
    mean_df.to_csv(mean_csv, index=False)
    pd.DataFrame(summary.get("paired_drop_tests", [])).to_csv(
        drop_tests_csv, index=False
    )
    save_json(summary_json, summary)

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
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def _run_constrained_retrain(
    source_best: dict[str, Any],
    pair_spec: dict[str, Any],
    *,
    out_dir: Path,
    device: str,
    random_state: int,
    retrain_epochs: int,
    retrain_patience: int,
    retrain_batch_size: int,
) -> list[dict[str, Any]]:
    source_run_dir = Path(source_best["run_dir"])
    cfg, meta = load_run_cfg_meta(source_run_dir)
    base_model = str(source_best["model_name"])
    experiment_family = str(source_best["experiment_family"])
    script = {
        "loso": "sst_campaign/experiments/train_zero_shot_loso.py",
        "baseline": "sst_campaign/experiments/train_repro_baseline.py",
    }.get(experiment_family)
    if script is None:
        raise RuntimeError(
            f"Unsupported source experiment family for constrained retraining: {experiment_family}"
        )

    results: list[dict[str, Any]] = []
    for job in pair_spec["train_jobs"]:
        run_args = ["--subjects", *[str(s) for s in cfg.get("subjects", ["all"])]]
        run_args.extend(
            [
                "--model",
                base_model,
                "--device",
                device,
                "--random-state",
                str(random_state),
                "--output-dir",
                str(out_dir),
                "--run-label",
                f"constrained_{job['label']}",
                "--epochs",
                str(retrain_epochs),
                "--patience",
                str(retrain_patience),
                "--batch-size",
                str(retrain_batch_size),
                "--val-fraction",
                str(cfg.get("val_fraction", 0.15)),
                "--crop-tmin",
                str(job["crop_tmin"]),
                "--crop-tmax",
                str(job["crop_tmax"]),
            ]
        )
        if cfg.get("matmul_precision") is not None:
            run_args.extend(["--matmul-precision", str(cfg["matmul_precision"])])
        if int(cfg.get("num_workers", 0)) > 0:
            run_args.extend(["--num-workers", str(cfg["num_workers"])])
            if int(cfg.get("prefetch_factor", 0)) > 0:
                run_args.extend(["--prefetch-factor", str(cfg["prefetch_factor"])])
        if bool(cfg.get("persistent_workers", False)):
            run_args.append("--persistent-workers")
        if bool(cfg.get("cudnn_benchmark", False)):
            run_args.append("--cudnn-benchmark")
        if bool(cfg.get("amp", False)):
            run_args.append("--amp")
        if cfg.get("amp_dtype") is not None:
            run_args.extend(["--amp-dtype", str(cfg["amp_dtype"])])

        result = run_python_script(
            script=script,
            args=run_args,
            cwd=ROOT,
            env={"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/matplotlib"},
        )
        entry = {
            "label": job["label"],
            "crop_tmin": float(job["crop_tmin"]),
            "crop_tmax": float(job["crop_tmax"]),
            "reproducible_command": result["reproducible_command"],
            "returncode": int(result["returncode"]),
            "run_dir": result["run_dir"],
            "stdout": result["stdout"][-4000:],
            "stderr": result["stderr"][-4000:],
        }
        if result["returncode"] == 0 and result["run_dir"]:
            run_meta_candidates = [
                Path(result["run_dir"]) / name for name in metadata_filenames()
            ]
            found = False
            for meta_path in run_meta_candidates:
                if meta_path.exists():
                    payload = json.loads(meta_path.read_text())
                    overall = payload.get("overall_metrics", {})
                    entry["mean_test_balanced_accuracy"] = overall.get(
                        "mean_test_balanced_accuracy"
                    )
                    entry["mean_test_auc"] = overall.get("mean_test_auc")
                    entry["mean_test_accuracy"] = overall.get("mean_test_accuracy")
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Successful constrained retrain produced no metadata file in {result['run_dir']}. "
                    f"Candidates: {[str(p) for p in run_meta_candidates]}"
                )
        results.append(entry)
    return results


def _maybe_replicate_post_hoc(
    *,
    replicate_families: list[str],
    threshold: float,
    cnn_summary: dict[str, Any],
    constrained_results: list[dict[str, Any]],
    source_campaign_root: Path,
    explicit_roots: dict[str, Path | None],
    root: Path,
    out_dir: Path,
    random_state: int,
    device: Any,
    cudnn_benchmark: bool,
    matmul_precision: str | None,
    p3_tmin: float = 0.25,
    p3_tmax: float = 0.45,
) -> list[dict[str, Any]]:
    strongest = cnn_summary.get("strongest_drop")
    if not strongest:
        return []
    if abs(float(strongest.get("mean_delta_bal_acc", 0.0))) < float(threshold):
        return []

    by_label = {row["label"]: row for row in constrained_results}
    early = by_label.get("early_only")
    late = by_label.get("late_only")
    if not early or not late:
        return []
    if int(early.get("returncode", 1)) != 0 or int(late.get("returncode", 1)) != 0:
        return []
    gap = float(late.get("mean_test_balanced_accuracy", float("nan"))) - float(
        early.get("mean_test_balanced_accuracy", float("nan"))
    )
    if not np.isfinite(gap) or gap < float(threshold):
        return []

    outputs: list[dict[str, Any]] = []
    for family in replicate_families:
        family = str(family)
        campaign_root = explicit_roots.get(family)
        if campaign_root is None:
            continue
        if not campaign_root.exists():
            raise FileNotFoundError(
                f"Replication campaign root for {family} does not exist: {campaign_root}"
            )
        best = load_best_training_run(
            campaign_root=campaign_root, expected_family=str(family)
        )
        df, summary = _run_post_hoc_ablation(
            run_dir=Path(best["run_dir"]),
            root=root,
            device=device,
            random_state=random_state,
            cudnn_benchmark=cudnn_benchmark,
            matmul_precision=matmul_precision,
            p3_tmin=float(p3_tmin),
            p3_tmax=float(p3_tmax),
        )
        family_dir = out_dir / f"replication_{family}"
        family_dir.mkdir(parents=True, exist_ok=True)
        _write_ablation_outputs(family_dir, df, summary)
        outputs.append(
            {
                "family": str(family),
                "source_run_dir": str(best["run_dir"]),
                "late_minus_early_gap_bal_acc": gap,
                "summary": summary,
                "output_dir": str(family_dir),
            }
        )
    return outputs


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    campaign_root = Path(args.source_campaign_root)
    if not campaign_root.is_absolute():
        campaign_root = root / campaign_root
    source_best = load_best_training_run(
        campaign_root=campaign_root, expected_family="cnn_transformer"
    )

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = root / out_base
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="causal_go_stop",
        config=vars(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")

    device = resolve_device(args.device)
    set_random_seed(int(args.random_state))
    ablation_df, ablation_summary = _run_post_hoc_ablation(
        run_dir=Path(source_best["run_dir"]),
        root=root,
        device=device,
        random_state=int(args.random_state),
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
        p3_tmin=float(args.p3_tmin),
        p3_tmax=float(args.p3_tmax),
    )
    _write_ablation_outputs(out_dir, ablation_df, ablation_summary)

    pair_spec = choose_constrained_retrain_pair(
        ablation_summary, p3_window_s=tuple(ablation_summary["p3_window_s"])
    )
    constrained_results = _run_constrained_retrain(
        source_best=source_best,
        pair_spec=pair_spec,
        out_dir=out_dir / "constrained_retraining",
        device=str(args.device),
        random_state=int(args.random_state),
        retrain_epochs=int(args.retrain_epochs),
        retrain_patience=int(args.retrain_patience),
        retrain_batch_size=int(args.retrain_batch_size),
    )
    constrained_csv = out_dir / "constrained_retraining_comparison.csv"
    pd.DataFrame(constrained_results).to_csv(constrained_csv, index=False)
    save_json(
        out_dir / "constrained_retraining_summary.json",
        {"pair_spec": pair_spec, "results": constrained_results},
    )

    replication_payload = _maybe_replicate_post_hoc(
        replicate_families=[str(x) for x in args.replicate_on],
        threshold=float(args.replication_threshold),
        cnn_summary=ablation_summary,
        constrained_results=constrained_results,
        source_campaign_root=campaign_root,
        explicit_roots={
            "pure_cnn": Path(args.replication_root_pure_cnn).resolve()
            if args.replication_root_pure_cnn
            else None,
            "enigma": Path(args.replication_root_enigma).resolve()
            if args.replication_root_enigma
            else None,
        },
        root=root,
        out_dir=out_dir,
        random_state=int(args.random_state),
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
        p3_tmin=float(args.p3_tmin),
        p3_tmax=float(args.p3_tmax),
    )
    save_json(
        out_dir / "replication_summary.json", {"replications": replication_payload}
    )

    summary_md = out_dir / "causal_go_stop_summary.md"
    strongest = ablation_summary.get("strongest_drop", {})
    lines = ["# Causal Go-vs-Stop Summary", ""]
    lines.append(f"- Source best run: `{source_best['run_dir']}`")
    lines.append(f"- Source experiment family: **{source_best['experiment_family']}**")
    if strongest:
        lines.append(
            f"- Strongest post-hoc drop: **{strongest['ablation_name']}** "
            f"(delta balanced accuracy {float(strongest['mean_delta_bal_acc']):.3f})"
        )
    lines.append(f"- Constrained retraining pair: **{pair_spec['pair_name']}**")
    if replication_payload:
        lines.append(
            f"- Cross-model replication executed for: **{', '.join(r['family'] for r in replication_payload)}**"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    save_json(
        out_dir / "run_metadata.json",
        {
            "source_campaign_root": str(campaign_root),
            "source_best_run": source_best,
            "ablation_summary": ablation_summary,
            "constrained_pair": pair_spec,
            "replication_threshold": float(args.replication_threshold),
        },
    )
    rendered = render_causal_go_stop_reports(run_dir=out_dir, output_root=out_base)
    print(f"Saved: {out_dir / 'ablation_metrics.csv'}")
    print(f"Saved: {constrained_csv}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {out_dir / 'run_metadata.json'}")
    for key, value in rendered.items():
        print(f"Saved report asset [{key}]: {value}")


if __name__ == "__main__":
    main()
