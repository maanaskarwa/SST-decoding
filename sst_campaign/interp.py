"""integrated gradients, temporal occlusion"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.interpretability.attribution import temporal_occlusion_curve
from pipeline.interpretability.plot import (
    plot_channel_time_heatmap,
    plot_erp_summary,
    plot_relevance_curves,
)
from pipeline.interpretability_helpers import mean_abs_ig, time_window_to_indices
from pipeline.misc import GO_LABEL, STOP_LABEL, save_json
from pipeline.perf import configure_torch_runtime, resolve_device
from pipeline.run_versioning import prepare_versioned_output_dir
from pipeline.train.driver import (
    map_roi_indices,
    predict_probabilities,
    select_indices,
    set_random_seed,
)
from sst_campaign.utils.common import emit_run_output_dir
from sst_campaign.utils.saved_run_context import (
    iter_saved_run_folds,
    load_saved_run_data,
)
from sst_campaign.utils.stop_windows import resolve_p3_window

DEFAULT_ROI_CHANNELS = ["Cz", "CP1", "CP2", "Pz", "P3", "P4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic interpretability analysis for SST campaign runs."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high", "medium"], default=None
    )
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-batch-size", type=int, default=32)
    parser.add_argument("--max-stop-samples", type=int, default=512)
    parser.add_argument("--max-go-samples", type=int, default=512)
    parser.add_argument("--include-misclassified", action="store_true")
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
    parser.add_argument("--roi-channels", nargs="+", default=DEFAULT_ROI_CHANNELS)
    parser.add_argument("--occlusion-window-ms", type=float, default=64.0)
    parser.add_argument("--occlusion-stride-ms", type=float, default=16.0)
    parser.add_argument("--random-state", type=int, default=9)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--disable-versioning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_data = load_saved_run_data(root=root, run_dir=run_dir)

    set_random_seed(int(args.random_state))
    rng = np.random.default_rng(int(args.random_state))

    times_s = run_data.times_s
    channel_names = run_data.channel_names
    p3_window = resolve_p3_window(
        y=run_data.y,
        stop_signal_onset_s=run_data.stop_signal_onset_s,
        p3_tmin=float(args.p3_tmin),
        p3_tmax=float(args.p3_tmax),
    )
    p3_mask = p3_window.mask(times_s)
    if not p3_mask.any():
        raise RuntimeError(
            f"P3 window has no samples in run time axis: {p3_window.window_s}"
        )

    roi_indices = map_roi_indices(
        channel_names=channel_names, roi_channels=[str(x) for x in args.roi_channels]
    )
    if not roi_indices:
        raise RuntimeError("None of the ROI channels were found")
    resolved_roi_channels = [channel_names[i] for i in roi_indices]
    pz_index = (
        channel_names.index("Pz")
        if "Pz" in channel_names
        else roi_indices[len(roi_indices) // 2]
    )

    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
    )
    window_size, stride = time_window_to_indices(
        times_s=times_s,
        window_ms=float(args.occlusion_window_ms),
        stride_ms=float(args.occlusion_stride_ms),
    )

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_p_stop: list[np.ndarray] = []
    all_raw_stop: list[np.ndarray] = []
    all_raw_go: list[np.ndarray] = []
    stop_abs_ig_weighted_sum = np.zeros(
        (run_data.X_raw.shape[1], run_data.X_raw.shape[2]), dtype=np.float64
    )
    go_abs_ig_weighted_sum = np.zeros(
        (run_data.X_raw.shape[1], run_data.X_raw.shape[2]), dtype=np.float64
    )
    stop_ig_count = 0
    go_ig_count = 0
    occ_weighted_sum: np.ndarray | None = None
    occ_times_s: np.ndarray | None = None
    occ_count = 0

    for fold in iter_saved_run_folds(run_data, device=device):
        p_stop, y_pred = predict_probabilities(
            model=fold.model,
            x_np=fold.X_te,
            device=device,
        )

        all_y_true.append(fold.y_te)
        all_y_pred.append(y_pred)
        all_p_stop.append(p_stop)
        all_raw_stop.append(fold.X_te_raw[fold.y_te == STOP_LABEL])
        all_raw_go.append(fold.X_te_raw[fold.y_te == GO_LABEL])

        stop_sel = select_indices(
            y_true=fold.y_te,
            y_pred=y_pred,
            target_label=STOP_LABEL,
            max_samples=max(
                1,
                int(math.ceil(int(args.max_stop_samples) / len(run_data.fold_dirs))),
            ),
            include_misclassified=bool(args.include_misclassified),
            rng=rng,
        )
        go_sel = select_indices(
            y_true=fold.y_te,
            y_pred=y_pred,
            target_label=GO_LABEL,
            max_samples=max(
                1,
                int(math.ceil(int(args.max_go_samples) / len(run_data.fold_dirs))),
            ),
            include_misclassified=bool(args.include_misclassified),
            rng=rng,
        )
        X_stop_sel = fold.X_te[stop_sel]
        X_go_sel = fold.X_te[go_sel]

        if len(X_stop_sel) > 0:
            stop_mean_abs_ig = mean_abs_ig(
                model=fold.model,
                x_np=X_stop_sel,
                device=device,
                ig_steps=int(args.ig_steps),
            )
            stop_abs_ig_weighted_sum += stop_mean_abs_ig * float(len(X_stop_sel))
            stop_ig_count += int(len(X_stop_sel))
            occ_local_n = min(len(X_stop_sel), max(32, int(args.ig_batch_size) * 4))
            occ_input = torch.Tensor(
                X_stop_sel[:occ_local_n].astype(np.float32, copy=False)
            ).to(device=device)

            centers, deltas = temporal_occlusion_curve(
                model=fold.model,
                x=occ_input,
                window_size=window_size,
                stride=stride,
            )
            local_occ_times_s = times_s[centers]
            if occ_weighted_sum is None:
                occ_weighted_sum = np.zeros_like(deltas, dtype=np.float64)
                occ_times_s = local_occ_times_s
            occ_weighted_sum += deltas * float(occ_local_n)
            occ_count += int(occ_local_n)

        if len(X_go_sel) > 0:
            go_mean_abs_ig = mean_abs_ig(
                model=fold.model,
                x_np=X_go_sel,
                device=device,
                ig_steps=int(args.ig_steps),
            )
            go_abs_ig_weighted_sum += go_mean_abs_ig * float(len(X_go_sel))
            go_ig_count += int(len(X_go_sel))

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    X_stop_raw = np.concatenate(all_raw_stop, axis=0)
    X_go_raw = np.concatenate(all_raw_go, axis=0)
    stop_roi_erp = X_stop_raw[:, roi_indices, :].mean(axis=(0, 1))
    go_roi_erp = X_go_raw[:, roi_indices, :].mean(axis=(0, 1))
    stop_minus_go_roi_erp = stop_roi_erp - go_roi_erp
    stop_pz_erp = X_stop_raw[:, pz_index, :].mean(axis=0)
    go_pz_erp = X_go_raw[:, pz_index, :].mean(axis=0)
    stop_minus_go_pz_erp = stop_pz_erp - go_pz_erp
    stop_abs_ig = stop_abs_ig_weighted_sum / float(stop_ig_count)
    go_abs_ig = go_abs_ig_weighted_sum / float(go_ig_count)
    stop_all_time = stop_abs_ig.mean(axis=0)
    stop_roi_time = stop_abs_ig[roi_indices].mean(axis=0)
    go_all_time = go_abs_ig.mean(axis=0)
    go_roi_time = go_abs_ig[roi_indices].mean(axis=0)
    stop_minus_go_all_time = stop_all_time - go_all_time
    stop_minus_go_roi_time = stop_roi_time - go_roi_time
    occ_deltas = (
        occ_weighted_sum / float(occ_count)
        if occ_weighted_sum is not None
        else np.array([])
    )
    if occ_times_s is None or occ_deltas.size == 0:
        raise RuntimeError("No temporal occlusion results were produced.")

    occ_p3_mask = p3_window.mask(occ_times_s)
    roi_erp_peak_idx = int(np.argmax(stop_minus_go_roi_erp[p3_mask]))
    pz_erp_peak_idx = int(np.argmax(stop_minus_go_pz_erp[p3_mask]))

    out_base = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (run_dir / "interpretability_results")
    )
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="sst_campaign_analyze_interpretability",
        config=vars(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")

    erp_roi_png = out_dir / "erp_roi_stop_vs_go.png"
    erp_pz_png = out_dir / "erp_pz_stop_vs_go.png"
    ig_curve_png = out_dir / "ig_temporal_relevance.png"
    ig_heatmap_png = out_dir / "ig_stop_channel_time_heatmap.png"
    occlusion_png = out_dir / "stop_occlusion_delta_pstop.png"
    erp_csv = out_dir / "erp_curves.csv"
    ig_csv = out_dir / "ig_temporal_curves.csv"
    occ_csv = out_dir / "occlusion_curve.csv"
    arrays_npz = out_dir / "interpretability_arrays.npz"
    summary_json = out_dir / "interpretability_summary.json"

    plot_erp_summary(
        times_s=times_s,
        go_curve=go_roi_erp,
        stop_curve=stop_roi_erp,
        diff_curve=stop_minus_go_roi_erp,
        out_path=erp_roi_png,
        title=f"ROI ERP: stop vs go ({', '.join(resolved_roi_channels)})",
        p3_window=p3_window.window_s,
    )
    plot_erp_summary(
        times_s=times_s,
        go_curve=go_pz_erp,
        stop_curve=stop_pz_erp,
        diff_curve=stop_minus_go_pz_erp,
        out_path=erp_pz_png,
        title=f"Pz ERP: stop vs go ({channel_names[pz_index]})",
        p3_window=p3_window.window_s,
    )
    plot_relevance_curves(
        times_s=times_s,
        curves={
            "stop IG (all channels)": stop_all_time,
            "stop IG (ROI)": stop_roi_time,
            "go IG (all channels)": go_all_time,
            "stop-go IG (ROI)": stop_minus_go_roi_time,
        },
        out_path=ig_curve_png,
        title="Integrated gradients for stop-vs-go evidence",
        ylabel="Mean absolute attribution",
        p3_window=p3_window.window_s,
    )
    plot_channel_time_heatmap(
        values=stop_abs_ig,
        times_s=times_s,
        channel_names=channel_names,
        out_path=ig_heatmap_png,
        title="Stop-trial integrated gradients heatmap",
        p3_window=p3_window.window_s,
    )
    plot_relevance_curves(
        times_s=occ_times_s,
        curves={"mean drop in P(stop)": occ_deltas},
        out_path=occlusion_png,
        title="Stop-trial temporal occlusion",
        ylabel="Delta P(stop)",
        p3_window=p3_window.window_s,
    )
    pd.DataFrame(
        {
            "time_s": times_s,
            "roi_go": go_roi_erp,
            "roi_stop": stop_roi_erp,
            "roi_stop_minus_go": stop_minus_go_roi_erp,
            "pz_go": go_pz_erp,
            "pz_stop": stop_pz_erp,
            "pz_stop_minus_go": stop_minus_go_pz_erp,
        }
    ).to_csv(erp_csv, index=False)
    pd.DataFrame(
        {
            "time_s": times_s,
            "stop_ig_all": stop_all_time,
            "stop_ig_roi": stop_roi_time,
            "go_ig_all": go_all_time,
            "go_ig_roi": go_roi_time,
            "stop_minus_go_ig_all": stop_minus_go_all_time,
            "stop_minus_go_ig_roi": stop_minus_go_roi_time,
        }
    ).to_csv(ig_csv, index=False)
    pd.DataFrame({"time_s": occ_times_s, "delta_p_stop": occ_deltas}).to_csv(
        occ_csv, index=False
    )
    np.savez_compressed(
        arrays_npz,
        stop_abs_ig=stop_abs_ig,
        go_abs_ig=go_abs_ig,
        times_s=times_s,
        occ_times_s=occ_times_s,
        occ_deltas=occ_deltas,
        roi_indices=np.asarray(roi_indices, dtype=np.int64),
    )
    summary = {
        "source_run_dir": str(run_dir),
        "model_name": run_data.model_name,
        "device": str(device),
        **p3_window.summary(),
        "roi_channels_resolved": resolved_roi_channels,
        "overall_accuracy": float((y_true_all == y_pred_all).mean()),
        "roi_erp_peak_time_s": float(times_s[p3_mask][roi_erp_peak_idx]),
        "roi_erp_peak_amp": float(stop_minus_go_roi_erp[p3_mask][roi_erp_peak_idx]),
        "pz_erp_peak_time_s": float(times_s[p3_mask][pz_erp_peak_idx]),
        "pz_erp_peak_amp": float(stop_minus_go_pz_erp[p3_mask][pz_erp_peak_idx]),
        "stop_ig_peak_time_s": float(times_s[int(np.argmax(stop_roi_time))]),
        "stop_ig_p3_share": float(
            stop_roi_time[p3_mask].sum() / max(stop_roi_time.sum(), 1e-12)
        ),
        "occlusion_peak_time_s": float(occ_times_s[int(np.argmax(occ_deltas))]),
        "occlusion_peak_delta": float(np.max(occ_deltas)),
        "occlusion_peak_in_p3_window": bool(occ_p3_mask[int(np.argmax(occ_deltas))]),
        "outputs": {
            "erp_roi_png": str(erp_roi_png),
            "erp_pz_png": str(erp_pz_png),
            "ig_curve_png": str(ig_curve_png),
            "ig_heatmap_png": str(ig_heatmap_png),
            "occlusion_png": str(occlusion_png),
            "erp_csv": str(erp_csv),
            "ig_csv": str(ig_csv),
            "occ_csv": str(occ_csv),
            "arrays_npz": str(arrays_npz),
        },
    }
    save_json(summary_json, summary)
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
