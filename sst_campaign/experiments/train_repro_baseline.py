"""CLI entrypoint for classic baseline training. Trains CNN/Transformer, pure CNN, or ENIGMA runs and writes folds, predictions, metrics, and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pipeline.data import (
    build_cv,
    save_split,
    split_train_val,
)
from pipeline.eval import plot_confusion, plot_training_curves
from pipeline.misc import save_json
from pipeline.perf import configure_torch_runtime, make_grad_scaler, resolve_device
from pipeline.run_versioning import prepare_versioned_output_dir
from pipeline.train import (
    run_loader,
    save_checkpoint,
    set_random_seed,
)
from pipeline.train.loss import build_loss_from_args
from sst_campaign.experiments.common import prepare_decoding_dataset
from sst_campaign.experiments.training_fold_loaders import (
    make_standardized_fold_loaders,
)
from sst_campaign.experiments.training_results import (
    aggregate_metric_columns,
    checkpoint_score_from_validation,
    dataset_count_fields,
    fold_metric_row,
    format_metric_triplet,
    write_per_subject_prediction_report,
)
from sst_campaign.utils.common import emit_run_output_dir
from sst_campaign.utils.model_loading import (
    build_model_from_training_args,
    model_metadata_from_training_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible deep-model training for go-vs-stop EEG decoding. "
            "Saves model checkpoints, fold splits, standardizers, and run manifests."
        )
    )
    from sst_campaign.experiments.common import add_common_training_args

    add_common_training_args(parser, loso=False)
    # Baseline-specific: a distinct default output directory.
    parser.set_defaults(output_dir="decoding_results_repro_baseline")
    return parser.parse_args()


def _as_jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = vars(args).copy()
    if isinstance(out.get("root"), Path):
        out["root"] = str(out["root"])
    return out


def train_run(args: argparse.Namespace, root: Path, out_dir: Path) -> None:
    arrays, data_manifest = prepare_decoding_dataset(args=args, root=root)
    X = arrays["X"]
    y = arrays["y"]
    groups = arrays["groups"]
    epoch_idx = arrays["epoch_idx"]
    subject_col = arrays["subject_col"]
    sample_id = arrays["sample_id"]

    cv, cv_groups = build_cv(
        y=y,
        groups=groups,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        cv_mode=str(args.cv_mode),
    )

    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        matmul_precision=args.matmul_precision,
    )
    print(f"Using device: {device}")

    fold_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        cv.split(X, y, groups=cv_groups), start=1
    ):
        tr_idx, val_idx = split_train_val(
            train_indices=np.asarray(trainval_idx),
            y=y,
            groups=groups,
            val_fraction=float(args.val_fraction),
            random_state=int(args.random_state) + fold_idx,
        )

        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        save_split(
            out_path=fold_dir / "split_indices.npz",
            train_idx=tr_idx,
            val_idx=val_idx,
            test_idx=np.asarray(test_idx),
            sample_id=sample_id,
            subject=subject_col,
        )

        loaders = make_standardized_fold_loaders(
            X=X,
            y=y,
            sample_id=sample_id,
            subject_ids=subject_col,
            train_idx=tr_idx,
            val_idx=val_idx,
            test_idx=np.asarray(test_idx),
            fold_dir=fold_dir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(args.persistent_workers),
            weighted_sampling=bool(args.weighted_sampling),
        )
        train_loader = loaders.train_loader
        val_loader = loaders.val_loader
        test_loader = loaders.test_loader

        model = build_model_from_training_args(
            args=args,
            in_channels=X.shape[1],
            n_times=X.shape[2],
        ).to(device)
        classification_loss = build_loss_from_args(
            args=args, y_train=y[tr_idx], device=device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        grad_scaler = make_grad_scaler(device=device, enabled=bool(args.amp))

        best_metric = -np.inf
        best_epoch = 0
        epochs_without_improve = 0
        rng = np.random.default_rng(int(args.random_state) + fold_idx)

        print(
            f"Fold {fold_idx}: train={len(tr_idx)} val={len(val_idx)} test={len(test_idx)} "
        )

        for epoch in range(1, int(args.epochs) + 1):
            train_loss, train_metrics, _ = run_loader(
                model=model,
                loader=train_loader,
                device=device,
                classification_loss=classification_loss,
                optimizer=optimizer,
                train_mode=True,
                noise_std=float(args.noise_std),
                time_mask_prob=float(args.time_mask_prob),
                channel_drop_prob=float(args.channel_drop_prob),
                rng=rng,
                amp_enabled=bool(args.amp),
                amp_dtype=str(args.amp_dtype),
                grad_scaler=grad_scaler,
                mixup_alpha=float(args.mixup_alpha),
                mixup_prob=float(args.mixup_prob),
            )

            val_loss, val_metrics, _ = run_loader(
                model=model,
                loader=val_loader,
                device=device,
                classification_loss=classification_loss,
                optimizer=None,
                train_mode=False,
                noise_std=0.0,
                time_mask_prob=0.0,
                channel_drop_prob=0.0,
                rng=rng,
                amp_enabled=bool(args.amp),
                amp_dtype=str(args.amp_dtype),
            )

            score = checkpoint_score_from_validation(
                val_loss=float(val_loss),
                val_metrics=val_metrics,
                checkpoint_metric=str(getattr(args, "checkpoint_metric", "auc")),
            )
            if score > best_metric + 1e-6:
                best_metric = float(score)
                best_epoch = int(epoch)
                epochs_without_improve = 0
                save_checkpoint(
                    path=fold_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    score=float(score),
                    extra={
                        "fold": int(fold_idx),
                    },
                )
            else:
                epochs_without_improve += 1

            history_rows.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_auc": float(train_metrics["auc"]),
                    "val_auc": float(val_metrics["auc"]),
                    "train_balanced_accuracy": float(
                        train_metrics["balanced_accuracy"]
                    ),
                    "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                }
            )

            print(
                f"  epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={val_metrics['auc']:.3f} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}"
            )

            if epochs_without_improve >= int(args.patience):
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

        ckpt = torch.load(fold_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        save_json(
            fold_dir / "fold_config.json",
            {
                "fold": int(fold_idx),
                "n_train": int(len(tr_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "best_epoch": int(best_epoch),
                "best_metric": float(best_metric),
            },
        )

        test_loss, test_metrics, test_preds = run_loader(
            model=model,
            loader=test_loader,
            device=device,
            classification_loss=classification_loss,
            optimizer=None,
            train_mode=False,
            noise_std=0.0,
            time_mask_prob=0.0,
            channel_drop_prob=0.0,
            rng=rng,
            amp_enabled=bool(args.amp),
            amp_dtype=str(args.amp_dtype),
        )

        lookup = pd.DataFrame(
            {
                "sample_id": sample_id[test_idx],
                "subject": subject_col[test_idx],
                "epoch_index_in_preprocessed_set": epoch_idx[test_idx],
                "fold": np.full(len(test_idx), fold_idx, dtype=int),
            }
        )
        test_preds = test_preds.merge(lookup, on="sample_id", how="left")
        prediction_rows.append(test_preds)

        fold_rows.append(
            fold_metric_row(
                fold=fold_idx,
                n_train=len(tr_idx),
                n_val=len(val_idx),
                n_test=len(test_idx),
                best_epoch=best_epoch,
                best_val_score=best_metric,
                test_loss=test_loss,
                test_metrics=test_metrics,
            )
        )

        print(
            f"Fold {fold_idx} test: auc={test_metrics['auc']:.3f} "
            f"ap={test_metrics['average_precision']:.3f} "
            f"acc={test_metrics['accuracy']:.3f} bal_acc={test_metrics['balanced_accuracy']:.3f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    history_df = pd.DataFrame(history_rows)
    pred_df = pd.concat(prediction_rows, axis=0, ignore_index=True)

    overall_metrics = {
        **aggregate_metric_columns(fold_df, std_ddof=0),
        **dataset_count_fields(y=y, groups=groups),
    }

    print("Overall CV: " + format_metric_triplet(overall_metrics))

    model_name = str(args.model).lower()
    # Use canonical name for file prefixes where possible; unknown models use their name as-is
    # (no silent fallback to cnn_transformer for unrecognized models).
    try:
        from sst_campaign.utils.model_specs import canonical_model_name

        prefix = canonical_model_name(model_name)
        # adjust for historical filename prefixes used by some core models
        if prefix == "enigma":
            prefix = "enigma_style"
    except ValueError, ImportError:
        prefix = model_name
    fold_csv = out_dir / f"{prefix}_fold_metrics.csv"
    history_csv = out_dir / f"{prefix}_history.csv"
    pred_csv = out_dir / f"{prefix}_predictions.csv"
    meta_json = out_dir / f"{prefix}_run_metadata.json"
    curve_png = out_dir / f"{prefix}_training_curves.png"
    cm_png = out_dir / f"{prefix}_confusion_matrix.png"
    run_cfg_json = out_dir / "run_config.json"
    manifest_json = out_dir / "dataset_manifest.json"

    fold_df.to_csv(fold_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)
    plot_training_curves(history_df, curve_png)
    plot_confusion(
        y_true=pred_df["y_true"].to_numpy(dtype=int),
        y_pred=pred_df["y_pred"].to_numpy(dtype=int),
        out_path=cm_png,
    )
    per_subject_report = write_per_subject_prediction_report(
        pred_df=pred_df,
        out_dir=out_dir,
        metrics_filename=f"{prefix}_per_subject_metrics.csv",
    )
    per_subject_csv = (
        None if per_subject_report is None else per_subject_report.metrics_csv
    )
    per_subject_cm_dir = (
        None if per_subject_report is None else per_subject_report.confusion_dir
    )

    model_metadata = model_metadata_from_training_args(args)

    metadata = {
        "model": model_metadata,
        "data": data_manifest,
        "training": {
            "model": str(args.model),
            "cv_mode": str(args.cv_mode),
            "n_splits": int(args.n_splits),
            "val_fraction": float(args.val_fraction),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "noise_std": float(args.noise_std),
            "time_mask_prob": float(args.time_mask_prob),
            "channel_drop_prob": float(args.channel_drop_prob),
            "weighted_sampling": bool(args.weighted_sampling),
            "class_weight_mode": str(args.class_weight_mode),
            "stop_loss_scale": float(args.stop_loss_scale),
            "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
            "checkpoint_metric": str(getattr(args, "checkpoint_metric", "auc")),
            "random_state": int(args.random_state),
            "device": str(device),
        },
        "overall_metrics": overall_metrics,
        "outputs": {
            "fold_metrics_csv": str(fold_csv),
            "history_csv": str(history_csv),
            "predictions_csv": str(pred_csv),
            "training_curves_png": str(curve_png),
            "confusion_matrix_png": str(cm_png),
            "per_subject_metrics_csv": str(per_subject_csv)
            if per_subject_csv
            else None,
            "per_subject_confusion_dir": str(per_subject_cm_dir)
            if per_subject_cm_dir
            else None,
        },
    }

    save_json(meta_json, metadata)
    save_json(run_cfg_json, _as_jsonable_args(args))
    save_json(manifest_json, data_manifest)

    print(f"Saved: {fold_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {pred_csv}")
    print(f"Saved: {curve_png}")
    print(f"Saved: {cm_png}")
    if per_subject_csv is not None:
        print(f"Saved: {per_subject_csv}")
    if per_subject_cm_dir is not None:
        print(f"Saved: {per_subject_cm_dir}")
    print(f"Saved: {meta_json}")
    print(f"Saved: {run_cfg_json}")
    print(f"Saved: {manifest_json}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    set_random_seed(int(args.random_state))

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = root / out_base
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="train_repro_baseline",
        config=_as_jsonable_args(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")

    train_run(args=args, root=root, out_dir=out_dir)


if __name__ == "__main__":
    main()
