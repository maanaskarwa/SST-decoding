from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.io import loadmat

from pipeline.misc import GO_LABEL, STOP_LABEL
from pipeline.types import SubjectDecodeData


def find_subjects(root: Path) -> list[int]:
    out: set[int] = set()
    # file names look like S21_preprocessed.set
    for path in sorted((root / "preprocessed").glob("S*_preprocessed.set")):
        stem = path.name.replace("_preprocessed.set", "")
        out.add(int(stem[1:]))
    return sorted(list(out))


def extract_lock_bins_from_set(set_path: Path) -> list[set[int]]:
    """Extract bin IDs at epoch-locking latency (0 ms) from EEGLAB epoch structs."""
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    if "epoch" not in mat:
        raise KeyError(f"{set_path} has no 'epoch' variable")

    epoch_structs = np.atleast_1d(mat["epoch"])
    all_bins: list[set[int]] = []

    for epoch in epoch_structs:
        event_bini = to_list(epoch.eventbini)
        event_latency = to_list(epoch.eventlatency)
        lock_indices = [
            idx
            for idx, latency in enumerate(event_latency)
            if abs(float(latency)) < 1e-12
        ]

        epoch_bins: set[int] = set()
        for idx in lock_indices:
            epoch_bins.update(flatten_positive_ints(event_bini[idx]))
        all_bins.append(epoch_bins)

    return all_bins


def _equals_event_type(event_type: object, target: str) -> bool:
    text = str(event_type).strip()
    if text == target:
        return True
    try:
        return float(text) == float(target)
    except ValueError:
        return False


def extract_stop_signal_onsets_from_set(
    set_path: Path, stop_event_type: str = "2"
) -> np.ndarray:
    """Extract positive stop-signal latencies from EEGLAB epoch structs, in seconds.

    The decoded SST epochs are go/start aligned. Marker code 2 is the stop
    signal; its positive latency is the trial-specific stop-signal delay.
    Epochs without a positive stop marker receive NaN.
    """
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    if "epoch" not in mat:
        raise KeyError(f"{set_path} has no 'epoch' variable")

    raw_onsets: list[float] = []
    positive_latencies: list[float] = []
    for epoch in np.atleast_1d(mat["epoch"]):
        event_types = to_list(getattr(epoch, "eventtype", []))
        event_latencies = to_list(getattr(epoch, "eventlatency", []))
        stop_latencies: list[float] = []
        for event_type, latency in zip(event_types, event_latencies, strict=False):
            if not _equals_event_type(event_type, stop_event_type):
                continue
            try:
                latency_value = float(latency)
            except TypeError, ValueError:
                continue
            if np.isfinite(latency_value) and latency_value > 0:
                stop_latencies.append(latency_value)
                positive_latencies.append(latency_value)
        raw_onsets.append(min(stop_latencies) if stop_latencies else np.nan)

    scale = (
        1e-3
        if positive_latencies and float(np.median(positive_latencies)) > 5.0
        else 1.0
    )
    return np.asarray(raw_onsets, dtype=np.float64) * scale


def load_subject_decode_data(
    root: Path,
    subject: int,
    go_bins: set[int],
    stop_bins: set[int],
    crop_tmin: float | None,
    crop_tmax: float | None,
) -> SubjectDecodeData:
    set_path = root / "preprocessed" / f"S{subject}_preprocessed.set"
    epochs = mne.read_epochs_eeglab(set_path, verbose="warning")

    if crop_tmin is not None or crop_tmax is not None:
        tmin = epochs.times[0] if crop_tmin is None else float(crop_tmin)
        tmax = epochs.times[-1] if crop_tmax is None else float(crop_tmax)
        if tmax <= tmin:
            raise ValueError(
                f"Invalid crop window for S{subject}: tmin={tmin}, tmax={tmax}"
            )
        epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)

    epoch_bins = extract_lock_bins_from_set(set_path)
    stop_signal_onsets = extract_stop_signal_onsets_from_set(set_path)
    if len(epoch_bins) != len(epochs):
        raise RuntimeError(
            f"S{subject}: bin metadata/epoch count mismatch in {set_path}: "
            f"bins={len(epoch_bins)}, epochs={len(epochs)}"
        )
    if len(stop_signal_onsets) != len(epochs):
        raise RuntimeError(
            f"S{subject}: stop-signal metadata/epoch count mismatch in {set_path}: "
            f"stop_onsets={len(stop_signal_onsets)}, epochs={len(epochs)}"
        )

    labels: list[int] = []
    keep_idx: list[int] = []
    original_epoch_idx: list[int] = []
    for idx, bins in enumerate(epoch_bins):
        label = label_from_bins(bins, go_bins=go_bins, stop_bins=stop_bins)
        if label is None:
            continue

        labels.append(label)
        keep_idx.append(idx)
        original_epoch_idx.append(idx)

    epochs_sel = epochs[keep_idx]
    X = epochs_sel.get_data(copy=True, picks="eeg").astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)

    return SubjectDecodeData(
        subject=subject,
        X=X,
        y=y,
        epoch_indices=np.asarray(original_epoch_idx, dtype=np.int64),
        times=epochs_sel.times.astype(np.float64, copy=True),
        ch_names=list(epochs_sel.ch_names),
        stop_signal_onset_s=stop_signal_onsets[np.asarray(keep_idx, dtype=np.int64)],
    )


def to_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def flatten_positive_ints(value: Any) -> list[int]:
    out: list[int] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, np.ndarray):
            stack.extend(item.ravel().tolist())
            continue
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        try:
            ivalue = int(item)
        except TypeError, ValueError:
            continue
        if ivalue > 0:
            out.append(ivalue)
    return out


def label_from_bins(
    bins: set[int], go_bins: set[int], stop_bins: set[int]
) -> int | None:
    has_go = bool(bins & go_bins)
    has_stop = bool(bins & stop_bins)
    if has_go and not has_stop:
        return GO_LABEL
    if has_stop and not has_go:
        return STOP_LABEL
    return None
