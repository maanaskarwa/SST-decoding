from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

from pipeline.data.utils import flatten_positive_ints, to_list

GO_TRIAL = "go"
STOP_SUCCESS_TRIAL = "stop_success"
STOP_FAILURE_TRIAL = "stop_failure"
RESPONSE_BINS = {5, 6}


@dataclass(frozen=True)
class BehavioralTrial:
    subject: int
    epoch_index: int
    trial_type: str
    response_latency_s: float | None
    stop_signal_delay_s: float | None
    lock_bins: tuple[int, ...]


def _event_type_matches(event_type: object, target: str) -> bool:
    text = str(event_type).strip()
    if text == target:
        return True
    try:
        return float(text) == float(target)
    except ValueError:
        return False


def _is_response_event(event_bins: list[int], event_type: str) -> bool:
    return (
        bool(set(event_bins) & RESPONSE_BINS)
        or "B5,6" in event_type
        or "(S9)" in event_type
        or event_type.strip() == "9"
    )


def _lock_bins(event_bini: list[Any], latencies: list[float]) -> set[int]:
    bins: set[int] = set()
    lock_indices = [
        idx
        for idx, latency in enumerate(latencies)
        if np.isfinite(latency) and abs(float(latency)) < 1e-12
    ]
    for idx in lock_indices:
        if idx < len(event_bini):
            bins.update(flatten_positive_ints(event_bini[idx]))
    return bins


def extract_behavioral_trials_from_epochs(
    epochs: Iterable[object],
    *,
    subject: int,
    go_bins: set[int] | None = None,
    stop_success_bins: set[int] | None = None,
    stop_failure_bins: set[int] | None = None,
    stop_event_type: str = "2",
) -> list[BehavioralTrial]:
    go_bins = {1} if go_bins is None else set(go_bins)
    stop_success_bins = {2} if stop_success_bins is None else set(stop_success_bins)
    stop_failure_bins = {3} if stop_failure_bins is None else set(stop_failure_bins)

    parsed: list[dict[str, Any]] = []

    for epoch_index, epoch in enumerate(epochs):
        event_bini = to_list(getattr(epoch, "eventbini", []))
        event_type = [str(value) for value in to_list(getattr(epoch, "eventtype", []))]
        latencies = [
            float(value) for value in to_list(getattr(epoch, "eventlatency", []))
        ]
        bins = _lock_bins(event_bini, latencies)

        has_go = bool(bins & go_bins)
        has_success = bool(bins & stop_success_bins)
        has_failure = bool(bins & stop_failure_bins)
        if has_go and not (has_success or has_failure):
            trial_type = GO_TRIAL
        elif has_success and not has_failure:
            trial_type = STOP_SUCCESS_TRIAL
        elif has_failure and not has_success:
            trial_type = STOP_FAILURE_TRIAL
        else:
            continue

        response_latency_raw: float | None = None
        stop_signal_raw: float | None = None
        n_events = max(len(event_bini), len(event_type), len(latencies))
        for idx in range(n_events):
            event_bins = (
                flatten_positive_ints(event_bini[idx]) if idx < len(event_bini) else []
            )
            event_text = event_type[idx] if idx < len(event_type) else ""
            latency = latencies[idx] if idx < len(latencies) else float("nan")
            if not np.isfinite(latency):
                continue
            if latency > 0 and _is_response_event(event_bins, event_text):
                if response_latency_raw is None or latency < response_latency_raw:
                    response_latency_raw = float(latency)
            if latency > 0 and _event_type_matches(event_text, stop_event_type):
                if stop_signal_raw is None or latency < stop_signal_raw:
                    stop_signal_raw = float(latency)

        parsed.append(
            {
                "subject": int(subject),
                "epoch_index": int(epoch_index),
                "trial_type": trial_type,
                "response_latency_raw": response_latency_raw,
                "stop_signal_raw": stop_signal_raw,
                "lock_bins": tuple(sorted(int(value) for value in bins)),
            }
        )

    trials: list[BehavioralTrial] = []
    for row in parsed:
        response_raw = row["response_latency_raw"]
        stop_raw = row["stop_signal_raw"]
        trials.append(
            BehavioralTrial(
                subject=int(row["subject"]),
                epoch_index=int(row["epoch_index"]),
                trial_type=str(row["trial_type"]),
                response_latency_s=None
                if response_raw is None
                else float(response_raw) * 1e-3,
                stop_signal_delay_s=None
                if stop_raw is None
                else float(stop_raw) * 1e-3,
                lock_bins=tuple(row["lock_bins"]),
            )
        )
    return trials


def load_behavioral_trials(
    root: Path, subjects: Iterable[int]
) -> list[BehavioralTrial]:
    trials: list[BehavioralTrial] = []
    for subject in subjects:
        set_path = Path(root) / "preprocessed" / f"S{int(subject)}_preprocessed.set"
        mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
        if "epoch" not in mat:
            raise KeyError(f"{set_path} has no 'epoch' variable")
        trials.extend(
            extract_behavioral_trials_from_epochs(
                np.atleast_1d(mat["epoch"]), subject=int(subject)
            )
        )
    return trials


def trial_table(trials: list[BehavioralTrial]) -> pd.DataFrame:
    rows = [asdict(trial) for trial in trials]
    df = pd.DataFrame(rows)
    if df.empty:
        columns = [
            "subject",
            "epoch_index",
            "trial_type",
            "response_latency_s",
            "stop_signal_delay_s",
            "lock_bins",
        ]
        return pd.DataFrame({column: [] for column in columns})
    df["lock_bins"] = df["lock_bins"].map(
        lambda values: ";".join(str(value) for value in values)
    )
    return df.sort_values(by=["subject", "epoch_index"]).reset_index(drop=True)


def _integration_ssrt(
    go_rt_s: np.ndarray, stop_response_probability: float, mean_ssd_s: float
) -> tuple[float, float]:
    if (
        go_rt_s.size == 0
        or not np.isfinite(stop_response_probability)
        or not np.isfinite(mean_ssd_s)
    ):
        return float("nan"), float("nan")
    if stop_response_probability <= 0.0 or stop_response_probability >= 1.0:
        return float("nan"), float("nan")
    ordered = np.sort(go_rt_s.astype(np.float64, copy=False))
    rank = int(np.ceil(stop_response_probability * ordered.size)) - 1
    rank = int(np.clip(rank, 0, ordered.size - 1))
    go_rt_at_quantile = float(ordered[rank])
    return go_rt_at_quantile - float(mean_ssd_s), go_rt_at_quantile


def summarize_subject_behavior(trials: list[BehavioralTrial]) -> pd.DataFrame:
    df = trial_table(trials)
    rows: list[dict[str, Any]] = []
    for subject, group in df.groupby("subject"):
        subject_id = int(str(subject))
        go = group[group["trial_type"] == GO_TRIAL]
        stop_success = group[group["trial_type"] == STOP_SUCCESS_TRIAL]
        stop_failure = group[group["trial_type"] == STOP_FAILURE_TRIAL]
        stop = group[group["trial_type"].isin([STOP_SUCCESS_TRIAL, STOP_FAILURE_TRIAL])]
        go_response_values = np.asarray(go["response_latency_s"], dtype=np.float64)
        failed_response_values = np.asarray(
            stop_failure["response_latency_s"], dtype=np.float64
        )
        ssd_values = np.asarray(stop["stop_signal_delay_s"], dtype=np.float64)
        go_rt = go_response_values[np.isfinite(go_response_values)]
        failed_stop_rt = failed_response_values[np.isfinite(failed_response_values)]
        ssd = ssd_values[np.isfinite(ssd_values)]
        n_stop = int(len(stop))
        n_failed = int(len(stop_failure))
        p_respond_stop = float(n_failed / n_stop) if n_stop else float("nan")
        mean_ssd = float(np.mean(ssd)) if ssd.size else float("nan")
        ssrt, go_rt_at_quantile = _integration_ssrt(go_rt, p_respond_stop, mean_ssd)
        rows.append(
            {
                "subject": subject_id,
                "n_go": int(len(go)),
                "n_go_with_response": int(go_rt.size),
                "n_stop": n_stop,
                "n_stop_success": int(len(stop_success)),
                "n_stop_failure": n_failed,
                "go_response_rate": float(go_rt.size / len(go))
                if len(go)
                else float("nan"),
                "stop_success_rate": float(len(stop_success) / n_stop)
                if n_stop
                else float("nan"),
                "stop_response_probability": p_respond_stop,
                "mean_go_rt_s": float(np.mean(go_rt)) if go_rt.size else float("nan"),
                "median_go_rt_s": float(np.median(go_rt))
                if go_rt.size
                else float("nan"),
                "mean_stop_failure_rt_s": float(np.mean(failed_stop_rt))
                if failed_stop_rt.size
                else float("nan"),
                "mean_ssd_s": mean_ssd,
                "integration_go_rt_quantile_s": go_rt_at_quantile,
                "ssrt_integration_s": ssrt,
            }
        )
    return pd.DataFrame(rows).sort_values(by=["subject"]).reset_index(drop=True)


def inhibition_function_table(trials: list[BehavioralTrial]) -> pd.DataFrame:
    df = trial_table(trials)
    stop = df[df["trial_type"].isin([STOP_SUCCESS_TRIAL, STOP_FAILURE_TRIAL])].copy()
    stop_ssd = np.asarray(stop["stop_signal_delay_s"], dtype=np.float64)
    stop = stop.loc[np.isfinite(stop_ssd)].copy()
    if stop.empty:
        columns = ["subject", "ssd_ms", "n_stop", "n_response", "p_response"]
        return pd.DataFrame({column: [] for column in columns})
    stop["ssd_ms"] = np.rint(
        np.asarray(stop["stop_signal_delay_s"], dtype=np.float64) * 1000.0
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for _, group in stop.groupby(["subject", "ssd_ms"]):
        subject_id = int(str(group["subject"].iloc[0]))
        ssd_ms = int(str(group["ssd_ms"].iloc[0]))
        n_stop = int(len(group))
        n_response = int((group["trial_type"] == STOP_FAILURE_TRIAL).sum())
        rows.append(
            {
                "subject": subject_id,
                "ssd_ms": ssd_ms,
                "n_stop": n_stop,
                "n_response": n_response,
                "p_response": float(n_response / n_stop) if n_stop else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(by=["ssd_ms", "subject"]).reset_index(drop=True)
    )
