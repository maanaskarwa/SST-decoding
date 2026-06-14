from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.misc import STOP_LABEL


@dataclass(frozen=True)
class StopRelativeWindow:
    reference: str
    relative_window_s: tuple[float, float]
    window_s: tuple[float, float]
    mean_stop_onset_s: float | None
    n_stop_trials: int

    def mask(self, times_s: np.ndarray) -> np.ndarray:
        return (times_s >= self.window_s[0]) & (times_s <= self.window_s[1])

    def summary(self) -> dict[str, object]:
        return {
            "p3_window_reference": self.reference,
            "p3_window_relative_to_stop_s": [
                float(self.relative_window_s[0]),
                float(self.relative_window_s[1]),
            ],
            "p3_window_s": [float(self.window_s[0]), float(self.window_s[1])],
            "mean_stop_onset_s": None
            if self.mean_stop_onset_s is None
            else float(self.mean_stop_onset_s),
            "n_stop_signal_onsets": int(self.n_stop_trials),
        }


def resolve_p3_window(
    *,
    y: np.ndarray,
    stop_signal_onset_s: np.ndarray,
    p3_tmin: float,
    p3_tmax: float,
    reference: str = "mean-stop",
) -> StopRelativeWindow:
    if p3_tmax <= p3_tmin:
        raise ValueError(f"Invalid P3 window: {p3_tmin}-{p3_tmax}")

    if reference != "mean-stop":
        raise ValueError(f"Unknown P3 window reference: {reference}")

    stop_onsets = np.asarray(stop_signal_onset_s, dtype=np.float64)
    labels = np.asarray(y)

    valid = (labels == STOP_LABEL) & np.isfinite(stop_onsets)
    mean_stop_onset = stop_onsets[valid].mean()
    return StopRelativeWindow(
        reference="mean-stop",
        relative_window_s=(p3_tmin, p3_tmax),
        window_s=(
            mean_stop_onset + p3_tmin,
            mean_stop_onset + p3_tmax,
        ),
        mean_stop_onset_s=mean_stop_onset,
        n_stop_trials=valid.sum(),
    )
