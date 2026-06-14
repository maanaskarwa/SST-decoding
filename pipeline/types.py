from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectDecodeData:
    subject: int
    X: np.ndarray  # (n_epochs, n_channels, n_times)
    y: np.ndarray  # (n_epochs,)
    epoch_indices: np.ndarray  # indices in original preprocessed set
    times: np.ndarray  # (n_times,)
    ch_names: list[str]
    stop_signal_onset_s: np.ndarray | None = (
        None  # stop-signal onset per epoch, in start-aligned seconds. None for go trials
    )
