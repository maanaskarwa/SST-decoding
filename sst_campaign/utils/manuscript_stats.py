"""Subject-level manuscript statistics for SST decoding reports."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    sd: float
    se: float
    ci95_low: float
    ci95_high: float
    n: int

    def as_dict(self, prefix: str) -> dict[str, float | int]:
        return {
            f"mean_{prefix}": self.mean,
            f"sd_{prefix}": self.sd,
            f"se_{prefix}": self.se,
            f"ci95_low_{prefix}": self.ci95_low,
            f"ci95_high_{prefix}": self.ci95_high,
            "n_subjects": self.n,
        }


def summarize_metric(
    values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int = 10000
) -> MetricSummary:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MetricSummary(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0
        )
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    se = float(sd / np.sqrt(finite.size)) if finite.size > 1 else 0.0
    if finite.size > 1 and n_bootstrap > 0:
        draws = rng.choice(finite, size=(int(n_bootstrap), finite.size), replace=True)
        boot_means = draws.mean(axis=1)
        ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    else:
        ci_low = ci_high = mean
    return MetricSummary(
        mean=mean,
        sd=sd,
        se=se,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        n=int(finite.size),
    )


def exact_sign_test_greater(
    values: np.ndarray, null_value: float = 0.0
) -> float | None:
    """One-sided exact sign test: median(values - null_value) > 0."""
    centered = np.asarray(values, dtype=np.float64) - null_value

    centered = centered[np.isfinite(centered)]
    nonzero = centered[np.abs(centered) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return None
    k = int((nonzero > 0).sum())
    return float(sum(comb(n, i) for i in range(k, n + 1)) / (2**n))


def paired_sign_test_greater(a: np.ndarray, b: np.ndarray) -> float | None:
    """One-sided exact sign test for paired values: a - b > 0."""
    return exact_sign_test_greater(
        np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), 0.0
    )


def paired_permutation_p_greater(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int = 10000,
) -> float | None:
    """One-sided paired sign-flip permutation test for mean(a - b) > 0."""
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    observed = float(diffs.mean())
    if diffs.size == 1:
        return 0.5 if observed > 0 else 1.0
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_permutations), diffs.size))
    null = (signs * diffs[None, :]).mean(axis=1)
    return float((np.count_nonzero(null >= observed) + 1) / (int(n_permutations) + 1))


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    indexed = [
        (idx, float(p))
        for idx, p in enumerate(p_values)
        if p is not None and np.isfinite(float(p))
    ]
    out: list[float | None] = [None for _ in p_values]
    if not indexed:
        return out
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[rank - 1] = running
    for (idx, _), adj in zip(indexed, adjusted):
        out[idx] = float(min(max(adj, 0.0), 1.0))
    return out


def comparison_record(
    *,
    comparison: str,
    metric: str,
    estimate: float,
    p_value: float | None,
    n: int,
    test: str,
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "metric": metric,
        "estimate": float(estimate),
        "p_value": None if p_value is None else float(p_value),
        "n_subjects": int(n),
        "test": test,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties (mergesort for stability)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation (centered, no scipy)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float("nan") if denom <= 0.0 else float(np.sum(a * b) / denom)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rank correlation using average ranks for ties."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return float("nan"), int(x.size)
    return pearson_correlation(average_ranks(x), average_ranks(y)), int(x.size)


def spearman_permutation_p_two_sided(
    x: np.ndarray, y: np.ndarray, *, rng: np.random.Generator, n_permutations: int
) -> float | None:
    """Two-sided permutation p-value for Spearman (permute y)."""
    observed, n = spearman_correlation(x, y)
    if n < 3 or not np.isfinite(observed):
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    count = 0
    for _ in range(int(n_permutations)):
        corr, _ = spearman_correlation(x, rng.permutation(y))
        if np.isfinite(corr) and abs(corr) >= abs(observed) - 1e-12:
            count += 1
    return float((count + 1) / (int(n_permutations) + 1))


def finite_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return only finite values from a column as float64 array."""
    values = np.asarray(df[column], dtype=np.float64)
    return values[np.isfinite(values)]
