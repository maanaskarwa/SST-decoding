from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_erp_summary(
    times_s: np.ndarray,
    go_curve: np.ndarray,
    stop_curve: np.ndarray,
    diff_curve: np.ndarray,
    out_path: Path,
    title: str,
    p3_window: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 6.5), sharex=True, height_ratios=[2.0, 1.2]
    )

    axes[0].plot(times_s, go_curve, label="go", linewidth=2)
    axes[0].plot(times_s, stop_curve, label="stop", linewidth=2)
    axes[0].axvspan(
        p3_window[0], p3_window[1], color="tab:orange", alpha=0.12, label="P3 window"
    )
    axes[0].axvline(0.0, color="k", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(times_s, diff_curve, color="tab:red", linewidth=2, label="stop - go")
    axes[1].axhline(0.0, color="k", linestyle=":", linewidth=1)
    axes[1].axvspan(p3_window[0], p3_window[1], color="tab:orange", alpha=0.12)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Difference")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_relevance_curves(
    times_s: np.ndarray,
    curves: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    ylabel: str,
    p3_window: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for label, values in curves.items():
        ax.plot(times_s, values, linewidth=2, label=label)

    ax.axvspan(
        p3_window[0], p3_window[1], color="tab:orange", alpha=0.12, label="P3 window"
    )
    ax.axvline(0.0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_channel_time_heatmap(
    values: np.ndarray,
    times_s: np.ndarray,
    channel_names: list[str],
    out_path: Path,
    title: str,
    p3_window: tuple[float, float],
    cmap: str = "magma",
    symmetric: bool = False,
    colorbar_label: str = "Value",
) -> None:
    vmin = None
    vmax = None
    if symmetric:
        vmax = float(np.max(np.abs(values)))
        vmin = -vmax

    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="lower",
        extent=(float(times_s[0]), float(times_s[-1]), 0, values.shape[0]),
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvspan(p3_window[0], p3_window[1], color="cyan", alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")

    tick_idx = np.arange(len(channel_names))
    ax.set_yticks(tick_idx + 0.5)
    ax.set_yticklabels(channel_names, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scalar_history(
    steps: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(steps, values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
