# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""GPU plots (single-config plot family).

Per-model, per-GPC bar grid with two stacked panels (load + frequency), one
column per model. Each model column shows one bar per GPC. Error bars use
clipped lower-half so the whisker cannot dip below zero.

Per-platform GPC counts and label sources:
  Xavier (1, Volta) — `gpu1_*`
  Orin   (2, Ampere) — `gpc1_*`, `gpc2_*`
  Thor   (3, Blackwell) — `gpc1_freq`, `gpc2_freq`, `gpc3_freq` for freq.
         Load: NVML `gpu_util` (single channel; tegrastats GR3D load is NaN
         on JP7). On Thor the load panel collapses to a single bar per model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style
from analysis.extras._bars import output_path_for


_GPC_LOAD_COLS: dict[str, list[tuple[str, str]]] = {
    "xavier": [("gpu1_load_prompt_mean",   "GR3D")],
    "orin":   [("gpc1_load_prompt_mean",   "GPC1"),
               ("gpc2_load_prompt_mean",   "GPC2")],
    # Thor: tegrastats does not expose per-GPC load on JP7. NVML reports a
    # single total-GPU utilization number. We replicate the NVML value as 3
    # bars labelled GPC1/2/3 so the load panel matches the freq panel's GPC
    # count visually. The bars are intentionally identical — the underlying
    # data source is one channel.
    "thor":   [("gpu_util_prompt_mean",    "GPC1"),
               ("gpu_util_prompt_mean",    "GPC2"),
               ("gpu_util_prompt_mean",    "GPC3")],
}
_GPC_FREQ_COLS: dict[str, list[tuple[str, str]]] = {
    "xavier": [("gpu1_freq_prompt_mean",   "GR3D")],
    "orin":   [("gpc1_freq_prompt_mean",   "GPC1"),
               ("gpc2_freq_prompt_mean",   "GPC2")],
    "thor":   [("gpc1_freq_prompt_mean",   "GPC1"),
               ("gpc2_freq_prompt_mean",   "GPC2"),
               ("gpc3_freq_prompt_mean",   "GPC3")],
}


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]


def _bar_panel(ax, means, stds, xlabels, color, ylabel, ylim=None,
               bar_width: float = 0.6):
    """Draw one bar panel with clipped lower error bars and centered bars.

    Bars are placed at integer positions 1..N and the x-axis is set to
    [0.5, N + 0.5] so a single bar appears centered with symmetric padding
    rather than touching the panel edges.
    """
    n = len(xlabels)
    means_arr = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
    stds_arr = np.nan_to_num(np.array(stds, dtype=float), nan=0.0)
    lower = np.minimum(stds_arr, np.maximum(means_arr, 0.0))
    yerr = np.array([lower, stds_arr])
    positions = np.arange(1, n + 1)
    ax.bar(
        positions, means_arr,
        width=bar_width,
        yerr=yerr,
        capsize=2,
        color=color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        error_kw={"elinewidth": 0.4, "ecolor": "black"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)


def gpu_barplot(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
    n_models = len(models)
    load_specs = _GPC_LOAD_COLS[platform]
    freq_specs = _GPC_FREQ_COLS[platform]

    figsize = (max(7.0, 0.9 * n_models), style.FIG_SIZE_2COL_TALL[1])
    # `sharey='row'` keeps load and freq y-scales comparable across models.
    # We deliberately do NOT share x across rows: load and freq panels can
    # have different bar counts on Thor (1 NVML load vs 3 GPC freq).
    fig, axs = plt.subplots(2, n_models, figsize=figsize, sharey="row")
    cmap = plt.get_cmap("tab20", n_models)

    for i, model in enumerate(models):
        rows = sub[sub["model"] == model]
        color = cmap(i)

        # Load panel (top row)
        ax_load = axs[0, i] if n_models > 1 else axs[0]
        means_load = [rows[col].mean() for col, _ in load_specs]
        stds_load  = [rows[col].std()  for col, _ in load_specs]
        _bar_panel(ax_load, means_load, stds_load,
                   [lbl for _, lbl in load_specs], color,
                   ylabel=(r"Mean load (\%)" if i == 0 else ""),
                   ylim=(0, 100))
        ax_load.set_title(style.MODEL_DISPLAY.get(model, model), fontsize=8)

        # Freq panel (bottom row, MHz -> GHz)
        ax_freq = axs[1, i] if n_models > 1 else axs[1]
        means_freq = [rows[col].mean() / 1000.0 for col, _ in freq_specs]
        stds_freq  = [rows[col].std()  / 1000.0 for col, _ in freq_specs]
        _bar_panel(ax_freq, means_freq, stds_freq,
                   [lbl for _, lbl in freq_specs], color,
                   ylabel=("Mean freq. (GHz)" if i == 0 else ""))
        ax_freq.set_xlabel("GPU")

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="gpu_barplot")
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    platforms = platforms or style.PLATFORM_ORDER
    written: list[Path] = []
    for plat in platforms:
        sub = df[(df["platform"] == plat) & (df["backend"] == "hf")]
        if sub.empty:
            continue
        for dtype in sub["dtype"].unique():
            written.append(gpu_barplot(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written
