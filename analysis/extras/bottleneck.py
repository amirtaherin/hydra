# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""bottleneck analysis — empirical "decode is memory-bound" check.

Two scatter figures:

  1. Decode tokens/s vs mean EMC load during decode. If decode is memory-
     bound, EMC should be near saturation regardless of how fast the GPU is.
     One point per (platform, dtype, model) cell; color = platform; marker
     = dtype.

  2. Decode tokens/s vs decode-phase mean GPU power. Tests "more compute
     -> more throughput": if the curve flattens, we're memory-bound; if it
     keeps rising, we're compute-bound.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


_DTYPE_MARKERS = {
    "torch.bfloat16": "o",
    "F16":            "s",
    "Q8_0":           "^",
    "Q6_K":           "D",
    "Q4_K_M":         "v",
}


def _scatter(df: pd.DataFrame, x_fn, y_fn, *,
             xlabel: str, ylabel: str, save: Path,
             ylim: tuple[float, float] | None = None,
             xlim: tuple[float, float] | None = None,
             title: str | None = None) -> Path:
    platforms = style.PLATFORM_ORDER
    fig, ax = plt.subplots(figsize=style.FIG_SIZE_2COL)
    used_dtypes = []
    for plat in platforms:
        for dt in _DTYPE_MARKERS:
            sub = df[(df["platform"] == plat) & (df["dtype"] == dt)]
            if sub.empty:
                continue
            xs, ys = [], []
            for m in style.MODEL_ORDER:
                cell = sub[sub["model"] == m]
                if cell.empty:
                    continue
                xs.append(x_fn(cell))
                ys.append(y_fn(cell))
            if not xs:
                continue
            ax.scatter(xs, ys, marker=_DTYPE_MARKERS[dt],
                       color=style.PLATFORM_COLORS[plat],
                       edgecolor="black", linewidth=0.4, alpha=0.8, s=42)
            if dt not in used_dtypes:
                used_dtypes.append(dt)

    plat_handles = [
        mlines.Line2D([], [], marker="s", linestyle="",
                      markerfacecolor=style.PLATFORM_COLORS[p],
                      markeredgecolor="black", markersize=8,
                      label=p.capitalize())
        for p in platforms
    ]
    dt_handles = [
        mlines.Line2D([], [], marker=_DTYPE_MARKERS[d], linestyle="",
                      markerfacecolor="grey", markeredgecolor="black",
                      markersize=6, label=style.DTYPE_DISPLAY.get(d, d))
        for d in used_dtypes
    ]
    leg1 = ax.legend(handles=plat_handles, loc="upper left",
                     fontsize=7, frameon=True, title="Platform")
    ax.add_artist(leg1)
    ax.legend(handles=dt_handles, loc="lower right",
              fontsize=7, frameon=True, title="Dtype", ncol=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    else: ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def decode_vs_emc(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> Path:
    return _scatter(
        df,
        x_fn=lambda c: c["generation_tokens_per_sec"].mean(),
        y_fn=lambda c: c["emc_load_decode_mean"].mean(),
        xlabel="Decode throughput (tokens/s)",
        ylabel=r"EMC load during decode (\%)",
        title="Memory-bandwidth utilization vs throughput",
        save=out_root / "bottleneck_decode_vs_emc.pdf",
        ylim=(0, 100),
    )


def decode_vs_gpu_power(df: pd.DataFrame, out_root: Path,
                        platforms: list[str] | None = None) -> Path:
    return _scatter(
        df,
        x_fn=lambda c: c["generation_tokens_per_sec"].mean(),
        y_fn=lambda c: c["gpu_power_mw_decode_mean"].mean() / 1000.0,
        xlabel="Decode throughput (tokens/s)",
        ylabel="GPU power during decode (W)",
        title="Compute power vs throughput",
        save=out_root / "bottleneck_decode_vs_gpu_power.pdf",
    )


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        decode_vs_emc(df, out_root, platforms=platforms),
        decode_vs_gpu_power(df, out_root, platforms=platforms),
    ]
