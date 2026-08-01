# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""efficiency — energy per token and the Pareto frontier.

Two figures:

  1. Tokens per Joule per (platform, model, dtype) — small-multiple, one
     panel per platform, x = model, y = tokens/J. Lines per dtype.
  2. Power vs throughput Pareto scatter — every (platform, dtype, model)
     cell is one point; x = mean throughput (tokens/s), y = mean total
     power (W). Color = platform; marker = dtype. The Pareto frontier
     visualizes which configs dominate the efficiency boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from analysis import style


_DTYPE_ALL = ["torch.bfloat16", "F16", "Q8_0", "Q6_K", "Q4_K_M"]
_DTYPE_MARKERS = {
    "torch.bfloat16": "o",
    "F16":            "s",
    "Q8_0":           "^",
    "Q6_K":           "D",
    "Q4_K_M":         "v",
}


def _per_cell(df: pd.DataFrame, value_fn) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for (plat, dt, model), grp in df.groupby(["platform", "dtype", "model"]):
        out[(plat, dt, model)] = value_fn(grp)
    return out


def tokens_per_joule(df: pd.DataFrame, out_root: Path,
                     platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    fig, axs = plt.subplots(1, len(platforms),
                            figsize=style.FIG_SIZE_FULL, sharey=True)
    if len(platforms) == 1:
        axs = [axs]

    legend_handles = None
    for ax, plat in zip(axs, platforms):
        sub = df[df["platform"] == plat]
        if sub.empty:
            ax.set_visible(False)
            continue
        models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
        x = np.arange(len(models))
        for dt in _DTYPE_ALL:
            sub_dt = sub[sub["dtype"] == dt]
            if sub_dt.empty:
                continue
            ys = []
            for m in models:
                cell = sub_dt[sub_dt["model"] == m]
                if cell.empty:
                    ys.append(np.nan)
                else:
                    tps = cell["generation_tokens_per_sec"].mean()
                    p_w = cell["total_power_mw_prompt_mean"].mean() / 1000.0
                    ys.append(tps / p_w if p_w and not np.isnan(p_w) else np.nan)
            ax.plot(x, ys, marker=_DTYPE_MARKERS.get(dt, "o"), markersize=4,
                    color=style.DTYPE_COLORS.get(dt, "grey"),
                    label=style.DTYPE_DISPLAY.get(dt, dt),
                    linewidth=1.0)
        ax.set_title(plat.capitalize(), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([style.MODEL_DISPLAY.get(m, m) for m in models],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        legend_handles = ax.get_legend_handles_labels()
    axs[0].set_ylabel("Tokens / Joule")
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=len(_DTYPE_ALL), frameon=False, fontsize=7)
    save = out_root / "efficiency_tokens_per_joule.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save)
    plt.close(fig)
    return save


def power_throughput_pareto(df: pd.DataFrame, out_root: Path,
                            platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    fig, ax = plt.subplots(figsize=style.FIG_SIZE_2COL)
    used_dtypes = []
    for plat in platforms:
        for dt in _DTYPE_ALL:
            sub = df[(df["platform"] == plat) & (df["dtype"] == dt)]
            if sub.empty:
                continue
            xs = []
            ys = []
            for m in style.MODEL_ORDER:
                cell = sub[sub["model"] == m]
                if cell.empty:
                    continue
                xs.append(cell["generation_tokens_per_sec"].mean())
                ys.append(cell["total_power_mw_prompt_mean"].mean() / 1000.0)
            if not xs:
                continue
            ax.scatter(xs, ys,
                       marker=_DTYPE_MARKERS.get(dt, "o"),
                       color=style.PLATFORM_COLORS[plat],
                       edgecolor="black", linewidth=0.4,
                       alpha=0.8, s=45)
            if dt not in used_dtypes:
                used_dtypes.append(dt)

    # Custom legend: platform color squares + dtype markers (gray).
    plat_handles = [
        mlines.Line2D([], [], marker="s", linestyle="",
                      markerfacecolor=style.PLATFORM_COLORS[p],
                      markeredgecolor="black", markersize=8,
                      label=p.capitalize())
        for p in platforms
    ]
    dt_handles = [
        mlines.Line2D([], [], marker=_DTYPE_MARKERS.get(d, "o"),
                      linestyle="", markerfacecolor="grey",
                      markeredgecolor="black", markersize=6,
                      label=style.DTYPE_DISPLAY.get(d, d))
        for d in used_dtypes
    ]
    leg1 = ax.legend(handles=plat_handles, loc="upper left",
                     fontsize=7, frameon=True, title="Platform")
    ax.add_artist(leg1)
    ax.legend(handles=dt_handles, loc="lower right",
              fontsize=7, frameon=True, title="Dtype", ncol=2)

    ax.set_xlabel("Mean throughput (tokens/s)")
    ax.set_ylabel("Mean total power (W)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    save = out_root / "efficiency_pareto.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        tokens_per_joule(df, out_root, platforms=platforms),
        power_throughput_pareto(df, out_root, platforms=platforms),
    ]
