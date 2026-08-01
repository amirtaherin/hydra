# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""memory pressure and Xavier F16 ≥7B failure annotation.

Two figures:

  1. RAM-used grid: rows = models, cols = (platform, dtype). Cells colored
     by mean RAM used (GB) during prompt; missing cells (Xavier F16 ≥7B)
     are explicitly hatched and labelled "OOM" so the NVMAP/cuBLAS finding
     is visible.

  2. RAM bars per platform x dtype: shows RAM as a function of model size,
     one curve per dtype, with OOM markers where data is missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


_DTYPE_ALL = ["torch.bfloat16", "F16", "Q8_0", "Q6_K", "Q4_K_M"]


def _all_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = []
    for plat in style.PLATFORM_ORDER:
        sub = df[df["platform"] == plat]
        for dt in _DTYPE_ALL:
            if dt in sub["dtype"].unique():
                cols.append((plat, dt))
    return cols


def ram_grid(df: pd.DataFrame, out_root: Path,
             platforms: list[str] | None = None) -> Path:
    rows = [m for m in style.MODEL_ORDER if not df[df["model"] == m].empty]
    cols = _all_columns(df)

    grid = np.full((len(rows), len(cols)), np.nan)
    for j, (plat, dt) in enumerate(cols):
        for i, model in enumerate(rows):
            cell = df[(df["platform"] == plat) & (df["dtype"] == dt)
                      & (df["model"] == model)]
            if not cell.empty:
                grid[i, j] = cell["ram_used_mb_prompt_mean"].mean() / 1024.0  # GB

    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 2.5,
                                    0.32 * len(rows) + 1.5))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("plasma")
    cmap.set_bad(color="#dcdcdc")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "OOM", ha="center", va="center",
                        color="red", fontsize=6, weight="bold")
                continue
            normalized = (v - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-9)
            color = "white" if normalized > 0.5 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=6.5)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [f"{p[:3].capitalize()}\n{style.DTYPE_DISPLAY.get(d, d)}"
         for p, d in cols], fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([style.MODEL_DISPLAY.get(m, m) for m in rows],
                       fontsize=7)
    ax.set_title("Mean RAM used during prompt (GB)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("GB", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    save = out_root / "memory_pressure_grid.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def ram_per_platform(df: pd.DataFrame, out_root: Path,
                     platforms: list[str] | None = None) -> Path:
    """Per-platform: RAM (GB) vs model size, one line per dtype, with
    explicit gap markers for OOM cells (Xavier F16 ≥7B).
    """
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
            ys = []
            xs = []
            ooms = []
            for i, m in enumerate(models):
                cell = sub[(sub["dtype"] == dt) & (sub["model"] == m)]
                if cell.empty:
                    if dt in sub["dtype"].unique():  # was attempted but missing
                        ooms.append(i)
                    continue
                xs.append(i)
                ys.append(cell["ram_used_mb_prompt_mean"].mean() / 1024.0)
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=4,
                    color=style.DTYPE_COLORS.get(dt, "grey"),
                    label=style.DTYPE_DISPLAY.get(dt, dt),
                    linewidth=1.0)
            for i in ooms:
                ax.scatter([i], [max(ys) * 1.05 if ys else 1],
                           marker="x", color="red", s=40, zorder=5)
        ax.set_title(plat.capitalize(), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([style.MODEL_DISPLAY.get(m, m) for m in models],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        legend_handles = ax.get_legend_handles_labels()
    axs[0].set_ylabel("Mean RAM used (GB)")
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=len(_DTYPE_ALL),
                   frameon=False, fontsize=7)
    save = out_root / "memory_pressure_per_platform.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        ram_grid(df, out_root, platforms=platforms),
        ram_per_platform(df, out_root, platforms=platforms),
    ]
