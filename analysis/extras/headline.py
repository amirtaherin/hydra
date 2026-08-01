# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""throughput grid headline figure.

Heatmap of mean decode throughput (tokens/s) across all (platform, dtype)
configurations, rows = models, cols = (platform, dtype). Designed to be the
one-figure paper headline that tells the cross-platform / cross-quantization
story at a glance.

Cell value = mean of `generation_tokens_per_sec`. Empty cells (e.g., the 5
Xavier F16 ≥7B combos) are left blank with a hatched marker.

Companion figure: per-cell TTFT heatmap (same axes), in ms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


def _make_grid(df: pd.DataFrame, value_col: str, scale: float = 1.0):
    """Build the 2D grid: rows=model, cols=(platform, dtype)."""
    columns: list[tuple[str, str]] = []
    for plat in style.PLATFORM_ORDER:
        if "torch.bfloat16" in df[df["platform"] == plat]["dtype"].unique():
            columns.append((plat, "torch.bfloat16"))
        for dt in style.DTYPE_ORDER:
            if dt == "torch.bfloat16":
                continue
            if dt in df[df["platform"] == plat]["dtype"].unique():
                columns.append((plat, dt))

    rows = [m for m in style.MODEL_ORDER
            if not df[df["model"] == m].empty]
    grid = np.full((len(rows), len(columns)), np.nan)
    for j, (plat, dt) in enumerate(columns):
        for i, model in enumerate(rows):
            cell = df[(df["platform"] == plat) & (df["dtype"] == dt)
                      & (df["model"] == model)]
            if not cell.empty:
                grid[i, j] = cell[value_col].mean() * scale
    return rows, columns, grid


def _draw_heatmap(rows, columns, grid, *, title: str, cbar_label: str,
                  cmap_name: str, fmt: str, save: Path):
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 2.5,
                                    0.32 * len(rows) + 1.5))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad(color="#dcdcdc")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    # Cell text annotations.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "x", ha="center", va="center",
                        color="grey", fontsize=7)
                continue
            # Choose text color by relative cell brightness.
            normalized = (v - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-9)
            color = "white" if normalized > 0.55 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color, fontsize=6.5)

    ax.set_xticks(range(len(columns)))
    col_labels = [
        f"{plat[:3].capitalize()}\n{style.DTYPE_DISPLAY.get(dt, dt)}"
        for plat, dt in columns
    ]
    ax.set_xticklabels(col_labels, fontsize=7, rotation=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([style.MODEL_DISPLAY.get(m, m) for m in rows],
                       fontsize=7)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)


def headline_throughput(df: pd.DataFrame, out_root: Path,
                        platforms: list[str] | None = None) -> Path:
    rows, columns, grid = _make_grid(df, "generation_tokens_per_sec")
    save = out_root / "headline_throughput.pdf"
    _draw_heatmap(rows, columns, grid,
                  title="Decode throughput (tokens/s)",
                  cbar_label="tokens/s",
                  cmap_name="viridis", fmt="{:.1f}",
                  save=save)
    return save


def headline_ttft(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> Path:
    rows, columns, grid = _make_grid(df, "time_to_first_token", scale=1000)
    save = out_root / "headline_ttft.pdf"
    _draw_heatmap(rows, columns, grid,
                  title="Time to first token (ms)",
                  cbar_label="TTFT (ms)",
                  cmap_name="viridis_r", fmt="{:.0f}",
                  save=save)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        headline_throughput(df, out_root, platforms=platforms),
        headline_ttft(df, out_root, platforms=platforms),
    ]
