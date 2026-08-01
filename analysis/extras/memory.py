# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Memory plot (single-config plot family).

Bubble scatter (one bubble per model):
  x = mean RAM used during prompt (GB)
  y = mean memory-controller (EMC) load (%)
  bubble size = mean end-to-end latency

Visual treatment matches the 2025 paper figure:
  - bubbles in the platform color, white edge
  - jittered x/y to reduce overlap
  - model labels placed via adjustText to avoid collision
  - 2-stddev confidence ellipse (red dashed) over the cloud
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd
from adjustText import adjust_text

from analysis import style
from analysis.extras._bars import output_path_for


def _bubble_size_legend(ax, size_values, scaled_sizes,
                        *, label: str, color: str,
                        loc: str = "lower left") -> None:
    """Custom bubble-size legend with configurable placement.

    Bubble-size legend helper; exposes `loc`
    (the legacy helper hardcoded `"upper left"`).
    """
    size_vals = np.percentile(size_values, [25, 50, 75])
    legend_sizes = [max(scaled_sizes) * (v / max(size_values)) for v in size_vals]
    scatter_points = [
        plt.scatter([], [], s=s, color=color, alpha=0.5, edgecolors="w")
        for s in legend_sizes
    ]
    ax.legend(
        scatter_points,
        [f"{val:.1f} s" for val in size_vals],
        title=label,
        loc=loc,
        scatterpoints=1,
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]


_LEGEND_LOC: dict[str, str] = {
    "xavier": "lower left",
    "orin":   "upper left",
    "thor":   "lower left",
}


def memory_bubble(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
    jitter_strength: float = 2.0,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    x_vals: list[float] = []
    y_vals: list[float] = []
    sizes:  list[float] = []
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        models.append(model)
        x_vals.append(rows["ram_used_mb_prompt_mean"].mean() / 1024.0)  # MB -> GB
        y_vals.append(rows["emc_load_prompt_mean"].mean())
        sizes.append(rows["end_to_end_latency"].mean())

    if not models:
        raise ValueError(f"No models present for {platform}/{backend}/{dtype}")

    max_size = max(sizes) if sizes else 1.0
    scaled_sizes = [500 * (s / max_size) for s in sizes]

    rng = np.random.default_rng(seed=42)
    jx = rng.normal(0, jitter_strength, len(x_vals))
    jy = rng.normal(0, jitter_strength, len(y_vals))
    xj = np.array(x_vals) + jx
    yj = np.array(y_vals) + jy

    fig, ax = plt.subplots(figsize=style.FIG_SIZE_1COL)
    color = style.PLATFORM_COLORS[platform]
    ax.scatter(
        xj, yj,
        s=scaled_sizes,
        alpha=0.8,
        color=color,
        edgecolors="white",
        linewidth=0.5,
    )
    _bubble_size_legend(ax, sizes, scaled_sizes,
                        label="Mean E2E latency",
                        color=color,
                        loc=_LEGEND_LOC.get(platform, "lower left"))

    texts = []
    for x, y, name in zip(xj, yj, models):
        display = style.MODEL_DISPLAY.get(name, name)
        texts.append(ax.text(x, y, display,
                             fontsize=7, ha="center", va="center",
                             color="black"))
    # Push labels apart aggressively so they don't collide with each other or
    # with the bubbles. Larger expand factors and a bit of pull-from-bubble
    # force keep the labels readable when models cluster.
    adjust_text(
        texts, ax=ax,
        x=xj, y=yj,
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
        expand_text=(2.5, 2.5),
        expand_points=(2.5, 2.5),
        expand_objects=(4.0, 4.0),
        force_text=(1.0, 1.0),
        force_points=(0.6, 0.6),
        only_move={"points": "y", "text": "xy"},
    )

    add_confidence_ellipse(xj, yj, ax, n_std=2.0,
                           edgecolor="red", linestyle="--")

    ax.set_xlabel("RAM usage (GB)")
    ax.set_ylabel(r"EMC load (\%)")
    # X-axis at [0, 30] GB across all platforms so the cross-platform RAM
    # comparison is visually consistent (Xavier/Orin both have 32 GB unified
    # memory; Thor has 128 GB but inference fits in <30 GB for these models).
    ax.set_xlim(0, 30)
    # Y-axis at [0, 120] so high EMC load runs aren't visually clipped at
    # the top edge.
    ax.set_ylim(0, 120)

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="memory_bubble")
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
            written.append(memory_bubble(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written


def add_confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    Add a confidence ellipse to the given Axes based on x and y data.
    n_std: Number of standard deviations to determine the ellipse's radii.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    cov = np.cov(x, y)
    if np.linalg.det(cov) == 0:
        return  # avoid singular matrix errors

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    transform = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transform + ax.transData)
    ax.add_patch(ellipse)
