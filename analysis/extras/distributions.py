# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""per-prompt distributions — predictability story.

For each platform, violin + scatter (per-prompt) of mean inter-token latency
across all 541 prompts, faceted by model. Tail latency is the predictability
story for real-time edge use.

Implementation: keep this lightweight. One violin column per dtype, one
small-multiple panel per platform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


def itl_violin(df: pd.DataFrame, out_root: Path,
               platforms: list[str] | None = None,
               focal_model: str = "Llama-3.2-3B") -> Path:
    """Violins of per-prompt mean ITL for one focal model, across (platform x dtype)."""
    platforms = platforms or style.PLATFORM_ORDER
    sub = df[df["model"] == focal_model]
    if sub.empty:
        raise ValueError(f"Focal model {focal_model} missing.")

    cols: list[tuple[str, str]] = []
    for plat in platforms:
        for dt in style.DTYPE_ORDER:
            if not sub[(sub["platform"] == plat) & (sub["dtype"] == dt)].empty:
                cols.append((plat, dt))

    fig, ax = plt.subplots(figsize=style.FIG_SIZE_2COL_TALL)
    positions = np.arange(1, len(cols) + 1)
    data = []
    colors = []
    for plat, dt in cols:
        cell = sub[(sub["platform"] == plat) & (sub["dtype"] == dt)]
        # Convert s -> ms.
        data.append(cell["mean_inter_token_latency"].dropna().values * 1000)
        colors.append(style.PLATFORM_COLORS[plat])

    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=True, widths=0.85)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.7)
        body.set_linewidth(0.5)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{p[:3].capitalize()}\n{style.DTYPE_DISPLAY.get(dt, dt)}"
         for p, dt in cols], fontsize=7)
    ax.set_ylabel("Mean inter-token latency (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"ITL distribution - {style.MODEL_DISPLAY.get(focal_model, focal_model)}",
                 fontsize=9)

    save = out_root / f"distributions_itl_{focal_model.replace('-', '_')}.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [itl_violin(df, out_root, platforms=platforms)]
