# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""cross-generation scaling.

For a focal model (default: Llama-3.2-3B — present on all platforms with
both backends and all four llama.cpp dtypes), plot how throughput, TTFT,
and energy-per-prompt scale across hardware generations:
  Xavier (Volta, JP5) -> Orin (Ampere, JP6) -> Thor (Blackwell, JP7).

One figure with three panels (one metric each), one line per dtype.
The "Thor regression on HF" finding shows up directly here as the bf16
line dropping at the Thor x position.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


FOCAL_MODEL = "Llama-3.2-3B"

_DTYPES = ["torch.bfloat16", "F16", "Q8_0", "Q6_K", "Q4_K_M"]


def _value_per_platform(df: pd.DataFrame, model: str, dtype: str,
                        value_fn) -> dict[str, float]:
    out: dict[str, float] = {}
    for plat in style.PLATFORM_ORDER:
        cell = df[(df["platform"] == plat) & (df["model"] == model)
                  & (df["dtype"] == dtype)]
        out[plat] = value_fn(cell) if not cell.empty else float("nan")
    return out


def cross_generation(df: pd.DataFrame, out_root: Path,
                     focal_model: str = FOCAL_MODEL,
                     platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER

    metrics = [
        ("Throughput (tokens/s)",
         lambda r: r["generation_tokens_per_sec"].mean(), 1.0),
        ("TTFT (ms)",
         lambda r: r["time_to_first_token"].mean(), 1000.0),
        ("Energy / prompt (J)",
         lambda r: (r["total_power_mw_prompt_mean"].mean() / 1000.0)
                   * r["end_to_end_latency"].mean(),
         1.0),
    ]

    fig, axs = plt.subplots(1, len(metrics), figsize=style.FIG_SIZE_FULL)
    for ax, (label, fn, scale) in zip(axs, metrics):
        for dt in _DTYPES:
            per_plat = _value_per_platform(df, focal_model, dt, fn)
            ys = [per_plat[p] * scale for p in platforms]
            if all(np.isnan(y) for y in ys):
                continue
            ax.plot(platforms, ys, marker="o", markersize=5,
                    color=style.DTYPE_COLORS.get(dt, "grey"),
                    label=style.DTYPE_DISPLAY.get(dt, dt),
                    linewidth=1.2)
        ax.set_xticks(range(len(platforms)))
        ax.set_xticklabels([p.capitalize() for p in platforms], fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Cross-generation scaling — {style.MODEL_DISPLAY.get(focal_model, focal_model)}",
                 fontsize=10, y=1.0)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.96),
               ncol=len(_DTYPES), frameon=False, fontsize=7)
    save = out_root / "cross_generation.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [cross_generation(df, out_root, platforms=platforms)]
