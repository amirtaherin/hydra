# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""scaling laws — how throughput, TTFT and energy scale with size.

Two figures:

  1. Log-log throughput vs parameter count, per platform per dtype. The
     slope quantifies "how badly does adding parameters hurt throughput?"
     Steeper-than -1 slope means the platform is memory-bandwidth-limited
     (decode is bandwidth-bound and weights grow linearly).

  2. TTFT vs input prompt length, per platform. Confirms prefill time is
     ~linear in input tokens (prefill is compute-bound).

Parameter counts come from `style.MODEL_PARAMS_B` if defined, else inferred
from `inputs/models/models.json` ordering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


# Approximate parameter counts (B). Used for log-log regression; ordering
# in style.MODEL_ORDER is authoritative.
_PARAMS_B: dict[str, float] = {
    "Llama-3.2-1B":     1.24,
    "Qwen2.5-1.5B":     1.54,
    "granite-3.3-2B":   2.53,
    "gemma-2B":         2.51,
    "Llama-3.2-3B":     3.21,
    "Qwen2.5-3B":       3.09,
    "Phi-3.5-mini":     3.82,
    "Qwen2.5-7B":       7.62,
    "moxin-7B":         7.02,
    "gemma-7B":         8.54,
    "Ministral-8B":     8.02,
    "Llama-3.1-8B":     8.03,
    "granite-3.3-8B":   8.17,
}

_DTYPES = ["torch.bfloat16", "F16", "Q8_0", "Q6_K", "Q4_K_M"]


def throughput_vs_size(df: pd.DataFrame, out_root: Path,
                       platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    fig, axs = plt.subplots(1, len(platforms),
                            figsize=style.FIG_SIZE_FULL, sharex=True, sharey=True)
    if len(platforms) == 1:
        axs = [axs]
    legend_handles = None
    for ax, plat in zip(axs, platforms):
        sub = df[df["platform"] == plat]
        if sub.empty:
            ax.set_visible(False)
            continue
        for dt in _DTYPES:
            sub_dt = sub[sub["dtype"] == dt]
            if sub_dt.empty:
                continue
            xs, ys = [], []
            for m in style.MODEL_ORDER:
                cell = sub_dt[sub_dt["model"] == m]
                if cell.empty or m not in _PARAMS_B:
                    continue
                xs.append(_PARAMS_B[m])
                ys.append(cell["generation_tokens_per_sec"].mean())
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=4,
                    color=style.DTYPE_COLORS.get(dt, "grey"),
                    label=style.DTYPE_DISPLAY.get(dt, dt),
                    linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(plat.capitalize(), fontsize=9)
        ax.set_xlabel("Parameter count (B)")
        ax.grid(True, which="both", alpha=0.3)
        legend_handles = ax.get_legend_handles_labels()
    axs[0].set_ylabel("Throughput (tokens/s)")
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=len(_DTYPES), frameon=False, fontsize=7)
    save = out_root / "scaling_throughput_vs_size.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save)
    plt.close(fig)
    return save


def ttft_vs_input_length(df: pd.DataFrame, out_root: Path,
                         platforms: list[str] | None = None,
                         focal_model: str = "Llama-3.2-3B") -> Path:
    """Per-prompt scatter: x = input tokens, y = TTFT (ms). One panel per platform.
    Multiple dtypes overlaid.
    """
    platforms = platforms or style.PLATFORM_ORDER
    sub = df[df["model"] == focal_model]
    if sub.empty:
        raise ValueError(f"Focal model {focal_model} missing.")

    fig, axs = plt.subplots(1, len(platforms),
                            figsize=style.FIG_SIZE_FULL, sharey=True)
    if len(platforms) == 1:
        axs = [axs]
    legend_handles = None
    for ax, plat in zip(axs, platforms):
        sub_p = sub[sub["platform"] == plat]
        if sub_p.empty:
            ax.set_visible(False)
            continue
        for dt in _DTYPES:
            cell = sub_p[sub_p["dtype"] == dt]
            if cell.empty:
                continue
            ax.scatter(cell["total_number_of_tokens_in_input"],
                       cell["time_to_first_token"] * 1000,
                       s=6, alpha=0.4,
                       color=style.DTYPE_COLORS.get(dt, "grey"),
                       label=style.DTYPE_DISPLAY.get(dt, dt))
        ax.set_xlabel("Input tokens")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(plat.capitalize(), fontsize=9)
        legend_handles = ax.get_legend_handles_labels()
    axs[0].set_ylabel("TTFT (ms)")
    fig.suptitle(f"TTFT vs input length — {style.MODEL_DISPLAY.get(focal_model, focal_model)}",
                 fontsize=10, y=1.0)
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.96),
                   ncol=len(_DTYPES), frameon=False, fontsize=7)
    save = out_root / "scaling_ttft_vs_input_length.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        throughput_vs_size(df, out_root, platforms=platforms),
        ttft_vs_input_length(df, out_root, platforms=platforms),
    ]
