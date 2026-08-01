# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""quantization scaling — `Q4_K_M -> Q6_K -> Q8_0 -> F16`.

For each platform, draw one line per model showing how throughput (decode
tokens/s) responds to quantization level. The x axis is ordered low-bit ->
high-bit so a line that monotonically falls right means "quantization helps"
on that platform. Multi-panel small-multiple, one panel per platform.

Companion figure: same shape but y = mean energy-per-prompt (J) so the
quantization-vs-energy story is visible. Both are llama.cpp-only since HF
data is single-precision (bfloat16) here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


# Quant levels in ascending precision (so the natural reading direction is
# "less aggressive quantization to the right").
_Q_ORDER = ["Q4_K_M", "Q6_K", "Q8_0", "F16"]


def _per_model_per_dtype(
    sub: pd.DataFrame, value_fn,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """For each model present in `sub`, compute `value_fn(rows)` per dtype in
    `_Q_ORDER`. Returns (models_in_order, {model: {dtype: value}}).
    """
    out: dict[str, dict[str, float]] = {}
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows_m = sub[sub["model"] == model]
        if rows_m.empty:
            continue
        models.append(model)
        out[model] = {}
        for dtype in _Q_ORDER:
            rows_d = rows_m[rows_m["dtype"] == dtype]
            out[model][dtype] = value_fn(rows_d) if not rows_d.empty else float("nan")
    return models, out


def _draw_panel(ax, title: str, models: list[str],
                values: dict[str, dict[str, float]],
                ylabel: str):
    cmap = style.MODEL_COLOR_CYCLE
    for i, model in enumerate(models):
        ys = [values[model].get(d, float("nan")) for d in _Q_ORDER]
        ax.plot(_Q_ORDER, ys,
                marker="o", markersize=4,
                color=cmap[i % len(cmap)],
                label=style.MODEL_DISPLAY.get(model, model),
                linewidth=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Quantization")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4)


def quant_throughput(df: pd.DataFrame, out_root: Path,
                     platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    fig, axs = plt.subplots(1, len(platforms), figsize=style.FIG_SIZE_FULL,
                            sharey=False)
    if len(platforms) == 1:
        axs = [axs]
    legend_handles = None
    for ax, plat in zip(axs, platforms):
        sub = df[(df["platform"] == plat) & (df["backend"] == "llamacpp")]
        if sub.empty:
            ax.set_visible(False)
            continue
        models, vals = _per_model_per_dtype(
            sub, lambda r: r["generation_tokens_per_sec"].mean()
        )
        _draw_panel(ax, plat.capitalize(), models, vals,
                    ylabel="Mean throughput (tokens/s)")
        legend_handles = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=5,
                   frameon=False, fontsize=7)
    save = out_root / "quant_throughput.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save)
    plt.close(fig)
    return save


def quant_energy(df: pd.DataFrame, out_root: Path,
                 platforms: list[str] | None = None) -> Path:
    """Energy per prompt = mean(total_power_W) * mean(end_to_end_latency_s)."""
    platforms = platforms or style.PLATFORM_ORDER
    fig, axs = plt.subplots(1, len(platforms), figsize=style.FIG_SIZE_FULL,
                            sharey=False)
    if len(platforms) == 1:
        axs = [axs]
    legend_handles = None
    for ax, plat in zip(axs, platforms):
        sub = df[(df["platform"] == plat) & (df["backend"] == "llamacpp")]
        if sub.empty:
            ax.set_visible(False)
            continue
        def value_fn(r):
            p_w = r["total_power_mw_prompt_mean"].mean() / 1000.0
            t_s = r["end_to_end_latency"].mean()
            return p_w * t_s
        models, vals = _per_model_per_dtype(sub, value_fn)
        _draw_panel(ax, plat.capitalize(), models, vals,
                    ylabel="Mean energy per prompt (J)")
        legend_handles = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.02), ncol=5,
                   frameon=False, fontsize=7)
    save = out_root / "quant_energy.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    return [
        quant_throughput(df, out_root, platforms=platforms),
        quant_energy(df, out_root, platforms=platforms),
    ]
