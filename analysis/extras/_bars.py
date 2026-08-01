# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Shared grouped-bar primitive used by the extras plot families.

This is the legacy `latency_analysis_*` / `cpu_analysis` / `gpu_analysis_barplots`
visual treatment, factored out so every Phase 1 plot looks consistent:

  - grouped bars per model, one bar per metric
  - error bars (yerr, capsize=5)
  - rotated value labels (90 deg), white inside the bar above a 0.4*max
    threshold, black above the bar+error otherwise
  - black bar edges, alpha 0.8
  - legend upper-left, ncol = len(metrics)
  - xticks at group centers, 15 deg rotation, centered alignment
  - groups separated by `group_spacing=1.5`

This
helper consolidates the formula so individual plot modules become tiny.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis import style


def grouped_bar(
    *,
    model_names: Sequence[str],            # already in display order
    means: dict[str, list[float]],          # metric_label -> per-model values
    stds: dict[str, list[float]] | None,    # same keys; None for no error bars
    colors: Sequence[str],                  # one per metric, in order
    legend_labels: Sequence[str],           # one per metric, in order
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float],
    save_path: Path,
    title: str | None = None,
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    label_threshold_frac: float = 0.4,
    label_pad_frac: float = 0.05,
    alpha: float = 0.8,
    group_spacing: float = 1.5,
    ylim: tuple[float, float] | None = None,
    non_negative: bool = True,
) -> Path:
    """Render one grouped-bar figure and write it to `save_path`.

    Single-metric mode (when `len(means) == 1`):
      - the legend is suppressed (no need to identify a single bar)
      - x-tick labels are rotated 45 deg (right-anchored) since they have
        more vertical room without a multi-bar group occupying it
    """
    metrics = list(means.keys())
    num_metrics = len(metrics)
    num_models = len(model_names)
    is_single_group = num_metrics == 1

    x = np.arange(num_models) * group_spacing
    width = 0.8 / num_metrics
    offset = width * (num_metrics - 1) / 2

    fig, ax = plt.subplots(figsize=figsize)

    # Compute global max for label-color threshold and padding.
    all_vals = [v for vals in means.values() for v in vals]
    global_max = max(all_vals) if all_vals else 1.0
    threshold = label_threshold_frac * global_max
    pad = label_pad_frac * global_max

    for i, metric in enumerate(metrics):
        positions = x - offset + i * width
        values = means[metric]
        errors = stds[metric] if stds is not None else None

        # For inherently non-negative quantities (latency, frequency, power,
        # temperature, etc.) clip the lower error bar so it cannot extend
        # below zero. Otherwise the whisker visually shows "negative time"
        # when the bar is small and the std is large.
        if errors is not None and non_negative:
            arr_err = np.array(errors, dtype=float)
            arr_val = np.array(values, dtype=float)
            lower_err = np.minimum(arr_err, np.maximum(arr_val, 0.0))
            yerr_arg = np.array([lower_err, arr_err])
        else:
            yerr_arg = errors

        ax.bar(
            positions, values, width,
            yerr=yerr_arg,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.6,
            alpha=alpha,
            label=legend_labels[i],
            capsize=2,
            error_kw={"elinewidth": 0.4, "ecolor": "black"},
        )

        if annotate:
            for j, (pos, val) in enumerate(zip(positions, values)):
                err = (errors[j] if errors is not None and not np.isnan(errors[j])
                       else 0.0)
                if val > threshold:
                    # White text inside bar near base
                    ax.text(pos, pad, annotate_fmt.format(val),
                            ha="center", va="bottom", rotation=90,
                            color="white", fontsize=7)
                else:
                    # Black text above the bar + error
                    ax.text(pos, val + err + pad, annotate_fmt.format(val),
                            ha="center", va="bottom", rotation=90,
                            color="black", fontsize=7)

    display_names = [style.MODEL_DISPLAY.get(m, m) for m in model_names]
    ax.set_xticks(x)
    if is_single_group:
        ax.set_xticklabels(display_names, rotation=45, ha="right")
    else:
        ax.set_xticklabels(display_names, rotation=15, ha="center")
    ax.set_xlim(x[0] - group_spacing / 2, x[-1] + group_spacing / 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif non_negative:
        # Lock the lower y-limit at 0 so visual artefacts can't push the axis
        # below zero (matches 2025 paper plot behaviour).
        ax.set_ylim(bottom=0)

    if not is_single_group:
        ax.legend(loc="upper left", ncol=num_metrics, frameon=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def output_path_for(
    out_root: Path,
    *,
    platform: str,
    backend: str,
    dtype: str,
    family: str,
) -> Path:
    """Build `<out_root>/<platform>/<backend>/<dtype>/<family>.pdf`.

    Standard layout for single-config (Phase 1) plots. Cross-config (Phase 2)
    plots will use a different helper.
    """
    return out_root / platform / backend / dtype / f"{family}.pdf"
