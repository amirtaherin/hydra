# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Latency plots (single-config plot family).

Three plot families per (platform, backend, dtype):
  - latency_stages:    grouped bars of {tokenization, prefill, TTFT}
  - latency_e2e:       single bar of end_to_end_latency per model
  - latency_per_token: grouped bars of {token_gen, token_decode, ITL}

Style conventions are in `analysis/extras/_bars.py` and `analysis/style.py`.
Output path: `<out>/<platform>/<backend>/<dtype>/<family>.pdf`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar, output_path_for


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    sub = df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]
    return sub


def _mean_std_per_model(
    sub: pd.DataFrame,
    columns: list[str],
    *,
    scale: float = 1.0,
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    """For each model in `style.MODEL_ORDER` present in `sub`, compute the
    per-prompt mean and std of each requested column. Returns
    (model_names_in_order, means, stds), with values multiplied by `scale`.
    """
    means: dict[str, list[float]] = {c: [] for c in columns}
    stds:  dict[str, list[float]] = {c: [] for c in columns}
    present_models: list[str] = []

    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        present_models.append(model)
        for col in columns:
            means[col].append(rows[col].mean() * scale)
            stds[col].append(rows[col].std() * scale)

    return present_models, means, stds


def _mean_sqrt_var_per_model(
    sub: pd.DataFrame,
    mean_columns: list[str],
    *,
    scale: float = 1.0,
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    """For per-token plots: std comes from sqrt(mean(variance_*)), as in the
    2025 latency_analysis_3. Each column name in `mean_columns` is the
    `mean_*` form; we look up the matching `variance_*` column for std.
    """
    var_columns = [c.replace("mean_", "variance_") for c in mean_columns]
    means: dict[str, list[float]] = {c: [] for c in mean_columns}
    stds:  dict[str, list[float]] = {c: [] for c in mean_columns}
    present_models: list[str] = []

    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        present_models.append(model)
        for mc, vc in zip(mean_columns, var_columns):
            means[mc].append(rows[mc].mean() * scale)
            stds[mc].append(np.sqrt(rows[vc].mean()) * scale)

    return present_models, means, stds


# -----------------------------------------------------------------------------
# 1. Latency stages — grouped bars of {tokenization, prefill, TTFT}
# -----------------------------------------------------------------------------
def latency_stages(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    cols = ["tokenization_time", "prefill_time", "time_to_first_token"]
    models, means, stds = _mean_std_per_model(sub, cols, scale=1000)  # s -> ms

    # Map raw column keys -> color/label.
    colors = [
        style.PERSIAN_PALETTE["Lajvard"],
        style.PERSIAN_PALETTE["Yashm"],
        style.PERSIAN_PALETTE["Zaferani"],
    ]
    legend = ["Tokenization", "Prefill", "TTFT"]

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="latency_stages")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=colors,
        legend_labels=legend,
        xlabel="Model",
        ylabel="Mean latency (ms)",
        figsize=style.FIG_SIZE_2COL,
        save_path=save,
    )


# -----------------------------------------------------------------------------
# 2. End-to-end latency — single-metric bar per model
# -----------------------------------------------------------------------------
def latency_e2e(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    cols = ["end_to_end_latency"]
    models, means, stds = _mean_std_per_model(sub, cols)  # already seconds

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="latency_e2e")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=[style.PLATFORM_COLORS[platform]],
        legend_labels=["End-to-end latency"],
        xlabel="Model",
        ylabel="Mean E2E latency (s)",
        figsize=style.FIG_SIZE_1COL,
        save_path=save,
    )


# -----------------------------------------------------------------------------
# 3. Per-token cost — grouped bars of {token_gen, token_decode, ITL}
# -----------------------------------------------------------------------------
def latency_per_token(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    cols = [
        "mean_token_generation_time",
        "mean_token_decode_time",
        "mean_inter_token_latency",
    ]
    models, means, stds = _mean_sqrt_var_per_model(sub, cols, scale=1000)  # s -> ms

    colors = [
        style.PERSIAN_PALETTE["Lajvard"],
        style.PERSIAN_PALETTE["Mesi"],
        style.PERSIAN_PALETTE["Zereshki"],
    ]
    legend = ["Token generation", "Token decode", "Inter-token latency"]

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="latency_per_token")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=colors,
        legend_labels=legend,
        xlabel="Model",
        ylabel="Mean latency (ms)",
        figsize=style.FIG_SIZE_2COL,
        save_path=save,
    )


# -----------------------------------------------------------------------------
# Driver — render all latency plots for HF on each available platform
# -----------------------------------------------------------------------------
def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    """Render the three latency families for HF/bf16 on every requested platform."""
    platforms = platforms or style.PLATFORM_ORDER
    written: list[Path] = []
    for plat in platforms:
        sub = df[(df["platform"] == plat) & (df["backend"] == "hf")]
        if sub.empty:
            continue
        for dtype in sub["dtype"].unique():
            kw = dict(platform=plat, backend="hf", dtype=dtype)
            written.append(latency_stages(df, out_root, **kw))
            written.append(latency_e2e(df, out_root, **kw))
            written.append(latency_per_token(df, out_root, **kw))
    return written
