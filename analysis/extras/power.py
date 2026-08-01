# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Power & energy plots (single-config plot family).

Two grouped bars per model:
  - Power panel:  GPU, CPU, Mem/IO mean power (W)
  - Energy panel: GPU, CPU, Mem/IO mean energy (J) over the full prompt

Cross-platform via canonical `gpu_power_mw`, `cpu_power_mw`, `sys_power_mw`
columns (see `analysis/schema.py` for the per-platform mapping). Note that
Thor's `cpu_power_mw` aggregates CPU+SoC+MSS rails; Xavier/Orin's is CPU only.

Energy per prompt approximated as `mean_power * mean_e2e_latency`. This is
the legacy formula; it underestimates energy slightly because
mean(P*t) >= mean(P)*mean(t) when P and t are positively correlated, but it
matches the 2025 paper's reporting and is fine for cross-platform trends.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar, output_path_for


# Three canonical rails, in plot order.
_POWER_COLS = [
    ("gpu_power_mw_prompt_mean", "GPU"),
    ("cpu_power_mw_prompt_mean", "CPU"),
    ("sys_power_mw_prompt_mean", "Mem/IO"),
]
_COLORS = [
    style.PERSIAN_PALETTE["Lajvard"],
    style.PERSIAN_PALETTE["Yashm"],
    style.PERSIAN_PALETTE["Zaferani"],
]


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]


def _per_model_power(sub: pd.DataFrame) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    """For each model in MODEL_ORDER (if present), compute mean and std of each
    canonical rail in Watts.
    """
    means: dict[str, list[float]] = {label: [] for _, label in _POWER_COLS}
    stds:  dict[str, list[float]] = {label: [] for _, label in _POWER_COLS}
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        models.append(model)
        for col, label in _POWER_COLS:
            v = rows[col]
            means[label].append((v.mean() or 0.0) / 1000.0)  # mW -> W
            stds[label].append((v.std() or 0.0) / 1000.0)
    return models, means, stds


def power_mean(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")
    models, means, stds = _per_model_power(sub)

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="power_mean")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=_COLORS,
        legend_labels=[label for _, label in _POWER_COLS],
        xlabel="Model",
        ylabel="Mean power (W)",
        figsize=style.FIG_SIZE_2COL,
        save_path=save,
    )


def energy_mean(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    """Energy per prompt (J) ~= mean_power_w * mean_e2e_latency_s."""
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    means: dict[str, list[float]] = {label: [] for _, label in _POWER_COLS}
    stds:  dict[str, list[float]] = {label: [] for _, label in _POWER_COLS}
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        models.append(model)
        e2e = rows["end_to_end_latency"]      # seconds
        e2e_mean = e2e.mean()
        for col, label in _POWER_COLS:
            p_w = rows[col] / 1000.0           # W
            energy_J = p_w * e2e               # per-prompt J (rough proxy)
            means[label].append(p_w.mean() * e2e_mean)
            stds[label].append(energy_J.std())

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="energy_mean")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=_COLORS,
        legend_labels=[label for _, label in _POWER_COLS],
        xlabel="Model",
        ylabel="Mean energy per prompt (J)",
        figsize=style.FIG_SIZE_2COL,
        save_path=save,
    )


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    platforms = platforms or style.PLATFORM_ORDER
    written: list[Path] = []
    for plat in platforms:
        sub = df[(df["platform"] == plat) & (df["backend"] == "hf")]
        if sub.empty:
            continue
        for dtype in sub["dtype"].unique():
            written.append(power_mean(df, out_root, platform=plat, backend="hf", dtype=dtype))
            written.append(energy_mean(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written
