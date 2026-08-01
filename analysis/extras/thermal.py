# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Thermal plot (single-config plot family).

Per-platform sensor zones differ:
  Xavier/Orin:  CPU, GPU, SoC0, SoC1, SoC2  (5 zones)
  Thor:         CPU, GPU, SoC012, SoC345    (4 zones, fewer SoC subgroups)

We render whatever the platform exposes. Cross-platform plots use
the canonical `cpu_temp_c`, `gpu_temp_c`, `temp_max_c` channels instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar, output_path_for


# Per-platform thermal channel sets (column-name root + display label).
_THERMAL_ZONES: dict[str, list[tuple[str, str]]] = {
    "xavier": [
        ("cpu_temp",   "CPU"),
        ("gpu_temp",   "GPU"),
        ("soc_0_temp", "SoC0"),
        ("soc_1_temp", "SoC1"),
        ("soc_2_temp", "SoC2"),
    ],
    "orin": [
        ("cpu_temp",   "CPU"),
        ("gpu_temp",   "GPU"),
        ("soc_0_temp", "SoC0"),
        ("soc_1_temp", "SoC1"),
        ("soc_2_temp", "SoC2"),
    ],
    "thor": [
        ("cpu_temp",     "CPU"),
        ("gpu_temp",     "GPU"),
        ("soc012_temp",  "SoC012"),
        ("soc345_temp",  "SoC345"),
    ],
}

_COLORS = [
    style.PERSIAN_PALETTE["Lajvard"],
    style.PERSIAN_PALETTE["Yashm"],
    style.PERSIAN_PALETTE["Zaferani"],
    style.PERSIAN_PALETTE["Mesi"],
    style.PERSIAN_PALETTE["Zereshki"],
]


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]


def temperature(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    zones = _THERMAL_ZONES.get(platform)
    if zones is None:
        raise ValueError(f"No thermal zone map for platform {platform!r}")

    means: dict[str, list[float]] = {label: [] for _, label in zones}
    stds:  dict[str, list[float]] = {label: [] for _, label in zones}
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        models.append(model)
        for root, label in zones:
            col = f"{root}_prompt_mean"
            v = rows[col] if col in rows.columns else None
            means[label].append(v.mean() if v is not None else float("nan"))
            stds[label].append(v.std() if v is not None else float("nan"))

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="temperature")
    return grouped_bar(
        model_names=models,
        means=means,
        stds=stds,
        colors=_COLORS[: len(zones)],
        legend_labels=[label for _, label in zones],
        xlabel="Model",
        ylabel="Mean temperature (C)",
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
            written.append(temperature(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written
