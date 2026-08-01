# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Throughput plot (single-config plot family).

Generation tokens per second, computed per prompt as
`output_length / total_generation_time` (the same formula the corrected
profiler now writes as `generation_tokens_per_sec`). Rendered as a
single-metric bar per model, no legend, 45 deg model labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar, output_path_for


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
    ]


def throughput(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
) -> Path:
    sub = _select(df, platform, backend, dtype)
    if sub.empty:
        raise ValueError(f"No data for {platform}/{backend}/{dtype}")

    means: list[float] = []
    stds:  list[float] = []
    models: list[str] = []
    for model in style.MODEL_ORDER:
        rows = sub[sub["model"] == model]
        if rows.empty:
            continue
        # Use the per-prompt corrected column when present, else compute it.
        if "generation_tokens_per_sec" in rows.columns and rows["generation_tokens_per_sec"].notna().any():
            tps = rows["generation_tokens_per_sec"]
        else:
            tps = rows["output_length"] / rows["total_generation_time"]
        means.append(tps.mean())
        stds.append(tps.std())
        models.append(model)

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="throughput")
    return grouped_bar(
        model_names=models,
        means={"throughput": means},
        stds={"throughput": stds},
        colors=[style.PLATFORM_COLORS[platform]],
        legend_labels=["Throughput"],
        xlabel="Model",
        ylabel="Mean throughput (tokens/s)",
        figsize=style.FIG_SIZE_1COL,
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
            written.append(throughput(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written
