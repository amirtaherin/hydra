# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""CPU plots (single-config plot family).

Per-core boxplots for one focal model, two stacked panels (load + frequency).
Generalized over the platform's actual CPU count (Xavier 8, Orin 12, Thor 14).

The legacy paper picked LLaMA-3.1-8B as the focal model. We do the same so the
figure-set parallels last year's exactly; if a different model is preferred
later, change `FOCAL_MODEL` in this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import style
from analysis.extras._bars import output_path_for
from analysis.schema import N_CPU_CORES


FOCAL_MODEL = "Llama-3.1-8B"


def _select(df: pd.DataFrame, platform: str, backend: str, dtype: str,
            model: str) -> pd.DataFrame:
    return df[
        (df["platform"] == platform)
        & (df["backend"] == backend)
        & (df["dtype"] == dtype)
        & (df["model"] == model)
    ]


def cpu_boxplot(
    df: pd.DataFrame, out_root: Path,
    *, platform: str, backend: str, dtype: str,
    focal_model: str = FOCAL_MODEL,
) -> Path:
    sub = _select(df, platform, backend, dtype, focal_model)
    if sub.empty:
        raise ValueError(f"No data for {focal_model} on {platform}/{backend}/{dtype}")

    n_cpu = N_CPU_CORES[platform]
    cores = [f"C{i}" for i in range(n_cpu)]
    load_data = [
        sub[f"cpu_{i}_load_prompt_mean"].dropna().values for i in range(n_cpu)
    ]
    freq_data = [
        # MHz -> GHz
        sub[f"cpu_{i}_freq_prompt_mean"].dropna().values / 1000.0
        for i in range(n_cpu)
    ]

    fig, axs = plt.subplots(2, 1, figsize=style.FIG_SIZE_2COL_TALL, sharex=True)

    # Load panel
    bp = axs[0].boxplot(
        load_data, patch_artist=True,
        boxprops=dict(facecolor="deepskyblue", alpha=0.8),
        flierprops=dict(marker="o", markerfacecolor="black", markersize=2),
    )
    axs[0].set_ylabel(r"Load (\%)")
    axs[0].set_ylim(0, 100)
    axs[0].set_xticks(range(1, n_cpu + 1))
    axs[0].set_xticklabels(cores)

    # Frequency panel
    bp = axs[1].boxplot(
        freq_data, patch_artist=True,
        boxprops=dict(facecolor="deepskyblue", alpha=0.8),
        flierprops=dict(marker="o", markerfacecolor="black", markersize=2),
    )
    axs[1].set_ylabel("Frequency (GHz)")
    # Fixed CPU-freq scale across all platforms (Xavier ~0.1-2.3 GHz,
    # Orin ~0.1-2.2 GHz, Thor ~1.0-2.6 GHz). 0.5-2.5 GHz captures the
    # in-use range and keeps the panel comparable across platforms.
    axs[1].set_ylim(0.5, 2.5)
    axs[1].set_xticks(range(1, n_cpu + 1))
    axs[1].set_xticklabels(cores)
    axs[1].set_xlabel("CPU core")

    save = output_path_for(out_root, platform=platform, backend=backend,
                           dtype=dtype, family="cpu_boxplot")
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
        sub = df[(df["platform"] == plat) & (df["backend"] == "hf")
                 & (df["model"] == FOCAL_MODEL)]
        if sub.empty:
            continue
        for dtype in sub["dtype"].unique():
            written.append(cpu_boxplot(df, out_root, platform=plat, backend="hf", dtype=dtype))
    return written
