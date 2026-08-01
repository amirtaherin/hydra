# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""HuggingFace vs llama.cpp at matched precision.

For each platform, paired bars per model:
  - HF bf16
  - llama.cpp F16

These are the closest-to-equivalent precision points across the two backends
(both 16-bit float). Differences in throughput here measure software-stack
overhead (Python+PyTorch dispatch and tokenizer wall-clock vs C++/CUDA only)
plus implementation differences.

Same shape, second figure: TTFT comparison. Often the C++ stack has lower
TTFT because it skips a tokenization layer's Python overhead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar


def _gather(df: pd.DataFrame, platform: str, value_col: str) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    sub = df[df["platform"] == platform]
    means: dict[str, list[float]] = {"HF bf16": [], "llama.cpp F16": []}
    stds:  dict[str, list[float]] = {"HF bf16": [], "llama.cpp F16": []}
    models: list[str] = []
    for model in style.MODEL_ORDER:
        hf = sub[(sub["model"] == model) & (sub["backend"] == "hf")
                 & (sub["dtype"] == "torch.bfloat16")]
        ll = sub[(sub["model"] == model) & (sub["backend"] == "llamacpp")
                 & (sub["dtype"] == "F16")]
        if hf.empty and ll.empty:
            continue
        models.append(model)
        for label, group in (("HF bf16", hf), ("llama.cpp F16", ll)):
            v = group[value_col] if not group.empty else pd.Series([np.nan])
            means[label].append(v.mean())
            stds[label].append(v.std())
    return models, means, stds


def _render_metric(df: pd.DataFrame, out_root: Path,
                   *, platforms: list[str], value_col: str,
                   ylabel: str, name: str, scale: float = 1.0) -> list[Path]:
    written = []
    for plat in platforms:
        models, means, stds = _gather(df, plat, value_col)
        if not models:
            continue
        for k in means:
            means[k] = [v * scale for v in means[k]]
            stds[k]  = [v * scale for v in stds[k]]
        save = out_root / f"backend_compare_{name}_{plat}.pdf"
        grouped_bar(
            model_names=models,
            means=means, stds=stds,
            colors=[style.BACKEND_COLORS["hf"], style.BACKEND_COLORS["llamacpp"]],
            legend_labels=list(means.keys()),
            xlabel="Model", ylabel=ylabel,
            figsize=style.FIG_SIZE_2COL,
            save_path=save,
            title=f"HF bf16 vs llama.cpp F16 - {plat.capitalize()}",
        )
        written.append(save)
    return written


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    platforms = platforms or style.PLATFORM_ORDER
    written: list[Path] = []
    written += _render_metric(df, out_root, platforms=platforms,
                              value_col="generation_tokens_per_sec",
                              ylabel="Throughput (tokens/s)",
                              name="throughput")
    written += _render_metric(df, out_root, platforms=platforms,
                              value_col="time_to_first_token",
                              ylabel="TTFT (ms)",
                              name="ttft", scale=1000)
    return written
