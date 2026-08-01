# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Performance Analysis figures for §5.1.

Two compact figures targeting two of the paper's headline contributions:

  F2 (`perf_transition_speedup`)
    The compute-to-bandwidth transition (C2). For each model, plot the
    observed throughput speedup Thor/Orin and Thor/Xavier as a function of
    model parameter count. Two horizontal reference lines per comparison
    mark the peak-compute and peak-bandwidth ratios from the hardware
    spec. The data points should descend from the compute reference at
    small sizes toward the bandwidth reference at 7-8B, visualizing the
    transition from compute-bound to memory-bound.

  F3 (`perf_arch_signatures`)
    Architecture-family signatures at matched parameter count (C4).
    Grouped bar of decode tokens/s on Orin HF bf16 for the six models in
    the 7--8.5 B band (Qwen2.5-7B, Moxin-7B, Ministral-8B, LLaMA-3.1-8B,
    Granite-3.3-8B, Gemma-7B). Even though parameter counts cluster
    tightly, decode throughput varies substantially due to FFN width,
    Q/KV head split, and attention variant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


# Per-model parameter count in billions (from Table 2 in the paper).
PARAMS_B: dict[str, float] = {
    "Llama-3.2-1B":     1.24,
    "Qwen2.5-1.5B":     1.54,
    "gemma-2B":         2.51,
    "granite-3.3-2B":   2.53,
    "Qwen2.5-3B":       3.09,
    "Llama-3.2-3B":     3.21,
    "Phi-3.5-mini":     3.82,
    "Qwen2.5-7B":       7.62,
    "Ministral-8B":     8.02,
    "Llama-3.1-8B":     8.03,
    "moxin-7B":         8.11,
    "granite-3.3-8B":   8.17,
    "gemma-7B":         8.54,
}

# Models in the 7--8.5 B band, ordered by parameter count.
MATCHED_BAND_78B: list[str] = [
    "Qwen2.5-7B", "Ministral-8B", "Llama-3.1-8B",
    "moxin-7B", "granite-3.3-8B", "gemma-7B",
]

# Hardware-spec reference ratios derived from Table 1.
#   Bandwidth: Xavier 137 GB/s, Orin 204.8 GB/s, Thor 273 GB/s
#   Compute (Tensor cores * clock as a rough peak-FP16 proxy):
#     Xavier: 64 cores * 1.37 GHz = 88 (unitless proxy)
#     Orin:   64 cores * 1.30 GHz = 83
#     Thor:   96 cores * 1.57 GHz = 151
#   Thor/Orin   compute ~ 1.81, bandwidth ~ 1.33
#   Thor/Xavier compute ~ 1.71, bandwidth ~ 2.00 (Xavier lacks bf16
#       tensor-core support; the realistic compute gap is much larger
#       in practice, but we report the pure spec ratio).
# Empirically the Thor/Orin large-model speedup tracks the bandwidth
# ratio and the small-model speedup tracks the compute ratio.
COMPUTE_RATIO = {"thor_over_orin": 1.81, "thor_over_xavier": 1.71}
BW_RATIO      = {"thor_over_orin": 1.33, "thor_over_xavier": 2.00}


def _select_hf_bf16(df: pd.DataFrame, plat: str) -> pd.DataFrame:
    return df[(df["platform"] == plat) & (df["backend"] == "hf")
              & (df["dtype"] == "torch.bfloat16")]


# -----------------------------------------------------------------------------
# F2: Compute-to-bandwidth transition
# -----------------------------------------------------------------------------
def perf_transition_speedup(df: pd.DataFrame, out_root: Path) -> Path:
    """Speedup vs model size (Thor/Orin and Thor/Xavier) with reference lines.

    Each point is one model's mean generation_tokens_per_sec ratio between
    the two platforms (HF, bfloat16). The dashed reference lines mark the
    peak compute and peak DRAM bandwidth ratios from the hardware spec.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    models = list(PARAMS_B.keys())
    x = np.array([PARAMS_B[m] for m in models])

    def per_model_throughput(plat: str) -> np.ndarray:
        sub = _select_hf_bf16(df, plat)
        out = []
        for m in models:
            v = sub[sub["model"] == m]["generation_tokens_per_sec"].mean()
            out.append(v)
        return np.array(out)

    tput_xavier = per_model_throughput("xavier")
    tput_orin   = per_model_throughput("orin")
    tput_thor   = per_model_throughput("thor")

    speed_thor_orin   = tput_thor / tput_orin
    speed_thor_xavier = tput_thor / tput_xavier

    # Sort by parameter count for line continuity.
    order = np.argsort(x)
    x_sorted = x[order]
    y_to    = speed_thor_orin[order]
    y_tx    = speed_thor_xavier[order]

    # Reference lines (dashed) first, so data points sit on top.
    ax.axhline(COMPUTE_RATIO["thor_over_xavier"],
               color=style.PLATFORM_COLORS["xavier"], linestyle=":",
               linewidth=0.9, alpha=0.7)
    ax.axhline(BW_RATIO["thor_over_xavier"],
               color=style.PLATFORM_COLORS["xavier"], linestyle="--",
               linewidth=0.9, alpha=0.7)
    ax.axhline(COMPUTE_RATIO["thor_over_orin"],
               color=style.PLATFORM_COLORS["orin"], linestyle=":",
               linewidth=0.9, alpha=0.7)
    ax.axhline(BW_RATIO["thor_over_orin"],
               color=style.PLATFORM_COLORS["orin"], linestyle="--",
               linewidth=0.9, alpha=0.7)

    # Data lines.
    ax.plot(x_sorted, y_tx, "o-",
            color=style.PLATFORM_COLORS["xavier"], markersize=4,
            linewidth=1.2, label="Thor / Xavier")
    ax.plot(x_sorted, y_to, "s-",
            color=style.PLATFORM_COLORS["orin"], markersize=4,
            linewidth=1.2, label="Thor / Orin")

    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters)", fontsize=8)
    ax.set_ylabel(r"Throughput speedup ($\times$)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    # Inline labels for the reference lines (left edge of plot).
    xmin = ax.get_xlim()[0]
    ax.text(xmin * 1.02, COMPUTE_RATIO["thor_over_xavier"] + 0.05,
            r"Xavier compute ($\sim$1.7$\times$)", fontsize=6,
            color=style.PLATFORM_COLORS["xavier"], ha="left", va="bottom")
    ax.text(xmin * 1.02, BW_RATIO["thor_over_xavier"] + 0.05,
            r"Xavier bandwidth (2.0$\times$)", fontsize=6,
            color=style.PLATFORM_COLORS["xavier"], ha="left", va="bottom")
    ax.text(xmin * 1.02, COMPUTE_RATIO["thor_over_orin"] + 0.05,
            r"Orin compute ($\sim$1.8$\times$)", fontsize=6,
            color=style.PLATFORM_COLORS["orin"], ha="left", va="bottom")
    ax.text(xmin * 1.02, BW_RATIO["thor_over_orin"] + 0.05,
            r"Orin bandwidth (1.3$\times$)", fontsize=6,
            color=style.PLATFORM_COLORS["orin"], ha="left", va="bottom")

    save = out_root / "perf_transition_speedup.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F3: Architecture-family signatures at matched parameter count
# -----------------------------------------------------------------------------
def perf_arch_signatures(df: pd.DataFrame, out_root: Path) -> Path:
    """Decode throughput at the 7-8 B band on Orin HF bf16.

    All six models cluster tightly in parameter count (7.6-8.5 B); the
    variance in measured throughput is due to architectural differences
    (FFN dim, Q/KV head split, MHA vs GQA).
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    sub = _select_hf_bf16(df, "orin")
    codes, vals = [], []
    for m in MATCHED_BAND_78B:
        codes.append(style.MODEL_CODE.get(m, m))
        vals.append(sub[sub["model"] == m]["generation_tokens_per_sec"].mean())

    x = np.arange(len(codes))
    # Color by family for visual cue.
    family_color = {
        "QW": style.PERSIAN_PALETTE["Lajvard"],
        "MI": style.PERSIAN_PALETTE["Yashm"],
        "LL": style.PERSIAN_PALETTE["Shangarfi"],
        "MO": style.PERSIAN_PALETTE["Arghavani"],
        "GR": style.PERSIAN_PALETTE["Mashi"],
        "GE": style.PERSIAN_PALETTE["Nili"],
    }
    colors = [family_color[c.split("-")[0]] for c in codes]

    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5,
           alpha=0.9, width=0.65)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(codes, fontsize=7.5)
    ax.set_ylabel("Decode tokens/s", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)

    save = out_root / "perf_arch_signatures.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F4: E2E latency grid (2 backends x 3 platforms)
# -----------------------------------------------------------------------------
def perf_e2e_latency_grid(df: pd.DataFrame, out_root: Path) -> Path:
    """2 x 3 grid of mean end-to-end latency.

    Row 1: HuggingFace Transformers, bfloat16, across Xavier / Orin / Thor.
    Row 2: llama.cpp, F16, across the same three platforms.
    x = 13 model codes; y = mean end-to-end latency (s) across 541 prompts.
    Shared x and y axes; missing cells (Xavier llama.cpp F16 7-8B) are
    naturally absent from the bars.
    """
    platforms = style.PLATFORM_ORDER
    rows = [
        ("hf",       "torch.bfloat16", "HF bf16"),
        ("llamacpp", "F16",            "llama.cpp F16"),
    ]
    nrows, ncols = len(rows), len(platforms)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.0, 2.2),
                            sharex=True, sharey=True)

    # Use model order from the spec table.
    models = list(PARAMS_B.keys())
    x = np.arange(len(models))

    # Pass 1: compute global y-axis upper limit across all panels so we
    # can decide per-bar whether the label fits inside (white) or has to
    # go above (black) --- matches the _bars.annotate convention.
    all_extents: list[float] = []
    cell_vals:  dict[tuple[int, int], list[float]] = {}
    cell_stds:  dict[tuple[int, int], list[float]] = {}
    for i, (backend, dtype, _) in enumerate(rows):
        for j, plat in enumerate(platforms):
            sub = df[(df["platform"] == plat) & (df["backend"] == backend)
                     & (df["dtype"] == dtype)]
            vals = [sub[sub["model"] == m]["end_to_end_latency"].mean()
                    for m in models]
            stds = [sub[sub["model"] == m]["end_to_end_latency"].std()
                    for m in models]
            cell_vals[(i, j)] = vals
            cell_stds[(i, j)] = stds
            all_extents.extend(
                (v + (s if pd.notna(s) else 0.0))
                for v, s in zip(vals, stds) if pd.notna(v)
            )
    y_max = max(all_extents) * 1.05 if all_extents else 1.0
    label_threshold = y_max * 0.4

    for i, (backend, dtype, row_label) in enumerate(rows):
        for j, plat in enumerate(platforms):
            ax = axs[i, j]
            vals = cell_vals[(i, j)]
            stds = cell_stds[(i, j)]
            ax.bar(x, vals, yerr=stds, color=style.PLATFORM_COLORS[plat],
                   edgecolor="black", linewidth=0.4, alpha=0.9, width=0.7,
                   error_kw=dict(elinewidth=0.5, capsize=1.2, ecolor="black"))
            # Per-bar annotation (rotated 90 deg).
            # Outside labels go above the upper error bar tip.
            pad = y_max * 0.012
            for xi, v, s in zip(x, vals, stds):
                if pd.isna(v) or v == 0:
                    continue
                s_eff = s if pd.notna(s) else 0.0
                if v >= label_threshold:
                    ax.text(xi, pad, f"{v:.1f}", ha="center", va="bottom",
                            rotation=90, color="white", fontsize=6.5)
                else:
                    ax.text(xi, v + s_eff + pad, f"{v:.1f}",
                            ha="center", va="bottom",
                            rotation=90, color="black", fontsize=6.5)
            ax.set_ylim(0, y_max)
            if i == 0:
                ax.set_title(plat.capitalize(), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{row_label}\nE2E latency\n(s)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)

    # Set x-tick labels once on the bottom row (shared x).
    for j in range(ncols):
        axs[-1, j].set_xticks(x)
        axs[-1, j].set_xticklabels(
            [style.MODEL_CODE.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=7,
        )

    save = out_root / "perf_e2e_latency_grid.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F5: Q4_K_M throughput, grouped bars (3 platforms per model)
# -----------------------------------------------------------------------------
def perf_q4_throughput_grouped(df: pd.DataFrame, out_root: Path) -> Path:
    """Single panel, 13 model groups, 3 platform bars per group.

    llama.cpp Q4_K_M decode throughput (generation tokens/s) on
    Xavier, Orin, and Thor. Color = platform; bars per group adjacent
    so the cross-platform speedup is direct per model.
    """
    platforms = style.PLATFORM_ORDER
    # Single-column width target. Bars are tight (39 in a 3.4 in panel)
    # so labels are sized down accordingly.
    fig, ax = plt.subplots(figsize=(3.4, 1.5))

    models = list(PARAMS_B.keys())
    x = np.arange(len(models))
    width = 0.27

    # Pre-compute all values + stds to set a global ylim for label placement.
    plat_vals: dict[str, list[float]] = {}
    plat_stds: dict[str, list[float]] = {}
    for plat in platforms:
        sub = df[(df["platform"] == plat) & (df["backend"] == "llamacpp")
                 & (df["dtype"] == "Q4_K_M")]
        plat_vals[plat] = [
            sub[sub["model"] == m]["generation_tokens_per_sec"].mean()
            for m in models
        ]
        plat_stds[plat] = [
            sub[sub["model"] == m]["generation_tokens_per_sec"].std()
            for m in models
        ]

    extents = [
        v + (s if pd.notna(s) else 0.0)
        for plat in platforms
        for v, s in zip(plat_vals[plat], plat_stds[plat])
        if pd.notna(v)
    ]
    y_max = max(extents) * 1.10 if extents else 1.0
    label_threshold = y_max * 0.4
    pad = y_max * 0.012

    for k, plat in enumerate(platforms):
        offset = (k - 1) * width
        vals = plat_vals[plat]
        stds = plat_stds[plat]
        ax.bar(x + offset, vals, width, yerr=stds,
               color=style.PLATFORM_COLORS[plat],
               edgecolor="black", linewidth=0.3, alpha=0.9,
               label=plat.capitalize(),
               error_kw=dict(elinewidth=0.4, capsize=0.8, ecolor="black"))
        for xi, v, s in zip(x, vals, stds):
            if pd.isna(v) or v == 0:
                continue
            s_eff = s if pd.notna(s) else 0.0
            if v >= label_threshold:
                ax.text(xi + offset, pad, f"{v:.0f}",
                        ha="center", va="bottom", rotation=90,
                        color="white", fontsize=4.5)
            else:
                ax.text(xi + offset, v + s_eff + pad, f"{v:.0f}",
                        ha="center", va="bottom", rotation=90,
                        color="black", fontsize=4.5)

    ax.set_xticks(x)
    ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in models],
                       rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Decode tok./s\n at Q4_K_M", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, y_max)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")

    save = out_root / "perf_q4_throughput_grouped.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F6: Phase-breakdown observability demo (Thor, llama.cpp Q8_0)
# -----------------------------------------------------------------------------
def perf_phase_breakdown_thor_q8(df: pd.DataFrame, out_root: Path) -> Path:
    """Two-row phase-breakdown figure for Thor / llama.cpp Q8_0.

    Row 1: latency stages per model --- tokenization, prefill, TTFT (ms).
    Row 2: per-token costs --- generation, decode, inter-token latency (ms).
    Shared x (models). Independent y per row because the two metric
    families differ by an order of magnitude.
    """
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True,
                            sharey=True)

    sub = df[(df["platform"] == "thor") & (df["backend"] == "llamacpp")
             & (df["dtype"] == "Q8_0")]
    models = [m for m in PARAMS_B if not sub[sub["model"] == m].empty]
    x = np.arange(len(models))

    # ----- Row 1: latency stages -----
    ax = axs[0]
    width = 0.27
    stages = [
        ("tokenization_time",  "Tokenization", style.PERSIAN_PALETTE["Yashm"]),
        ("prefill_time",       "Prefill",      style.PERSIAN_PALETTE["Lajvard"]),
        ("time_to_first_token","TTFT",         style.PERSIAN_PALETTE["Shangarfi"]),
    ]
    for k, (col, lbl, color) in enumerate(stages):
        vals = [sub[sub["model"] == m][col].mean() * 1000.0 for m in models]
        offset = (k - 1) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.9, label=lbl)
    ax.set_ylabel("Latency stages (ms)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")

    # ----- Row 2: per-token costs -----
    ax = axs[1]
    per_token = [
        ("mean_token_generation_time", "Generation", style.PERSIAN_PALETTE["Yashm"]),
        ("mean_token_decode_time",     "Decode",     style.PERSIAN_PALETTE["Lajvard"]),
        ("mean_inter_token_latency",   "ITL",        style.PERSIAN_PALETTE["Shangarfi"]),
    ]
    for k, (col, lbl, color) in enumerate(per_token):
        vals = [sub[sub["model"] == m][col].mean() * 1000.0 for m in models]
        offset = (k - 1) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.9, label=lbl)
    ax.set_ylabel("Per-token costs (ms)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")

    # Shared x labels (bottom row only).
    ax.set_xticks(x)
    ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in models],
                       rotation=45, ha="right", fontsize=6)

    save = out_root / "perf_phase_breakdown_thor_q8.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F7: Phase-breakdown observability demo (Thor, HuggingFace bf16)
# -----------------------------------------------------------------------------
def perf_phase_breakdown_thor_hf(df: pd.DataFrame, out_root: Path) -> Path:
    """Two-row phase-breakdown figure for Thor / HF bf16.

    Row 1: latency stages per model --- tokenization, prefill, TTFT (ms).
    Row 2: per-token costs --- generation, decode, inter-token latency (ms).
    Shared x only; independent y per row.
    """
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)

    sub = df[(df["platform"] == "thor") & (df["backend"] == "hf")
             & (df["dtype"] == "torch.bfloat16")]
    models = [m for m in PARAMS_B if not sub[sub["model"] == m].empty]
    x = np.arange(len(models))

    width = 0.27
    ax = axs[0]
    stages = [
        ("tokenization_time",  "Tokenization", style.PERSIAN_PALETTE["Yashm"]),
        ("prefill_time",       "Prefill",      style.PERSIAN_PALETTE["Lajvard"]),
        ("time_to_first_token","TTFT",         style.PERSIAN_PALETTE["Shangarfi"]),
    ]
    for k, (col, lbl, color) in enumerate(stages):
        vals = [sub[sub["model"] == m][col].mean() * 1000.0 for m in models]
        offset = (k - 1) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.9, label=lbl)
    ax.set_ylabel("Latency stages (ms)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")

    ax = axs[1]
    per_token = [
        ("mean_token_generation_time", "Generation", style.PERSIAN_PALETTE["Yashm"]),
        ("mean_token_decode_time",     "Decode",     style.PERSIAN_PALETTE["Lajvard"]),
        ("mean_inter_token_latency",   "ITL",        style.PERSIAN_PALETTE["Shangarfi"]),
    ]
    for k, (col, lbl, color) in enumerate(per_token):
        vals = [sub[sub["model"] == m][col].mean() * 1000.0 for m in models]
        offset = (k - 1) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.9, label=lbl)
    ax.set_ylabel("Per-token costs (ms)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in models],
                       rotation=45, ha="right", fontsize=6)

    save = out_root / "perf_phase_breakdown_thor_hf.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# F8: Side-by-side phase-breakdown observability demo (Thor, Qwen family)
# -----------------------------------------------------------------------------
def perf_phase_breakdown_thor_qwen(df: pd.DataFrame, out_root: Path) -> Path:
    """2x2 phase-breakdown figure: HF bf16 (left) vs llama.cpp F16 (right).

    Rows: latency stages / per-token costs (different colors per row to
    signal they are distinct pipeline phases). Cols: backends. y is
    shared within each row, x within each column. Bars carry std across
    prompts as error bars and rotated value labels (phase-1 style).
    Single-column width.

    Models: Qwen family only --- QW-1.5B, QW-3B, QW-7B.
    """
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 2.6),
                            sharex="col", sharey="row")

    qwen_models = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]

    # Row 0 colors (one-shot stages) vs Row 1 colors (per-token costs).
    # Bottom row uses darker faces than the top row to signal a distinct
    # phase family. All six distinct from each other and from platform
    # colors (Lajvard / Yashm / Shangarfi).
    row_colors = [
        [style.PERSIAN_PALETTE["Zaferani"],   # saffron
         style.PERSIAN_PALETTE["Nili"],       # indigo
         style.PERSIAN_PALETTE["Choobi"]],    # wood
        [style.PERSIAN_PALETTE["Mesi"],          # copper
         style.PERSIAN_PALETTE["Shahedanaei"],   # hemp green / olive
         "#2F5D80"],                             # steel blue
    ]
    width = 0.27

    backends = [
        ("hf",       "torch.bfloat16", "HuggingFace (bf16)"),
        ("llamacpp", "F16",            "llama.cpp (F16)"),
    ]
    rows = [
        ("Prefill-Stage Costs (ms)",
         [("tokenization_time",   "Tokenization"),
          ("prefill_time",        "Prefill"),
          ("time_to_first_token", "TTFT")]),
        ("Per-token costs (ms)",
         [("mean_token_generation_time", "Generation"),
          ("mean_token_decode_time",     "De-tokenization"),
          ("mean_inter_token_latency",   "ITL")]),
    ]

    # First pass: collect per-row y-max so value labels can pick
    # white-inside vs black-above based on a shared threshold.
    row_ymax = [0.0, 0.0]
    for col_idx, (backend, dtype, _) in enumerate(backends):
        sub = df[(df["platform"] == "thor") & (df["backend"] == backend)
                 & (df["dtype"] == dtype)]
        for row_idx, (_, cols) in enumerate(rows):
            for col, _ in cols:
                for m in qwen_models:
                    sm = sub[sub["model"] == m]
                    if sm.empty:
                        continue
                    v = sm[col].mean() * 1000.0
                    s = sm[col].std()  * 1000.0
                    row_ymax[row_idx] = max(row_ymax[row_idx],
                                            (v + s) * 1.05)

    for col_idx, (backend, dtype, title) in enumerate(backends):
        sub = df[(df["platform"] == "thor") & (df["backend"] == backend)
                 & (df["dtype"] == dtype)]
        models = [m for m in qwen_models if not sub[sub["model"] == m].empty]
        x = np.arange(len(models))

        for row_idx, (ylabel, cols) in enumerate(rows):
            ax = axs[row_idx, col_idx]
            for k, (col, lbl) in enumerate(cols):
                means = [sub[sub["model"] == m][col].mean() * 1000.0 for m in models]
                stds  = [sub[sub["model"] == m][col].std()  * 1000.0 for m in models]
                offset = (k - 1) * width
                ax.bar(x + offset, means, width, yerr=stds,
                       color=row_colors[row_idx][k],
                       edgecolor="black", linewidth=0.3, alpha=0.9, label=lbl,
                       error_kw=dict(elinewidth=0.5, capsize=1.0, ecolor="black"))

                # Phase-1 style rotated value labels.
                # Inside-tall: anchored near the bar bottom (white text).
                # Outside-short: positioned just above the upper error bar tip
                # (black text). Sub-0.1 ms values are suppressed.
                thresh = row_ymax[row_idx] * 0.35
                for xi, v, s in zip(x + offset, means, stds):
                    if v < 0.01:
                        # Bar is basically invisible at this y-scale; mark
                        # it as essentially zero so the reader knows the
                        # bar is real (not missing data).
                        ax.text(xi, row_ymax[row_idx] * 0.02, r"$\approx$0",
                                ha="center", va="bottom", rotation=90,
                                fontsize=4.5, color="black")
                        continue
                    if v >= thresh:
                        ax.text(xi, row_ymax[row_idx] * 0.02, f"{v:.1f}",
                                ha="center", va="bottom", rotation=90,
                                fontsize=4.5, color="white")
                    else:
                        ax.text(xi, v + s + row_ymax[row_idx] * 0.02, f"{v:.1f}",
                                ha="center", va="bottom", rotation=90,
                                fontsize=4.5, color="black")

            ax.set_ylim(0, row_ymax[row_idx])
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=5.5)
            if row_idx == 0:
                ax.set_title(title, fontsize=7)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=5.5)
                ax.legend(fontsize=5, frameon=False, ncol=1, loc="upper left",
                          handlelength=1.0, columnspacing=0.6, handletextpad=0.3,
                          labelspacing=0.25, borderpad=0.2)
            if row_idx == len(rows) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in models],
                                   rotation=0, ha="center", fontsize=6)

    save = out_root / "perf_phase_breakdown_thor_qwen.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


def render_all(df: pd.DataFrame, out_root: Path) -> list[Path]:
    return [
        perf_transition_speedup(df, out_root),
        perf_arch_signatures(df, out_root),
        perf_e2e_latency_grid(df, out_root),
        perf_q4_throughput_grouped(df, out_root),
        perf_phase_breakdown_thor_q8(df, out_root),
        perf_phase_breakdown_thor_hf(df, out_root),
        perf_phase_breakdown_thor_qwen(df, out_root),
    ]
