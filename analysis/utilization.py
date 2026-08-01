# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Plots for IISWC 2026 paper, Section 5.2 System Resource Utilization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


PLATFORM_CORES = {"xavier": 8, "orin": 12, "thor": 14}


# -----------------------------------------------------------------------------
# Per-core CPU load and frequency boxplots: HF bf16 vs llama.cpp F16
# -----------------------------------------------------------------------------
def system_cpu_per_core_qwen7b(df: pd.DataFrame, out_root: Path) -> Path:
    """4 x 3 grid showing per-core CPU activity for Qwen2.5-7B.

    Rows: HF bf16 load, HF bf16 freq, llama.cpp F16 load, llama.cpp F16 freq.
    Cols: Xavier, Orin, Thor.
    Each panel: boxplot over prompts of the per-prompt mean per CPU core
    (prompt-window aggregate). Qwen2.5-7B is used because its weight
    footprint (14.2 GB) fits under Xavier's JetPack-5 cuBLAS/NVMAP cap, so
    all 12 panels carry real data --- the rest of the 7-8B class fails to
    load on Xavier (see Section 5.1).
    """
    MODEL = "Qwen2.5-7B"
    platforms = ["xavier", "orin", "thor"]
    backend_specs = [
        ("hf",       "torch.bfloat16", "HF bf16"),
        ("llamacpp", "F16",            "llama.cpp F16"),
    ]

    nrows, ncols = 4, len(platforms)
    # Width of each column is proportional to its core count
    # (Xavier=8, Orin=12, Thor=14). This keeps per-core boxes the same
    # visual width across platforms instead of stretching Xavier wide.
    width_ratios = [PLATFORM_CORES[p] for p in platforms]
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.0, 3.4),
                            sharex="col", sharey="row",
                            gridspec_kw={"width_ratios": width_ratios})

    # Per-backend y-axis ranges (HF and llama.cpp differ by an order of
    # magnitude on load and ~half on freq, so they get their own scales).
    HF_LOAD_YLIM    = (0,    75)
    HF_FREQ_YLIM    = (0.6,  2.5)
    LCPP_LOAD_YLIM  = (0,    15)
    LCPP_FREQ_YLIM  = (0.6,  1.5)

    box_kw = dict(
        widths=0.6, patch_artist=True,
        flierprops=dict(marker=".", markersize=1.8, linestyle="none",
                        markerfacecolor="black", markeredgecolor="none",
                        alpha=0.5),
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
        boxprops=dict(linewidth=0.4),
    )

    for b_idx, (backend, dtype, b_label) in enumerate(backend_specs):
        for p_idx, plat in enumerate(platforms):
            sub = df[(df["platform"] == plat) & (df["backend"] == backend)
                     & (df["dtype"] == dtype) & (df["model"] == MODEL)]
            ncores = PLATFORM_CORES[plat]
            color = style.PLATFORM_COLORS[plat]

            ax_load = axs[b_idx * 2,     p_idx]
            ax_freq = axs[b_idx * 2 + 1, p_idx]

            load_data, freq_data = [], []
            for c in range(ncores):
                lc = f"cpu_{c}_load_prompt_mean"
                fc = f"cpu_{c}_freq_prompt_mean"
                load_data.append(sub[lc].dropna().values
                                 if lc in sub.columns else np.array([]))
                freq_data.append((sub[fc].dropna() / 1000.0).values
                                 if fc in sub.columns else np.array([]))

            empty_panel = sub.empty or all(len(d) == 0 for d in load_data)

            if empty_panel:
                for ax in (ax_load, ax_freq):
                    ax.text(0.5, 0.5,
                            "model fails to load\n(see \\S5.1)",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=7, color="gray", style="italic")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_alpha(0.3)
            else:
                bp = ax_load.boxplot(load_data,
                                     positions=list(range(ncores)),
                                     **box_kw)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                ax_load.set_xticks(range(ncores))
                ax_load.tick_params(axis="y", labelsize=8)
                ax_load.grid(True, axis="y", alpha=0.3)

                bp = ax_freq.boxplot(freq_data,
                                     positions=list(range(ncores)),
                                     **box_kw)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                ax_freq.set_xticks(range(ncores))
                ax_freq.tick_params(axis="y", labelsize=8)
                ax_freq.grid(True, axis="y", alpha=0.3)

            # x-tick labels only on the bottom row of the figure.
            if b_idx * 2 + 1 == nrows - 1 and not empty_panel:
                ax_freq.set_xticklabels([f"C{c}" for c in range(ncores)],
                                        fontsize=8, rotation=90)

            if p_idx == 0:
                ax_load.set_ylabel("Load (\\%)", fontsize=9)
                ax_freq.set_ylabel("Freq (GHz)", fontsize=9)

    # Apply per-row y-limits. sharey="row" propagates ylim across the row.
    axs[0, 0].set_ylim(*HF_LOAD_YLIM)
    axs[1, 0].set_ylim(*HF_FREQ_YLIM)
    axs[2, 0].set_ylim(*LCPP_LOAD_YLIM)
    axs[3, 0].set_ylim(*LCPP_FREQ_YLIM)

    for p_idx, plat in enumerate(platforms):
        axs[0, p_idx].set_title(plat.capitalize(), fontsize=11)

    fig.tight_layout(rect=(0.025, 0, 1, 1))

    # Backend group labels: centered vertically across each two-row group,
    # placed snug against the leftmost ylabels.
    pos_hf_top    = axs[0, 0].get_position()
    pos_hf_bot    = axs[1, 0].get_position()
    pos_lcpp_top  = axs[2, 0].get_position()
    pos_lcpp_bot  = axs[3, 0].get_position()
    y_hf   = (pos_hf_top.y1   + pos_hf_bot.y0)   / 2
    y_lcpp = (pos_lcpp_top.y1 + pos_lcpp_bot.y0) / 2
    fig.text(0.012, y_hf,   "HF bf16",                rotation=90,
             ha="left", va="center", fontsize=10)
    fig.text(0.012, y_lcpp, "llama.cpp F16", rotation=90,
             ha="left", va="center", fontsize=10)
    save = out_root / "system_cpu_per_core_qwen7b.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# Thor GPU effective utilization across precision ladder (Qwen family)
# -----------------------------------------------------------------------------
def system_gpu_ueff_thor_qwen(df: pd.DataFrame, out_root: Path) -> Path:
    """4 x 1 grid: rows = HF bf16, llama.cpp Q8_0/Q6_K/Q4_K_M on Thor.

    Each row: 3 Qwen models (1.5B/3B/7B), 2 phase bars per model
    (prefill, decode). y = GPU effective utilization (%), defined as
    U_eff = load(%) * freq / freq_peak. On Thor freq is pinned at
    1574 MHz across every (model, dtype, phase), so U_eff numerically
    equals Load(%); the metric still earns its name on platforms whose
    GPU DVFS actually moves.
    """
    PEAK_FREQ_MHZ = 1575
    qwen_models = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
    rows = [
        ("hf",       "torch.bfloat16", "HF bf16"),
        ("llamacpp", "Q8_0",           "llama.cpp Q8_0"),
        ("llamacpp", "Q6_K",           "llama.cpp Q6_K"),
        ("llamacpp", "Q4_K_M",         "llama.cpp Q4_K_M"),
    ]
    phases = [("prefill", "Prefill"), ("decode", "Decode")]

    nrows = len(rows)
    fig, axs = plt.subplots(nrows, 1, figsize=(3.4, 4.6),
                            sharex=True, sharey=True)

    phase_colors = [
        style.PERSIAN_PALETTE["Zaferani"],   # saffron (prefill)
        style.PERSIAN_PALETTE["Nili"],       # indigo  (decode)
    ]
    width = 0.36
    x = np.arange(len(qwen_models))

    Y_MAX = 100.0
    for r_idx, (backend, dtype, label) in enumerate(rows):
        ax = axs[r_idx]
        sub = df[(df["platform"] == "thor") & (df["backend"] == backend)
                 & (df["dtype"] == dtype)]
        for k, (phase, plbl) in enumerate(phases):
            load_col = f"gpu_load_{phase}_mean"
            freq_col = f"gpu_freq_{phase}_mean"
            means, stds = [], []
            for m in qwen_models:
                sm = sub[sub["model"] == m]
                if sm.empty:
                    means.append(np.nan); stds.append(np.nan); continue
                ueff = (sm[load_col] * sm[freq_col] / PEAK_FREQ_MHZ).dropna()
                means.append(ueff.mean())
                stds.append(ueff.std())
            offset = (k - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   color=phase_colors[k],
                   edgecolor="black", linewidth=0.3, alpha=0.9, label=plbl,
                   error_kw=dict(elinewidth=0.5, capsize=1.2, ecolor="black"))
            # Value labels: phase-1 style.
            thresh = Y_MAX * 0.35
            for xi, v, s in zip(x + offset, means, stds):
                if pd.isna(v):
                    continue
                s_eff = s if pd.notna(s) else 0.0
                if v >= thresh:
                    ax.text(xi, Y_MAX * 0.02, f"{v:.1f}",
                            ha="center", va="bottom", rotation=90,
                            fontsize=5, color="white")
                else:
                    ax.text(xi, v + s_eff + Y_MAX * 0.02, f"{v:.1f}",
                            ha="center", va="bottom", rotation=90,
                            fontsize=5, color="black")

        ax.set_ylim(0, Y_MAX)
        ax.set_ylabel(label, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        if r_idx == 0:
            ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left",
                      handlelength=1.0, columnspacing=0.6, handletextpad=0.3,
                      borderpad=0.2)
        if r_idx == nrows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in qwen_models],
                               fontsize=7)

    # Shared overall y-axis label on the figure margin.
    fig.text(0.005, 0.5, r"GPU $U_\mathrm{eff}$ (\%)",
             rotation=90, ha="left", va="center", fontsize=8)

    fig.tight_layout(rect=(0.04, 0, 1, 1))
    save = out_root / "system_gpu_ueff_thor_qwen.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save)
    plt.close(fig)
    return save


def render_all(df: pd.DataFrame, out_root: Path) -> list[Path]:
    return [
        system_cpu_per_core_qwen7b(df, out_root),
        system_gpu_ueff_thor_qwen(df, out_root),
    ]
