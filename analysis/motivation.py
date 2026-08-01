# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""phase decomposition — the methodology contribution rendered.

Three figures:

  1. Per-phase telemetry contrast: prefill-window vs decode-window means of
     GPU load, CPU load, GPU power, EMC load. Small-multiple grid, one row
     per metric, one column per platform; bars in each panel are paired
     (Prefill, Decode) per model. Demonstrates how the pipeline phases
     stress the SoC differently — the central methodology insight.

  2. Stacked-bar prefill vs decode time per (platform, dtype, model).

  3. GPU-saturation-by-phase: bar with prefill and decode load side-by-side
     per model; shows whether GPU is saturated during decode
     (memory-bound) vs during prefill (compute-bound).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style
from analysis.extras._bars import grouped_bar


def _select_hf(df: pd.DataFrame, plat: str) -> pd.DataFrame:
    return df[(df["platform"] == plat) & (df["backend"] == "hf")
              & (df["dtype"] == "torch.bfloat16")]


# -----------------------------------------------------------------------------
# 1. Per-phase telemetry contrast — small-multiple
# -----------------------------------------------------------------------------
_TELEMETRY_ROWS = [
    ("gpu_load",      r"GPU load (\%)",          1.0),
    ("cpu_load_mean", r"CPU load (\%)",          1.0),
    ("gpu_power_mw",  r"GPU power (W)",          1/1000.0),
    ("emc_load",      r"EMC load (\%)",          1.0),
]


def per_phase_telemetry(df: pd.DataFrame, out_root: Path,
                        platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    nrows = len(_TELEMETRY_ROWS)
    ncols = len(platforms)
    # sharey="row" forces honest cross-platform comparison within each
    # telemetry metric (e.g. CPU load on Xavier vs Orin is on one scale).
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.0, 1.6 * nrows + 0.9),
                            sharex="col", sharey="row")
    if nrows == 1: axs = np.array([axs])
    if ncols == 1: axs = axs.reshape(-1, 1)

    # Pass 1: collect all bar values keyed by (row, col) so we can compute
    # the per-row maximum across platforms before drawing. Otherwise the
    # shared row y-axis is auto-scaled by whichever panel matplotlib sees
    # last, which can clip platforms with larger ranges (observed: Orin GPU
    # power up to 26W being clipped at Thor's ~20W max).
    cell_data: dict[tuple[int, int], dict] = {}
    for j, plat in enumerate(platforms):
        sub = _select_hf(df, plat)
        if sub.empty:
            cell_data[("empty", j)] = True
            continue
        models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
        for i, (root, ylabel, scale) in enumerate(_TELEMETRY_ROWS):
            prefill_col = f"{root}_prefill_mean"
            decode_col = f"{root}_decode_mean"
            prefill_vals = [sub[sub["model"] == m][prefill_col].mean() * scale
                            for m in models]
            decode_vals = [sub[sub["model"] == m][decode_col].mean() * scale
                           for m in models]
            cell_data[(i, j)] = {
                "models": models,
                "prefill": prefill_vals,
                "decode": decode_vals,
            }

    # Row maxima across all platforms (with a small headroom margin).
    row_max = []
    for i in range(nrows):
        vals = []
        for j in range(ncols):
            cell = cell_data.get((i, j))
            if isinstance(cell, dict):
                vals.extend(cell["prefill"])
                vals.extend(cell["decode"])
        m = max(vals) if vals else 1.0
        row_max.append(m * 1.08)

    # Pass 2: draw with explicit ylim per row.
    width = 0.4
    for j, plat in enumerate(platforms):
        if cell_data.get(("empty", j)):
            for i in range(nrows):
                axs[i, j].set_visible(False)
            continue
        models = cell_data[(0, j)]["models"]
        x = np.arange(len(models))
        for i, (root, ylabel, scale) in enumerate(_TELEMETRY_ROWS):
            ax = axs[i, j]
            cell = cell_data[(i, j)]
            ax.bar(x - width/2, cell["prefill"], width,
                   color=style.PERSIAN_PALETTE["Lajvard"],
                   edgecolor="black", linewidth=0.4,
                   alpha=0.85, label="Prefill")
            ax.bar(x + width/2, cell["decode"], width,
                   color=style.PERSIAN_PALETTE["Mesi"],
                   edgecolor="black", linewidth=0.4,
                   alpha=0.85, label="Decode")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            if i == 0:
                ax.set_title(plat.capitalize(), fontsize=9)
            ax.set_ylim(0, row_max[i])
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(True, axis="y", alpha=0.3)
        # Bottom row: model labels.
        axs[-1, j].set_xticks(x)
        axs[-1, j].set_xticklabels(
            [style.MODEL_DISPLAY.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=6,
        )

    # Header: state backend + precision explicitly so the figure is
    # self-describing (_select_hf above pins HF / bf16).
    fig.suptitle(
        "HuggingFace Transformers, bfloat16 "
        r"\textemdash{} mean across 541 IFEval prompts",
        fontsize=8.5, y=0.995,
    )
    # Legend just below the suptitle.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=8)
    save = out_root / "phase_decomp_telemetry.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# 2. Stacked-bar prefill vs decode time per platform (HF bf16)
# -----------------------------------------------------------------------------
def phase_time_breakdown(df: pd.DataFrame, out_root: Path,
                         platforms: list[str] | None = None) -> Path:
    platforms = platforms or style.PLATFORM_ORDER
    fig, axs = plt.subplots(1, len(platforms), figsize=style.FIG_SIZE_FULL,
                            sharey=True)
    if len(platforms) == 1:
        axs = [axs]
    for ax, plat in zip(axs, platforms):
        sub = _select_hf(df, plat)
        if sub.empty:
            ax.set_visible(False)
            continue
        models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
        # Prefill time = tokenization + prefill (matches the phase
        # boundary used by the unifier for telemetry slicing).
        prefill = [
            (sub[sub["model"] == m]["tokenization_time"].mean()
             + sub[sub["model"] == m]["prefill_time"].mean())
            for m in models
        ]
        # Decode time = total_generation_time - first-token slice,
        # approximated by total_generation_time minus the small
        # gen[0]+decode[0] tail. We use total_generation_time directly
        # since the first-token cost is well under 1% on this scale.
        decode = [sub[sub["model"] == m]["total_generation_time"].mean()
                  for m in models]
        x = np.arange(len(models))
        ax.bar(x, prefill, color=style.PERSIAN_PALETTE["Lajvard"],
               edgecolor="black", linewidth=0.5,
               alpha=0.85, label="Prefill")
        ax.bar(x, decode, bottom=prefill,
               color=style.PERSIAN_PALETTE["Mesi"],
               edgecolor="black", linewidth=0.5,
               alpha=0.85, label="Decode")
        ax.set_title(plat.capitalize(), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([style.MODEL_DISPLAY.get(m, m) for m in models],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("Mean per-prompt time (s)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save = out_root / "phase_decomp_time.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# 3. GPU saturation by phase (bar, prefill vs decode load)
# -----------------------------------------------------------------------------
def gpu_saturation_by_phase(df: pd.DataFrame, out_root: Path,
                            platforms: list[str] | None = None) -> list[Path]:
    platforms = platforms or style.PLATFORM_ORDER
    written: list[Path] = []
    for plat in platforms:
        sub = _select_hf(df, plat)
        if sub.empty:
            continue
        models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
        means: dict[str, list[float]] = {"Prefill": [], "Decode": []}
        stds:  dict[str, list[float]] = {"Prefill": [], "Decode": []}
        for m in models:
            r = sub[sub["model"] == m]
            means["Prefill"].append(r["gpu_load_prefill_mean"].mean())
            means["Decode"].append(r["gpu_load_decode_mean"].mean())
            stds["Prefill"].append(r["gpu_load_prefill_mean"].std())
            stds["Decode"].append(r["gpu_load_decode_mean"].std())
        save = out_root / f"phase_decomp_gpu_saturation_{plat}.pdf"
        grouped_bar(
            model_names=models,
            means=means, stds=stds,
            colors=[style.PERSIAN_PALETTE["Lajvard"],
                    style.PERSIAN_PALETTE["Mesi"]],
            legend_labels=["Prefill", "Decode"],
            xlabel="Model",
            ylabel=r"GPU load (\%)",
            figsize=style.FIG_SIZE_2COL,
            save_path=save,
            title=f"GPU saturation by phase - {plat.capitalize()}",
            ylim=(0, 100),
        )
        written.append(save)
    return written


# -----------------------------------------------------------------------------
# 4. Phase-detail figure — per-model breakdown for §5 Analysis / Discussion
# -----------------------------------------------------------------------------
def phase_detail(df: pd.DataFrame, out_root: Path,
                 platforms: list[str] | None = None) -> Path:
    """Three-row figure with per-model granularity. Intended for §5 Analysis
    or §6 Discussion (it carries all 13 models on the x-axis and is too
    information-dense for the §2 motivation slot — the compact bin-grid
    `phase_motivation` plays that role).

    Row 1: per-token latency (log scale).
      - Prefill bar  = mean over prompts of (prefill_time / n_input_tokens) ms.
      - Decode bar   = mean over prompts of (total_generation_time / output_length) ms.
      Per-token rate of the relevant operation in each phase. Captures the
      30--100x phase divergence that mean-over-phase utilization metrics miss.

    Row 2: effective DRAM bandwidth (GB/s) per phase.
      - DAC formula: peak_BW * (emc_freq / freq_max) * (emc_load / 100).
      - Xavier UNIFIED CSVs predate the parser fix and lack `emc_freq`;
        EMC is empirically pinned at peak under inference load on Xavier,
        so we fall back to peak_BW * (emc_load / 100) for that platform.
      - Peak-BW reference label is anchored at the left edge of each panel
        to keep the rightmost bars clear.

    Row 3: GPU power (W) per phase.
      - Reveals the TDP envelope per generation (Xavier ~30W cap, Orin 60W,
        Thor ~168W) and exposes that decode often consumes nearly as much
        GPU power as prefill, despite very different arithmetic intensity.
    """
    platforms = platforms or style.PLATFORM_ORDER
    nrows, ncols = 3, len(platforms)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.0, 4.0),
                            sharex="col", sharey="row")
    if ncols == 1: axs = axs.reshape(-1, 1)

    # Pass 1: collect cell data per row so we can set ylim explicitly.
    cell_data: dict[tuple[int, int], dict] = {}
    for j, plat in enumerate(platforms):
        sub = _select_hf(df, plat)
        if sub.empty:
            cell_data[("empty", j)] = True
            continue
        models = [m for m in style.MODEL_ORDER if not sub[sub["model"] == m].empty]
        peak_bw = style.PEAK_MEM_BW_GBPS[plat]
        peak_emc = style.PEAK_EMC_FREQ_MHZ[plat]
        has_freq = "emc_freq_prefill_mean" in sub.columns and sub["emc_freq_prefill_mean"].notna().any()

        ms_prefill, ms_decode = [], []
        bw_prefill, bw_decode = [], []
        gw_prefill, gw_decode = [], []
        for m in models:
            r = sub[sub["model"] == m]
            # ---- Per-token latency (ms) ----
            ms_prefill.append(
                (r["prefill_time"] / r["total_number_of_tokens_in_input"]).mean() * 1000.0
            )
            ms_decode.append(
                (r["total_generation_time"] / r["output_length"]).mean() * 1000.0
            )
            # ---- Effective DRAM BW (GB/s) ----
            load_p = r["emc_load_prefill_mean"].mean() / 100.0
            load_d = r["emc_load_decode_mean"].mean() / 100.0
            if has_freq:
                freq_p = r["emc_freq_prefill_mean"].mean() / peak_emc
                freq_d = r["emc_freq_decode_mean"].mean() / peak_emc
            else:
                freq_p = freq_d = 1.0  # Xavier fallback (empirically always at peak)
            bw_prefill.append(peak_bw * freq_p * load_p)
            bw_decode.append(peak_bw * freq_d * load_d)
            # ---- GPU power (W) ----
            gw_prefill.append(r["gpu_power_mw_prefill_mean"].mean() / 1000.0)
            gw_decode.append(r["gpu_power_mw_decode_mean"].mean() / 1000.0)

        cell_data[(0, j)] = {"models": models, "prefill": ms_prefill, "decode": ms_decode}
        cell_data[(1, j)] = {"models": models, "prefill": bw_prefill, "decode": bw_decode,
                             "peak": peak_bw, "has_freq": has_freq}
        cell_data[(2, j)] = {"models": models, "prefill": gw_prefill, "decode": gw_decode}

    # ----- Row limits -----
    # Row 0 (log): decade-aligned across platforms.
    r0_vals = [v for j in range(ncols)
                 if isinstance(cell_data.get((0, j)), dict)
                 for v in cell_data[(0, j)]["prefill"] + cell_data[(0, j)]["decode"]]
    r0_lo = max(0.1, 10 ** np.floor(np.log10(min(r0_vals)) - 0.1)) if r0_vals else 0.1
    r0_hi = 10 ** np.ceil(np.log10(max(r0_vals)) + 0.1) if r0_vals else 1000.0

    # Row 1 (linear): max peak BW across platforms, with headroom.
    r1_hi = max(style.PEAK_MEM_BW_GBPS[p] for p in platforms) * 1.08

    # Row 2 (linear): max GPU power across platforms, with headroom.
    r2_vals = [v for j in range(ncols)
                 if isinstance(cell_data.get((2, j)), dict)
                 for v in cell_data[(2, j)]["prefill"] + cell_data[(2, j)]["decode"]]
    r2_hi = max(r2_vals) * 1.08 if r2_vals else 30.0

    width = 0.4
    for j, plat in enumerate(platforms):
        if cell_data.get(("empty", j)):
            for i in range(nrows):
                axs[i, j].set_visible(False)
            continue
        models = cell_data[(0, j)]["models"]
        x = np.arange(len(models))

        # ----- Row 0: per-token latency (log) -----
        ax = axs[0, j]
        ax.bar(x - width/2, cell_data[(0, j)]["prefill"], width,
               color=style.PERSIAN_PALETTE["Nili"],
               edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Prefill (per input token)")
        ax.bar(x + width/2, cell_data[(0, j)]["decode"], width,
               color=style.PERSIAN_PALETTE["Mashi"],
               edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Decode (per output token)")
        ax.set_yscale("log")
        ax.set_ylim(r0_lo, r0_hi)
        if j == 0: ax.set_ylabel("Per-token latency (ms)", fontsize=8)
        ax.set_title(plat.capitalize(), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", which="both", alpha=0.3)

        # ----- Row 1: effective DRAM bandwidth (linear) -----
        ax = axs[1, j]
        ax.bar(x - width/2, cell_data[(1, j)]["prefill"], width,
               color=style.PERSIAN_PALETTE["Nili"],
               edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(1, j)]["decode"], width,
               color=style.PERSIAN_PALETTE["Mashi"],
               edgecolor="black", linewidth=0.4, alpha=0.9)
        peak = cell_data[(1, j)]["peak"]
        # Peak reference: dashed dark-red line at full opacity so the
        # "ceiling" reads at a glance against the indigo bars.
        ax.axhline(peak, color=style.PERSIAN_PALETTE["Zereshki"],
                   linestyle="--", linewidth=1.1, alpha=0.95)
        # Label sits above the line where there is headroom (Xavier, Orin),
        # and below the line on Thor where the peak is close to the top edge
        # of the panel and an above-line label crowds the panel border.
        label_va = "top" if plat == "thor" else "bottom"
        ax.text(-0.4, peak, f"peak {peak:g} GB/s", fontsize=8,
                ha="left", va=label_va,
                color=style.PERSIAN_PALETTE["Zereshki"])
        ax.set_ylim(0, r1_hi)
        if j == 0: ax.set_ylabel("Effective BW (GB/s)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)

        # ----- Row 2: GPU power (W) -----
        ax = axs[2, j]
        ax.bar(x - width/2, cell_data[(2, j)]["prefill"], width,
               color=style.PERSIAN_PALETTE["Nili"],
               edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(2, j)]["decode"], width,
               color=style.PERSIAN_PALETTE["Mashi"],
               edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.set_ylim(0, r2_hi)
        if j == 0: ax.set_ylabel("GPU power (W)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [style.MODEL_CODE.get(m, m) for m in models],
            rotation=45, ha="right", fontsize=7,
        )

    # No suptitle — figure metadata (backend, dtype, n_prompts) belongs in
    # the LaTeX caption, not on the figure itself.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    save = out_root / "phase_detail.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save)
    plt.close(fig)
    return save


# -----------------------------------------------------------------------------
# 5. Motivation figure — compact bin-grid version (§2.2)
# -----------------------------------------------------------------------------
# Models partitioned into two parameter bins, used by `phase_motivation`.
SIZE_BINS: dict[str, set[str]] = {
    "1–4 B": {"Llama-3.2-1B", "Qwen2.5-1.5B", "granite-3.3-2B", "gemma-2B",
                 "Llama-3.2-3B", "Qwen2.5-3B", "Phi-3.5-mini"},
    "5–8 B": {"Qwen2.5-7B", "moxin-7B", "gemma-7B", "Ministral-8B",
                 "Llama-3.1-8B", "granite-3.3-8B"},
}


def _per_token_ms(sub: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pref = (sub["prefill_time"] / sub["total_number_of_tokens_in_input"]) * 1000.0
    dec = (sub["total_generation_time"] / sub["output_length"]) * 1000.0
    return pref, dec


def _effective_bw_gbps(sub: pd.DataFrame, plat: str) -> tuple[pd.Series, pd.Series]:
    peak_bw = style.PEAK_MEM_BW_GBPS[plat]
    peak_emc = style.PEAK_EMC_FREQ_MHZ[plat]
    has_freq = ("emc_freq_prefill_mean" in sub.columns
                and sub["emc_freq_prefill_mean"].notna().any())
    if has_freq:
        freq_p = sub["emc_freq_prefill_mean"] / peak_emc
        freq_d = sub["emc_freq_decode_mean"] / peak_emc
    else:
        freq_p = freq_d = 1.0
    return (sub["emc_load_prefill_mean"] / 100.0 * freq_p * peak_bw,
            sub["emc_load_decode_mean"] / 100.0 * freq_d * peak_bw)


def _gpu_power_w(sub: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return (sub["gpu_power_mw_prefill_mean"] / 1000.0,
            sub["gpu_power_mw_decode_mean"] / 1000.0)


def phase_motivation(df: pd.DataFrame, out_root: Path,
                     platforms: list[str] | None = None) -> Path:
    """Compact 3-row x 3-col motivation figure for §2.2.

    Same three metrics as `phase_detail` (per-token latency on log scale,
    effective DRAM bandwidth, GPU power) and three Jetson generations, but
    the x-axis collapses the 13 models into two size bins (1-4B, 5-8B).
    Each bin shows mean prefill and decode values. The result fits a
    single column and serves as a teaser for the per-model `phase_detail`
    figure used in §5 Analysis.
    """
    platforms = platforms or style.PLATFORM_ORDER
    bin_labels = list(SIZE_BINS)
    nrows, ncols = 3, len(platforms)
    # Single-column width (~3.4 in for IEEE 10pt). Slightly taller than
    # wide to keep the 3 stacked rows readable.
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4, 3.2),
                            sharex="col", sharey="row")
    if ncols == 1:
        axs = axs.reshape(-1, 1)

    # Pass 1: collect per-cell values per size bin.
    cell_data: dict[tuple, dict] = {}
    for j, plat in enumerate(platforms):
        sub = df[(df["platform"] == plat) & (df["backend"] == "hf")
                 & (df["dtype"] == "torch.bfloat16")]
        if sub.empty:
            cell_data[("empty", j)] = True
            continue
        pref_lat, dec_lat = [], []
        pref_bw,  dec_bw  = [], []
        pref_pw,  dec_pw  = [], []
        for bin_lbl in bin_labels:
            sub_bin = sub[sub["model"].isin(SIZE_BINS[bin_lbl])]
            pl, dl = _per_token_ms(sub_bin)
            pb, db = _effective_bw_gbps(sub_bin, plat)
            pp, dp = _gpu_power_w(sub_bin)
            pref_lat.append(pl.mean()); dec_lat.append(dl.mean())
            pref_bw.append(pb.mean());  dec_bw.append(db.mean())
            pref_pw.append(pp.mean());  dec_pw.append(dp.mean())
        cell_data[(0, j)] = {"prefill": pref_lat, "decode": dec_lat}
        cell_data[(1, j)] = {"prefill": pref_bw,  "decode": dec_bw,
                             "peak": style.PEAK_MEM_BW_GBPS[plat]}
        cell_data[(2, j)] = {"prefill": pref_pw,  "decode": dec_pw}

    # Row-axis limits (log row, two linear rows).
    r0_vals = [v for j in range(ncols)
                 if isinstance(cell_data.get((0, j)), dict)
                 for v in cell_data[(0, j)]["prefill"] + cell_data[(0, j)]["decode"]]
    r0_lo = max(0.1, 10 ** np.floor(np.log10(min(r0_vals)) - 0.1)) if r0_vals else 0.1
    r0_hi = 10 ** np.ceil(np.log10(max(r0_vals)) + 0.1) if r0_vals else 1000.0
    r1_hi = max(style.PEAK_MEM_BW_GBPS[p] for p in platforms) * 1.08
    r2_vals = [v for j in range(ncols)
                 if isinstance(cell_data.get((2, j)), dict)
                 for v in cell_data[(2, j)]["prefill"] + cell_data[(2, j)]["decode"]]
    r2_hi = max(r2_vals) * 1.08 if r2_vals else 30.0

    phase_colors = (style.PERSIAN_PALETTE["Nili"],
                    style.PERSIAN_PALETTE["Mashi"])
    width = 0.35
    x = np.arange(len(bin_labels))
    for j, plat in enumerate(platforms):
        if cell_data.get(("empty", j)):
            for i in range(nrows):
                axs[i, j].set_visible(False)
            continue

        # Row 0 — log per-token latency
        ax = axs[0, j]
        ax.bar(x - width/2, cell_data[(0, j)]["prefill"], width,
               color=phase_colors[0], edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Prefill")
        ax.bar(x + width/2, cell_data[(0, j)]["decode"], width,
               color=phase_colors[1], edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Decode")
        ax.set_yscale("log")
        ax.set_ylim(r0_lo, r0_hi)
        if j == 0: ax.set_ylabel("Per-token\nlatency (ms)", fontsize=7)
        ax.set_title(plat.capitalize(), fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", which="both", alpha=0.3)

        # Row 1 — effective bandwidth
        ax = axs[1, j]
        ax.bar(x - width/2, cell_data[(1, j)]["prefill"], width,
               color=phase_colors[0], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(1, j)]["decode"], width,
               color=phase_colors[1], edgecolor="black", linewidth=0.4, alpha=0.9)
        peak = cell_data[(1, j)]["peak"]
        ax.axhline(peak, color=style.PERSIAN_PALETTE["Zereshki"],
                   linestyle="--", linewidth=1.1, alpha=0.95)
        # Center the peak label horizontally on the dashed line, with a
        # white opaque background so it's readable on top of the line.
        ax.text(np.mean(x), peak, f"peak {peak:g}", fontsize=7,
                ha="center", va="center",
                color=style.PERSIAN_PALETTE["Zereshki"],
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.85))
        ax.set_ylim(0, r1_hi)
        if j == 0: ax.set_ylabel("Effective\nBW (GB/s)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)

        # Row 2 — GPU power
        ax = axs[2, j]
        ax.bar(x - width/2, cell_data[(2, j)]["prefill"], width,
               color=phase_colors[0], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(2, j)]["decode"], width,
               color=phase_colors[1], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.set_ylim(0, r2_hi)
        if j == 0: ax.set_ylabel("GPU\nPower (W)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        # Only the middle (Orin) column gets the x-axis label.
        if j == 1:
            ax.set_xlabel("Model size", fontsize=7)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7)
    save = out_root / "phase_motivation.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save)
    plt.close(fig)
    return save


def render(df: pd.DataFrame, out_root: Path,
                  platforms: list[str] | None = None) -> list[Path]:
    written = [
        per_phase_telemetry(df, out_root, platforms=platforms),
        phase_time_breakdown(df, out_root, platforms=platforms),
        phase_detail(df, out_root, platforms=platforms),
        phase_motivation(df, out_root, platforms=platforms),
    ]
    written += gpu_saturation_by_phase(df, out_root, platforms=platforms)
    return written
