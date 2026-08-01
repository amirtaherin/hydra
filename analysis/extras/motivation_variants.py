# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Four compact motivation-figure variants for §2.2 design exploration.

The canonical full-grid figure is `phase_decomp.phase_motivation` (3 metrics
x 3 platforms x 13 models x 2 phases). It works for §5 Analysis but is too
much for §2 Motivation. These variants are smaller teasers proposed during
discussion 2026-05-16:

  A. Size-binned (1-4B vs 5-8B), 3 metrics, 3 platforms; phase collapsed.
  B. Single focal model (Llama-3.1-8B), per-token latency on a log scale,
     3 platforms x 2 phases = 6 bars.
  C. Three mini metric bars, each platform = 1 bar (mean across all 13
     models and both phases).
  E. Simplest possible: 2 bars showing mean prefill vs mean decode
     per-token latency aggregated across everything (log scale).

Option D in the discussion was "no figure, prose-only" — nothing to render.

All variants use the same Nili+Mashi (indigo+gold) prefill/decode colors
and the MODEL_CODE x-axis labelling convention used by the canonical
figure for visual consistency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


def _select_hf(df: pd.DataFrame, plat: str) -> pd.DataFrame:
    return df[(df["platform"] == plat) & (df["backend"] == "hf")
              & (df["dtype"] == "torch.bfloat16")]


SIZE_BINS = {
    "1--4 B": {"Llama-3.2-1B", "Qwen2.5-1.5B", "granite-3.3-2B", "gemma-2B",
               "Llama-3.2-3B", "Qwen2.5-3B", "Phi-3.5-mini"},
    "5--8 B": {"Qwen2.5-7B", "moxin-7B", "gemma-7B", "Ministral-8B",
               "Llama-3.1-8B", "granite-3.3-8B"},
}
FOCAL_MODEL = "Llama-3.1-8B"     # used by option B
PHASE_COLORS = (style.PERSIAN_PALETTE["Nili"],
                style.PERSIAN_PALETTE["Mashi"])


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


# ---------------------------------------------------------------------------
# Option A: bins (1-4B, 5-8B) x 3 platforms, 3 metric panels, phase collapsed
# ---------------------------------------------------------------------------
def variant_A_bins(df: pd.DataFrame, out_root: Path) -> Path:
    platforms = style.PLATFORM_ORDER
    bin_labels = list(SIZE_BINS)
    metrics = ("Per-token latency (ms, log)", "Effective BW (GB/s)", "GPU power (W)")

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.0))
    width = 0.25
    x = np.arange(len(bin_labels))

    for ax, metric_name in zip(axs, metrics):
        for j, plat in enumerate(platforms):
            sub = _select_hf(df, plat)
            if sub.empty:
                continue
            values = []
            for bin_lbl in bin_labels:
                sub_bin = sub[sub["model"].isin(SIZE_BINS[bin_lbl])]
                if metric_name.startswith("Per-token"):
                    pref, dec = _per_token_ms(sub_bin)
                    v = pd.concat([pref, dec]).mean()
                elif metric_name.startswith("Effective"):
                    pref, dec = _effective_bw_gbps(sub_bin, plat)
                    v = pd.concat([pref, dec]).mean()
                else:
                    pref, dec = _gpu_power_w(sub_bin)
                    v = pd.concat([pref, dec]).mean()
                values.append(v)
            ax.bar(x + (j - 1) * width, values, width,
                   color=style.PLATFORM_COLORS[plat],
                   edgecolor="black", linewidth=0.4, alpha=0.9,
                   label=plat.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_title(metric_name, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        if metric_name.startswith("Per-token"):
            ax.set_yscale("log")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
    save = out_root / "motivation_A_bins.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save)
    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Option B: single focal model, log latency, 3 platforms x 2 phases
# ---------------------------------------------------------------------------
def variant_B_focal(df: pd.DataFrame, out_root: Path,
                    focal: str = FOCAL_MODEL) -> Path:
    platforms = style.PLATFORM_ORDER
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    width = 0.35
    x = np.arange(len(platforms))

    prefill_vals, decode_vals = [], []
    for plat in platforms:
        sub = _select_hf(df, plat)
        sub = sub[sub["model"] == focal]
        if sub.empty:
            prefill_vals.append(np.nan)
            decode_vals.append(np.nan)
            continue
        pref, dec = _per_token_ms(sub)
        prefill_vals.append(pref.mean())
        decode_vals.append(dec.mean())

    ax.bar(x - width/2, prefill_vals, width,
           color=PHASE_COLORS[0], edgecolor="black", linewidth=0.4,
           alpha=0.9, label="Prefill")
    ax.bar(x + width/2, decode_vals, width,
           color=PHASE_COLORS[1], edgecolor="black", linewidth=0.4,
           alpha=0.9, label="Decode")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in platforms], fontsize=9)
    ax.set_ylabel("Per-token latency (ms)", fontsize=8)
    ax.set_title(style.MODEL_DISPLAY.get(focal, focal), fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    save = out_root / "motivation_B_focal.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Option C: 3 mini metric panels, one bar per platform (mean over all models)
# ---------------------------------------------------------------------------
def variant_C_minibars(df: pd.DataFrame, out_root: Path) -> Path:
    platforms = style.PLATFORM_ORDER
    metrics = (
        ("Per-token latency (ms)", "log"),
        ("Effective BW (GB/s)",    "linear"),
        ("GPU power (W)",          "linear"),
    )
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9))
    x = np.arange(len(platforms))
    for ax, (metric_name, yscale) in zip(axs, metrics):
        vals = []
        for plat in platforms:
            sub = _select_hf(df, plat)
            if metric_name.startswith("Per-token"):
                pref, dec = _per_token_ms(sub)
                vals.append(pd.concat([pref, dec]).mean())
            elif metric_name.startswith("Effective"):
                pref, dec = _effective_bw_gbps(sub, plat)
                vals.append(pd.concat([pref, dec]).mean())
            else:
                pref, dec = _gpu_power_w(sub)
                vals.append(pd.concat([pref, dec]).mean())
        colors = [style.PLATFORM_COLORS[p] for p in platforms]
        ax.bar(x, vals, color=colors,
               edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in platforms], fontsize=8)
        ax.set_ylabel(metric_name, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        if yscale == "log":
            ax.set_yscale("log")

    save = out_root / "motivation_C_minibars.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Option E: simplest possible — 2 bars (mean prefill vs mean decode)
# ---------------------------------------------------------------------------
def variant_E_simplest(df: pd.DataFrame, out_root: Path) -> Path:
    platforms = style.PLATFORM_ORDER
    pref_all, dec_all = [], []
    for plat in platforms:
        sub = _select_hf(df, plat)
        pref, dec = _per_token_ms(sub)
        pref_all.append(pref)
        dec_all.append(dec)
    pref_concat = pd.concat(pref_all)
    dec_concat = pd.concat(dec_all)

    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    bars = ax.bar(["Prefill", "Decode"],
                  [pref_concat.mean(), dec_concat.mean()],
                  color=PHASE_COLORS,
                  edgecolor="black", linewidth=0.5, alpha=0.9, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Per-token latency (ms)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # Annotate the ratio.
    ratio = dec_concat.mean() / pref_concat.mean()
    ax.text(0.5, 0.97,
            f"$\\sim${ratio:.0f}$\\times$ slower",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="lightgray"))

    save = out_root / "motivation_E_simplest.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    return save


# ---------------------------------------------------------------------------
# Option F: same 3-metric x 3-platform grid as the canonical motivation but
# with x-axis collapsed to 2 size bins (1-4B, 5-8B), each bin showing the
# mean prefill and decode value for that bin. Keeps the phase-divergence
# story and the cross-generation story; drops per-model granularity.
# ---------------------------------------------------------------------------
def variant_F_bin_grid(df: pd.DataFrame, out_root: Path) -> Path:
    platforms = style.PLATFORM_ORDER
    bin_labels = list(SIZE_BINS)
    nrows, ncols = 3, len(platforms)
    # Single-column width target (~3.4 in for IEEE 10pt). Slightly taller
    # than wide to keep the 3 stacked rows readable.
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4, 3.4),
                            sharex="col", sharey="row")
    if ncols == 1:
        axs = axs.reshape(-1, 1)

    # Pass 1: collect per (row, col) cell values keyed by bin label.
    cell_data: dict[tuple[int, int], dict] = {}
    for j, plat in enumerate(platforms):
        sub = _select_hf(df, plat)
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
            pref_lat.append(pl.mean());  dec_lat.append(dl.mean())
            pref_bw.append(pb.mean());   dec_bw.append(db.mean())
            pref_pw.append(pp.mean());   dec_pw.append(dp.mean())
        cell_data[(0, j)] = {"prefill": pref_lat, "decode": dec_lat}
        cell_data[(1, j)] = {"prefill": pref_bw,  "decode": dec_bw,
                             "peak": style.PEAK_MEM_BW_GBPS[plat]}
        cell_data[(2, j)] = {"prefill": pref_pw,  "decode": dec_pw}

    # Row 0 (log) decade-aligned limits across platforms.
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
               color=PHASE_COLORS[0], edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Prefill")
        ax.bar(x + width/2, cell_data[(0, j)]["decode"], width,
               color=PHASE_COLORS[1], edgecolor="black", linewidth=0.4,
               alpha=0.9, label="Decode")
        ax.set_yscale("log")
        ax.set_ylim(r0_lo, r0_hi)
        if j == 0: ax.set_ylabel("Per-token latency (ms)", fontsize=7)
        ax.set_title(plat.capitalize(), fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", which="both", alpha=0.3)

        # Row 1 — effective BW (linear)
        ax = axs[1, j]
        ax.bar(x - width/2, cell_data[(1, j)]["prefill"], width,
               color=PHASE_COLORS[0], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(1, j)]["decode"], width,
               color=PHASE_COLORS[1], edgecolor="black", linewidth=0.4, alpha=0.9)
        peak = cell_data[(1, j)]["peak"]
        ax.axhline(peak, color=style.PERSIAN_PALETTE["Zereshki"],
                   linestyle="--", linewidth=1.1, alpha=0.95)
        label_va = "top" if plat == "thor" else "bottom"
        ax.text(-0.4, peak, f"peak {peak:g} GB/s", fontsize=7,
                ha="left", va=label_va,
                color=style.PERSIAN_PALETTE["Zereshki"])
        ax.set_ylim(0, r1_hi)
        if j == 0: ax.set_ylabel("Effective BW (GB/s)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)

        # Row 2 — GPU power
        ax = axs[2, j]
        ax.bar(x - width/2, cell_data[(2, j)]["prefill"], width,
               color=PHASE_COLORS[0], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.bar(x + width/2, cell_data[(2, j)]["decode"], width,
               color=PHASE_COLORS[1], edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.set_ylim(0, r2_hi)
        if j == 0: ax.set_ylabel("GPU power (W)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_xlabel("Model size", fontsize=7)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7)
    save = out_root / "motivation_F_bin_grid.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(save)
    plt.close(fig)
    return save


def render_all(df: pd.DataFrame, out_root: Path) -> list[Path]:
    return [
        variant_A_bins(df, out_root),
        variant_B_focal(df, out_root),
        variant_C_minibars(df, out_root),
        variant_E_simplest(df, out_root),
        variant_F_bin_grid(df, out_root),
    ]
