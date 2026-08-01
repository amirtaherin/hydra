# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Plots for IISWC 2026 paper, sensitivity sweeps (S1 input-length, S2 output-length)."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import style


MODELS = [
    "Llama-3.2-1B",
    "Qwen2.5-1.5B",
    "Llama-3.2-3B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
    "Llama-3.1-8B",
]

TIERS = [1000, 3000, 5000]
TIER_LABELS = ["1k", "3k", "5k"]
# Row 0 (input sweep): warm earth-tone progression.
INPUT_TIER_COLORS = [
    style.PERSIAN_PALETTE["Khaki"],       # 1k (light tan)
    style.PERSIAN_PALETTE["Choobi"],      # 3k (medium brown)
    style.PERSIAN_PALETTE["Bademjani"],   # 5k (dark eggplant)
]
# Row 1 (output sweep): cool blue progression. Distinct base hue so the
# reader can tell the two rows apart at a glance.
OUTPUT_TIER_COLORS = [
    "#BDD7E7",   # 1k (light blue)
    "#6BAED6",   # 3k (medium blue)
    "#2171B5",   # 5k (dark blue)
]


def _parse_model_from_filename(stem: str) -> str:
    """Strip the trailing dtype suffix from 'UNIFIED_<model>_<dtype>_'."""
    base = stem.replace("UNIFIED_", "").rstrip("_")
    if "torch" in base:
        return base.split("_torch.")[0]
    for suffix in ["_Q4_K_M", "_Q6_K", "_Q8_0", "_F16"]:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def load_sensitivity(sweep_root: str | Path) -> pd.DataFrame:
    """Walk results_sensitivity/ and return one DataFrame with sweep/platform/
    backend/dtype/outlen/model attached as columns."""
    sweep_root = Path(sweep_root)
    rows = []
    for d in sorted(glob.glob(str(sweep_root / "results_sensitivity" / "S*"))):
        name = Path(d).name
        parts = name.split("_")
        sweep, plat, backend = parts[0], parts[1], parts[2]
        if sweep == "S2":
            outlen = int(parts[-1].replace("out", ""))
            dtype = "_".join(parts[3:-1])
        else:
            outlen = None
            dtype = "_".join(parts[3:])
        for f in glob.glob(f"{d}/UNIFIED_*.csv"):
            model = _parse_model_from_filename(Path(f).stem)
            d2 = pd.read_csv(f, low_memory=False)
            # Single assign avoids pandas' fragmentation PerformanceWarning
            # from repeated column inserts on a wide (415-column) frame.
            d2 = d2.assign(sweep=sweep, platform=plat, backend=backend,
                           dtype=dtype, outlen=outlen, model=model)
            rows.append(d2)
    return pd.concat(rows, ignore_index=True)


def _bin_input(x: pd.Series) -> pd.Series:
    """Bin input-token counts into 1k/3k/5k tiers."""
    return pd.cut(x, bins=[0, 2000, 4000, 9999], labels=TIERS)


# -----------------------------------------------------------------------------
# 2x2 grid: TTFT vs input length (row 1) and decode tok/s vs output length (row 2);
# columns are Orin and Thor. Backend: llama.cpp Q4_K_M (focal precision).
# -----------------------------------------------------------------------------
def sensitivity_ttft_and_throughput(df_all: pd.DataFrame,
                                    out_root: Path) -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 2.0),
                            sharex="col", sharey="row")
    width = 0.27
    x = np.arange(len(MODELS))

    backend, dtype = "llamacpp", "Q4_K_M"
    platforms = [("orin", "Orin"), ("thor", "Thor")]

    def _annotate(ax, xs, vals, stds, y_max, fmt="{:.0f}"):
        """Phase-1 style value labels: inside-tall (white near bottom),
        outside-short (black above error bar)."""
        thresh = y_max * 0.35
        for xi, v, s in zip(xs, vals, stds):
            if v is None or np.isnan(v) or v == 0:
                continue
            s_eff = s if (s is not None and not np.isnan(s)) else 0.0
            if v >= thresh:
                ax.text(xi, y_max * 0.02, fmt.format(v),
                        ha="center", va="bottom", rotation=90,
                        fontsize=6, color="white")
            else:
                ax.text(xi, v + s_eff + y_max * 0.02, fmt.format(v),
                        ha="center", va="bottom", rotation=90,
                        fontsize=6, color="black")

    # ----- Row 0: Panel A - TTFT vs input length -----
    s1 = df_all[(df_all["sweep"] == "S1") & (df_all["backend"] == backend)
                & (df_all["dtype"] == dtype)].copy()
    s1["tier"] = _bin_input(s1["total_number_of_tokens_in_input"])

    # First pass: compute shared y-max across both Orin and Thor for row 0.
    row0_ymax = 0.0
    for plat, _ in platforms:
        sub = s1[s1["platform"] == plat]
        for tier in TIERS:
            for m in MODELS:
                cell = sub[(sub["model"] == m) & (sub["tier"] == tier)]
                v = cell["time_to_first_token"].mean() * 1000.0
                s = cell["time_to_first_token"].std()  * 1000.0
                if not pd.isna(v):
                    row0_ymax = max(row0_ymax, (v + (s if pd.notna(s) else 0.0)) * 1.10)

    for col_idx, (plat, plat_label) in enumerate(platforms):
        ax = axs[0, col_idx]
        sub = s1[s1["platform"] == plat]
        for k, tier in enumerate(TIERS):
            means = []
            stds = []
            for m in MODELS:
                cell = sub[(sub["model"] == m) & (sub["tier"] == tier)]
                means.append(cell["time_to_first_token"].mean() * 1000.0)
                stds.append(cell["time_to_first_token"].std() * 1000.0)
            offset = (k - 1) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   color=INPUT_TIER_COLORS[k],
                   edgecolor="black", linewidth=0.3, alpha=0.9,
                   label=f"Inp {TIER_LABELS[k]} (out 500)",
                   error_kw=dict(elinewidth=0.5, capsize=1.0, ecolor="black"))
            _annotate(ax, x + offset, means, stds, row0_ymax)
        ax.set_ylim(0, row0_ymax)
        ax.set_title(plat_label, fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel("TTFT (ms)", fontsize=7)
        if col_idx == 1:
            ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left",
                      handlelength=1.0, handletextpad=0.4,
                      columnspacing=0.8, borderpad=0.3)

    # ----- Row 1: Panel B - decode tok/s vs output length -----
    s2 = df_all[(df_all["sweep"] == "S2") & (df_all["backend"] == backend)
                & (df_all["dtype"] == dtype)].copy()

    row1_ymax = 0.0
    for plat, _ in platforms:
        sub = s2[s2["platform"] == plat]
        for outlen in TIERS:
            for m in MODELS:
                cell = sub[(sub["model"] == m) & (sub["outlen"] == outlen)]
                v = cell["generation_tokens_per_sec"].mean()
                s = cell["generation_tokens_per_sec"].std()
                if not pd.isna(v):
                    row1_ymax = max(row1_ymax, (v + (s if pd.notna(s) else 0.0)) * 1.10)

    for col_idx, (plat, plat_label) in enumerate(platforms):
        ax = axs[1, col_idx]
        sub = s2[s2["platform"] == plat]
        for k, outlen in enumerate(TIERS):
            means = []
            stds = []
            for m in MODELS:
                cell = sub[(sub["model"] == m) & (sub["outlen"] == outlen)]
                means.append(cell["generation_tokens_per_sec"].mean())
                stds.append(cell["generation_tokens_per_sec"].std())
            offset = (k - 1) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   color=OUTPUT_TIER_COLORS[k],
                   edgecolor="black", linewidth=0.3, alpha=0.9,
                   label=f"Out {TIER_LABELS[k]} (in $\\sim$50)",
                   error_kw=dict(elinewidth=0.5, capsize=1.0, ecolor="black"))
            _annotate(ax, x + offset, means, stds, row1_ymax)
        ax.set_ylim(0, row1_ymax)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([style.MODEL_CODE.get(m, m) for m in MODELS],
                           rotation=0, ha="center", fontsize=6)
        if col_idx == 0:
            ax.set_ylabel("Decode tok/s", fontsize=7)
        if col_idx == 1:
            ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right",
                      handlelength=1.0, handletextpad=0.4,
                      columnspacing=0.8, borderpad=0.3)

    fig.tight_layout()
    save = out_root / "sensitivity_ttft_throughput.pdf"
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save)
    plt.close(fig)
    return save


def render_all(df_all: pd.DataFrame, out_root: Path) -> list[Path]:
    return [
        sensitivity_ttft_and_throughput(df_all, out_root),
    ]
