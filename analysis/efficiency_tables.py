# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Regenerate the IISWC 2026 paper's data tables (Tables 4-7) from the
released UNIFIED per-prompt corpus.

Artifact-evaluation companion to the figure modules in `analysis/extras/`.
Each table function returns a pandas DataFrame whose cells are formatted
"mean(std)" strings matching the paper's layout; the CLI writes one
markdown file and one CSV per table.

Formulas (verified against the published values):
  U_eff  [%]    = gpu_load * gpu_freq / gpu_freq_peak
  BW_eff [GB/s] = peak_BW * (emc_freq / emc_freq_peak) * (emc_load / 100)
  Power  [W]    = (gpu_power_mw + sys_power_mw) / 1000          (decode window)
  mJ/tok        = Power * mean_inter_token_latency * 1000
  Total J       = (gpu_power_mw + sys_power_mw)/1000 * end_to_end_latency
                                                             (prompt window)

Usage:
    python3 -m analysis.efficiency_tables \
        --results-root /path/to/hydra-results --out tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import schema, style
from analysis.loader import load_all
from analysis.sensitivity import load_sensitivity

# Peak GPU clocks (MHz) used for effective utilization. Orin matches
# Table 4 / Fig. gpu_ueff; Thor value from plots/system_analysis.py.
PEAK_GPU_FREQ_MHZ = {"xavier": 1377.0, "orin": 1300.0, "thor": 1575.0}

QWEN_MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
FORMATS = [  # (backend, dtype, paper label)
    ("hf", "torch.bfloat16", "HF bf16"),
    ("llamacpp", "F16", "F16"),
    ("llamacpp", "Q8_0", "Q8_0"),
    ("llamacpp", "Q6_K", "Q6_K"),
    ("llamacpp", "Q4_K_M", "Q4_K_M"),
]


def _ms(series: pd.Series, nd: int = 1) -> str:
    """Format a per-prompt series as 'mean(std)' with nd decimals."""
    return f"{series.mean():.{nd}f}({series.std():.{nd}f})"


def _ueff(sub: pd.DataFrame, plat: str, phase: str) -> pd.Series:
    return (sub[f"gpu_load_{phase}_mean"] * sub[f"gpu_freq_{phase}_mean"]
            / PEAK_GPU_FREQ_MHZ[plat])


def _bweff(sub: pd.DataFrame, plat: str, phase: str) -> pd.Series:
    return (style.PEAK_MEM_BW_GBPS[plat]
            * sub[f"emc_freq_{phase}_mean"] / style.PEAK_EMC_FREQ_MHZ[plat]
            * sub[f"emc_load_{phase}_mean"] / 100.0)


def _power_w(sub: pd.DataFrame, phase: str) -> pd.Series:
    return (sub[f"gpu_power_mw_{phase}_mean"]
            + sub[f"sys_power_mw_{phase}_mean"]) / 1000.0


def _cell(df: pd.DataFrame, plat: str, backend: str, model: str,
          dtype: str) -> pd.DataFrame:
    return df[(df.platform == plat) & (df.backend == backend)
              & (df.model == model) & (df.dtype == dtype)]


# --------------------------------------------------------------------------
# Table 4 — Orin: phase-split GPU U_eff and DRAM BW_eff, Qwen family.
# --------------------------------------------------------------------------
def table4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in QWEN_MODELS:
        row: dict[str, str] = {"Model": model}
        for backend, dtype, label in FORMATS:
            sub = _cell(df, "orin", backend, model, dtype)
            if sub.empty:
                row[f"{label} U_eff"] = row[f"{label} BW_eff"] = "--"
                continue
            row[f"{label} U_eff"] = (
                f"{_ms(_ueff(sub, 'orin', 'prefill'))}/"
                f"{_ms(_ueff(sub, 'orin', 'decode'))}")
            row[f"{label} BW_eff"] = (
                f"{_ms(_bweff(sub, 'orin', 'prefill'))}/"
                f"{_ms(_bweff(sub, 'orin', 'decode'))}")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 5 — HF bf16 U_eff on Orin/Thor across S1 (input) and S2 (output).
# --------------------------------------------------------------------------
def _bin_input(tokens: pd.Series) -> pd.Series:
    return pd.cut(tokens, [0, 2000, 4000, 10**9], labels=["1k", "3k", "5k"])


def table5(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in QWEN_MODELS:
        for plat in ["orin", "thor"]:
            for phase in ["prefill", "decode"]:
                row = {"Model": model, "Platf.": plat,
                       "Phase": "Pre" if phase == "prefill" else "Dec"}
                # S1: input sweep, output fixed at 500.
                m = sens[(sens.sweep == "S1") & (sens.platform == plat)
                         & (sens.backend == "hf") & (sens.model == model)]
                m = m.assign(
                    in_bin=_bin_input(m["total_number_of_tokens_in_input"]))
                for tier in ["1k", "3k", "5k"]:
                    sub = m[m.in_bin == tier]
                    row[f"In {tier}"] = (
                        _ms(_ueff(sub, plat, phase)) if len(sub) else "--")
                # S2: output sweep, input fixed at ~50.
                for outlen in [1000, 3000, 5000]:
                    sub = sens[(sens.sweep == "S2") & (sens.platform == plat)
                               & (sens.backend == "hf")
                               & (sens.model == model)
                               & (sens.outlen == outlen)]
                    row[f"Out {outlen//1000}k"] = (
                        _ms(_ueff(sub, plat, phase)) if len(sub) else "--")
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 6 — decode-phase efficiency on Orin/Thor, Qwen family.
# --------------------------------------------------------------------------
def table6(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in QWEN_MODELS:
        for backend, dtype, label in FORMATS:
            row = {"Model": model, "Format": label}
            for metric in ["Power (W)", "mJ/tok",
                           "T_GPU (C)", "T_CPU (C)"]:
                vals = []
                for plat in ["orin", "thor"]:
                    sub = _cell(df, plat, backend, model, dtype)
                    if sub.empty:
                        vals.append("--")
                        continue
                    if metric == "Power (W)":
                        vals.append(_ms(_power_w(sub, "decode")))
                    elif metric == "mJ/tok":
                        mj = (_power_w(sub, "decode")
                              * sub["mean_inter_token_latency"] * 1000.0)
                        vals.append(_ms(mj, nd=0))
                    elif metric == "T_GPU (C)":
                        vals.append(_ms(sub["gpu_temp_c_decode_mean"]))
                    else:
                        vals.append(_ms(sub["cpu_temp_c_decode_mean"]))
                row[metric] = "/".join(vals)   # Orin/Thor
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 7 — total energy per prompt across S1/S2 sweeps.
# --------------------------------------------------------------------------
def table7(sens: pd.DataFrame) -> pd.DataFrame:
    combos = [("hf", "HF bf16"), ("llamacpp", "lcpp Q4")]
    rows = []
    for model in QWEN_MODELS:
        for plat in ["orin", "thor"]:
            for backend, label in combos:
                row = {"Model": model, "Plat.": plat, "Format": label}
                m1 = sens[(sens.sweep == "S1") & (sens.platform == plat)
                          & (sens.backend == backend) & (sens.model == model)]
                m1 = m1.assign(
                    in_bin=_bin_input(m1["total_number_of_tokens_in_input"]))
                for tier in ["1k", "3k", "5k"]:
                    sub = m1[m1.in_bin == tier]
                    if not len(sub):
                        row[f"In {tier}"] = "--"
                        continue
                    joules = _power_w(sub, "prompt") * sub["end_to_end_latency"]
                    row[f"In {tier}"] = _ms(joules, nd=0)
                for outlen in [1000, 3000, 5000]:
                    sub = sens[(sens.sweep == "S2") & (sens.platform == plat)
                               & (sens.backend == backend)
                               & (sens.model == model)
                               & (sens.outlen == outlen)]
                    if not len(sub):
                        row[f"Out {outlen//1000}k"] = "--"
                        continue
                    joules = _power_w(sub, "prompt") * sub["end_to_end_latency"]
                    row[f"Out {outlen//1000}k"] = _ms(joules, nd=0)
                rows.append(row)
    return pd.DataFrame(rows)


def _to_markdown(tbl: pd.DataFrame) -> str:
    """Plain markdown table without the optional `tabulate` dependency."""
    cols = list(tbl.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in tbl.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path,
                    default=Path("/media/amir/data/hydra-results"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading main corpus from {args.results_root} ...")
    df = load_all(args.results_root)
    print(f"  {len(df)} main-corpus records")

    print("Loading sensitivity corpus ...")
    sens_raw = load_sensitivity(args.results_root)
    # Normalize per platform (canonical telemetry columns).
    sens = pd.concat(
        [schema.normalize(g.copy(), plat)
         for plat, g in sens_raw.groupby("platform")],
        ignore_index=True)
    print(f"  {len(sens)} sensitivity records")

    for name, tbl in [("table4_gpu_mem_orin", table4(df)),
                      ("table5_ueff_sweeps", table5(sens)),
                      ("table6_efficiency", table6(df)),
                      ("table7_total_energy", table7(sens))]:
        md = args.out / f"{name}.md"
        csv = args.out / f"{name}.csv"
        tbl.to_csv(csv, index=False)
        md.write_text(_to_markdown(tbl))
        print(f"  wrote {md.name}, {csv.name}")
    print("Done. Compare against Tables 4-7 in the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
