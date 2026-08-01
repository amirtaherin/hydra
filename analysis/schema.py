# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Per-platform telemetry schema → canonical metric names.

Each Jetson generation exposes tegrastats with a different shape: different CPU
core count, different number of GPCs, different power-rail labels. Plotters
should not have to care. This module provides a single function `normalize`
that takes a UNIFIED DataFrame for one (platform, backend, model, dtype) cell
and adds canonical columns derivable from the platform's raw columns.

Canonical columns added (each in three phase variants: `_prompt_mean`,
`_prefill_mean`, `_decode_mean`; `_*_std` variants are added analogously
where the source columns have stds). The phase boundary is end-of-prefill
(`tokenization_time + prefill_time`), matching the prefill/decode split
used in mainstream LLM-serving phase-aware work (Splitwise, DistServe,
TokenPowerBench); `time_to_first_token` remains a reported latency
metric but is not used as the phase boundary.

  cpu_load_mean    — mean load across all CPU cores (%)
  cpu_load_max     — max load across cores (%)
  cpu_freq_mean    — mean clock across all CPU cores (MHz)
  gpu_load         — mean load across all GPCs (%)
  gpu_freq         — mean clock across all GPCs (MHz)
  gpu_temp_c       — GPU temperature (deg C)
  cpu_temp_c       — CPU temperature (deg C)
  temp_max_c       — max of all available thermal zones
  ram_used_mb      — RAM in use
  gpu_power_mw     — GPU rail power
  cpu_power_mw     — CPU rail power (Thor combines CPU+SoC+MSS — see note below)
  total_power_mw   — sum of all measured rails (cross-platform proxy)

NVML extensions (Thor only — emit NaN on Xavier/Orin):
  nvml_gpu_util    — NVML-reported GPU utilization (%)
  nvml_mem_util    — NVML-reported memory controller utilization (%)

GPC counts are physical:
  Xavier: 1 (Volta) — tegrastats prints `gpu1_*`
  Orin:   2 (Ampere) — `gpc1_*`, `gpc2_*`
  Thor:   3 (Blackwell) — `gpc1_*`, `gpc2_*`, `gpc3_*`

GPU load source caveat:
  JetPack 7 tegrastats reports per-GPC freq but NOT per-GPC load — the
  `gpc{1,2,3}_load` columns are NaN on Thor. The Thor logger appends NVML
  GPU utilization (`gpu_util` column) precisely to fill this gap. So canonical
  `gpu_load` is sourced from tegrastats GR3D on Xavier/Orin and from NVML on
  Thor. The two are not strictly identical metrics — NVML is sample-time
  occupancy, GR3D is bin-period busy fraction — but both are the standard
  "is the GPU saturated?" indicator on each platform.

Power-rail caveat:
  Thor's CPU rail label is `vdd_cpu_soc_mss_*` — combined CPU+SoC+MSS. This
  is intrinsically broader than Xavier/Orin's `vdd_cpu_cv_*`. Cross-platform
  CPU-power comparisons are approximate; for absolute numbers, prefer
  per-platform plots that use the raw rail name.

Total board power:
  Thor exposes `vin_*` (full board input) directly; Xavier/Orin do not. For
  cross-platform "total power" we use sum of (gpu, cpu, 5V0) rails as a
  consistent proxy. Thor-only plots can use `vin_cur` for the true board total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PHASE_SUFFIXES = ("_prompt_mean", "_prefill_mean", "_decode_mean")


# -----------------------------------------------------------------------------
# Per-platform definition tables
# -----------------------------------------------------------------------------
N_CPU_CORES: dict[str, int] = {"xavier": 8, "orin": 12, "thor": 14}
N_GPCS: dict[str, int] = {"xavier": 1, "orin": 2, "thor": 3}

# GPU load source columns per platform — see "GPU load source caveat" above.
# Thor falls back to NVML because JP7 tegrastats lacks per-GPC load values.
_GPU_LOAD_COLS: dict[str, list[str]] = {
    "xavier": ["gpu1_load"],                          # GR3D only; GR3D2 is unused
    "orin":   ["gpc1_load", "gpc2_load"],
    "thor":   ["gpu_util"],                           # NVML (gpc*_load is NaN)
}
_GPU_FREQ_COLS: dict[str, list[str]] = {
    "xavier": ["gpu1_freq"],
    "orin":   ["gpc1_freq", "gpc2_freq"],
    "thor":   ["gpc1_freq", "gpc2_freq", "gpc3_freq"],
}

# Power rail column mapping (current-draw column).
_GPU_POWER_COL:   dict[str, str] = {"xavier": "vdd_gpu_soc_cur",      "orin": "vdd_gpu_soc_cur",      "thor": "vdd_gpu_cur"}
_CPU_POWER_COL:   dict[str, str] = {"xavier": "vdd_cpu_cv_cur",       "orin": "vdd_cpu_cv_cur",       "thor": "vdd_cpu_soc_mss_cur"}
_5V0_POWER_COL:   dict[str, str] = {"xavier": "sys5v0_cur",           "orin": "vin_sys_5v0_cur",      "thor": "vin_sys_5v0_cur"}

# Optional Thor-only "true board" rail.
_BOARD_POWER_COL: dict[str, str | None] = {"xavier": None, "orin": None, "thor": "vin_cur"}

# Temperature zones available per platform — used to compute temp_max_c.
_TEMP_ZONES: dict[str, list[str]] = {
    "xavier": ["cpu_temp", "gpu_temp", "tboard_temp", "tdiode_temp",
               "soc_0_temp", "soc_1_temp", "soc_2_temp", "tj_temp"],
    "orin":   ["cpu_temp", "gpu_temp", "tboard_temp", "tdiode_temp",
               "soc_0_temp", "soc_1_temp", "soc_2_temp", "tj_temp"],
    "thor":   ["cpu_temp", "gpu_temp", "tj_temp", "soc012_temp", "soc345_temp"],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _phase_cols(base: str, suffix: str, n: int) -> list[str]:
    """Return [base 0_suffix, base 1_suffix, ..., base (n-1)_suffix]."""
    return [f"{base.format(i=i)}{suffix}" for i in range(n)]


def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise mean across cols that exist in df. Returns NaN where all missing."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].mean(axis=1, skipna=True)


def _safe_max(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].max(axis=1, skipna=True)


def _safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise sum across cols that exist in df. NaN if all missing."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].sum(axis=1, skipna=True, min_count=1)


def _passthrough(df: pd.DataFrame, source: str | None) -> pd.Series:
    """Return df[source] if source is a column, else NaN-filled column."""
    if source is None or source not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[source]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def normalize(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Add canonical telemetry columns to a UNIFIED DataFrame for one platform.

    The original columns are preserved (untouched) so per-platform plots that
    need raw rail names still work. Adds canonical columns for each phase
    suffix the source columns expose (`_prompt_mean`, `_prefill_mean`,
    `_decode_mean`).

    Parameters
    ----------
    df : pd.DataFrame
        Loaded UNIFIED_*.csv for a single (platform, backend, model, dtype) cell.
    platform : {"xavier", "orin", "thor"}
        Platform tag — selects the column mapping.

    Returns
    -------
    pd.DataFrame
        Same df with canonical columns added in place.
    """
    if platform not in N_CPU_CORES:
        raise ValueError(f"Unknown platform: {platform!r}")

    df = df.copy()
    n_cpu = N_CPU_CORES[platform]

    for suffix in PHASE_SUFFIXES:
        # CPU aggregates across cores
        load_cols = [f"cpu_{i}_load{suffix}" for i in range(n_cpu)]
        freq_cols = [f"cpu_{i}_freq{suffix}" for i in range(n_cpu)]
        df[f"cpu_load_mean{suffix}"] = _safe_mean(df, load_cols)
        df[f"cpu_load_max{suffix}"]  = _safe_max(df, load_cols)
        df[f"cpu_freq_mean{suffix}"] = _safe_mean(df, freq_cols)

        # GPU aggregates across GPCs
        gpu_load_cols = [c + suffix for c in _GPU_LOAD_COLS[platform]]
        gpu_freq_cols = [c + suffix for c in _GPU_FREQ_COLS[platform]]
        df[f"gpu_load{suffix}"] = _safe_mean(df, gpu_load_cols)
        df[f"gpu_freq{suffix}"] = _safe_mean(df, gpu_freq_cols)

        # Thermal
        df[f"gpu_temp_c{suffix}"] = _passthrough(df, f"gpu_temp{suffix}")
        df[f"cpu_temp_c{suffix}"] = _passthrough(df, f"cpu_temp{suffix}")
        temp_zone_cols = [z + suffix for z in _TEMP_ZONES[platform]]
        df[f"temp_max_c{suffix}"] = _safe_max(df, temp_zone_cols)

        # Memory
        df[f"ram_used_mb{suffix}"] = _passthrough(df, f"ram_used{suffix}")

        # Power
        gpu_p = _GPU_POWER_COL[platform] + suffix
        cpu_p = _CPU_POWER_COL[platform] + suffix
        sys_p = _5V0_POWER_COL[platform] + suffix
        df[f"gpu_power_mw{suffix}"]   = _passthrough(df, gpu_p)
        df[f"cpu_power_mw{suffix}"]   = _passthrough(df, cpu_p)
        df[f"sys_power_mw{suffix}"]   = _passthrough(df, sys_p)
        df[f"total_power_mw{suffix}"] = _safe_sum(df, [gpu_p, cpu_p, sys_p])

        if _BOARD_POWER_COL[platform] is not None:
            df[f"board_power_mw{suffix}"] = _passthrough(
                df, _BOARD_POWER_COL[platform] + suffix
            )
        else:
            df[f"board_power_mw{suffix}"] = pd.Series(np.nan, index=df.index)

        # NVML — Thor only
        df[f"nvml_gpu_util{suffix}"] = _passthrough(df, f"gpu_util{suffix}")
        df[f"nvml_mem_util{suffix}"] = _passthrough(df, f"mem_util{suffix}")

        # EMC load — JetPack 5 (Xavier) tegrastats can report >100% (we have
        # observed up to ~107% on memory-bound 7--8B prompts). This is a JP5
        # reporting artifact: the EMC controller's reported utilization can
        # briefly exceed the supplied frequency budget. Orin (JP6) and Thor
        # (JP7) cap at 100% in our data. We clip the canonical view at 100
        # so that "saturated" reads as 100%. Raw per-sample EMC values are
        # preserved upstream in the parsed TGS CSV and the unprocessed
        # UNIFIED aggregates; this clip applies only after normalization.
        if f"emc_load{suffix}" in df.columns:
            df[f"emc_load{suffix}"] = df[f"emc_load{suffix}"].clip(
                lower=0, upper=100
            )

    return df
