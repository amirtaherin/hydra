# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Load UNIFIED_*.csv files into a single long-form DataFrame.

Walks `<results-root>/<platform>/results_<backend>_<platform>/UNIFIED_*.csv`,
parses each filename for `(model, dtype)`, attaches `(platform, backend)`
from the directory, applies `analysis.schema.normalize` for canonical
telemetry column names, and concatenates everything into one DataFrame
indexed by (platform, backend, model, dtype, prompt_number).

Layout assumed:

    <results-root>/
        orin/   results_hf_orin/        UNIFIED_<model>_torch.bfloat16_.csv
                results_llamacpp_orin/  UNIFIED_<model>_<F16|Q8_0|Q6_K|Q4_K_M>_.csv
        xavier/ results_hf_xavier/      ...
                results_llamacpp_xavier/...
        thor/   results_hf_thor/        ...
                results_llamacpp_thor/  ...

Both `<results-root>` and the project's `results/` symlink point at the same
directory tree on the user's machine.

Filename conventions:
    UNIFIED_<model_name>_<dtype>_.csv
    where dtype is one of: torch.bfloat16 (HF) or {F16, Q8_0, Q6_K, Q4_K_M} (llama.cpp).

Memory: ~190 files × 541 rows × ~400 cols ≈ 40M cells, roughly 1.5 GB if all
columns kept as float64. To stay tractable for paper-figure scripts we keep
the full schema by default; pass `columns=["..."]` to subset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis.schema import normalize


_PLATFORMS = ("orin", "xavier", "thor")
_BACKENDS = ("hf", "llamacpp")

# UNIFIED_<model>_<dtype>_.csv  (note trailing underscore before .csv)
_FILENAME_RE = re.compile(r"^UNIFIED_(?P<model>.+)_(?P<dtype>[^_]+(?:_[^_]+)*)_\.csv$")


def _parse_filename(name: str) -> tuple[str, str] | None:
    """Return (model, dtype) parsed from a UNIFIED filename, or None if no match.

    Filenames look like:
        UNIFIED_Llama-3.2-1B_torch.bfloat16_.csv  -> ("Llama-3.2-1B", "torch.bfloat16")
        UNIFIED_Qwen2.5-7B_Q4_K_M_.csv            -> ("Qwen2.5-7B",   "Q4_K_M")
        UNIFIED_gemma-2B_F16_.csv                 -> ("gemma-2B",     "F16")

    The model name segment may contain dashes and dots (e.g. "Llama-3.2-1B",
    "Phi-3.5-mini", "Qwen2.5-1.5B"). The dtype segment may contain underscores
    (e.g. "Q4_K_M"). We disambiguate by anchoring on the trailing `_.csv` and
    the known dtype tokens.
    """
    if not name.startswith("UNIFIED_") or not name.endswith("_.csv"):
        return None
    body = name[len("UNIFIED_"):-len("_.csv")]
    # Match by known dtype suffix list to avoid greedy-split ambiguity on Q4_K_M.
    for dtype in ("torch.bfloat16", "torch.float16",
                  "F16", "Q8_0", "Q6_K", "Q4_K_M"):
        if body.endswith("_" + dtype):
            return body[:-len("_" + dtype)], dtype
    return None


def discover_files(results_root: Path) -> list[dict]:
    """Walk results-root and return a list of dicts:
        {path, platform, backend, model, dtype}

    Skips directories that don't match the convention, and files that don't
    parse. Does not read any CSVs.
    """
    out: list[dict] = []
    for platform in _PLATFORMS:
        plat_dir = results_root / platform
        if not plat_dir.is_dir():
            continue
        for backend in _BACKENDS:
            backend_dir = plat_dir / f"results_{backend}_{platform}"
            if not backend_dir.is_dir():
                continue
            for csv in sorted(backend_dir.glob("UNIFIED_*.csv")):
                parsed = _parse_filename(csv.name)
                if parsed is None:
                    continue
                model, dtype = parsed
                out.append({
                    "path":     csv,
                    "platform": platform,
                    "backend":  backend,
                    "model":    model,
                    "dtype":    dtype,
                })
    return out


def load_one(meta: dict, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one UNIFIED CSV, normalize, attach identifier columns."""
    df = pd.read_csv(meta["path"])
    df = normalize(df, platform=meta["platform"])
    df["platform"] = meta["platform"]
    df["backend"]  = meta["backend"]
    df["model"]    = meta["model"]
    df["dtype"]    = meta["dtype"]
    if columns is not None:
        ident = ["platform", "backend", "model", "dtype", "prompt_number"]
        keep = list(dict.fromkeys(list(ident) + list(columns)))
        df = df[[c for c in keep if c in df.columns]]
    return df


def load_all(
    results_root: Path,
    *,
    platforms: Iterable[str] | None = None,
    backends: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load every UNIFIED file under results-root into one long-form DataFrame.

    Parameters
    ----------
    results_root : Path
        Top of the results tree (e.g. /media/amir/data/hydra-results).
    platforms : iterable of {"orin","xavier","thor"} or None
        If given, restrict to these. Default: all.
    backends : iterable of {"hf","llamacpp"} or None
        If given, restrict to these. Default: all.
    columns : sequence of column names or None
        If given, only keep these columns plus the identifier columns. Useful
        when memory matters and you only need a few telemetry channels.

    Returns
    -------
    pd.DataFrame
        Long-form: one row per (platform, backend, model, dtype, prompt_number),
        with all canonical and platform-specific columns intact.
    """
    metas = discover_files(results_root)
    if platforms is not None:
        platforms = set(platforms)
        metas = [m for m in metas if m["platform"] in platforms]
    if backends is not None:
        backends = set(backends)
        metas = [m for m in metas if m["backend"] in backends]
    if not metas:
        raise FileNotFoundError(f"No UNIFIED files found under {results_root}")

    frames = [load_one(m, columns=columns) for m in metas]
    return pd.concat(frames, ignore_index=True)
