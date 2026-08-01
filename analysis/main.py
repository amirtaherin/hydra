# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""CLI driver for the Hydra analysis pipeline (IISWC 2026 figure set).

Loads UNIFIED CSVs, applies canonical-schema normalization, dispatches to one
or more plot families, and writes PDFs to the output directory.

The paper's own figures and tables are driven by `analysis.ae_figures`
and `analysis.efficiency_tables`; this CLI renders the additional
exploratory plot families over the same corpus.

Usage:
    python3 -m analysis.main --all --out figures/extras/
    python3 -m analysis.main --plots latency,throughput \
                             --platforms orin,thor --out figures/test/

See `analysis/README.md` for the design rationale and plot inventory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis import style
from analysis.loader import load_all
from analysis.extras import (
    backend_compare,
    bottleneck,
    cpu,
    cross_gen,
    distributions,
    efficiency,
    gpu,
    headline,
    latency,
    memory,
    memory_pressure,
    power,
    quantization,
    scaling,
    thermal,
    throughput,
)


PLOT_FAMILIES = {
    # Single-config families (per platform, backend, dtype).
    "latency":    latency.render,
    "throughput": throughput.render,
    "memory":     memory.render,
    "cpu":        cpu.render,
    "gpu":        gpu.render,
    "thermal":    thermal.render,
    "power":      power.render,
    # Cross-platform / cross-backend / quantization families.
    "headline":         headline.render,
    "quantization":     quantization.render,
    "backend_compare":  backend_compare.render,
    "efficiency":       efficiency.render,
    "memory_pressure":  memory_pressure.render,
    "cross_gen":        cross_gen.render,
    "bottleneck":       bottleneck.render,
    "distributions":    distributions.render,
    "scaling":          scaling.render,
}

def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render Hydra analysis figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("/media/amir/data/hydra-results"),
        help="Top of the results tree (where <platform>/results_<backend>_<platform>/ live).",
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="Directory to write figure PDFs into.")
    ap.add_argument("--all", action="store_true",
                    help="Render every plot family.")
    ap.add_argument("--plots", type=_parse_csv, default=[],
                    help="Comma-separated list of plot family names "
                         "(e.g., 'latency,throughput').")
    ap.add_argument("--platforms", type=_parse_csv, default=None,
                    help="Restrict to these platforms. Default: all available.")
    ap.add_argument("--backends", type=_parse_csv, default=None,
                    help="Restrict to these backends. Default: all.")
    args = ap.parse_args()

    if not args.all and not args.plots:
        ap.error("Specify --all or --plots.")

    families = list(args.plots)
    if args.all:
        families.extend(PLOT_FAMILIES)
    families = list(dict.fromkeys(families))  # de-dupe, preserve order

    unknown = [f for f in families if f not in PLOT_FAMILIES]
    if unknown:
        ap.error(f"Unknown plot families: {unknown}. "
                 f"Available: {sorted(PLOT_FAMILIES)}")

    style.apply_paper_defaults()

    print(f"Loading UNIFIED data from {args.results_root}")
    df = load_all(
        args.results_root,
        platforms=args.platforms,
        backends=args.backends,
    )
    print(f"  {len(df):,} prompt-level rows across "
          f"{df.groupby(['platform','backend','model','dtype']).ngroups} cells")

    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    for fam in families:
        print(f"\nRendering family: {fam}")
        paths = PLOT_FAMILIES[fam](df, args.out, platforms=args.platforms)
        for p in paths:
            print(f"  wrote {p}")
        written.extend(paths)

    print(f"\nDone — {len(written)} figures in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
