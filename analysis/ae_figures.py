# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Render exactly the six data-derived figures used in the IISWC 2026 paper.

Artifact-evaluation companion to `analysis/main.py` (which drives the wider
exploratory plot families). This module invokes only the figure functions
whose outputs appear in the paper, with the paper's filenames:

  Fig. 1  phase_motivation.pdf               (plots/motivation.py)
  Fig. 3  perf_e2e_latency_grid.pdf          (plots/performance.py)
  Fig. 4  perf_q4_throughput_grouped.pdf     (plots/performance.py)
  Fig. 5  perf_phase_breakdown_thor_qwen.pdf (plots/performance.py)
  Fig. 6  sensitivity_ttft_throughput.pdf    (plots/sensitivity.py)
  Fig. 7  system_cpu_per_core_qwen7b.pdf     (plots/utilization.py)

(Fig. 2 is the hand-drawn pipeline diagram and has no data dependency.)

Usage:
    python3 -m analysis.ae_figures \
        --results-root /path/to/hydra-results --out figures_ae/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis import style
from analysis.loader import load_all
from analysis import (motivation, performance, sensitivity,
                      utilization)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path,
                    default=Path("/media/amir/data/hydra-results"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style.apply_paper_defaults()

    print(f"Loading main corpus from {args.results_root} ...")
    df = load_all(args.results_root)
    print(f"  {len(df)} main-corpus records")

    print("Fig. 1  phase_motivation ...")
    motivation.phase_motivation(df, args.out)

    print("Figs. 3-5  perf_e2e / q4_throughput / phase_breakdown ...")
    performance.perf_e2e_latency_grid(df, args.out)
    performance.perf_q4_throughput_grouped(df, args.out)
    performance.perf_phase_breakdown_thor_qwen(df, args.out)

    print("Fig. 7  system_cpu_per_core_qwen7b ...")
    utilization.system_cpu_per_core_qwen7b(df, args.out)

    print("Fig. 6  sensitivity_ttft_throughput ...")
    sens = sensitivity.load_sensitivity(args.results_root)
    print(f"  {len(sens)} sensitivity records")
    sensitivity.render_all(sens, args.out)

    made = sorted(p.name for p in args.out.glob("*.pdf"))
    print(f"Done - {len(made)} PDFs in {args.out}:")
    for name in made:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
