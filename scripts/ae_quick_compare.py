# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Compare fresh ae_quick spot-check measurements against the released
corpus reference values (expected_results/ae_quick_reference.csv).

Reports, per (backend, dtype) cell: decode power (W), inter-token latency
(ms), energy per token (mJ), and generation throughput (tok/s), side by
side with the corpus means, plus the percentage deviation.

Pass criteria (printed at the end):
  * Qualitative invariants (hold exactly, run-to-run):
      - llama.cpp Q6_K decode power > Q8_0 decode power
      - Q4_K_M lowest mJ/token among the llama.cpp formats
      - llama.cpp ITL < HF bf16 ITL for this model class
  * Quantitative: values typically within +/-15% of corpus means
    (thermal state and background load shift absolute numbers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import schema  # noqa: E402

TOL_PCT = 15.0


def load_cell(results_root: Path, platform: str, backend: str,
              dtype: str) -> pd.DataFrame | None:
    d = results_root / f"results_{backend}_{platform}"
    hits = list(d.glob(f"UNIFIED_*_{dtype}_.csv"))
    if not hits:
        return None
    df = pd.read_csv(hits[0], low_memory=False)
    return schema.normalize(df, platform)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--platform", required=True,
                    choices=["xavier", "orin", "thor"])
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    ref = ref[ref.platform == args.platform]
    if ref.empty:
        print(f"No reference rows for platform {args.platform}")
        return 1

    print(f"\n{'cell':<22}{'metric':<12}{'measured':>10}{'corpus':>10}"
          f"{'dev%':>8}")
    print("-" * 62)
    measured: dict[tuple[str, str], dict[str, float]] = {}
    for _, r in ref.iterrows():
        # NOTE: r["dtype"] not r.dtype - the attribute is pandas' own dtype.
        backend, dtype = r["backend"], r["dtype"]
        note = str(r.get("note", "") or "")
        if note == "nan":
            note = ""
        sub = load_cell(args.results_root, args.platform, backend, dtype)
        label = f"{backend}/{dtype}"
        if sub is None or sub.empty:
            print(f"{label:<22}{'MISSING (run ae_quick.sh first?)'}")
            continue
        power = ((sub["gpu_power_mw_decode_mean"]
                  + sub["sys_power_mw_decode_mean"]) / 1000.0).mean()
        itl = (sub["mean_inter_token_latency"] * 1000.0).mean()
        mj = power * itl
        tps = sub["generation_tokens_per_sec"].mean()
        measured[(backend, dtype)] = {"power": power, "itl": itl,
                                          "mj": mj, "tps": tps}
        for metric, meas, refv in [
                ("power (W)", power, r["decode_power_w_mean"]),
                ("ITL (ms)", itl, r["itl_ms_mean"]),
                ("mJ/token", mj, r["mj_per_token_mean"]),
                ("tok/s", tps, r["gen_tok_per_s_mean"])]:
            dev = 100.0 * (meas - refv) / refv if refv else float("nan")
            if note:
                flag = "  (+)" if abs(dev) > TOL_PCT else ""
            else:
                flag = "" if abs(dev) <= TOL_PCT else "  <-- outside +/-15%"
            print(f"{label:<22}{metric:<12}{meas:>10.1f}{refv:>10.1f}"
                  f"{dev:>+8.1f}{flag}")
        if note:
            print(f"{'':<22}(+) reference cell: {note}")
        print()

    # ---- Qualitative invariants ----
    print("Qualitative invariants:")
    ok = True

    def check(name: str, cond: bool | None) -> None:
        nonlocal ok
        if cond is None:
            print(f"  [SKIP] {name} (missing cells)")
            return
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    q8 = measured.get(("llamacpp", "Q8_0"))
    q6 = measured.get(("llamacpp", "Q6_K"))
    q4 = measured.get(("llamacpp", "Q4_K_M"))
    hf = measured.get(("hf", "torch.bfloat16"))
    check("Q6_K decode power > Q8_0 (bit-width non-monotonicity)",
          None if not (q6 and q8) else q6["power"] > q8["power"])
    check("Q4_K_M lowest mJ/token among llama.cpp formats",
          None if not (q4 and q6 and q8)
          else q4["mj"] < min(q6["mj"], q8["mj"]))
    check("llama.cpp ITL < HF bf16 ITL (runtime-overhead gap)",
          None if not (hf and q8) else q8["itl"] < hf["itl"])

    if not measured:
        print("\nResult: NO MEASUREMENTS FOUND")
        return 1
    print("\nResult:", "INVARIANTS PASS" if ok else "INVARIANT FAILURE")
    print("Quantitative deviations within +/-15% are expected; see"
          " AE_README.md.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
