# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Verify actual token counts of the sensitivity-sweep prompts on every
target model's tokenizer (not just the Llama-3.2-1B used to generate them).

For each S1 and S2 prompt, we tokenize with all six tokenizers (Llama-3.2-
1B/3B, Llama-3.1-8B, Qwen2.5-1.5B/3B/7B) and report the mean and spread
per target tier. This catches any per-model divergence in actual input
length before we burn Thor runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HYDRA_ROOT = Path(__file__).resolve().parent.parent
MODELS_CACHE = ("/media/amir/data/Xavier_BackUp_After_Hydra_IISWC/"
                "jetson-containers/data/models/huggingface")

TOKENIZER_PATHS = {
    "Llama-3.2-1B": f"{MODELS_CACHE}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/",
    "Llama-3.2-3B": f"{MODELS_CACHE}/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/",
    "Llama-3.1-8B": f"{MODELS_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/",
    "Qwen2.5-1.5B": f"{MODELS_CACHE}/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/",
    "Qwen2.5-3B":   f"{MODELS_CACHE}/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/",
    "Qwen2.5-7B":   f"{MODELS_CACHE}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/",
}


def load_tokenizers(names: list[str]) -> dict:
    from transformers import AutoTokenizer
    out = {}
    for n in names:
        out[n] = AutoTokenizer.from_pretrained(TOKENIZER_PATHS[n])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-jsonl", type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s1_ruler.jsonl")
    ap.add_argument("--s1-meta",  type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s1_ruler.token_counts.csv")
    ap.add_argument("--s2-jsonl", type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s2_ifeval.jsonl")
    ap.add_argument("--out-csv",  type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_token_counts_all_models.csv")
    args = ap.parse_args()

    print("Loading 6 tokenizers...", file=sys.stderr)
    tok = load_tokenizers(list(TOKENIZER_PATHS))
    print("done.", file=sys.stderr)

    # Load S1 metadata for task / target tier per key.
    s1_meta: dict[int, dict] = {}
    if args.s1_meta.exists():
        with open(args.s1_meta) as f:
            for row in csv.DictReader(f):
                s1_meta[int(row["key"])] = row

    rows = []
    # --- S1 ---
    if args.s1_jsonl.exists():
        with open(args.s1_jsonl) as f:
            for line in f:
                d = json.loads(line)
                key = d["key"]
                meta = s1_meta.get(key, {})
                base = {
                    "sweep": "S1",
                    "key": key,
                    "task": meta.get("task", ""),
                    "target_tokens": meta.get("target_input_tokens", ""),
                }
                for name, t in tok.items():
                    base[name] = len(t.encode(d["prompt"]))
                rows.append(base)
    # --- S2 ---
    if args.s2_jsonl.exists():
        with open(args.s2_jsonl) as f:
            for line in f:
                d = json.loads(line)
                base = {
                    "sweep": "S2",
                    "key": d["key"],
                    "task": "ifeval",
                    "target_tokens": "",
                }
                for name, t in tok.items():
                    base[name] = len(t.encode(d["prompt"]))
                rows.append(base)

    # Write CSV.
    cols = ["sweep", "key", "task", "target_tokens"] + list(TOKENIZER_PATHS)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- Summary print ----
    # S1: by (target_tokens, model)
    print()
    print("=" * 88)
    print("S1: per-model token counts at each target tier")
    print("=" * 88)
    s1_by_tier: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["sweep"] != "S1":
            continue
        for m in TOKENIZER_PATHS:
            s1_by_tier[r["target_tokens"]][m].append(r[m])
    print(f"{'target':<8} {'model':<14} {'n':<3} {'min':<6} {'mean':<6} {'max':<6} {'%off':<6}")
    for target in sorted(s1_by_tier, key=lambda x: int(x) if x else 0):
        for m in TOKENIZER_PATHS:
            vals = s1_by_tier[target][m]
            n = len(vals)
            mn, mx = min(vals), max(vals)
            mean = sum(vals) / n
            pct_off = (mean - int(target)) / int(target) * 100 if target else 0
            print(f"{target:<8} {m:<14} {n:<3} {mn:<6} {mean:<6.0f} {mx:<6} {pct_off:<+6.1f}")
        print()

    # S2: single tier
    print("=" * 88)
    print("S2: per-model input token counts (tight band, 30 prompts)")
    print("=" * 88)
    print(f"{'model':<14} {'n':<3} {'min':<6} {'mean':<6} {'max':<6}")
    s2_vals: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["sweep"] != "S2":
            continue
        for m in TOKENIZER_PATHS:
            s2_vals[m].append(r[m])
    for m in TOKENIZER_PATHS:
        vals = s2_vals[m]
        if not vals:
            continue
        print(f"{m:<14} {len(vals):<3} {min(vals):<6} {sum(vals)/len(vals):<6.0f} {max(vals):<6}")

    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
