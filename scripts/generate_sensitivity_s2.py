# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Generate the S2 sensitivity sweep prompt set from IFEval.

S2 tests output-length sensitivity of decode behavior. To isolate that
variable cleanly, we hold input length tightly controlled: we filter IFEval
to prompts in a narrow input-length band (default 40-60 tokens, the median
+/- 10) and take the first 30 by line order. This removes
KV-cache-starting-size as a confound while staying within IFEval (consistent
with the main results).

Output:
  inputs/prompts/sensitivity_s2_ifeval.jsonl       - Hydra profiler input format
  inputs/prompts/sensitivity_s2_ifeval.line_ids.txt - selected IFEval `key` values

See findings/sensitivity_sweep_design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HYDRA_ROOT = Path(__file__).resolve().parent.parent
IFEVAL_JSONL = HYDRA_ROOT / "dataset" / "input_data.jsonl"
DEFAULT_TOKENIZER = (
    "/media/amir/data/Xavier_BackUp_After_Hydra_IISWC/jetson-containers/data/"
    "models/huggingface/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6/"
)

# Per the design note: tight band around the IFEval median (~47 tokens).
DEFAULT_MIN_TOKENS = 40
DEFAULT_MAX_TOKENS = 60
N_PROMPTS = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-jsonl", type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s2_ifeval.jsonl")
    ap.add_argument("--out-ids",   type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s2_ifeval.line_ids.txt")
    ap.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER)
    ap.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    args = ap.parse_args()

    if not IFEVAL_JSONL.exists():
        sys.exit(f"IFEval not found: {IFEVAL_JSONL}")
    if not Path(args.tokenizer_path).exists():
        sys.exit(f"tokenizer not found: {args.tokenizer_path}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    selected: list[dict] = []
    with open(IFEVAL_JSONL) as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"]
            n_tok = len(tok.encode(prompt))
            if args.min_tokens <= n_tok <= args.max_tokens:
                selected.append({
                    "key": d["key"],
                    "prompt": prompt,
                    "n_tokens": n_tok,
                })
                if len(selected) >= args.n:
                    break

    if len(selected) < args.n:
        sys.exit(f"only {len(selected)} prompts in [{args.min_tokens}, "
                 f"{args.max_tokens}] tokens; need {args.n}")

    # Write Hydra-format JSONL. Re-key 1..N to keep `prompt_number` stable.
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as out:
        for i, r in enumerate(selected, start=1):
            out.write(json.dumps({"key": i, "prompt": r["prompt"]}) + "\n")

    # Record source IFEval keys for traceability.
    with open(args.out_ids, "w") as out:
        out.write("# IFEval keys selected for S2 (output-length sensitivity)\n")
        out.write(f"# Filter: {args.min_tokens} <= n_tokens <= {args.max_tokens}, "
                  f"first {args.n} by line order\n")
        out.write("# Tokenizer: Llama-3.2-1B-Instruct\n")
        for r in selected:
            out.write(f"{r['key']}\t{r['n_tokens']}\n")

    n_tokens = [r["n_tokens"] for r in selected]
    print(f"Wrote {len(selected)} prompts to {args.out_jsonl}")
    print(f"  source-key list: {args.out_ids}")
    print(f"  input-length band: {min(n_tokens)}--{max(n_tokens)} tokens "
          f"(mean {sum(n_tokens)/len(n_tokens):.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
