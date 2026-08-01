# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Generate the S1 sensitivity sweep prompt set using RULER.

S1 tests input-length sensitivity of prefill behavior. We generate 30 prompts
per input-length tier, balanced across 3 RULER task families (10 each) for
prompt-shape diversity. The output is a single JSONL in the Hydra profiler
input format (`{"key": int, "prompt": str}`) plus a sidecar CSV recording the
target length, task family, and actual tokenized length for every prompt.

Configuration:
  - Target input lengths (tokens): 1000, 3000, 5000.
  - Tasks: niah_single_1, niah_multikey_2, vt (10 prompts each per length).
    niah_single_1 uses a noise haystack which gives tight length control
    (<1% spread); the essay-haystack variants underestimate target lengths
    because the Paul-Graham-essay paragraphs are too coarse a granularity
    for the binary search at our target tiers.
  - Tokenizer for RULER's length budgeting: Llama-3.2-1B-Instruct (the
    smallest Llama in our sweep). The actual token count is verified
    per-model in scripts/verify_sensitivity_token_counts.py.

See findings/sensitivity_sweep_design.md for the design rationale.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Locations.
HYDRA_ROOT = Path(__file__).resolve().parent.parent
RULER_DATA = HYDRA_ROOT / "external" / "RULER" / "scripts" / "data"
DEFAULT_TOKENIZER = (
    "/media/amir/data/Xavier_BackUp_After_Hydra_IISWC/jetson-containers/data/"
    "models/huggingface/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6/"
)

# Per the design note. RULER's `max_seq_length` is total tokens including its
# own answer budget; we add 128 to the target input to leave room for that.
INPUT_LENGTHS = [1000, 3000, 5000]
TASKS = ["niah_single_1", "niah_multikey_2", "vt"]
SAMPLES_PER_TASK_PER_LENGTH = 10
RULER_ANSWER_BUDGET = 128  # niah default; vt is 30 but we use the same upper.


def run_ruler(task: str, target_input: int, num_samples: int,
              tokenizer_path: str, work_dir: Path) -> Path:
    """Invoke RULER prepare.py for one (task, length). Returns the JSONL path."""
    out_dir = work_dir / f"{task}_{target_input}"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seq = target_input + RULER_ANSWER_BUDGET
    cmd = [
        "python3", "prepare.py",
        "--save_dir", str(out_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(max_seq),
        "--model_template_type", "base",
        "--num_samples", str(num_samples),
    ]
    print(f"[ruler] {task} target_input={target_input} num={num_samples}")
    subprocess.run(cmd, cwd=RULER_DATA, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_dir / task / "validation.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-jsonl", type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s1_ruler.jsonl")
    ap.add_argument("--out-csv", type=Path,
                    default=HYDRA_ROOT / "dataset" / "sensitivity_s1_ruler.token_counts.csv")
    ap.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER)
    ap.add_argument("--work-dir", type=Path,
                    default=Path("/tmp/ruler_s1_workdir"))
    args = ap.parse_args()

    if not Path(args.tokenizer_path).exists():
        sys.exit(f"tokenizer path not found: {args.tokenizer_path}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Import the tokenizer once to verify actual token counts inline.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rows: list[dict] = []
    key = 1
    for target_input in INPUT_LENGTHS:
        for task in TASKS:
            ruler_jsonl = run_ruler(task, target_input,
                                    SAMPLES_PER_TASK_PER_LENGTH,
                                    args.tokenizer_path, args.work_dir)
            with open(ruler_jsonl) as f:
                for line in f:
                    d = json.loads(line)
                    text = d["input"]
                    actual = len(tok.encode(text))
                    rows.append({
                        "key": key,
                        "task": task,
                        "target_input_tokens": target_input,
                        "actual_input_tokens_llama_3_2_1b": actual,
                        "prompt": text,
                    })
                    key += 1

    # Write Hydra-format JSONL.
    with open(args.out_jsonl, "w") as out:
        for r in rows:
            out.write(json.dumps({"key": r["key"], "prompt": r["prompt"]}) + "\n")

    # Write sidecar CSV with metadata (task, target length, actual length).
    with open(args.out_csv, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["key", "task", "target_input_tokens",
                    "actual_input_tokens_llama_3_2_1b"])
        for r in rows:
            w.writerow([r["key"], r["task"],
                        r["target_input_tokens"],
                        r["actual_input_tokens_llama_3_2_1b"]])

    print()
    print("=" * 56)
    print(f"  Wrote {len(rows)} prompts to {args.out_jsonl}")
    print(f"  Metadata sidecar: {args.out_csv}")
    print("=" * 56)
    # Summary per (task, target_length).
    from collections import defaultdict
    g: dict[tuple, list] = defaultdict(list)
    for r in rows:
        g[(r["task"], r["target_input_tokens"])].append(
            r["actual_input_tokens_llama_3_2_1b"]
        )
    print()
    print(f"  {'task':<22} {'target':<8} {'n':<4} {'min':<6} {'mean':<6} {'max':<6} {'std%':<6}")
    for (task, target), vals in sorted(g.items()):
        n = len(vals)
        mn = min(vals); mx = max(vals)
        mean = sum(vals) / n
        # Spread as %.
        spread = (mx - mn) / target * 100.0
        print(f"  {task:<22} {target:<8d} {n:<4d} {mn:<6d} {mean:<6.0f} {mx:<6d} {spread:<6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
