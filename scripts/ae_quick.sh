#!/usr/bin/env bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# ==========================================================================
# IISWC 2026 Artifact Evaluation - on-device measurement spot-check.
#
# Runs a small, representative subset of the paper's measurements on the
# Jetson board it is executed on, then compares the fresh measurements
# against the released corpus (expected_results/ae_quick_reference.csv).
#
# Coverage rationale: the selected cells (Qwen2.5-1.5B under HF bf16 and
# llama.cpp Q8_0 / Q6_K / Q4_K_M) span both instrumented backends, the
# quantized-format power non-monotonicity (Q6_K > Q8_0), and - when run
# on more than one board - cross-generation behavior, while completing in
# well under an hour per board.
#
# Expected match: qualitative invariants hold exactly (see compare step);
# quantitative values typically fall within +/-15% of corpus means
# depending on thermal state and background load.
#
# Usage:   bash scripts/ae_quick.sh          # ~20 prompts x 4 configs
#          N_PROMPTS=10 bash scripts/ae_quick.sh   # faster variant
# Tip: run inside tmux/screen; the run takes ~20-40 min per board.
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_NAME="Qwen2.5-7B"
N_PROMPTS="${N_PROMPTS:-20}"
MAX_NEW_TOKENS=500
INPUT_DATA="$PROJECT_ROOT/inputs/prompts/input_data.jsonl"
MODELS_JSON="$PROJECT_ROOT/inputs/models/models.json"
PROFILER_BIN="$PROJECT_ROOT/inference/llamacpp_profiler/build/cpp_api_pipeline"

# --- Detect platform ---
HOSTNAME=$(hostname)
case "$HOSTNAME" in
    *[Oo]rin*)   PLATFORM="orin" ;;
    *[Xx]avier*) PLATFORM="xavier" ;;
    *[Tt]hor*)   PLATFORM="thor" ;;
    *) echo "ERROR: unrecognized hostname '$HOSTNAME' (need xavier/orin/thor)"; exit 1 ;;
esac

# Preflight: fail fast if the llama.cpp profiler is not built yet, before
# spending ~15 min in the HF phase.
[ -f "$PROFILER_BIN" ] || { echo "ERROR: llama.cpp profiler not built."; \
  echo "Build it with:  bash scripts/build_llamacpp_profiler.sh"; \
  echo "then re-run ae_quick.sh."; exit 1; }

OUT_ROOT="$PROJECT_ROOT/ae_quick_results/$PLATFORM"
HF_DIR="$OUT_ROOT/results_hf_$PLATFORM"
LC_DIR="$OUT_ROOT/results_llamacpp_$PLATFORM"
mkdir -p "$HF_DIR" "$LC_DIR"

# --- Single-model registry entry ---
TMP_JSON=$(mktemp /tmp/ae_quick_model_XXXXXX.json)
python3 -c "
import json
with open('$MODELS_JSON') as f:
    data = json.load(f)
entry = [m for m in data['models'] if m['name'] == '$MODEL_NAME']
assert entry, 'model $MODEL_NAME not in registry'
print(json.dumps({'models': entry}, indent=2))
" > "$TMP_JSON"

echo "=============================================="
echo "  Hydra AE quick spot-check - $PLATFORM"
echo "  model=$MODEL_NAME prompts=$N_PROMPTS tokens=$MAX_NEW_TOKENS"
echo "=============================================="

echo "== [1/4] HuggingFace bf16 =="
python3 -m inference.hf_profiler.main \
    --input "$INPUT_DATA" --model-names "$TMP_JSON" \
    --dtype bfloat16 --max-new-tokens $MAX_NEW_TOKENS \
    --max-prompts "$N_PROMPTS" --result-dir "$HF_DIR"

echo "== [2/4] llama.cpp Q8_0 / Q6_K / Q4_K_M =="
for DTYPE in Q8_0 Q6_K Q4_K_M; do
    echo "--- $DTYPE (download if missing) ---"
    "$PROFILER_BIN" --model-names "$TMP_JSON" --dtype "$DTYPE" --download-only
    echo "--- $DTYPE ---"
    "$PROFILER_BIN" \
        --input "$INPUT_DATA" --model-names "$TMP_JSON" \
        --dtype "$DTYPE" --max-new-tokens $MAX_NEW_TOKENS --n-ctx 2048 \
        --max-prompts "$N_PROMPTS" --result-dir "$LC_DIR/"
done
rm -f "$TMP_JSON"

echo "== [3/4] Fusing timing + telemetry (unifier) =="
python3 -m analysis.unifier.data_unifier_parallel --platform "$PLATFORM" --results-dir "$HF_DIR" --force
python3 -m analysis.unifier.data_unifier_parallel --platform "$PLATFORM" --results-dir "$LC_DIR" --force

echo "== [4/4] Comparing against released corpus =="
python3 scripts/ae_quick_compare.py \
    --platform "$PLATFORM" --results-root "$OUT_ROOT" \
    --reference "$PROJECT_ROOT/expected_results/ae_quick_reference.csv"
