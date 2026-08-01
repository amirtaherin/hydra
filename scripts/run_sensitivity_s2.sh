#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# S2 sensitivity sweep — output-length sensitivity on Thor.
#
# Runs the HF (bfloat16) and llama.cpp (Q4_K_M) profilers over the
# 6-model IFEval tight-band prompt set (inputs/prompts/sensitivity_s2_ifeval.jsonl,
# 30 prompts) at THREE different max_new_tokens targets: 1000, 3000, 5000.
# Input length is held tight (40-60 tokens) so output length is the
# isolated variable.
#
# Output layout (under results_sensitivity/):
#   S2_<platform>_hf_bf16_out1000/
#   S2_<platform>_hf_bf16_out3000/
#   S2_<platform>_hf_bf16_out5000/
#   S2_<platform>_llamacpp_Q4_K_M_out1000/
#   S2_<platform>_llamacpp_Q4_K_M_out3000/
#   S2_<platform>_llamacpp_Q4_K_M_out5000/
#
# Designed to be run on Thor. Resume-safe (per-model skip on complete CSVs).
#
# Usage:
#   bash scripts/run_sensitivity_s2.sh                    # all backends, all output tiers
#   bash scripts/run_sensitivity_s2.sh --backend hf
#   bash scripts/run_sensitivity_s2.sh --out-tokens 5000  # one tier only

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_JSON="$PROJECT_ROOT/inputs/models/models_sensitivity.json"
INPUT_DATA="$PROJECT_ROOT/inputs/prompts/sensitivity_s2_ifeval.jsonl"
HF_DTYPE="bfloat16"
HF_DTYPE_CSV="torch.bfloat16"
LLAMACPP_DTYPE="Q4_K_M"

# --- Defaults ---
BACKEND="all"
OUT_TOKEN_TIERS=(1000 3000 5000)

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)    BACKEND="$2"; shift 2 ;;
        --out-tokens) OUT_TOKEN_TIERS=("$2"); shift 2 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done

# --- Detect platform ---
HOSTNAME=$(hostname)
if echo "$HOSTNAME" | grep -qi "thor"; then
    PLATFORM="thor"
elif echo "$HOSTNAME" | grep -qi "orin"; then
    PLATFORM="orin"
elif echo "$HOSTNAME" | grep -qi "xavier"; then
    PLATFORM="xavier"
else
    PLATFORM="$HOSTNAME"
fi

RESULTS_ROOT="$PROJECT_ROOT/results_sensitivity"
mkdir -p "$RESULTS_ROOT"

echo "============================================"
echo "  Hydra S2 (output-length) — ${PLATFORM}"
echo "============================================"
echo "Prompt set:     $INPUT_DATA"
echo "Models JSON:    $MODELS_JSON"
echo "Out tiers:      ${OUT_TOKEN_TIERS[*]}"
echo "Backend(s):     $BACKEND"
echo ""

# --- Expected row count ---
EXPECTED_ROWS=$(wc -l < "$INPUT_DATA")
EXPECTED_LINES=$((EXPECTED_ROWS + 1))
echo "Prompts/tier:   $EXPECTED_ROWS"
echo ""

# --- Model names ---
MODEL_NAMES=$(python3 -c "
import json
with open('$MODELS_JSON') as f: data = json.load(f)
for m in data['models']: print(m['name'])
")
TOTAL=$(echo "$MODEL_NAMES" | wc -l)
echo "Models:         $TOTAL"
echo ""


# ============================================================
#   One (backend, out_tokens) sweep over all models
# ============================================================
run_one_tier() {
    local BACKEND_TAG="$1"     # hf | llamacpp
    local OUT_TOKENS="$2"
    local DTYPE_CSV="$3"        # used in CSV filename
    local RESULT_DIR="$RESULTS_ROOT/S2_${PLATFORM}_${BACKEND_TAG}_${DTYPE_CSV//torch./}_out${OUT_TOKENS}"
    # ^ strips the "torch." prefix from HF dtype tag so dir names stay short.
    mkdir -p "$RESULT_DIR"

    echo "--------------------------------------------"
    echo "  S2 ${BACKEND_TAG} out=${OUT_TOKENS}"
    echo "  -> $RESULT_DIR"
    echo "--------------------------------------------"

    local CURRENT=0 SKIPPED=0 COMPLETED=0 FAILED=0
    for MODEL_NAME in $MODEL_NAMES; do
        CURRENT=$((CURRENT + 1))
        local CSV_FILE="$RESULT_DIR/INFO_${MODEL_NAME}_${DTYPE_CSV}_.csv"

        if [ -f "$CSV_FILE" ]; then
            local LC=$(wc -l < "$CSV_FILE")
            if [ "$LC" -eq "$EXPECTED_LINES" ]; then
                echo "[$CURRENT/$TOTAL] SKIP $MODEL_NAME (complete)"
                SKIPPED=$((SKIPPED + 1)); continue
            else
                echo "[$CURRENT/$TOTAL] REDO $MODEL_NAME (partial: $((LC - 1))/$EXPECTED_ROWS)"
                rm -f "$CSV_FILE" "$RESULT_DIR/TGS_${MODEL_NAME}_${DTYPE_CSV}_.csv"
            fi
        else
            echo "[$CURRENT/$TOTAL] RUN  $MODEL_NAME"
        fi

        local TMP_JSON=$(mktemp /tmp/hydra_s2_XXXXXX.json)
        python3 -c "
import json
with open('$MODELS_JSON') as f: data = json.load(f)
for m in data['models']:
    if m['name'] == '$MODEL_NAME':
        print(json.dumps({'models':[m]}, indent=2)); break
" > "$TMP_JSON"

        local OK=0
        if [ "$BACKEND_TAG" = "hf" ]; then
            cd "$PROJECT_ROOT"
            python3 -m inference.hf_profiler.main \
                --input "$INPUT_DATA" \
                --model-names "$TMP_JSON" \
                --dtype "$HF_DTYPE" \
                --max-new-tokens "$OUT_TOKENS" \
                --result-dir "$RESULT_DIR" 2>&1 && OK=1
        else
            cd "$PROJECT_ROOT/llama_cpp_profiler"
            "$PROJECT_ROOT/inference/llamacpp_profiler/build/cpp_api_pipeline" \
                --input "$INPUT_DATA" \
                --model-names "$TMP_JSON" \
                --dtype "$LLAMACPP_DTYPE" \
            --n-ctx 8192 \
                --max-new-tokens "$OUT_TOKENS" \
                --result-dir "$RESULT_DIR/" 2>&1 && OK=1
        fi

        if [ "$OK" = "1" ] && [ -f "$CSV_FILE" ] && [ "$(wc -l < "$CSV_FILE")" -eq "$EXPECTED_LINES" ]; then
            echo "  DONE: $MODEL_NAME"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  FAIL: $MODEL_NAME"
            FAILED=$((FAILED + 1))
        fi
        rm -f "$TMP_JSON"
    done

    echo "  S2 ${BACKEND_TAG} out=${OUT_TOKENS}: done=$COMPLETED skip=$SKIPPED fail=$FAILED"
    echo ""
}


# ============================================================
#   Dispatch
# ============================================================
PROFILER_BIN="$PROJECT_ROOT/inference/llamacpp_profiler/build/cpp_api_pipeline"

case "$BACKEND" in
    all|hf|llamacpp) ;;
    *) echo "ERROR: --backend must be one of {all, hf, llamacpp}" >&2; exit 2 ;;
esac

if [ "$BACKEND" = "all" ] || [ "$BACKEND" = "hf" ]; then
    for OUT in "${OUT_TOKEN_TIERS[@]}"; do
        run_one_tier "hf" "$OUT" "$HF_DTYPE_CSV"
    done
fi

if [ "$BACKEND" = "all" ] || [ "$BACKEND" = "llamacpp" ]; then
    if [ ! -x "$PROFILER_BIN" ]; then
        echo "ERROR: llama.cpp profiler not built at $PROFILER_BIN" >&2
        exit 1
    fi
    for OUT in "${OUT_TOKEN_TIERS[@]}"; do
        run_one_tier "llamacpp" "$OUT" "$LLAMACPP_DTYPE"
    done
fi

echo "============================================"
echo "  S2 sweep complete on $PLATFORM"
echo "  Results under: $RESULTS_ROOT"
echo "============================================"
