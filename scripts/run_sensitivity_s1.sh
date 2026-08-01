#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# S1 sensitivity sweep — input-length sensitivity on Thor.
#
# Runs the HF (bfloat16) and llama.cpp (Q4_K_M) profilers over the
# 6-model RULER prompt set (inputs/prompts/sensitivity_s1_ruler.jsonl, 90 prompts)
# with max_new_tokens=500 fixed. The 3 input-length tiers (1000/3000/5000)
# are encoded *inside* the prompt file; one run produces all 90 per model.
#
# Output layout (under results_sensitivity/):
#   S1_<platform>_hf_bf16/             INFO_*.csv, TGS_*.csv per model
#   S1_<platform>_llamacpp_Q4_K_M/     INFO_*.csv, TGS_*.csv per model
#
# Designed to be run on Thor. Resume-safe (per-model skip on complete CSVs).
#
# Usage:
#   bash scripts/run_sensitivity_s1.sh                 # both backends
#   bash scripts/run_sensitivity_s1.sh --backend hf
#   bash scripts/run_sensitivity_s1.sh --backend llamacpp

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_JSON="$PROJECT_ROOT/inputs/models/models_sensitivity.json"
INPUT_DATA="$PROJECT_ROOT/inputs/prompts/sensitivity_s1_ruler.jsonl"
HF_DTYPE="bfloat16"
HF_DTYPE_CSV="torch.bfloat16"
LLAMACPP_DTYPE="Q4_K_M"
MAX_NEW_TOKENS=500

# --- Args ---
BACKEND="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
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
echo "  Hydra S1 (input-length) — ${PLATFORM}"
echo "============================================"
echo "Prompt set:     $INPUT_DATA"
echo "Models JSON:    $MODELS_JSON"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Backend(s):     $BACKEND"
echo ""

# --- Expected row count ---
EXPECTED_ROWS=$(wc -l < "$INPUT_DATA")
EXPECTED_LINES=$((EXPECTED_ROWS + 1))
echo "Prompts to run: $EXPECTED_ROWS"
echo ""

# --- Extract model names ---
MODEL_NAMES=$(python3 -c "
import json
with open('$MODELS_JSON') as f: data = json.load(f)
for m in data['models']: print(m['name'])
")
TOTAL=$(echo "$MODEL_NAMES" | wc -l)
echo "Models:         $TOTAL"
echo ""


# ============================================================
#                       HuggingFace
# ============================================================
run_hf() {
    local RESULT_DIR="$RESULTS_ROOT/S1_${PLATFORM}_hf_bf16"
    mkdir -p "$RESULT_DIR"

    echo "============================================"
    echo "  S1 HF (bf16) -> $RESULT_DIR"
    echo "============================================"

    local CURRENT=0 SKIPPED=0 COMPLETED=0 FAILED=0
    for MODEL_NAME in $MODEL_NAMES; do
        CURRENT=$((CURRENT + 1))
        local CSV_FILE="$RESULT_DIR/INFO_${MODEL_NAME}_${HF_DTYPE_CSV}_.csv"

        if [ -f "$CSV_FILE" ]; then
            local LC=$(wc -l < "$CSV_FILE")
            if [ "$LC" -eq "$EXPECTED_LINES" ]; then
                echo "[$CURRENT/$TOTAL] SKIP $MODEL_NAME (complete)"
                SKIPPED=$((SKIPPED + 1)); continue
            else
                echo "[$CURRENT/$TOTAL] REDO $MODEL_NAME (partial: $((LC - 1))/$EXPECTED_ROWS)"
                rm -f "$CSV_FILE" "$RESULT_DIR/TGS_${MODEL_NAME}_${HF_DTYPE_CSV}_.csv"
            fi
        else
            echo "[$CURRENT/$TOTAL] RUN  $MODEL_NAME"
        fi

        local TMP_JSON=$(mktemp /tmp/hydra_s1_hf_XXXXXX.json)
        python3 -c "
import json
with open('$MODELS_JSON') as f: data = json.load(f)
for m in data['models']:
    if m['name'] == '$MODEL_NAME':
        print(json.dumps({'models':[m]}, indent=2)); break
" > "$TMP_JSON"

        cd "$PROJECT_ROOT"
        if python3 -m inference.hf_profiler.main \
            --input "$INPUT_DATA" \
            --model-names "$TMP_JSON" \
            --dtype "$HF_DTYPE" \
            --max-new-tokens $MAX_NEW_TOKENS \
            --result-dir "$RESULT_DIR" 2>&1; then
            if [ -f "$CSV_FILE" ] && [ "$(wc -l < "$CSV_FILE")" -eq "$EXPECTED_LINES" ]; then
                echo "  DONE: $MODEL_NAME"
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  WARN: $MODEL_NAME incomplete"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  FAIL: $MODEL_NAME (profiler error)"
            FAILED=$((FAILED + 1))
        fi
        rm -f "$TMP_JSON"
    done

    echo ""
    echo "  S1 HF: done=$COMPLETED skip=$SKIPPED fail=$FAILED"
    echo ""
}


# ============================================================
#                       llama.cpp
# ============================================================
run_llamacpp() {
    local PROFILER_BIN="$PROJECT_ROOT/inference/llamacpp_profiler/build/cpp_api_pipeline"
    local RESULT_DIR="$RESULTS_ROOT/S1_${PLATFORM}_llamacpp_${LLAMACPP_DTYPE}"
    mkdir -p "$RESULT_DIR"

    if [ ! -x "$PROFILER_BIN" ]; then
        echo "ERROR: llama.cpp profiler not built at $PROFILER_BIN" >&2
        echo "       Run scripts/run_llamacpp_experiments.sh first or build manually." >&2
        return 1
    fi

    echo "============================================"
    echo "  S1 llama.cpp ($LLAMACPP_DTYPE) -> $RESULT_DIR"
    echo "============================================"

    local CURRENT=0 SKIPPED=0 COMPLETED=0 FAILED=0
    for MODEL_NAME in $MODEL_NAMES; do
        CURRENT=$((CURRENT + 1))
        local CSV_FILE="$RESULT_DIR/INFO_${MODEL_NAME}_${LLAMACPP_DTYPE}_.csv"

        if [ -f "$CSV_FILE" ]; then
            local LC=$(wc -l < "$CSV_FILE")
            if [ "$LC" -eq "$EXPECTED_LINES" ]; then
                echo "[$CURRENT/$TOTAL] SKIP $MODEL_NAME (complete)"
                SKIPPED=$((SKIPPED + 1)); continue
            else
                echo "[$CURRENT/$TOTAL] REDO $MODEL_NAME (partial: $((LC - 1))/$EXPECTED_ROWS)"
                rm -f "$CSV_FILE" "$RESULT_DIR/TGS_${MODEL_NAME}_${LLAMACPP_DTYPE}_.csv"
            fi
        else
            echo "[$CURRENT/$TOTAL] RUN  $MODEL_NAME"
        fi

        local TMP_JSON=$(mktemp /tmp/hydra_s1_llcpp_XXXXXX.json)
        python3 -c "
import json
with open('$MODELS_JSON') as f: data = json.load(f)
for m in data['models']:
    if m['name'] == '$MODEL_NAME':
        print(json.dumps({'models':[m]}, indent=2)); break
" > "$TMP_JSON"

        cd "$PROJECT_ROOT/llama_cpp_profiler"
        if "$PROFILER_BIN" \
            --input "$INPUT_DATA" \
            --model-names "$TMP_JSON" \
            --dtype "$LLAMACPP_DTYPE" \
            --n-ctx 8192 \
            --max-new-tokens $MAX_NEW_TOKENS \
            --result-dir "$RESULT_DIR/" 2>&1; then
            if [ -f "$CSV_FILE" ] && [ "$(wc -l < "$CSV_FILE")" -eq "$EXPECTED_LINES" ]; then
                echo "  DONE: $MODEL_NAME"
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  WARN: $MODEL_NAME incomplete"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  FAIL: $MODEL_NAME (profiler error)"
            FAILED=$((FAILED + 1))
        fi
        rm -f "$TMP_JSON"
    done

    echo ""
    echo "  S1 llama.cpp: done=$COMPLETED skip=$SKIPPED fail=$FAILED"
    echo ""
}


# --- Dispatch ---
case "$BACKEND" in
    all)      run_hf; run_llamacpp ;;
    hf)       run_hf ;;
    llamacpp) run_llamacpp ;;
    *) echo "ERROR: --backend must be one of {all, hf, llamacpp}" >&2; exit 2 ;;
esac

echo "============================================"
echo "  S1 sweep complete on $PLATFORM"
echo "  Results under: $RESULTS_ROOT"
echo "============================================"
