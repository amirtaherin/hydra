#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# Run HF profiler experiments with resume support.
# Runs bfloat16 on all models. Skips models with complete results.
# Safe to Ctrl+C between models — re-run to continue from where you left off.
#
# Usage: bash scripts/run_hf_experiments.sh

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_JSON="$PROJECT_ROOT/inputs/models/models.json"
INPUT_DATA="$PROJECT_ROOT/inputs/prompts/input_data.jsonl"
DTYPE="bfloat16"
# HF dtype string as it appears in CSV filenames
DTYPE_CSV="torch.bfloat16"
MAX_NEW_TOKENS=500

# --- Detect platform ---
HOSTNAME=$(hostname)
if echo "$HOSTNAME" | grep -qi "orin"; then
    PLATFORM="orin"
elif echo "$HOSTNAME" | grep -qi "xavier"; then
    PLATFORM="xavier"
elif echo "$HOSTNAME" | grep -qi "thor"; then
    PLATFORM="thor"
else
    PLATFORM="$HOSTNAME"
fi

RESULT_DIR="$PROJECT_ROOT/results_hf_${PLATFORM}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo "  Hydra HF Profiler — ${PLATFORM}"
echo "============================================"
echo "Result dir:     $RESULT_DIR"
echo "Dtype:          $DTYPE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo ""

# --- Derive expected row count from dataset ---
EXPECTED_ROWS=$(wc -l < "$INPUT_DATA")
EXPECTED_LINES=$((EXPECTED_ROWS + 1))  # +1 for CSV header
echo "Input prompts:  $EXPECTED_ROWS"
echo ""

# --- Extract model names ---
MODEL_NAMES=$(python3 -c "
import json
with open('$MODELS_JSON') as f:
    data = json.load(f)
for m in data['models']:
    print(m['name'])
")

TOTAL=$(echo "$MODEL_NAMES" | wc -l)
echo "Models to process: $TOTAL"
echo ""

# --- Step 1: Download all models ---
echo "--- Step 1: Downloading models ---"
cd "$PROJECT_ROOT"
python3 -m inference.hf_profiler.main \
    --model-names "$MODELS_JSON" \
    --download-only 2>&1
echo ""

# --- Step 2: Run experiments model by model ---
echo "--- Step 2: Running experiments ---"
SKIPPED=0
COMPLETED=0
FAILED=0
CURRENT=0

for MODEL_NAME in $MODEL_NAMES; do
    CURRENT=$((CURRENT + 1))
    CSV_FILE="$RESULT_DIR/INFO_${MODEL_NAME}_${DTYPE_CSV}_.csv"

    # Check if already complete
    if [ -f "$CSV_FILE" ]; then
        LINE_COUNT=$(wc -l < "$CSV_FILE")
        if [ "$LINE_COUNT" -eq "$EXPECTED_LINES" ]; then
            echo "[$CURRENT/$TOTAL] SKIP $MODEL_NAME (complete: $EXPECTED_ROWS rows)"
            SKIPPED=$((SKIPPED + 1))
            continue
        else
            echo "[$CURRENT/$TOTAL] REDO $MODEL_NAME (partial: $((LINE_COUNT - 1))/$EXPECTED_ROWS rows)"
            rm -f "$CSV_FILE"
            rm -f "$RESULT_DIR/TGS_${MODEL_NAME}_${DTYPE_CSV}_.csv"
        fi
    else
        echo "[$CURRENT/$TOTAL] RUN  $MODEL_NAME"
    fi

    # Create temp single-model JSON
    TMP_JSON=$(mktemp /tmp/hydra_model_XXXXXX.json)
    python3 -c "
import json
with open('$MODELS_JSON') as f:
    data = json.load(f)
for m in data['models']:
    if m['name'] == '$MODEL_NAME':
        print(json.dumps({'models': [m]}, indent=2))
        break
" > "$TMP_JSON"

    # Run profiler
    cd "$PROJECT_ROOT"
    if python3 -m inference.hf_profiler.main \
        --input "$INPUT_DATA" \
        --model-names "$TMP_JSON" \
        --dtype "$DTYPE" \
        --max-new-tokens $MAX_NEW_TOKENS \
        --result-dir "$RESULT_DIR" 2>&1; then

        # Verify completion
        if [ -f "$CSV_FILE" ]; then
            LINE_COUNT=$(wc -l < "$CSV_FILE")
            if [ "$LINE_COUNT" -eq "$EXPECTED_LINES" ]; then
                echo "  DONE: $MODEL_NAME ($EXPECTED_ROWS rows)"
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  WARN: $MODEL_NAME incomplete ($((LINE_COUNT - 1))/$EXPECTED_ROWS rows)"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  FAIL: $MODEL_NAME (no CSV produced)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  FAIL: $MODEL_NAME (profiler exited with error)"
        FAILED=$((FAILED + 1))
    fi

    rm -f "$TMP_JSON"
done

echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
echo "  Completed: $COMPLETED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "  Total:     $TOTAL"
echo "============================================"
