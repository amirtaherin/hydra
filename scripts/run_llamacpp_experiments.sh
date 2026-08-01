#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# Run llama.cpp profiler experiments with resume support.
# Runs 4 dtypes (F16, Q8_0, Q6_K, Q4_K_M) on all models.
# Skips models with complete results.
# Safe to Ctrl+C between models — re-run to continue from where you left off.
#
# Usage: bash scripts/run_llamacpp_experiments.sh

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_JSON="$PROJECT_ROOT/inputs/models/models.json"
INPUT_DATA="$PROJECT_ROOT/inputs/prompts/input_data.jsonl"
PROFILER_BIN="$PROJECT_ROOT/inference/llamacpp_profiler/build/cpp_api_pipeline"
DTYPES=("F16" "Q8_0" "Q6_K" "Q4_K_M")
MAX_NEW_TOKENS=500

# --- Detect platform & CUDA arch ---
HOSTNAME=$(hostname)
if echo "$HOSTNAME" | grep -qi "orin"; then
    PLATFORM="orin"
    CUDA_ARCH=87
elif echo "$HOSTNAME" | grep -qi "xavier"; then
    PLATFORM="xavier"
    CUDA_ARCH=72
elif echo "$HOSTNAME" | grep -qi "thor"; then
    PLATFORM="thor"
    CUDA_ARCH=110  # use 101 if CUDA < 13.0
else
    PLATFORM="$HOSTNAME"
    CUDA_ARCH=87
    echo "WARNING: Unknown hostname '$HOSTNAME', defaulting CUDA_ARCH=$CUDA_ARCH"
fi

RESULT_DIR="$PROJECT_ROOT/results_llamacpp_${PLATFORM}"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo "  Hydra llama.cpp Profiler — ${PLATFORM}"
echo "============================================"
echo "Result dir:     $RESULT_DIR"
echo "Dtypes:         ${DTYPES[*]}"
echo "CUDA arch:      $CUDA_ARCH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo ""

# --- Build via shared script (submodule check, json.hpp, cmake) ---
if [ ! -f "$PROFILER_BIN" ]; then
    bash "$SCRIPT_DIR/build_llamacpp_profiler.sh"
fi

# --- Check submodule ---
LLAMA_CPP_DIR="$PROJECT_ROOT/inference/llamacpp_profiler/llama.cpp"
if [ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
    echo "ERROR: llama.cpp submodule not initialized."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# --- Check nlohmann/json ---
JSON_HPP="$PROJECT_ROOT/inference/llamacpp_profiler/nlohmann/json.hpp"
if [ ! -f "$JSON_HPP" ]; then
    echo "Downloading nlohmann/json.hpp..."
    mkdir -p "$PROJECT_ROOT/inference/llamacpp_profiler/nlohmann"
    wget -q -O "$JSON_HPP" \
      https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    echo "  Done."
fi

# --- Build profiler if needed ---
if [ ! -f "$PROFILER_BIN" ]; then
    echo "Building llama.cpp profiler (CUDA_ARCH=$CUDA_ARCH)..."
    mkdir -p "$PROJECT_ROOT/inference/llamacpp_profiler/build"
    cd "$PROJECT_ROOT/inference/llamacpp_profiler/build"
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
    make -j$(nproc) cpp_api_pipeline
    cd "$PROJECT_ROOT"
    echo "  Build complete."
    echo ""
fi

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

NUM_MODELS=$(echo "$MODEL_NAMES" | wc -l)
TOTAL=$((NUM_MODELS * ${#DTYPES[@]}))
echo "Models:         $NUM_MODELS"
echo "Total runs:     $TOTAL (${#DTYPES[@]} dtypes x $NUM_MODELS models)"
echo ""

# --- Step 1: Download all models for all dtypes ---
echo "--- Step 1: Downloading models ---"
for DTYPE in "${DTYPES[@]}"; do
    echo "Downloading dtype=$DTYPE..."
    cd "$PROJECT_ROOT"
    "$PROFILER_BIN" \
        --model-names "$MODELS_JSON" \
        --dtype "$DTYPE" \
        --download-only 2>&1
    echo ""
done

# --- Step 2: Run experiments ---
echo "--- Step 2: Running experiments ---"
SKIPPED=0
COMPLETED=0
FAILED=0
CURRENT=0

for DTYPE in "${DTYPES[@]}"; do
    echo ""
    echo "=== dtype: $DTYPE ==="
    for MODEL_NAME in $MODEL_NAMES; do
        CURRENT=$((CURRENT + 1))
        CSV_FILE="$RESULT_DIR/INFO_${MODEL_NAME}_${DTYPE}_.csv"

        # Check if already complete
        if [ -f "$CSV_FILE" ]; then
            LINE_COUNT=$(wc -l < "$CSV_FILE")
            if [ "$LINE_COUNT" -eq "$EXPECTED_LINES" ]; then
                echo "[$CURRENT/$TOTAL] SKIP $MODEL_NAME/$DTYPE (complete: $EXPECTED_ROWS rows)"
                SKIPPED=$((SKIPPED + 1))
                continue
            else
                echo "[$CURRENT/$TOTAL] REDO $MODEL_NAME/$DTYPE (partial: $((LINE_COUNT - 1))/$EXPECTED_ROWS rows)"
                rm -f "$CSV_FILE"
                rm -f "$RESULT_DIR/TGS_${MODEL_NAME}_${DTYPE}_.csv"
            fi
        else
            echo "[$CURRENT/$TOTAL] RUN  $MODEL_NAME/$DTYPE"
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
        if "$PROFILER_BIN" \
            --input "$INPUT_DATA" \
            --model-names "$TMP_JSON" \
            --dtype "$DTYPE" \
            --max-new-tokens $MAX_NEW_TOKENS \
            --result-dir "$RESULT_DIR/" 2>&1; then

            # Verify completion
            if [ -f "$CSV_FILE" ]; then
                LINE_COUNT=$(wc -l < "$CSV_FILE")
                if [ "$LINE_COUNT" -eq "$EXPECTED_LINES" ]; then
                    echo "  DONE: $MODEL_NAME/$DTYPE ($EXPECTED_ROWS rows)"
                    COMPLETED=$((COMPLETED + 1))
                else
                    echo "  WARN: $MODEL_NAME/$DTYPE incomplete ($((LINE_COUNT - 1))/$EXPECTED_ROWS rows)"
                    FAILED=$((FAILED + 1))
                fi
            else
                echo "  FAIL: $MODEL_NAME/$DTYPE (no CSV produced)"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  FAIL: $MODEL_NAME/$DTYPE (profiler exited with error)"
            FAILED=$((FAILED + 1))
        fi

        rm -f "$TMP_JSON"
    done
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
