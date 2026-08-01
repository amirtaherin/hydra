#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# Run the data unifier across every (platform, backend) combination.
#
# For each of the 6 combos {orin,xavier,thor} x {hf,llamacpp}:
#   - resolves <results-root>/<platform>/results_<backend>_<platform>
#   - skips combos whose directory does not exist (with a MISSING line)
#   - invokes `python3 -m analysis.unifier.data_unifier_parallel` with the right --platform
#   - tracks per-combo exit codes and prints a final summary
#
# Idempotent: the unifier itself skips INFO files whose UNIFIED already exists.
# Use --force to re-run everything.
#
# Usage:
#   bash scripts/run_unifiers.sh
#   bash scripts/run_unifiers.sh --force
#   bash scripts/run_unifiers.sh --results-root /some/other/path
#

set -u

# --- Resolve project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Defaults ---
RESULTS_ROOT="/media/amir/data/hydra-results"
FORCE_FLAG=""
EXTRA_ARGS=()

# --- Parse args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --max-processes)
            EXTRA_ARGS+=("--max-processes" "$2")
            shift 2
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

PLATFORMS=("orin" "xavier" "thor")
BACKENDS=("hf" "llamacpp")

echo "============================================"
echo "  Hydra unifier — all platforms x backends"
echo "============================================"
echo "Project root:  $PROJECT_ROOT"
echo "Results root:  $RESULTS_ROOT"
echo "Force re-run:  ${FORCE_FLAG:-no}"
echo ""

# --- Per-combo run ---
declare -a COMBO_NAMES
declare -a COMBO_STATUS

cd "$PROJECT_ROOT"

for plat in "${PLATFORMS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        combo="${plat}/${backend}"
        target="${RESULTS_ROOT}/${plat}/results_${backend}_${plat}"

        echo "--- $combo ---"
        echo "    dir: $target"

        if [ ! -d "$target" ]; then
            echo "    MISSING — skipping"
            COMBO_NAMES+=("$combo")
            COMBO_STATUS+=("MISSING")
            echo ""
            continue
        fi

        if python3 -m analysis.unifier.data_unifier_parallel \
            --platform "$plat" \
            --results-dir "$target" \
            $FORCE_FLAG \
            "${EXTRA_ARGS[@]}"; then
            COMBO_NAMES+=("$combo")
            COMBO_STATUS+=("OK")
        else
            COMBO_NAMES+=("$combo")
            COMBO_STATUS+=("FAIL")
        fi
        echo ""
    done
done

# --- Summary ---
echo "============================================"
echo "  Overall summary"
echo "============================================"
ANY_FAIL=0
for i in "${!COMBO_NAMES[@]}"; do
    printf "  %-18s %s\n" "${COMBO_NAMES[$i]}" "${COMBO_STATUS[$i]}"
    if [ "${COMBO_STATUS[$i]}" = "FAIL" ]; then
        ANY_FAIL=1
    fi
done
echo "============================================"

exit $ANY_FAIL
