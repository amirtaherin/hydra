#!/usr/bin/env bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# Run the per-prompt unifier on every sensitivity-sweep subdirectory.
#
# Layout assumed:
#   <root>/results_sensitivity/{S1,S2}_<platform>_<backend>_<dtype>[ _out<N> ]/
#
# Each subdir has INFO_*.csv + TGS_*.csv. UNIFIED_*.csv lands next to them.
# Platform is parsed from the dir name (orin|thor|xavier).

set -euo pipefail

ROOT="${1:-/media/amir/data/hydra-results}"
SWEEP_ROOT="$ROOT/results_sensitivity"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$SWEEP_ROOT" ]]; then
    echo "ERROR: $SWEEP_ROOT does not exist" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

declare -i ok=0 fail=0 skip=0
declare -a failed_dirs=()

for d in "$SWEEP_ROOT"/{S1,S2}_*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    # Parse platform from the name (second underscore-separated token).
    plat="$(echo "$name" | awk -F_ '{print $2}')"
    if [[ "$plat" != "orin" && "$plat" != "thor" && "$plat" != "xavier" ]]; then
        echo "SKIP  $name (cannot parse platform: '$plat')"
        skip+=1
        continue
    fi
    echo "----- $name (platform=$plat) -----"
    if python3 -m analysis.unifier.data_unifier_parallel \
            --platform "$plat" \
            --results-dir "$d"; then
        ok+=1
    else
        fail+=1
        failed_dirs+=("$name")
    fi
done

echo
echo "=========== summary ==========="
echo "ok:      $ok"
echo "failed:  $fail"
echo "skipped: $skip"
if (( fail > 0 )); then
    echo "failed dirs:"
    for d in "${failed_dirs[@]}"; do echo "  - $d"; done
    exit 1
fi
