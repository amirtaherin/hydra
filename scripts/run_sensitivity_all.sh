#!/bin/bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# Top-level driver: run S1 then S2 sensitivity sweeps end-to-end on Thor.
#
# Both sub-runners are resume-safe, so re-running this script is cheap if
# a sub-run is interrupted. Forwards --backend to both children.
#
# Usage:
#   bash scripts/run_sensitivity_all.sh
#   bash scripts/run_sensitivity_all.sh --backend hf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        *) echo "ERROR: unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "############################################"
echo "#  Hydra sensitivity sweeps — S1 then S2  #"
echo "############################################"

bash "$SCRIPT_DIR/run_sensitivity_s1.sh" --backend "$BACKEND"
bash "$SCRIPT_DIR/run_sensitivity_s2.sh" --backend "$BACKEND"

echo "############################################"
echo "#  All sensitivity sweeps complete         #"
echo "############################################"
