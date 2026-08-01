#!/usr/bin/env bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# ==========================================================================
# IISWC 2026 Artifact Evaluation - one-command reproduction driver.
#
# Reassembles the released UNIFIED per-prompt corpus shipped in this repo
# (data/unified/, split xz tarball), then regenerates every data-derived
# figure (Figs. 1, 3-7) and data table (Tables 4-7) of the paper.
#
# Requires: python3 with pandas, numpy, matplotlib, seaborn. No GPU, no
# Jetson hardware - this is the analysis-from-corpus path. Runtime: a few
# minutes on a laptop. See AE_README.md for the output -> paper mapping.
#
# Usage:   bash scripts/ae_reproduce.sh [output_dir]
# Default output_dir: ./ae_output
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/ae_output}"
CORPUS_DIR="$OUT_DIR/corpus"
RESULTS_ROOT="$CORPUS_DIR/media/amir/data/hydra-results"

mkdir -p "$OUT_DIR" "$CORPUS_DIR"
cd "$REPO_ROOT"

echo "== [1/5] Verifying and extracting the released corpus =="
echo "   tarball:   data/unified/hydra_unified.tar.xz (2 parts, 115 MB)"
# Portable sha256 verification (GNU sha256sum is absent on stock macOS).
if python3 - <<'PYEOF'
import hashlib, pathlib, sys
root = pathlib.Path("data/unified")
for line in (root / "SHA256SUMS").read_text().splitlines():
    want, name = line.split()
    h = hashlib.sha256()
    with open(root / name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        sys.exit(1)
PYEOF
then
    echo "   sha256:    OK (matches data/unified/SHA256SUMS)"
else
    echo "   sha256:    FAILED - tarball parts corrupt or modified"; exit 1
fi
if [ -d "$RESULTS_ROOT" ] && \
   [ "$(find "$RESULTS_ROOT" -name 'UNIFIED_*.csv' | wc -l)" -eq 286 ]; then
    echo "   extracted: corpus already present - skipping extraction"
else
    cat data/unified/hydra_unified.tar.xz.part* > "$OUT_DIR/hydra_unified.tar.xz"
    tar -xJf "$OUT_DIR/hydra_unified.tar.xz" -C "$CORPUS_DIR/"
    rm -f "$OUT_DIR/hydra_unified.tar.xz"
fi

# ---- Corpus manifest (hard integrity gate) ----
FILES_TOTAL=$(find "$RESULTS_ROOT" -name 'UNIFIED_*.csv' | wc -l)
FILES_SENS=$(find "$RESULTS_ROOT" -path '*results_sensitivity*' -name 'UNIFIED_*.csv' | wc -l)
FILES_MAIN=$((FILES_TOTAL - FILES_SENS))
LINES_TOTAL=$(find "$RESULTS_ROOT" -name 'UNIFIED_*.csv' -exec cat {} + | wc -l)
REC_TOTAL=$((LINES_TOTAL - FILES_TOTAL))    # minus one header per file
LINES_SENS=$(find "$RESULTS_ROOT" -path '*results_sensitivity*' -name 'UNIFIED_*.csv' -exec cat {} + | wc -l)
REC_SENS=$((LINES_SENS - FILES_SENS))
REC_MAIN=$((REC_TOTAL - REC_SENS))
SIZE=$(du -sh "$RESULTS_ROOT" | cut -f1)
COLS_MIN=999999; COLS_MAX=0
while IFS= read -r f; do
    c=$(head -1 "$f" | tr ',' '\n' | wc -l)
    [ "$c" -lt "$COLS_MIN" ] && COLS_MIN=$c
    [ "$c" -gt "$COLS_MAX" ] && COLS_MAX=$c
done < <(find "$RESULTS_ROOT" -name 'UNIFIED_*.csv')

echo ""
echo "   Corpus manifest"
echo "   ------------------------------------------------------------"
echo "   per-prompt records ......... $REC_TOTAL   (expected 107110)"
echo "     main IFEval corpus ....... $REC_MAIN in $FILES_MAIN configurations"
echo "       platforms .............. xavier, orin, thor"
echo "       backends ............... hf, llamacpp"
echo "       formats ................ bf16, F16, Q8_0, Q6_K, Q4_K_M"
echo "     sensitivity sweeps ....... $REC_SENS in $FILES_SENS files (S1 input / S2 output)"
echo "   schema ..................... 25 timing fields + phase-attributed"
echo "                                telemetry aggregates ($COLS_MIN-$COLS_MAX cols,"
echo "                                platform/backend-dependent)"
echo "   extracted size ............. $SIZE ($FILES_TOTAL UNIFIED CSV files)"
echo "   ------------------------------------------------------------"
if [ "$REC_TOTAL" -eq 107110 ] && [ "$FILES_TOTAL" -eq 286 ] && \
   [ "$FILES_MAIN" -eq 190 ] && [ "$REC_SENS" -eq 4320 ]; then
    echo "   CORPUS VERIFICATION PASSED"
else
    echo "   CORPUS VERIFICATION FAILED (records=$REC_TOTAL files=$FILES_TOTAL"
    echo "   main=$FILES_MAIN sens_records=$REC_SENS) - aborting"
    exit 1
fi

echo "== [2/5] Rendering paper figures (Figs. 1, 3-7) =="
python3 -m analysis.ae_figures \
    --results-root "$RESULTS_ROOT" --out "$OUT_DIR/figures"

echo "== [3/5] Regenerating paper tables (Tables 4-7) =="
python3 -m analysis.efficiency_tables \
    --results-root "$RESULTS_ROOT" --out "$OUT_DIR/tables"

echo "== [4/5] Output -> paper mapping =="
cat <<'EOF'
   figures/phase_motivation.pdf                -> Fig. 1
   figures/perf_e2e_latency_grid.pdf           -> Fig. 3
   figures/perf_q4_throughput_grouped.pdf      -> Fig. 4
   figures/perf_phase_breakdown_thor_qwen.pdf  -> Fig. 5
   figures/sensitivity_ttft_throughput.pdf     -> Fig. 6
   figures/system_cpu_per_core_qwen7b.pdf      -> Fig. 7
   tables/table4_gpu_mem_orin.{md,csv}         -> Table 4
   tables/table5_ueff_sweeps.{md,csv}          -> Table 5
   tables/table6_efficiency.{md,csv}           -> Table 6
   tables/table7_total_energy.{md,csv}         -> Table 7
   (Fig. 2 is a hand-drawn pipeline diagram; Tables 1-3, 8 are static /
    external - see AE_README.md.)
EOF
echo "== [5/5] Comparing regenerated tables against committed references =="
if diff -r "$OUT_DIR/tables" "$REPO_ROOT/expected_results/tables" > /dev/null 2>&1; then
    echo "   RESULT: every value in the regenerated tables is identical to"
    echo "   expected_results/tables - the paper's Tables 4-7 are reproduced."
else
    echo "   RESULT: regenerated tables DIFFER from expected_results/tables:"
    diff -r "$OUT_DIR/tables" "$REPO_ROOT/expected_results/tables" | head -20
    exit 1
fi
echo "Done. All outputs under: $OUT_DIR"
