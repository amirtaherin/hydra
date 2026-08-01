# IISWC 2026 Artifact Evaluation — Hydra

Artifact for **"Hydra: Phase-Aware Workload Characterization of LLM Inference
across Edge SoC Generations, Backends, and Quantization Levels"** (paper #234).

The artifact has two layers:

1. **Evaluation I — analysis-from-corpus (no hardware needed).** This repo ships the full
   released measurement corpus — **107,110 per-prompt records** (286 UNIFIED
   CSVs: 190 main-corpus configurations + S1/S2 sensitivity sweeps) — as a
   split xz tarball under `data/unified/`. One command regenerates every
   data-derived figure and table in the paper from it. *This is the
   reproduction path we ask reviewers to run.*
2. **Evaluation II — measurement validation (Jetson hardware).** The profilers that produced
   the corpus (`inference/hf_profiler/` for HuggingFace, `inference/llamacpp_profiler/` for
   llama.cpp, `inference/telemetry/` + `analysis/unifier/` for
   tegrastats fusion) run on NVIDIA Jetson AGX
   Xavier/Orin/Thor. SSH access to all three boards is available on request
   during the evaluation window (see paper appendix).

## Evaluation I — reproduce the paper's results from the corpus

Requirements: Linux or macOS (validated on Linux), `python3` (3.10+) with
`pandas>=2.0`, `numpy`, `matplotlib`, `seaborn`. ~1 GB free disk. No GPU.
Runtime: a few minutes. (Evaluation I can also run on the provided Jetson boards: skip the venv
creation and `pip install` — the pre-installed environment activates on
login and already includes the analysis stack — and just run
`bash scripts/ae_reproduce.sh` in `~/hydra`. Prefer a separate machine,
and **never run Evaluation I on a board while a measurement
(Evaluation II) is in progress there**: the analysis load perturbs the
telemetry being recorded.)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn
bash scripts/ae_reproduce.sh            # outputs to ./ae_output
```

The script (1) **verifies and extracts the corpus** — sha256 check of the
release tarball, then an integrity manifest (107,110 per-prompt records in
286 files: 190 main-corpus configurations + S1/S2 sweeps; 349–419 columns
depending on platform/backend) that hard-stops on any mismatch; (2) renders the six data-derived
paper figures; (3) regenerates the four data tables; and (4) prints the
output→paper mapping.

Tables are emitted in two formats: `.md` for visual comparison against
the published tables, and `.csv` for machine processing.

**Success criteria:** the manifest prints `CORPUS VERIFICATION PASSED`
(automated); six figure PDFs render (observed); and the driver's final
step reports that **every value in the regenerated tables is identical to
the reference tables shipped in `expected_results/`** (automated). You can also verify by hand with
`diff -r ae_output/tables expected_results/tables` — note that `diff`
**prints nothing when the tables are identical**; empty output is the
expected success result. ~5 minutes, no GPU.
(How the shipped references relate to the published tables is described
under "Expected match" below.)

## Output → paper mapping

| Output | Paper element |
|---|---|
| `figures/phase_motivation.pdf` | Fig. 1 (motivation) |
| `figures/perf_e2e_latency_grid.pdf` | Fig. 3 (E2E latency) |
| `figures/perf_q4_throughput_grouped.pdf` | Fig. 4 (Q4 throughput) |
| `figures/perf_phase_breakdown_thor_qwen.pdf` | Fig. 5 (phase breakdown) |
| `figures/sensitivity_ttft_throughput.pdf` | Fig. 6 (length sensitivity) |
| `figures/system_cpu_per_core_qwen7b.pdf` | Fig. 7 (per-core CPU) |
| `tables/table4_gpu_mem_orin.{md,csv}` | Table 4 (U_eff / BW_eff, Orin) |
| `tables/table5_ueff_sweeps.{md,csv}` | Table 5 (U_eff, length sweeps) |
| `tables/table6_efficiency.{md,csv}` | Table 6 (power / mJ/tok / thermal) |
| `tables/table7_total_energy.{md,csv}` | Table 7 (total energy, sweeps) |

**Expected match.** For Evaluation I, the driver performs two automated
checks: it first verifies the released corpus itself (the checksum and
record manifest of step 1), and in its final step it verifies that the
regenerated tables are identical to the reference tables shipped in
`expected_results/` — this must hold exactly, since the analysis pipeline
is deterministic. Two further comparisons are manual, made by the
reviewer against the paper: the regenerated `.md` tables can be compared
with the published Tables 4–7 — they match at the printed precision,
though in a handful of cells the last printed digit differs by one
(e.g., 86.3 vs 86.2) due to rounding during manuscript preparation. The figures the driver generates in `ae_output/figures/` are produced
by the same scripts from the same data as the published Figs. 1 and 3–7,
so they should look identical to the figures in the paper; reference
copies also ship in `expected_results/figures/` for side-by-side viewing.
Figure PDFs are compared visually, not with `diff` — PDF files are never
byte-identical across systems (fonts, timestamps).

**Not covered by this path** (documented for completeness):

- *Fig. 2* — hand-drawn pipeline diagram (no data dependency).
- *Tables 1, 2, 8* — static hardware/model/related-work summaries.
- *Table 3 (accuracy)* — produced with the public
  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
  (WikiText-2 perplexity; zero-shot Winogrande and HellaSwag) on the GGUF
  models, reported as deltas of each quantized format vs. `F16`. This is a
  standard third-party tool run, requires downloading all model weights
  (~200 GB across formats), and is independent of the telemetry corpus —
  it is therefore outside the artifact's reproduction scope. The evaluated
  GGUF files are the public quantizations of the models listed in
  `inputs/models/models.json`.

## Expected results (shipped reference outputs)

`expected_results/` contains the expected outputs so reviewers
can `diff` instead of judging by eye:

- `expected_results/figures/` — the six paper figures as generated by
  `ae_reproduce.sh`, for side-by-side visual comparison with your
  `ae_output/figures/` and with the paper.
- `expected_results/tables/` — Tables 4–7 as `.md`/`.csv`; your
  `ae_output/tables/` files should match these exactly.
- `expected_results/ae_quick_reference.csv` — corpus reference values for
  the hardware spot-check below.

## Evaluation II — validate the measurement pipeline (Jetson, SSH)

**Quick spot-check (recommended for reviewers; ~30–60 min on Orin/Thor,
up to ~2 h on Xavier, the oldest platform):**

On the provided boards the llama.cpp profiler is **pre-compiled** — you
can run the spot-check directly. If you wish to verify the build yourself
(or are on your own Jetson), `bash scripts/build_llamacpp_profiler.sh`
rebuilds it from the pinned submodule (~15–30 min; requires cmake, the
CUDA toolkit on PATH, and a C++17 compiler — see the README's build
section for troubleshooting).

```bash
tmux new -s ae            # keeps the run alive if SSH drops
bash scripts/ae_quick.sh
```

(`tmux` basics: detach with `Ctrl-b` then `d`; reattach later with
`tmux attach -t ae`. The run continues while detached.)

On the provided boards the Python environment (including the NVIDIA
JetPack build of PyTorch) is pre-installed in a virtualenv that activates
automatically on login — no setup is needed. To recreate it on your own
Jetson, see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

On the Jetson it is run on, this measures Qwen2.5-7B — the paper's
flagship analysis model — under HF bf16 and llama.cpp Q8_0/Q6_K/Q4_K_M
(~20 IFEval prompts, 500-token decode), producing a **fresh mini-corpus**
(20 prompts × 4 configurations, unified exactly like the released corpus)
and a comparison table against the corpus references.

The comparison applies **two reproducibility checks**:

1. **Quantitative agreement** — every fresh value (decode power, ITL,
   energy/token, throughput) is printed next to its corpus reference with
   the percentage deviation; values typically fall within ±15% of the
   corpus means. Thermal state and background load shift absolute
   numbers, so this band is guidance, not a hard gate.
2. **Qualitative findings** — the binding pass/fail criterion (printed as
   `Qualitative invariants`; the script's exit code follows it). Three
   findings of the paper must reproduce on every board:
   - Q6_K draws more decode power than Q8_0 (bit-width non-monotonicity);
   - Q4_K_M has the lowest energy per token among the llama.cpp formats;
   - llama.cpp ITL < HF bf16 ITL (runtime-overhead gap).

Expected outcome: the table ends in `Result: INVARIANTS PASS`.

**Full collection (days of device time; not expected of reviewers):**

```bash
git submodule update --init --recursive     # pinned llama.cpp (build dep)
bash scripts/build_llamacpp_profiler.sh     # compile the profiler (auto CUDA arch)
bash scripts/run_hf_experiments.sh          # HuggingFace backend
bash scripts/run_llamacpp_experiments.sh    # llama.cpp backend
bash scripts/run_unifiers.sh <results_root>             # fuse INFO+TGS
bash scripts/run_unifiers_sensitivity.sh <results_root> # S1/S2 sweeps
```

on Jetson AGX Xavier (JetPack 5), Orin (JetPack 6), and Thor (JetPack 7).
SSH access to the three boards, with the environment pre-provisioned, is
available on request during the evaluation window.

## Repository map (AE-relevant)

```
data/unified/                  released corpus (split tarball + README)
scripts/ae_reproduce.sh        one-command reproduction driver
analysis/ae_figures.py         renders the six paper figures
analysis/efficiency_tables.py  regenerates Tables 4-7
analysis/                      loader, canonical schema, unifier (see its README)
inference/hf_profiler/         HuggingFace profiler (collection)
inference/llamacpp_profiler/   llama.cpp profiler (collection)
inference/telemetry/           tegrastats control + per-platform parsers
inputs/                        prompts + model registry
```
