# Analysis Pipeline

Turns the per-prompt UNIFIED CSVs (the released corpus under
`data/unified/`, or your own collection runs) into the figures and tables
of the paper, plus additional exploratory plot families.

## Layout

```
unifier/                 INFO (timing) + TGS (telemetry) -> UNIFIED CSVs
loader.py                discover & load UNIFIED_*.csv into one long-form df
schema.py                per-platform column -> canonical cross-platform names
style.py                 palette, figure sizes, model/dtype/platform orderings
performance.py           paper Figs. 3-5  (E2E latency, Q4 throughput, phase breakdown)
utilization.py           paper Fig. 7     (per-core CPU load/frequency)
sensitivity.py           paper Fig. 6     (S1/S2 input/output-length sweeps)
motivation.py            paper Fig. 1     (phase-aware motivation grid)
efficiency_tables.py     paper Tables 4-7 (U_eff/BW_eff, sweep U_eff, power/energy/thermal)
ae_figures.py            driver: renders exactly the six paper figures
main.py                  driver: renders the exploratory plot families below
extras/                  additional plot families (latency, throughput, memory,
                         cpu, gpu, thermal, power, quantization, backend_compare,
                         efficiency, memory_pressure, cross_gen, bottleneck,
                         distributions, scaling, headline) - not used in the
                         paper, provided for further exploration of the corpus
```

## The canonical schema

Every plot consumes a long-form DataFrame indexed by
`(platform, backend, model, dtype, prompt_number)` with two column layers:

- **Timing columns** (backend-agnostic): the 25 profiler INFO fields
  (`tokenization_time`, `prefill_time`, `time_to_first_token`, ...).
- **Telemetry columns**, in two flavours: raw platform-specific names
  (e.g., `vdd_gpu_soc_cur_decode_mean`) and **canonical cross-platform
  names** added by `schema.normalize` (e.g., `gpu_power_mw_decode_mean`) -
  every cross-platform figure/table uses the canonical layer.

Each telemetry signal carries three phase suffixes: `_prompt_mean` (full
prompt window), `_prefill_mean` (tokenization + prefill), and
`_decode_mean` (prefill end -> prompt end). The phase boundary is the end
of prefill (not TTFT), matching the prefill/decode split used in
LLM-serving systems.

## Usage

```bash
# The paper's figures and tables (see also scripts/ae_reproduce.sh):
python3 -m analysis.ae_figures        --results-root <corpus> --out figs/
python3 -m analysis.efficiency_tables --results-root <corpus> --out tables/

# Exploratory families over the same corpus:
python3 -m analysis.main --all --out figs_extras/
python3 -m analysis.main --plots latency,thermal --platforms orin,thor --out figs_test/
```
