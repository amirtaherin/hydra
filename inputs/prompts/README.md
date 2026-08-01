# Prompt Sets — Provenance and Licensing

## `input_data.jsonl` — IFEval (main corpus)

The 541 instruction-following prompts are the IFEval dataset, copied
unmodified from
[google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
(Zhou et al., *Instruction-Following Evaluation for Large Language Models*,
arXiv:2311.07911).

Datasets in the google-research repository are released under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license
(<https://creativecommons.org/licenses/by/4.0/>). The file is redistributed
here, unchanged, under that license with attribution to Google Research.

## `sensitivity_s2_ifeval.jsonl` — S2 output-length sweep

A **filtered subset** of the IFEval prompts above (30 prompts within a
40–60 input-token band; see `sensitivity_s2_ifeval.line_ids.txt` for the
selected line ids). Modified (subset selection only) from the CC BY 4.0
original; same license and attribution apply.

## `sensitivity_s1_ruler.jsonl` — S1 input-length sweep

90 synthetic prompts **generated using** the
[NVIDIA RULER](https://github.com/NVIDIA/RULER) benchmark generators
(Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context
Models?*, arXiv:2404.06654), balanced across three NIAH/tracking task
families at 1k/3k/5k-token tiers. RULER is licensed under **Apache-2.0**;
these generated prompts are distributed with attribution to NVIDIA.

## Token-count sidecars

`*.token_counts.csv` and `sensitivity_token_counts_all_models.csv` are
derived metadata produced by this project (per-model token counts of the
above prompt sets); MIT, like the rest of this repository.
