# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
"""Shared visual style for Hydra analysis plots (IISWC 2026 submission).

One source of truth for:
  - colors (Persian palette, https://github.com/nafasebra/iranian-colors)
  - figure sizes (IISWC 2-column ACM template)
  - canonical model display order (size-sorted)
  - canonical dtype display order
  - canonical platform display order (oldest -> newest gen)

Importing from this module rather than hardcoding choices in each plot keeps
the figure set visually consistent across the paper.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Persian palette — full list, then curated assignments
# -----------------------------------------------------------------------------
PERSIAN_PALETTE = {
    # Nature
    "Lajvard":          "#0047AB",  # deep blue (lapis lazuli)
    "Firouzeh":         "#40E0D0",  # turquoise
    "Yashm":            "#00A86B",  # jade green
    "Khaki":            "#C3B091",
    "Mashi":            "#D4AF37",
    "Hennaei":          "#A52A2A",  # henna brown-red
    "Khormaei":         "#8B4513",  # date brown
    "Choobi":           "#966F33",  # wood
    # Foods & plants
    "Zaferani":         "#F4C430",  # saffron
    "Zereshki":         "#900020",  # barberry deep red
    "Annabi":           "#800000",  # jujube maroon
    "Bademjani":        "#5D3A3A",  # eggplant
    "GolBehi":          "#FFB07C",  # quince blossom peach
    "Shahedanaei":      "#78866B",  # hemp green
    # Architecture & handicrafts
    "Nili":             "#4B0082",  # indigo
    "Arghavani":        "#800080",  # judas-tree purple
    "Goli":             "#FF69B4",
    "Mesi":             "#B87333",  # copper
    "ToosiIrani":       "#708090",  # slate
    "LajvardFirouzeh":  "#007BA7",
    "Shangarfi":        "#E32636",  # cinnabar red
    "Berenji":          "#D4AF37",
    # Lesser-known
    "Kahgeli":          "#D2B48C",
    "NoghreiIrani":     "#C0C0C0",  # silver
    "SabzSeyedi":       "#4CAF50",
    "Daryaei":          "#4682B4",  # sea blue
    "Fili":             "#808080",
    "Mokhammali":       "#560319",  # velvet
}


# -----------------------------------------------------------------------------
# Curated assignments — these are what plotters import
# -----------------------------------------------------------------------------

# Generation order: oldest to newest. Hue progression blue -> green -> red.
PLATFORM_COLORS: dict[str, str] = {
    "xavier": PERSIAN_PALETTE["Lajvard"],      # JP5 / Volta
    "orin":   PERSIAN_PALETTE["Yashm"],        # JP6 / Ampere
    "thor":   PERSIAN_PALETTE["Shangarfi"],    # JP7 / Blackwell
}
PLATFORM_ORDER: list[str] = ["xavier", "orin", "thor"]

# Peak theoretical DRAM bandwidth per platform (GB/s). Used to convert the
# tegrastats EMC load percentage into effective bandwidth.
#   Xavier: LPDDR4x @ 2133 MHz, 256-bit  -> 137 GB/s
#   Orin:   LPDDR5  @ 3200 MHz, 256-bit  -> 204.8 GB/s
#   Thor:   LPDDR5X @ 4267 MHz, 256-bit  -> 273 GB/s
PEAK_MEM_BW_GBPS: dict[str, float] = {
    "xavier": 137.0,
    "orin":   204.8,
    "thor":   273.0,
}

# Peak EMC frequency per platform (MHz). Used by the DAC effective-bandwidth
# formula: BW_eff = peak_BW * (emc_freq / freq_max) * (emc_load / 100). On
# Xavier and Orin the EMC always runs at peak under inference load, so this
# is a no-op; on Thor early-prefill samples can dip to 3200 MHz (~75% of
# peak) and the freq term matters.
PEAK_EMC_FREQ_MHZ: dict[str, float] = {
    "xavier": 2133.0,
    "orin":   3200.0,
    "thor":   4267.0,
}

# Backend split: HF (PyTorch) vs llama.cpp.
BACKEND_COLORS: dict[str, str] = {
    "hf":       PERSIAN_PALETTE["Arghavani"],
    "llamacpp": PERSIAN_PALETTE["Mesi"],
}
BACKEND_ORDER: list[str] = ["hf", "llamacpp"]

# Dtypes — HF emits torch.bfloat16; llama.cpp emits {F16, Q8_0, Q6_K, Q4_K_M}.
# Color assignment goes blue (least quantized) -> warm (most quantized).
DTYPE_COLORS: dict[str, str] = {
    "torch.bfloat16": PERSIAN_PALETTE["Zereshki"],   # HF baseline
    "F16":            PERSIAN_PALETTE["Lajvard"],
    "Q8_0":           PERSIAN_PALETTE["Firouzeh"],
    "Q6_K":           PERSIAN_PALETTE["Mesi"],
    "Q4_K_M":         PERSIAN_PALETTE["Zaferani"],
}
DTYPE_ORDER: list[str] = ["torch.bfloat16", "F16", "Q8_0", "Q6_K", "Q4_K_M"]
DTYPE_DISPLAY: dict[str, str] = {
    "torch.bfloat16": "bf16 (HF)",
    "F16":            "F16",
    "Q8_0":           "Q8_0",
    "Q6_K":           "Q6_K",
    "Q4_K_M":         "Q4_K_M",
}

# Model display order — single source of truth is `inputs/models/models.json`, where
# the entries are hand-curated in ascending parameter-count order. We read
# that list verbatim instead of re-deriving from parameter counts (avoids
# disagreements when several models share the same nominal size).
import json as _json
import pathlib as _pl

_MODELS_JSON = _pl.Path(__file__).parent.parent / "inputs" / "models" / "models.json"
with open(_MODELS_JSON) as _f:
    _models_data = _json.load(_f)
MODEL_ORDER: list[str] = [m["name"] for m in _models_data["models"]]

# Pretty names for axis tick labels and captions (matches 2025 paper figures).
MODEL_DISPLAY: dict[str, str] = {
    "Llama-3.2-1B":     "LLaMA-3.2-1B",
    "Qwen2.5-1.5B":     "Qwen2.5-1.5B",
    "granite-3.3-2B":   "Granite-3.3-2B",
    "gemma-2B":         "Gemma-2B",
    "Llama-3.2-3B":     "LLaMA-3.2-3B",
    "Qwen2.5-3B":       "Qwen2.5-3B",
    "Phi-3.5-mini":     "Phi-3.5-mini-4B",
    "Qwen2.5-7B":       "Qwen2.5-7B",
    "moxin-7B":         "Moxin-7B",
    "gemma-7B":         "Gemma-7B",
    "Ministral-8B":     "Ministral-8B",
    "Llama-3.1-8B":     "LLaMA-3.1-8B",
    "granite-3.3-8B":   "Granite-3.3-8B",
}

# Short two-letter family code + parameter count, used as x-axis tick labels
# on dense figures (e.g., the §2.2 motivation figure). The paper carries a
# model-spec table that maps these codes to the full names + architectural
# detail so reviewers can decode at a glance.
#   LL = LLaMA, QW = Qwen, GR = Granite, GE = Gemma,
#   PH = Phi,   MO = Moxin, MI = Ministral (Mistral family)
MODEL_CODE: dict[str, str] = {
    "Llama-3.2-1B":     "LL-1B",
    "Qwen2.5-1.5B":     "QW-1.5B",
    "granite-3.3-2B":   "GR-2B",
    "gemma-2B":         "GE-2B",
    "Llama-3.2-3B":     "LL-3B",
    "Qwen2.5-3B":       "QW-3B",
    "Phi-3.5-mini":     "PH-4B",
    "Qwen2.5-7B":       "QW-7B",
    "moxin-7B":         "MO-7B",
    "gemma-7B":         "GE-7B",
    "Ministral-8B":     "MI-8B",
    "Llama-3.1-8B":     "LL-8B",
    "granite-3.3-8B":   "GR-8B",
}

# Per-model categorical color cycle — useful when a single platform/dtype
# plot needs to distinguish models. 13 distinct hues, print-friendly.
MODEL_COLOR_CYCLE: list[str] = [
    PERSIAN_PALETTE[name] for name in (
        "Lajvard", "Firouzeh", "Yashm", "Zereshki", "Zaferani",
        "Arghavani", "Mesi", "Shangarfi", "Daryaei", "SabzSeyedi",
        "Khormaei", "Goli", "ToosiIrani",
    )
]


# -----------------------------------------------------------------------------
# Figure sizes — IISWC uses ACM 2-column. 1-col ~ 3.4", 2-col ~ 7.0".
# -----------------------------------------------------------------------------
FIG_SIZE_1COL = (3.4, 2.4)
FIG_SIZE_2COL = (7.0, 2.6)
FIG_SIZE_2COL_TALL = (7.0, 3.6)
FIG_SIZE_FULL = (7.0, 5.0)


# -----------------------------------------------------------------------------
# matplotlib defaults — apply once at the top of main.py
# -----------------------------------------------------------------------------
def apply_paper_defaults(use_latex: bool = False) -> None:
    """Apply IISWC-friendly matplotlib rcParams. Call once in main.

    LaTeX text rendering is on by default to match the 2025 paper figure style.
    `use_latex=True` requires a full TeX toolchain (latex + dvipng); the
    default renders every label with matplotlib's built-in mathtext so the
    artifact has no TeX dependency.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":     "serif",
        "font.size":       9,
        "axes.titlesize":  10,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
        "axes.grid":       True,
        "axes.axisbelow":  True,        # gridlines behind bars, not in front
        "grid.alpha":      0.4,
        "grid.linewidth":  0.5,
        "figure.dpi":      150,
        "savefig.dpi":     600,         # match 2025 figures
        "savefig.bbox":    "tight",
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })
    if use_latex:
        mpl.rcParams.update({
            "text.usetex":          True,
            "font.family":          "serif",
            "font.serif":           ["Computer Modern Roman"],
            "text.latex.preamble":  r"\usepackage{amsmath}",
        })


def latex_safe(text: str) -> str:
    """Return text unchanged (labels are mathtext-safe; no TeX escaping)."""
    return text
