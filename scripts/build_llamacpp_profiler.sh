#!/usr/bin/env bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# ==========================================================================
# Build the llama.cpp profiler (cpp_api_pipeline) for the current Jetson.
#
# Detects the platform from the hostname and selects the CUDA architecture
# (Xavier sm_72, Orin sm_87, Thor sm_110), checks the pinned llama.cpp
# submodule, fetches nlohmann/json.hpp if missing, and compiles.
#
# Requirements: cmake >= 3.18, CUDA toolkit (nvcc on PATH), C++17 compiler:
#   sudo apt install cmake build-essential wget
#
# Usage:   bash scripts/build_llamacpp_profiler.sh
#          CUDA_ARCH=101 bash scripts/build_llamacpp_profiler.sh  # override
# Build time: ~15-30 min on-device. Output:
#   inference/llamacpp_profiler/build/cpp_api_pipeline
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROFILER_DIR="$PROJECT_ROOT/inference/llamacpp_profiler"

# --- CUDA arch by platform (override with CUDA_ARCH=...) ---
if [ -z "${CUDA_ARCH:-}" ]; then
    HOSTNAME=$(hostname)
    case "$HOSTNAME" in
        *[Xx]avier*) CUDA_ARCH=72 ;;
        *[Oo]rin*)   CUDA_ARCH=87 ;;
        *[Tt]hor*)   CUDA_ARCH=110 ;;  # use CUDA_ARCH=101 if CUDA < 13.0
        *) echo "Unknown hostname '$HOSTNAME' - set CUDA_ARCH explicitly"; exit 1 ;;
    esac
fi

# --- Submodule check ---
if [ ! -f "$PROFILER_DIR/llama.cpp/CMakeLists.txt" ]; then
    echo "llama.cpp submodule not initialized - run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

# --- nlohmann/json.hpp ---
JSON_HPP="$PROFILER_DIR/nlohmann/json.hpp"
if [ ! -f "$JSON_HPP" ]; then
    echo "Fetching nlohmann/json.hpp..."
    mkdir -p "$PROFILER_DIR/nlohmann"
    wget -q -O "$JSON_HPP" \
      https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
fi

# --- Build ---
echo "Building cpp_api_pipeline (CUDA_ARCH=$CUDA_ARCH)..."
mkdir -p "$PROFILER_DIR/build"
cd "$PROFILER_DIR/build"
cmake .. -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
make -j"$(nproc)" cpp_api_pipeline
echo "Done: $PROFILER_DIR/build/cpp_api_pipeline"
