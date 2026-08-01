# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
#############################################################################
# data_unifier_parallel.py
#
# Unify INFO timing CSVs and TGS tegrastats logs produced by the Hydra
# profilers (HuggingFace and llama.cpp) into per-prompt UNIFIED CSVs that
# carry both timing and hardware-telemetry aggregates.
#
# Per platform:
#   - Orin   : inference.telemetry.tegrastats_parser_orin
#   - Xavier : inference.telemetry.tegrastats_parser_xavier
#   - Thor   : inference.telemetry.tegrastats_parser_thor
#
# All three parsers expose the same `parser(input_file, output_file)` API and
# emit a `timestamp` column in int64 nanoseconds since epoch UTC, so once the
# right parser is selected the rest of the pipeline is platform-agnostic.
#
# The TGS file format is also backend-agnostic on a given platform: HF's
# Python tegrastats logger and llama.cpp's C++ logger produce the same
# wall-clock prefix, so the same parser handles both backends.
#
# CLI:
#   python3 -m analysis.unifier.data_unifier_parallel \
#       --platform {orin,xavier,thor} \
#       --results-dir <path-to-INFO/TGS-csvs> \
#       [--force] [--max-processes N]
#
# Output: one UNIFIED_<rest>.csv per INFO_<rest>.csv, written next to the
# inputs. Idempotent — files whose UNIFIED already exists are skipped unless
# --force is passed.
#############################################################################

import argparse
import os
import shutil
import sys
import tempfile
import warnings
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _get_parser(platform: str):
    """Return the platform-specific tegrastats parser callable.

    Lazy-imports so an unused parser doesn't pull in its module.
    """
    if platform == "orin":
        from inference.telemetry.tegrastats_parser_orin import parser
        return parser
    if platform == "xavier":
        from inference.telemetry.tegrastats_parser_xavier import parser
        return parser
    if platform == "thor":
        from inference.telemetry.tegrastats_parser_thor import parser
        return parser
    raise ValueError(f"Unknown platform: {platform!r}")


def unify_data(info_path: Path, tmp_dir: Path, platform: str, force: bool):
    """Unify one INFO/TGS pair into a UNIFIED CSV next to the INFO file.

    Runs in a subprocess. Any exception is caught and re-raised as
    SystemExit(1) so the parent can attribute failures via process.exitcode.
    The exception traceback is printed first so the cause is visible.
    """
    try:
        results_dir = info_path.parent
        unified_path = results_dir / info_path.name.replace("INFO", "UNIFIED")
        if unified_path.exists() and not force:
            print(f"[PID {os.getpid()}] SKIP {info_path.name} (UNIFIED exists)")
            return

        tgs_path = results_dir / info_path.name.replace("INFO", "TGS")
        if not tgs_path.exists():
            raise FileNotFoundError(f"TGS file missing: {tgs_path}")

        df_info = pd.read_csv(info_path)
        # NOTE: the legacy HF profiler set START once before the prompt loop,
        # so all rows shared the same start_time. We used to reconstruct
        # per-prompt starts here via end_time - estimated_e2e * 1e9. That
        # reconstruction is now baked into the INFO CSVs themselves by
        # scripts/correct_old_hf_results.py. New profiler runs already write
        # per-prompt start_time. So no in-unifier reconstruction is needed.

        # Parse TGS into a tmp CSV (cached per run via mkdtemp).
        tgs_tmp = tmp_dir / (tgs_path.name + ".tmp")
        if not tgs_tmp.exists():
            print(f"[PID {os.getpid()}] PARSE {tgs_path.name}")
            tegrastats_parser = _get_parser(platform)
            tegrastats_parser(tgs_path, tgs_tmp)

        df_tgs = pd.read_csv(tgs_tmp)

        # Downsample very large parsed TGS files (>=5 GiB) to keep memory sane.
        if tgs_tmp.stat().st_size >> 30 >= 5:
            df_tgs.drop(
                [i for i in range(0, df_tgs.shape[0], 5)],
                axis=0,
                inplace=True,
            )

        # 1. PROMPT interval — full [start_time, end_time] window per prompt.
        prompt_intervals = pd.IntervalIndex.from_arrays(
            df_info["start_time"], df_info["end_time"], closed="both"
        )
        df_tgs["prompt_id"] = prompt_intervals.get_indexer(df_tgs["timestamp"])

        # 2. PREFILL interval — start_time -> end of prefill GPU work.
        # We use (tokenization_time + prefill_time) as the phase boundary
        # rather than time_to_first_token. Prefill/decode is the mainstream
        # phase split used in LLM-serving literature (Splitwise, DistServe,
        # TokenPowerBench): prefill is compute-bound, decode is memory-bound.
        # TTFT additionally includes the first generated token's argmax +
        # text-decode, which is decode-like work; including it would
        # contaminate the "compute-bound" window with decode-like behavior.
        # We still report time_to_first_token as a first-class latency
        # metric — it is just no longer the phase boundary.
        prefill_durations = (
            (df_info["tokenization_time"] + df_info["prefill_time"]).values * 1e9
        )
        prefill_ends = df_info["start_time"].values + prefill_durations.astype(np.int64)
        prefill_intervals = pd.IntervalIndex.from_arrays(
            df_info["start_time"], prefill_ends, closed="both"
        )
        df_tgs["prefill_id"] = prefill_intervals.get_indexer(df_tgs["timestamp"])

        # 3. DECODE interval — prefill-end -> end_time.
        decode_intervals = pd.IntervalIndex.from_arrays(
            prefill_ends, df_info["end_time"].values, closed="both"
        )
        df_tgs["decode_id"] = decode_intervals.get_indexer(df_tgs["timestamp"])

        # Aggregate per phase. groupby drops the -1 indexer rows (samples
        # outside any interval) by default for negative integer keys; we
        # filter explicitly to keep behavior obvious.
        in_prompt = df_tgs[df_tgs["prompt_id"] >= 0]
        in_prefill = df_tgs[df_tgs["prefill_id"] >= 0]
        in_decode = df_tgs[df_tgs["decode_id"] >= 0]

        # pandas 2.0+ raises on mean/std over object columns; numeric_only=True
        # mirrors the legacy implicit-skip behavior.
        df_prompt_mean = (
            in_prompt.groupby("prompt_id").mean(numeric_only=True)
            .add_suffix("_prompt_mean")
        )
        df_prompt_std = (
            in_prompt.groupby("prompt_id").std(numeric_only=True)
            .add_suffix("_prompt_std")
        )
        df_prefill_mean = (
            in_prefill.groupby("prefill_id").mean(numeric_only=True)
            .add_suffix("_prefill_mean")
        )
        df_prefill_std = (
            in_prefill.groupby("prefill_id").std(numeric_only=True)
            .add_suffix("_prefill_std")
        )
        df_decode_mean = (
            in_decode.groupby("decode_id").mean(numeric_only=True)
            .add_suffix("_decode_mean")
        )
        df_decode_std = (
            in_decode.groupby("decode_id").std(numeric_only=True)
            .add_suffix("_decode_std")
        )

        merged_stats = pd.concat(
            [
                df_prompt_mean,
                df_prompt_std,
                df_prefill_mean,
                df_prefill_std,
                df_decode_mean,
                df_decode_std,
            ],
            axis=1,
        )

        # Concat by row index — df_info index is 0..N-1, group keys are the
        # same prompt-row indices, so axis=1 alignment is correct.
        new_df_info = pd.concat([df_info, merged_stats], axis=1)
        new_df_info.to_csv(unified_path, index=False)
        print(f"[PID {os.getpid()}] OK   {unified_path.name}")

    except Exception as e:
        import traceback
        print(
            f"[PID {os.getpid()}] FAIL {info_path.name}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc()
        # Non-zero exit so parent's p.exitcode reflects failure.
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Unify INFO timing CSVs and TGS tegrastats logs into UNIFIED CSVs."
    )
    ap.add_argument(
        "--platform",
        required=True,
        choices=["orin", "xavier", "thor"],
        help="Platform whose tegrastats parser to use",
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Directory containing INFO_*.csv and TGS_*.csv files",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if UNIFIED_*.csv already exists",
    )
    ap.add_argument(
        "--max-processes",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Concurrent worker cap (default: half of CPU count)",
    )
    args = ap.parse_args()

    if not args.results_dir.exists():
        sys.exit(f"results-dir does not exist: {args.results_dir}")

    info_files = sorted(args.results_dir.glob("INFO_*.csv"))
    if not info_files:
        sys.exit(f"No INFO_*.csv files in {args.results_dir}")

    print(f"Platform:       {args.platform}")
    print(f"Results dir:    {args.results_dir}")
    print(f"INFO files:     {len(info_files)}")
    print(f"Max workers:    {args.max_processes}")
    print(f"Force re-run:   {args.force}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="hydra_unifier_"))
    print(f"Tmp parse dir:  {tmp_dir}")

    processes: list[tuple[Process, Path]] = []
    completed: list[Path] = []
    failed: list[Path] = []

    def reap_one(blocking: bool = True):
        """Wait for at least one process to finish, then collect outcomes."""
        nonlocal processes
        still_running = []
        for proc, path in processes:
            if blocking and proc is processes[0][0]:
                proc.join()
            elif proc.is_alive():
                still_running.append((proc, path))
                continue
            if proc.exitcode == 0:
                completed.append(path)
            else:
                failed.append(path)
        processes = still_running

    try:
        for info_path in info_files:
            # Pre-skip locally so we don't fork a process just to skip.
            unified = info_path.parent / info_path.name.replace("INFO", "UNIFIED")
            if unified.exists() and not args.force:
                print(f"SKIP {info_path.name} (UNIFIED exists)")
                continue

            while len(processes) >= args.max_processes:
                # Block on the oldest process to free a worker slot.
                proc, path = processes.pop(0)
                proc.join()
                if proc.exitcode == 0:
                    completed.append(path)
                else:
                    failed.append(path)

            p = Process(
                target=unify_data,
                args=(info_path, tmp_dir, args.platform, args.force),
            )
            p.start()
            processes.append((p, info_path))
            print(f"START {info_path.name} (pid {p.pid})")

        # Drain remaining workers.
        for proc, path in processes:
            proc.join()
            if proc.exitcode == 0:
                completed.append(path)
            else:
                failed.append(path)
        processes = []

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Skipped = INFO files for which UNIFIED already existed and we didn't fork.
    n_skipped = len(info_files) - len(completed) - len(failed)

    print()
    print("=" * 56)
    print(f"  Unifier summary — {args.platform} @ {args.results_dir.name}")
    print("=" * 56)
    print(f"  Completed: {len(completed)}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        for f in failed:
            print(f"    FAIL: {f.name}")
    print("=" * 56)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
