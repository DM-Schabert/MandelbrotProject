#!/usr/bin/env python3
"""
Benchmark sweep for Mandelbrot.py.

Runs Mandelbrot.py under mpirun for every combination of ranks, schedule,
comm mode and chunk size, a few times each, and writes one CSV row per run.
At the end it prints the median time per combination and the speed-up
compared with 1 rank.

Examples:

    # Default sweep (ranks 1..your core count, 500x500, 3 repeats)
    python3 benchmark.py

    # Quick check that everything works (tiny image, 1 repeat)
    python3 benchmark.py --quick

    # Choose exactly what to run
    python3 benchmark.py --ranks 1 2 4 --modes nonblocking nonblocking_perchunk \\
        --chunk-sizes 1 10 50 --size 1000x1000 --repeats 5

    # Negative limits need "=" so they aren't read as options
    python3 benchmark.py --xlim=-0.8:-0.7 --ylim=0.05:0.15

Notes:
  - Timings with more ranks than CPU cores are meaningless; the script warns
    if you ask for that.
  - The repeats are the outer loop, so slow drift (other programs, the CPU
    heating up) affects all combinations equally instead of just some.
  - Each row is written as soon as its run finishes, so a crash or Ctrl-C
    keeps everything measured so far.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "Mandelbrot.py"

COMM_MODES = ("blocking", "nonblocking", "blocking_perchunk", "nonblocking_perchunk")
SCHEDULES = ("block", "chunk")

# Columns of the RESULT line printed by Mandelbrot.py, in order
RESULT_FIELDS = ["ranks", "schedule", "comm_mode", "chunk_size", "size",
                 "t_total", "compute_max", "compute_mean", "imbalance", "t_wait"]
CSV_FIELDS = ["repeat"] + RESULT_FIELDS


# ============================================================
# Machine and MPI
# ============================================================

def usable_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def mpi_vendor():
    out = subprocess.run(
        [sys.executable, "-c", "from mpi4py import MPI; print(MPI.get_vendor()[0])"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() or "unknown"


def default_ranks(cores):
    """1, 2, 4, 8, ... up to the core count, plus the core count itself."""
    ranks, n = [], 1
    while n <= cores:
        ranks.append(n)
        n *= 2
    if ranks[-1] != cores:
        ranks.append(cores)
    return ranks


# ============================================================
# One run
# ============================================================

# mpirun merges the output of all ranks, and another rank's text can end up
# glued to the RESULT line (e.g. "Rank 1: 0.02 s for 30 rowsRESULT,2,..."),
# so search for the exact field pattern anywhere instead of whole lines.
_NUM = r"(\d+(?:\.\d+)?)"
RESULT_PATTERN = re.compile(
    r"RESULT,(\d+),(\w+),(\w+),(\d+),(\d+x\d+)," + ",".join([_NUM] * 5)
)


def parse_result(stdout):
    match = RESULT_PATTERN.search(stdout)
    if not match:
        raise RuntimeError("no RESULT line in the output")
    return dict(zip(RESULT_FIELDS, match.groups()))


def run_once(ranks, schedule, comm_mode, chunk_size, args, mpi_flags, workdir):
    cmd = ["mpirun", *mpi_flags, "-n", str(ranks),
           sys.executable, str(PROGRAM),
           str(chunk_size), args.size, args.xlim, args.ylim, schedule, comm_mode]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                          timeout=args.timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
        raise RuntimeError(f"exit code {proc.returncode}:\n{tail}")
    return parse_result(proc.stdout)


# ============================================================
# Summary
# ============================================================

def print_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["schedule"], row["comm_mode"], int(row["chunk_size"]), int(row["ranks"]))
        groups[key].append(row)

    median = {}
    for key, runs in groups.items():
        median[key] = (statistics.median(float(r["t_total"]) for r in runs),
                       statistics.median(float(r["imbalance"]) for r in runs),
                       len(runs))

    print("\nMedian over repeats (speed-up = time with 1 rank / time with n ranks,")
    print("for the same schedule, mode and chunk size):\n")
    header = f"{'schedule':8s} {'comm_mode':22s} {'chunk':>5s} {'ranks':>5s} " \
             f"{'t_total [s]':>11s} {'speed-up':>8s} {'imbalance':>9s} {'runs':>4s}"
    print(header)
    print("-" * len(header))
    for key in sorted(median):
        schedule, mode, chunk, ranks = key
        t, imb, n = median[key]
        base = median.get((schedule, mode, chunk, 1))
        speedup = f"{base[0] / t:8.2f}" if base else f"{'-':>8s}"
        print(f"{schedule:8s} {mode:22s} {chunk:5d} {ranks:5d} "
              f"{t:11.3f} {speedup} {imb:9.3f} {n:4d}")


# ============================================================
# Main
# ============================================================

def main():
    cores = usable_cores()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranks", type=int, nargs="+", default=None,
                   help=f"rank counts (default: {' '.join(map(str, default_ranks(cores)))})")
    p.add_argument("--schedules", nargs="+", default=list(SCHEDULES), choices=SCHEDULES)
    p.add_argument("--modes", nargs="+", default=list(COMM_MODES), choices=COMM_MODES)
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 10, 100])
    p.add_argument("--size", default="500x500", help="WIDTHxHEIGHT (default 500x500)")
    p.add_argument("--xlim", default="-2.2:0.75")
    p.add_argument("--ylim", default="-1.3:1.3")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600, help="seconds per run (default 600)")
    p.add_argument("--out", default=None,
                   help="CSV file (default: results/benchmark_<date>_<time>.csv)")
    p.add_argument("--quick", action="store_true",
                   help="tiny sanity sweep: 60x40 image, 1 repeat, chunk sizes 1 and 10")
    args = p.parse_args()

    if args.quick:
        args.size, args.repeats, args.chunk_sizes = "60x40", 1, [1, 10]
    ranks_list = args.ranks or default_ranks(cores)

    # Open MPI refuses more ranks than cores without --oversubscribe.
    vendor = mpi_vendor()
    mpi_flags = []
    if max(ranks_list) > cores:
        print(f"WARNING: up to {max(ranks_list)} ranks on {cores} cores. Those timings "
              f"are not meaningful (ranks share cores).", file=sys.stderr)
        if vendor == "Open MPI":
            mpi_flags = ["--oversubscribe"]

    out = Path(args.out) if args.out else \
        HERE / "results" / time.strftime("benchmark_%Y%m%d_%H%M%S.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    combos = list(product(ranks_list, args.schedules, args.modes, args.chunk_sizes))
    total = len(combos) * args.repeats
    print(f"MPI: {vendor}, cores: {cores}, image: {args.size}, "
          f"{len(combos)} combinations x {args.repeats} repeats = {total} runs")
    print(f"Writing to {out}\n")

    rows, failures, done = [], 0, 0
    started = time.time()
    with open(out, "w", newline="") as f, tempfile.TemporaryDirectory() as workdir:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            for repeat in range(1, args.repeats + 1):
                for ranks, schedule, mode, chunk in combos:
                    done += 1
                    label = f"[{done}/{total}] repeat {repeat}: {ranks} ranks, " \
                            f"{schedule}, {mode}, chunk {chunk}"
                    try:
                        result = run_once(ranks, schedule, mode, chunk, args,
                                          mpi_flags, workdir)
                    except (RuntimeError, subprocess.TimeoutExpired) as err:
                        failures += 1
                        print(f"{label}  FAILED: {err}", file=sys.stderr)
                        continue
                    row = {"repeat": repeat, **result}
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
                    print(f"{label}  {float(result['t_total']):.3f} s", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Everything measured so far is in the CSV.", file=sys.stderr)

    minutes = (time.time() - started) / 60
    print(f"\n{len(rows)} runs saved to {out} ({failures} failed, {minutes:.1f} min)")
    if rows:
        print_summary(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())