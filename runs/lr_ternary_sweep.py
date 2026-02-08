"""
LR Ternary Search Sweep

Finds optimal --matrix-lr for each (TPP, batch_size) config at d12.
Assumes val_bpb is convex in LR. Runs on a single node (8 GPUs).
Launches quickrun_muonh.sh for each run (handles venv, data, tokenizer).

Usage:
    python runs/lr_ternary_sweep.py

Configs ordered shortest-first (by training iterations):
    TPP=12/1M, TPP=12/0.5M, TPP=24/1M, TPP=24/0.5M,
    TPP=36/1M, TPP=36/0.5M, TPP=70/1M, TPP=70/0.5M

48 total training runs (8 configs * 3 ternary iters * 2 runs/iter).
Total: 268,380 training iterations.
"""

import os
import re
import csv
import subprocess
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

DEPTH = 12
NUM_TERNARY_ITERS = 3

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "xingyu20")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "nanochat_lr_sweep")

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = CACHE_DIR / "lr_sweep_logs"
RESULTS_DIR = CACHE_DIR / "lr_sweep_results"
QUICKRUN_SCRIPT = PROJECT_ROOT / "runs" / "quickrun_muonh.sh"

for d in [LOG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sweep configs ───────────────────────────────────────────────────────────
# (TPP, batch_size, lr_lo, lr_hi)
# NOTE: base_train.py internally scales matrix_lr by sqrt(batch_size / 524288),
# so we use the SAME CLI arg range for both batch sizes.
# Sorted shortest-first by training time (iters ~ TPP / BS)

CONFIGS = [
    (6,  1048576, 0.0120, 0.0250),  #    630 iters/run
    (6,  524288,  0.0120, 0.0250),  #  1,260 iters/run
    (12, 1048576, 0.0120, 0.0250),  #  1,260 iters/run
    (12, 524288,  0.0120, 0.0250),  #  2,520 iters/run
    (24, 1048576, 0.0100, 0.0150),  #  2,520 iters/run
    (24, 524288,  0.0100, 0.0150),  #  5,040 iters/run
    (36, 1048576, 0.0100, 0.0150),  #  3,780 iters/run
    (36, 524288,  0.0100, 0.0150),  #  7,560 iters/run
]


def fmt_bs(bs: int) -> str:
    if bs == 524288:
        return "0.5M"
    elif bs == 1048576:
        return "1.0M"
    return str(bs)


def run_training(tpp: int, bs: int, lr: float, run_name: str, log_file: Path) -> int:
    """Run training via quickrun_muonh.sh with env var overrides."""
    env = os.environ.copy()
    env.update({
        "DEPTH": str(DEPTH),
        "TARGET_RATIO": str(tpp),
        "TOTAL_BATCH_SIZE": str(bs),
        "MATRIX_LR": f"{lr:.6f}",
        "WANDB_RUN": run_name,
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "CORE_METRIC_EVERY": "-1",
        "SAMPLE_EVERY": "-1",
        "SAVE_EVERY": "-1",
    })

    cmd = ["bash", str(QUICKRUN_SCRIPT)]
    print(f"  RUN: {run_name}")
    print(f"  ENV: DEPTH={DEPTH} TARGET_RATIO={tpp} TOTAL_BATCH_SIZE={bs} MATRIX_LR={lr:.6f}")
    print(f"  LOG: {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), env=env,
        )
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            f.write(decoded)
        proc.wait()
    return proc.returncode


def extract_bpb(log_file: Path) -> float | None:
    """Extract minimum validation bpb from a training log file."""
    if not log_file.exists():
        return None
    text = log_file.read_text()
    matches = re.findall(r"Minimum validation bpb:\s*([\d.]+)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"Validation bpb:\s*([\d.]+)", text)
    if matches:
        return min(float(m) for m in matches)
    return None


def ternary_search(tpp: int, bs: int, lr_lo: float, lr_hi: float) -> dict:
    """Run ternary search for a single (TPP, batch_size) config."""
    bs_fmt = fmt_bs(bs)
    print()
    print("=" * 64)
    print(f"  TPP={tpp}  Batch={bs_fmt}  LR=[{lr_lo:.6f}, {lr_hi:.6f}]")
    print("=" * 64)

    lo, hi = lr_lo, lr_hi
    all_results = []  # (lr, bpb) pairs

    for it in range(1, NUM_TERNARY_ITERS + 1):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3

        print(f"\n  Iter {it}/{NUM_TERNARY_ITERS}  range=[{lo:.6f}, {hi:.6f}]")
        print(f"  m1={m1:.6f}  m2={m2:.6f}")

        # Run m1
        run_m1 = f"sweep_tpp{tpp}_bs{bs_fmt}_iter{it}_m1"
        log_m1 = LOG_DIR / f"{run_m1}.log"
        print(f"\n  --- Running m1 (LR={m1:.6f}) ---")
        run_training(tpp, bs, m1, run_m1, log_m1)
        bpb_m1 = extract_bpb(log_m1)
        if bpb_m1 is None:
            print(f"  WARNING: could not parse bpb from {log_m1}")
            bpb_m1 = float(input(f"  Enter val_bpb for m1 (LR={m1:.6f}): "))
        all_results.append((m1, bpb_m1))

        # Run m2
        run_m2 = f"sweep_tpp{tpp}_bs{bs_fmt}_iter{it}_m2"
        log_m2 = LOG_DIR / f"{run_m2}.log"
        print(f"\n  --- Running m2 (LR={m2:.6f}) ---")
        run_training(tpp, bs, m2, run_m2, log_m2)
        bpb_m2 = extract_bpb(log_m2)
        if bpb_m2 is None:
            print(f"  WARNING: could not parse bpb from {log_m2}")
            bpb_m2 = float(input(f"  Enter val_bpb for m2 (LR={m2:.6f}): "))
        all_results.append((m2, bpb_m2))

        print(f"\n  m1(LR={m1:.6f}) bpb={bpb_m1:.6f}  |  m2(LR={m2:.6f}) bpb={bpb_m2:.6f}")

        if bpb_m1 < bpb_m2:
            hi = m2
            print(f"  -> m1 wins. New range: [{lo:.6f}, {hi:.6f}]")
        else:
            lo = m1
            print(f"  -> m2 wins. New range: [{lo:.6f}, {hi:.6f}]")

    optimal_lr = (lo + hi) / 2
    best_lr, best_bpb = min(all_results, key=lambda x: x[1])

    print(f"\n  Done: TPP={tpp} Batch={bs_fmt}")
    print(f"    optimal LR ~ {optimal_lr:.6f} (range=[{lo:.6f}, {hi:.6f}])")
    print(f"    best observed: LR={best_lr:.6f} bpb={best_bpb:.6f}")

    return {
        "tpp": tpp,
        "batch_size": bs,
        "optimal_lr": f"{optimal_lr:.6f}",
        "final_lo": f"{lo:.6f}",
        "final_hi": f"{hi:.6f}",
        "best_lr": f"{best_lr:.6f}",
        "best_val_bpb": f"{best_bpb:.6f}",
    }


def main():
    print("=" * 64)
    print(f"  LR Ternary Search Sweep — single node")
    print(f"  d{DEPTH}, {NUM_TERNARY_ITERS} iters/config, {len(CONFIGS)} configs")
    print(f"  {len(CONFIGS) * NUM_TERNARY_ITERS * 2} total training runs")
    print("=" * 64)
    for i, (tpp, bs, lo, hi) in enumerate(CONFIGS):
        print(f"    {i+1}. TPP={tpp} Batch={fmt_bs(bs)} LR=[{lo:.6f}, {hi:.6f}]")

    results = []
    for cfg_idx, (tpp, bs, lr_lo, lr_hi) in enumerate(CONFIGS):
        print(f"\n{'#' * 64}")
        print(f"  Config {cfg_idx+1}/{len(CONFIGS)}")
        print(f"{'#' * 64}")
        result = ternary_search(tpp, bs, lr_lo, lr_hi)
        results.append(result)

    # Write results
    results_file = RESULTS_DIR / "results.csv"
    fieldnames = ["tpp", "batch_size", "optimal_lr", "final_lo", "final_hi", "best_lr", "best_val_bpb"]
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 64)
    print(f"  Sweep complete! Results: {results_file}")
    print("=" * 64)
    print()
    print(f"{'TPP':>5} {'Batch':>6} {'Optimal LR':>12} {'Best LR':>10} {'Best BPB':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['tpp']:>5} {fmt_bs(int(r['batch_size'])):>6} {r['optimal_lr']:>12} "
              f"{r['best_lr']:>10} {r['best_val_bpb']:>10}")


if __name__ == "__main__":
    main()
