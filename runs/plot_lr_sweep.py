"""
Plot LR sweep results with quadratic fits.

For each (TPP, batch_size) config:
  1. Extract all (LR, val_bpb) data points from log files
  2. Fit a quadratic: bpb = a*lr^2 + b*lr + c
  3. Plot the quadratic curve with data points

Then fit a power law to the quadratic-predicted optimal LRs vs TPP.

Usage:
    python runs/plot_lr_sweep.py
"""

import re
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "cache" / "lr_sweep_logs"
PLOT_DIR = PROJECT_ROOT / "cache" / "lr_sweep_results"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def extract_lr_and_bpb(log_file: Path):
    """Extract Matrix LR and Minimum validation bpb from a log file."""
    text = log_file.read_text()
    # Matrix LR from quickrun header (may have ANSI codes)
    lr_match = re.search(r"Matrix LR:\s*([\d.]+)", text)
    # Minimum validation bpb from training output
    bpb_match = re.search(r"Minimum validation bpb:\s*([\d.]+)", text)
    if lr_match and bpb_match:
        return float(lr_match.group(1)), float(bpb_match.group(1))
    return None, None


def parse_log_filename(name: str):
    """Parse sweep_tpp{TPP}_bs{BS}_iter{I}_{m1|m2}.log"""
    m = re.match(r"sweep_(?:n\d+_)?tpp(\d+)_bs([\d.]+M)_iter(\d+)_(m[12])\.log", name)
    if m:
        tpp = int(m.group(1))
        bs_str = m.group(2)
        bs = 524288 if bs_str == "0.5M" else 1048576
        return tpp, bs
    return None, None


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def power_law(x, a, b):
    return a * np.power(x, b)


def main():
    # Collect all (LR, bpb) data points grouped by (tpp, bs)
    data = defaultdict(list)  # (tpp, bs) -> [(lr, bpb), ...]

    for log_file in sorted(LOG_DIR.glob("sweep_*.log")):
        tpp, bs = parse_log_filename(log_file.name)
        if tpp is None:
            continue
        lr, bpb = extract_lr_and_bpb(log_file)
        if lr is not None and bpb is not None:
            data[(tpp, bs)].append((lr, bpb))

    if not data:
        print("No data found!")
        return

    # Sort configs
    configs = sorted(data.keys())
    tpp_values = sorted(set(t for t, _ in configs))
    bs_values = sorted(set(b for _, b in configs))

    print(f"Found {len(configs)} configs: {configs}")
    for key, points in sorted(data.items()):
        print(f"  {key}: {len(points)} points")
        for lr, bpb in sorted(points):
            print(f"    LR={lr:.6f}  bpb={bpb:.6f}")

    # ── Quadratic fits and collect optimal LRs ──────────────────────────────
    quad_results = {}  # (tpp, bs) -> (optimal_lr, optimal_bpb, popt)

    # Plot quadratics: one subplot per (tpp, bs) config
    n_configs = len(configs)
    ncols = 2
    nrows = (n_configs + 1) // 2
    fig1, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for idx, (tpp, bs) in enumerate(configs):
        ax = axes[idx]
        points = sorted(data[(tpp, bs)])
        lrs = np.array([p[0] for p in points])
        bpbs = np.array([p[1] for p in points])
        bs_fmt = "0.5M" if bs == 524288 else "1.0M"

        ax.scatter(lrs, bpbs, color="tab:blue", s=60, zorder=5, label="data")

        # Fit quadratic
        try:
            popt, _ = curve_fit(quadratic, lrs, bpbs, p0=[1000, -30, 1.0])
            a, b, c = popt

            # Quadratic minimum: lr* = -b / (2a)
            opt_lr = -b / (2 * a)
            opt_bpb = quadratic(opt_lr, a, b, c)
            quad_results[(tpp, bs)] = (opt_lr, opt_bpb, popt)

            # Plot fit curve
            lr_range = np.linspace(min(lrs) * 0.9, max(lrs) * 1.1, 200)
            bpb_fit = quadratic(lr_range, a, b, c)
            ax.plot(lr_range, bpb_fit, "r--", alpha=0.7, label=f"quad fit")
            ax.axvline(opt_lr, color="green", linestyle=":", alpha=0.7,
                       label=f"opt LR={opt_lr:.5f}")
            ax.scatter([opt_lr], [opt_bpb], color="green", s=100, marker="*", zorder=6)

            print(f"\nTPP={tpp} BS={bs_fmt}: bpb = {a:.1f}*lr^2 + {b:.2f}*lr + {c:.4f}")
            print(f"  Optimal LR = {opt_lr:.6f}, Predicted BPB = {opt_bpb:.6f}")
        except Exception as e:
            print(f"Quadratic fit failed for TPP={tpp} BS={bs_fmt}: {e}")

        ax.set_xlabel("Matrix LR")
        ax.set_ylabel("Min val BPB")
        ax.set_title(f"TPP={tpp}, BS={bs_fmt}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(configs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig1.savefig(PLOT_DIR / "quadratic_fits.png", dpi=150, bbox_inches="tight")
    print(f"\nQuadratic fits plot saved to: {PLOT_DIR / 'quadratic_fits.png'}")

    # ── Power law fit: optimal LR vs TPP ────────────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for bs, color, marker in [(524288, "tab:blue", "o"), (1048576, "tab:orange", "s")]:
        bs_fmt = "0.5M" if bs == 524288 else "1.0M"
        tpps, opt_lrs, opt_bpbs = [], [], []
        for tpp in tpp_values:
            if (tpp, bs) in quad_results:
                opt_lr, opt_bpb, _ = quad_results[(tpp, bs)]
                tpps.append(tpp)
                opt_lrs.append(opt_lr)
                opt_bpbs.append(opt_bpb)

        tpps = np.array(tpps)
        opt_lrs = np.array(opt_lrs)
        opt_bpbs = np.array(opt_bpbs)

        # Plot 1: Optimal LR vs TPP
        ax1.scatter(tpps, opt_lrs, color=color, s=80, marker=marker, zorder=5)

        try:
            popt, pcov = curve_fit(power_law, tpps, opt_lrs, p0=[0.05, -0.3])
            a, b = popt
            perr = np.sqrt(np.diag(pcov))
            tpp_fit = np.linspace(min(tpps) * 0.8, max(tpps) * 1.2, 100)
            ax1.plot(tpp_fit, power_law(tpp_fit, a, b), color=color, linestyle="--",
                     label=f"BS={bs_fmt}: {a:.4f} * TPP^({b:.3f}±{perr[1]:.3f})")
            print(f"\nPower law (BS={bs_fmt}): lr = {a:.6f} * TPP^({b:.4f} ± {perr[1]:.4f})")
        except Exception as e:
            ax1.plot(tpps, opt_lrs, color=color, linestyle="-", alpha=0.5,
                     label=f"BS={bs_fmt}")
            print(f"Power law fit failed for BS={bs_fmt}: {e}")

        # Plot 2: Optimal BPB vs TPP
        ax2.scatter(tpps, opt_bpbs, color=color, s=80, marker=marker, zorder=5)
        try:
            popt, _ = curve_fit(power_law, tpps, opt_bpbs, p0=[1.0, -0.05])
            a, b = popt
            tpp_fit = np.linspace(min(tpps) * 0.8, max(tpps) * 1.2, 100)
            ax2.plot(tpp_fit, power_law(tpp_fit, a, b), color=color, linestyle="--",
                     label=f"BS={bs_fmt}: {a:.4f} * TPP^({b:.3f})")
            print(f"Power law BPB (BS={bs_fmt}): bpb = {a:.6f} * TPP^({b:.4f})")
        except Exception as e:
            ax2.plot(tpps, opt_bpbs, color=color, linestyle="-", alpha=0.5,
                     label=f"BS={bs_fmt}")

    ax1.set_xlabel("TPP (tokens per parameter)", fontsize=12)
    ax1.set_ylabel("Optimal matrix LR (quadratic min)", fontsize=12)
    ax1.set_title("Optimal LR vs TPP (d12)", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("TPP (tokens per parameter)", fontsize=12)
    ax2.set_ylabel("Predicted optimal BPB", fontsize=12)
    ax2.set_title("Optimal BPB vs TPP (d12)", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(PLOT_DIR / "power_law_fits.png", dpi=150, bbox_inches="tight")
    print(f"Power law plot saved to: {PLOT_DIR / 'power_law_fits.png'}")


if __name__ == "__main__":
    main()
