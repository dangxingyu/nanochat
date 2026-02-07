#!/usr/bin/env python3
"""
Analyze the LR and data size sweep results and fit a scaling law:
    best_lr ~ (target_ratio)^{-alpha}

This script:
1. Fetches runs from wandb matching the sweep pattern
2. Identifies the best LR for each target_ratio (based on final validation loss)
3. Fits a power law: best_lr = C * (target_ratio)^{-alpha}
4. Plots the results with the fitted curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import wandb

# Configuration
WANDB_ENTITY = "xingyu20"
WANDB_PROJECT = "nanochat"
SWEEP_PREFIX = "d12_sweep"

def power_law(x, C, alpha):
    """Power law: y = C * x^{-alpha}"""
    return C * x ** (-alpha)

def fetch_sweep_results():
    """Fetch all runs from the sweep and organize by target_ratio and lr."""
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    sweep_data = {}

    for run in runs:
        if run.name.startswith(SWEEP_PREFIX):
            # Parse run name: d12_sweep_ratio{ratio}_lr{lr}
            try:
                parts = run.name.split("_")
                ratio_str = [p for p in parts if p.startswith("ratio")][0]
                lr_str = [p for p in parts if p.startswith("lr")][0]

                target_ratio = float(ratio_str.replace("ratio", ""))
                lr = float(lr_str.replace("lr", ""))

                # Get final validation loss
                history = run.scan_history(keys=["val_loss"])
                val_losses = [row["val_loss"] for row in history if "val_loss" in row]

                if val_losses:
                    final_val_loss = val_losses[-1]

                    if target_ratio not in sweep_data:
                        sweep_data[target_ratio] = {}

                    sweep_data[target_ratio][lr] = {
                        "val_loss": final_val_loss,
                        "run_name": run.name,
                        "run_id": run.id
                    }
                    print(f"Found: {run.name} - ratio={target_ratio}, lr={lr}, val_loss={final_val_loss:.4f}")
            except (IndexError, ValueError) as e:
                print(f"Skipping run {run.name}: {e}")
                continue

    return sweep_data

def find_best_lr_per_ratio(sweep_data):
    """Find the LR with the lowest validation loss for each target_ratio."""
    best_lrs = {}

    for target_ratio, lr_results in sweep_data.items():
        best_lr = min(lr_results.items(), key=lambda x: x[1]["val_loss"])
        best_lrs[target_ratio] = {
            "lr": best_lr[0],
            "val_loss": best_lr[1]["val_loss"],
            "run_name": best_lr[1]["run_name"]
        }
        print(f"Target ratio {target_ratio}: best LR = {best_lr[0]}, val_loss = {best_lr[1]['val_loss']:.4f}")

    return best_lrs

def fit_scaling_law(best_lrs):
    """Fit power law: best_lr = C * (target_ratio)^{-alpha}"""
    ratios = np.array(sorted(best_lrs.keys()))
    lrs = np.array([best_lrs[r]["lr"] for r in ratios])

    # Fit the power law
    popt, pcov = curve_fit(power_law, ratios, lrs, p0=[0.02, 0.5])
    C, alpha = popt

    # Calculate R^2
    residuals = lrs - power_law(ratios, C, alpha)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((lrs - np.mean(lrs))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\n{'='*60}")
    print(f"Scaling Law Fit: best_lr = C * (target_ratio)^(-alpha)")
    print(f"{'='*60}")
    print(f"C     = {C:.6f}")
    print(f"alpha = {alpha:.6f}")
    print(f"R^2   = {r_squared:.6f}")
    print(f"{'='*60}\n")

    return C, alpha, r_squared, ratios, lrs

def plot_results(ratios, lrs, C, alpha, r_squared):
    """Plot the data points and fitted curve."""
    plt.figure(figsize=(10, 6))

    # Data points
    plt.scatter(ratios, lrs, s=100, alpha=0.7, label="Best LR (from sweep)")

    # Fitted curve
    ratio_range = np.linspace(ratios.min() * 0.9, ratios.max() * 1.1, 100)
    fitted_lrs = power_law(ratio_range, C, alpha)
    plt.plot(ratio_range, fitted_lrs, 'r--', linewidth=2,
             label=f"Fit: {C:.4f} × (ratio)^(-{alpha:.3f})\n$R^2$ = {r_squared:.4f}")

    # Formatting
    plt.xlabel("Target Ratio (params/data)", fontsize=12)
    plt.ylabel("Best Learning Rate", fontsize=12)
    plt.title("Learning Rate Scaling with Data Size (d12)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_path = "lr_scaling_fit.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

    # Also try log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ratios, lrs, 'o', markersize=10, alpha=0.7, label="Best LR (from sweep)")
    plt.loglog(ratio_range, fitted_lrs, 'r--', linewidth=2,
               label=f"Fit: {C:.4f} × (ratio)^(-{alpha:.3f})\n$R^2$ = {r_squared:.4f}")
    plt.xlabel("Target Ratio (params/data)", fontsize=12)
    plt.ylabel("Best Learning Rate", fontsize=12)
    plt.title("Learning Rate Scaling (Log-Log Plot)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    output_path_log = "lr_scaling_fit_loglog.png"
    plt.savefig(output_path_log, dpi=300)
    print(f"Log-log plot saved to: {output_path_log}")

def main():
    print("Fetching sweep results from wandb...")
    sweep_data = fetch_sweep_results()

    if not sweep_data:
        print("No sweep results found. Make sure the sweep has completed and runs are logged to wandb.")
        return

    print(f"\nFound results for {len(sweep_data)} target ratios")

    print("\nFinding best LR for each target ratio...")
    best_lrs = find_best_lr_per_ratio(sweep_data)

    if len(best_lrs) < 2:
        print("Need at least 2 data points to fit a curve.")
        return

    print("\nFitting scaling law...")
    C, alpha, r_squared, ratios, lrs = fit_scaling_law(best_lrs)

    print("Generating plots...")
    plot_results(ratios, lrs, C, alpha, r_squared)

    print("\nAnalysis complete!")
    print(f"\nRecommended LR formula: best_lr = {C:.6f} * (target_ratio)^(-{alpha:.3f})")

if __name__ == "__main__":
    main()
