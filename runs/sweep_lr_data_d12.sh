#!/bin/bash
# Learning rate and data size sweep for d12
# Sweeps over target_ratio (8, 12, 16, 20) and learning rate (0.01, 0.015, 0.02)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed parameters for d12
export DEPTH=12
export NUM_SHARDS=100  # Adjust based on d12 data needs

# Arrays for sweep
target_ratios=(8 10 12 16 20)
learning_rates=(0.01 0.015 0.02)

echo "=============================================="
echo "LR & Data Size Sweep for d12"
echo "=============================================="
echo "Target ratios: ${target_ratios[@]}"
echo "Learning rates: ${learning_rates[@]}"
echo "Total runs: $((${#target_ratios[@]} * ${#learning_rates[@]}))"
echo "=============================================="
echo ""

run_count=0
total_runs=$((${#target_ratios[@]} * ${#learning_rates[@]}))

# Nested loop over target ratios and learning rates
for target_ratio in "${target_ratios[@]}"; do
    for lr in "${learning_rates[@]}"; do
        run_count=$((run_count + 1))

        echo ""
        echo "=============================================="
        echo "Run $run_count/$total_runs"
        echo "Target ratio: $target_ratio, LR: $lr"
        echo "=============================================="

        # Set environment variables
        export TARGET_RATIO=$target_ratio
        export MATRIX_LR=$lr
        export WANDB_PROJECT="nanochat_scaling"
        export WANDB_ENTITY="xingyu20"
        export WANDB_RUN="d12_sweep_ratio${target_ratio}_lr${lr}"

        # Run the training
        bash "${SCRIPT_DIR}/quickrun_muonh.sh"

        echo ""
        echo "Completed run $run_count/$total_runs"
        echo ""
    done
done

echo ""
echo "=============================================="
echo "Sweep complete! All $total_runs runs finished."
echo "=============================================="
echo ""
echo "To analyze results and fit scaling law:"
echo "  python runs/analyze_lr_scaling.py"
