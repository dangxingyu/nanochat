#!/bin/bash

# Sweep Phase 1 constant LR multiplier for d6 -> d12 stacking
# Runs 3 experiments sequentially with different seed-lr-multiplier values

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$NANOCHAT_BASE_DIR/torch_inductor"
export TRITON_CACHE_DIR="$NANOCHAT_BASE_DIR/triton"
export TMPDIR="$NANOCHAT_BASE_DIR/tmp"
mkdir -p "$NANOCHAT_BASE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

export WANDB_ENTITY="${WANDB_ENTITY:-xingyu20}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    NPROC_PER_NODE=1
fi

# FP8
FP8_ARGS=""
if [ "${FP8:-1}" -eq 1 ]; then
    FP8_ARGS="--fp8 --fp8-recipe=${FP8_RECIPE:-tensorwise}"
fi

cd "$PROJECT_ROOT"
source .venv/bin/activate

# Shared config
COMMON_ARGS=(
    --seed-depth=6
    --target-depth=12
    --seed-tpp=2.0
    --aspect-ratio=64
    --window-pattern=SSSL
    --target-param-data-ratio=10.5
    --device-batch-size=16
    --total-batch-size=524288
    --matrix-optimizer=hyperball
    --matrix-lr=0.02
    --warmdown-ratio=0.3
    --matrix-warmdown-ratio=1.0
    --embedding-lr=0.3
    --unembedding-lr=0.004
    --norm-lr=0.1
    --scalar-lr=0.5
    --core-metric-every=2000
    --sample-every=-1
    --save-every=-1
)

for SEED_LR in 1.0; do
    echo ""
    echo "=============================================="
    echo "Sweep: seed-lr-multiplier=${SEED_LR}"
    echo "=============================================="

    RUN_NAME="stack_d6_d12_seedlr${SEED_LR}"
    MODEL_TAG="d12_stacked_seedlr${SEED_LR}"

    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.stack_train -- \
            "${COMMON_ARGS[@]}" \
            --seed-lr-multiplier=$SEED_LR \
            --run=$RUN_NAME \
            --model-tag=$MODEL_TAG \
            $FP8_ARGS
    else
        python -m scripts.stack_train \
            "${COMMON_ARGS[@]}" \
            --seed-lr-multiplier=$SEED_LR \
            --run=$RUN_NAME \
            --model-tag=$MODEL_TAG \
            $FP8_ARGS
    fi
done

echo ""
echo "=============================================="
echo "Sweep complete!"
echo "=============================================="
