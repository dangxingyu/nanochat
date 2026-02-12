#!/bin/bash

# Stack training: train a 6-layer seed model, then stack layers to target depth and continue training.
#
# The seed model uses d12 width (768 dim, 6 heads) and trains for 1 TPP.
# After seed training, layers are stacked as [0,1,2,3,4,5, 0,1,2,3,4,5, ...] to target depth.
# Optimizer state is properly duplicated and re-sharded across ranks.
#
# Examples:
#   bash runs/stack_train.sh                     # default: stack to 24 layers
#   TARGET_DEPTH=12 bash runs/stack_train.sh     # stack to 12 layers
#   TARGET_DEPTH=48 bash runs/stack_train.sh     # stack to 48 layers

set -e

# -----------------------------------------------------------------------------
# Config

SEED_DEPTH="${SEED_DEPTH:-6}"
TARGET_DEPTH="${TARGET_DEPTH:-24}"
SEED_TPP="${SEED_TPP:-1.0}"
TARGET_RATIO="${TARGET_RATIO:-10.5}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"

# Seed uses d12 width: depth=6, aspect_ratio=128 -> 6*128=768
SEED_ASPECT_RATIO=128

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    NPROC_PER_NODE=1
fi

# Optimizer
MATRIX_OPTIMIZER="${MATRIX_OPTIMIZER:-hyperball}"
SCALAR_LR="${SCALAR_LR:-0.5}"
MATRIX_LR="${MATRIX_LR:-0.02}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.3}"
MATRIX_WARMDOWN_RATIO="${MATRIX_WARMDOWN_RATIO:-1.0}"

# AdamW
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
NORM_LR="${NORM_LR:-0.1}"

# Wandb
export WANDB_ENTITY="${WANDB_ENTITY:-xingyu20}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat}"

# FP8 (default enabled)
FP8="${FP8:-1}"
FP8_ARGS=""
if [ "${FP8:-0}" -eq 1 ]; then
    FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
    FP8_ARGS="--fp8 --fp8-recipe=${FP8_RECIPE}"
fi

# Shards
NUM_SHARDS="${NUM_SHARDS:-370}"

# -----------------------------------------------------------------------------
# Paths and cache

export OMP_NUM_THREADS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$NANOCHAT_BASE_DIR/torch_inductor"
export TRITON_CACHE_DIR="$NANOCHAT_BASE_DIR/triton"
export TMPDIR="$NANOCHAT_BASE_DIR/tmp"
mkdir -p "$NANOCHAT_BASE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

SEED_TAG="d${SEED_DEPTH}_seed"
SEED_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$SEED_TAG"
STACKED_TAG="${MODEL_TAG:-d${TARGET_DEPTH}_stacked}"

# -----------------------------------------------------------------------------
# Print summary

echo "=============================================="
echo "Stack Training"
echo "=============================================="
echo "Project root:      $PROJECT_ROOT"
echo "Seed depth:        $SEED_DEPTH (aspect_ratio=$SEED_ASPECT_RATIO -> width 768)"
echo "Seed TPP:          $SEED_TPP"
echo "Target depth:      $TARGET_DEPTH"
echo "Target ratio:      $TARGET_RATIO"
echo "Window pattern:    $WINDOW_PATTERN"
echo "Num GPUs:          $NPROC_PER_NODE"
echo "Matrix optimizer:  $MATRIX_OPTIMIZER"
echo "Seed checkpoint:   $SEED_DIR"
echo "Stacked tag:       $STACKED_TAG"
if [ "${FP8:-0}" -eq 1 ]; then
    echo "FP8:               enabled ($FP8_RECIPE)"
fi
echo "=============================================="

cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Python venv

if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Data + tokenizer

echo ""
echo "Downloading $NUM_SHARDS data shards..."
python -m nanochat.dataset -n "$NUM_SHARDS"

echo ""
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Tokenizer already exists at $TOKENIZER_DIR, skipping training."
else
    echo "Training tokenizer..."
    python -m scripts.tok_train --max-chars=500000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
# Phase 1: Train seed model (6 layers, 768 width, 1 TPP)

echo ""
echo "=============================================="
echo "Phase 1: Training seed model (depth=$SEED_DEPTH, TPP=$SEED_TPP)"
echo "=============================================="

SEED_WANDB_RUN="${WANDB_RUN:-stack_seed_d${SEED_DEPTH}_tpp${SEED_TPP}}"

SEED_ARGS=(
    --depth=$SEED_DEPTH
    --aspect-ratio=$SEED_ASPECT_RATIO
    --run=$SEED_WANDB_RUN
    --model-tag=$SEED_TAG
    --window-pattern=$WINDOW_PATTERN
    --target-param-data-ratio=$SEED_TPP
    --device-batch-size=$DEVICE_BATCH_SIZE
    --total-batch-size=$TOTAL_BATCH_SIZE
    --matrix-optimizer=$MATRIX_OPTIMIZER
    --matrix-lr=$MATRIX_LR
    --warmdown-ratio=$WARMDOWN_RATIO
    --matrix-warmdown-ratio=$MATRIX_WARMDOWN_RATIO
    --embedding-lr=$EMBEDDING_LR
    --unembedding-lr=$UNEMBEDDING_LR
    --norm-lr=$NORM_LR
    --scalar-lr=$SCALAR_LR
    --core-metric-every=-1
    --sample-every=-1
    --save-every=-1
    --eval-every=50
)

if [ -d "$SEED_DIR" ] && ls "$SEED_DIR"/model_*.pt 1> /dev/null 2>&1; then
    echo "Seed checkpoint already exists at $SEED_DIR, skipping seed training."
else
    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            "${SEED_ARGS[@]}" $FP8_ARGS
    else
        python -m scripts.base_train \
            "${SEED_ARGS[@]}" $FP8_ARGS
    fi
fi

# -----------------------------------------------------------------------------
# Phase 2: Stack and continue training

echo ""
echo "=============================================="
echo "Phase 2: Stacking $SEED_DEPTH -> $TARGET_DEPTH layers and continuing training"
echo "=============================================="

STACK_WANDB_RUN="${WANDB_RUN:-stack_d${TARGET_DEPTH}_from_d${SEED_DEPTH}}"

STACK_ARGS=(
    --depth=$TARGET_DEPTH
    --run=$STACK_WANDB_RUN
    --model-tag=$STACKED_TAG
    --stack-from=$SEED_DIR
    --window-pattern=$WINDOW_PATTERN
    --target-param-data-ratio=$TARGET_RATIO
    --device-batch-size=$DEVICE_BATCH_SIZE
    --total-batch-size=$TOTAL_BATCH_SIZE
    --matrix-optimizer=$MATRIX_OPTIMIZER
    --matrix-lr=$MATRIX_LR
    --warmdown-ratio=$WARMDOWN_RATIO
    --matrix-warmdown-ratio=$MATRIX_WARMDOWN_RATIO
    --embedding-lr=$EMBEDDING_LR
    --unembedding-lr=$UNEMBEDDING_LR
    --norm-lr=$NORM_LR
    --scalar-lr=$SCALAR_LR
    --core-metric-every=${CORE_METRIC_EVERY:-2000}
    --sample-every=${SAMPLE_EVERY:--1}
    --save-every=${SAVE_EVERY:--1}
    --warmup-ratio=0.05
)

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        "${STACK_ARGS[@]}" $FP8_ARGS
else
    python -m scripts.base_train \
        "${STACK_ARGS[@]}" $FP8_ARGS
fi

echo ""
echo "=============================================="
echo "Stack training complete!"
echo "=============================================="
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/base_checkpoints/${STACKED_TAG}/"
