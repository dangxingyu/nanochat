#!/bin/bash

# Debug: Stack 6 -> 12 with seed TPP=2.0

set -e

# -----------------------------------------------------------------------------
# Config

SEED_DEPTH="${SEED_DEPTH:-6}"
TARGET_DEPTH="${TARGET_DEPTH:-12}"
SEED_TPP="${SEED_TPP:-4.0}"
TARGET_RATIO="${TARGET_RATIO:-10.5}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"  # n_embd will be auto-computed from target_depth
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"

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
STACK_WARMUP_RATIO="${STACK_WARMUP_RATIO:-0.05}"

# AdamW
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
NORM_LR="${NORM_LR:-0.1}"

# Wandb
export WANDB_ENTITY="${WANDB_ENTITY:-xingyu20}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat}"
WANDB_RUN="${WANDB_RUN:-stack_d6_d12_4.0_seed}"

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

# -----------------------------------------------------------------------------
# Print summary

echo "=============================================="
echo "Stack Training Debug: 6 -> 12"
echo "=============================================="
echo "Project root:      $PROJECT_ROOT"
echo "Seed depth:        $SEED_DEPTH (n_embd=$N_EMBD)"
echo "Seed TPP:          $SEED_TPP"
echo "Target depth:      $TARGET_DEPTH"
echo "Target ratio:      $TARGET_RATIO"
echo "Window pattern:    $WINDOW_PATTERN"
echo "Num GPUs:          $NPROC_PER_NODE"
echo "Matrix optimizer:  $MATRIX_OPTIMIZER"
echo "Stack warmup:      $STACK_WARMUP_RATIO"
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
# Train

echo ""
echo "Starting stack training (6 -> 12, seed TPP=2.0)..."

TRAIN_ARGS=(
    --seed-depth=$SEED_DEPTH
    --target-depth=$TARGET_DEPTH
    --seed-tpp=$SEED_TPP
    --aspect-ratio=$ASPECT_RATIO
    --run=$WANDB_RUN
    --model-tag=${MODEL_TAG:-d12_stacked_debug}
    --window-pattern=$WINDOW_PATTERN
    --target-param-data-ratio=$TARGET_RATIO
    --device-batch-size=$DEVICE_BATCH_SIZE
    --total-batch-size=$TOTAL_BATCH_SIZE
    --matrix-optimizer=$MATRIX_OPTIMIZER
    --matrix-lr=$MATRIX_LR
    --warmdown-ratio=$WARMDOWN_RATIO
    --matrix-warmdown-ratio=$MATRIX_WARMDOWN_RATIO
    --stack-warmup-ratio=$STACK_WARMUP_RATIO
    --embedding-lr=$EMBEDDING_LR
    --unembedding-lr=$UNEMBEDDING_LR
    --norm-lr=$NORM_LR
    --scalar-lr=$SCALAR_LR
    --core-metric-every=${CORE_METRIC_EVERY:-2000}
    --sample-every=${SAMPLE_EVERY:--1}
    --save-every=${SAVE_EVERY:--1}
)

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.stack_train -- \
        "${TRAIN_ARGS[@]}" $FP8_ARGS
else
    python -m scripts.stack_train \
        "${TRAIN_ARGS[@]}" $FP8_ARGS
fi

echo ""
echo "=============================================="
echo "Stack training complete!"
echo "=============================================="
