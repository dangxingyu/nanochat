"""
Layer stacking training: train a shallow seed model, stack layers in-memory, continue training.

Usage:
    python -m scripts.stack_train --seed-depth=6 --target-depth=24
    torchrun --nproc_per_node=8 -m scripts.stack_train --seed-depth=6 --target-depth=24
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, has_ve
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, create_stacked_model_state
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Layer stacking pretraining")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
# Stacking config
parser.add_argument("--seed-depth", type=int, default=6, help="depth of the seed model")
parser.add_argument("--target-depth", type=int, default=24, help="target depth after stacking")
parser.add_argument("--seed-tpp", type=float, default=1.0, help="tokens-per-parameter for seed training")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--n-embd", type=int, default=None, help="model width (default: auto-compute from target depth)")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
# Training horizon for Phase 2
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="TPP for Phase 2 continued training")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--weight-decay", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--matrix-optimizer", type=str, default="hyperball", choices=["muon", "hyperball"])
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--norm-lr", type=float, default=0.1)
parser.add_argument("--adam-beta1", type=float, default=0.8)
parser.add_argument("--adam-beta2", type=float, default=0.95)
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.3)
parser.add_argument("--matrix-warmup-ratio", type=float, default=0.0)
parser.add_argument("--matrix-warmdown-ratio", type=float, default=1.0)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--seed-lr-multiplier", type=float, default=1.0, help="Constant LR multiplier for Phase 1 seed training")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=40*524288)
parser.add_argument("--core-metric-every", type=int, default=2000)
parser.add_argument("--core-metric-max-per-task", type=int, default=500)
parser.add_argument("--sample-every", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=-1)
# Output
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()

# Auto-compute n_embd from target depth if not provided (same as base_train)
if args.n_embd is None:
    base_dim = args.target_depth * args.aspect_ratio
    args.n_embd = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    print(f"Auto-computed n_embd={args.n_embd} from target_depth={args.target_depth}, aspect_ratio={args.aspect_ratio}")

user_config = vars(args).copy()

assert args.target_depth % args.seed_depth == 0, \
    f"Target depth {args.target_depth} must be a multiple of seed depth {args.seed_depth}"

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# Wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_project = os.environ.get("WANDB_PROJECT", "nanochat")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=wandb_project, name=args.run, config=user_config)

if HAS_FA3:
    print0("Using Flash Attention 3")
else:
    print0("WARNING: Flash Attention 3 not available, using SDPA fallback")

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Checkpoint dir
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.target_depth}_stacked"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

# -----------------------------------------------------------------------------
# Helpers

def build_model(depth, n_embd):
    """Build a model on meta device with given depth and width."""
    num_heads = n_embd // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=n_embd,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        m = GPT(config)
    return m

def get_scaling_params(m):
    pc = m.num_scaling_params()
    return pc['transformer_matrices'] + pc['lm_head']

def setup_fp8(model):
    """Convert model to FP8 if enabled. Returns the model."""
    if not args.fp8 or device_type != "cuda":
        return model
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn
    def fp8_filter(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_filter)
    n_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    print0(f"FP8 enabled: converted {n_fp8} layers")
    return model

@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear to nn.Linear for BF16 eval."""
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))
    if not fp8_locations:
        yield
        return
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(fp8_module.in_features, fp8_module.out_features,
                          bias=fp8_module.bias is not None,
                          device=fp8_module.weight.device, dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

def compute_training_params(model, target_ratio, total_batch_size):
    """Compute num_iterations, scaled LRs, weight decay for a given model and training config."""
    num_scaling = get_scaling_params(model)
    target_tokens = int(target_ratio * num_scaling)

    # Reference d12
    d12_ref = build_model(12, args.n_embd)
    D_REF = target_ratio * get_scaling_params(d12_ref)
    B_REF = 2**19

    # Batch LR scaling
    batch_lr_scale = (total_batch_size / B_REF) ** 0.5

    # Weight decay scaling
    wd_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

    # Matrix LR scaling
    matrix_lr = args.matrix_lr * batch_lr_scale
    if args.matrix_optimizer == "hyperball":
        D_REF_LR = 10.5 * get_scaling_params(d12_ref)
        matrix_lr = matrix_lr * (D_REF_LR / target_tokens) ** 0.35

    num_iterations = target_tokens // total_batch_size
    return num_iterations, batch_lr_scale, matrix_lr, wd_scaled

def create_optimizer(model, batch_lr_scale, matrix_lr, wd_scaled):
    """Create optimizer for a model."""
    return model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=matrix_lr,
        weight_decay=wd_scaled,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        scalar_lr=args.scalar_lr * batch_lr_scale,
        norm_lr=args.norm_lr * batch_lr_scale,
        matrix_optimizer=args.matrix_optimizer,
    )

# -----------------------------------------------------------------------------
# Stacking: in-memory optimizer state transfer

def gather_all_optimizer_states(optimizer):
    """Gather optimizer state dicts from all ranks. Returns list of state_dicts."""
    my_sd = optimizer.state_dict()
    if not ddp or ddp_world_size == 1:
        return [my_sd]
    # Use all_gather_object for simplicity (seed model is small)
    all_sds = [None] * ddp_world_size
    dist.all_gather_object(all_sds, my_sd)
    return all_sds

def create_stacked_optimizer_state(seed_optim_states, seed_model, target_model,
                                    seed_n_layer, target_n_layer, rank, world_size):
    """
    Build stacked optimizer state dict from gathered seed optimizer states.
    Maps seed optimizer state to a deeper model by repeating layer states.
    """
    from nanochat.checkpoint_manager import _get_param_groups_info
    num_repeats = target_n_layer // seed_n_layer
    seed_sd = seed_optim_states[0]
    seed_groups = seed_sd['param_groups']

    seed_infos = _get_param_groups_info(seed_model, seed_n_layer)
    target_infos = _get_param_groups_info(target_model, target_n_layer)

    new_state = {}
    new_param_groups = []
    param_counter = 0

    for gi, (seed_group, seed_info, target_info) in enumerate(
        zip(seed_groups, seed_infos, target_infos)
    ):
        kind = seed_group['kind']
        target_n_params = target_info['n_params']

        # Copy group hyperparams, update param indices
        new_group = {k: v for k, v in seed_group.items() if k != 'params'}
        new_group['params'] = list(range(param_counter, param_counter + target_n_params))

        if kind == 'adamw':
            _stack_adamw_state(new_state, seed_optim_states, seed_group,
                               seed_info, target_info, param_counter,
                               rank, world_size, num_repeats)
        elif kind in ('muon', 'hyperball'):
            _stack_matrix_state(new_state, seed_optim_states, seed_group, kind,
                                seed_info, target_info, param_counter,
                                rank, world_size, num_repeats)

        new_param_groups.append(new_group)
        param_counter += target_n_params

    return {'state': new_state, 'param_groups': new_param_groups}


def _stack_adamw_state(new_state, seed_optim_states, seed_group,
                        seed_info, target_info, param_offset,
                        rank, world_size, num_repeats):
    seed_indices = seed_group['params']
    category = seed_info['category']
    rank_sd = seed_optim_states[rank]

    if category == 'shared':
        for i, seed_idx in enumerate(seed_indices):
            if seed_idx in rank_sd['state']:
                new_state[param_offset + i] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in rank_sd['state'][seed_idx].items()
                }

    elif category == 'scalar_expand':
        for i, seed_idx in enumerate(seed_indices):
            if seed_idx in rank_sd['state']:
                entry = {}
                for k, v in rank_sd['state'][seed_idx].items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 1:
                        entry[k] = v.repeat(num_repeats)[:target_info['shapes'][i][0]]
                    else:
                        entry[k] = v
                new_state[param_offset + i] = entry

    elif category in ('per_layer_ve', 'per_layer_1d'):
        seed_n = len(seed_indices)
        for i in range(target_info['n_params']):
            seed_idx = seed_indices[i % seed_n]
            if seed_idx in rank_sd['state']:
                new_state[param_offset + i] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in rank_sd['state'][seed_idx].items()
                }


def _stack_matrix_state(new_state, seed_optim_states, seed_group, kind,
                         seed_info, target_info, param_offset,
                         rank, world_size, num_repeats):
    seed_n_params = seed_info['n_params']
    target_n_params = target_info['n_params']

    seed_chunk = (seed_n_params + world_size - 1) // world_size
    target_chunk = (target_n_params + world_size - 1) // world_size

    def _gather_full(key):
        chunks = []
        target_device = None
        for r in range(world_size):
            first_idx = seed_group['params'][0]
            r_state = seed_optim_states[r]['state']
            if first_idx in r_state and key in r_state[first_idx]:
                tensor = r_state[first_idx][key]
                if target_device is None:
                    target_device = torch.device('cuda', rank)
                chunks.append(tensor.to(target_device))
            else:
                return None
        return torch.cat(chunks, dim=0)[:seed_n_params]

    full_mom = _gather_full('momentum_buffer')
    full_sec = _gather_full('second_momentum_buffer')
    full_pnorm = _gather_full('p_norm') if kind == 'hyperball' else None

    if full_mom is None:
        return

    # Repeat for stacking
    stacked_mom = full_mom.repeat(num_repeats, *([1] * (full_mom.ndim - 1)))[:target_n_params]
    stacked_sec = full_sec.repeat(num_repeats, *([1] * (full_sec.ndim - 1)))[:target_n_params]

    # Shard for this rank
    start = rank * target_chunk
    num_owned = min(target_chunk, max(0, target_n_params - start))

    chunk_mom = torch.zeros(target_chunk, *stacked_mom.shape[1:],
                            dtype=stacked_mom.dtype, device=stacked_mom.device)
    chunk_sec = torch.zeros(target_chunk, *stacked_sec.shape[1:],
                            dtype=stacked_sec.dtype, device=stacked_sec.device)
    if num_owned > 0:
        chunk_mom[:num_owned] = stacked_mom[start:start + num_owned]
        chunk_sec[:num_owned] = stacked_sec[start:start + num_owned]

    entry = {'momentum_buffer': chunk_mom, 'second_momentum_buffer': chunk_sec}

    if full_pnorm is not None:
        stacked_pn = full_pnorm.repeat(num_repeats, *([1] * (full_pnorm.ndim - 1)))[:target_n_params]
        chunk_pn = torch.zeros(target_chunk, *stacked_pn.shape[1:],
                               dtype=stacked_pn.dtype, device=stacked_pn.device)
        if num_owned > 0:
            chunk_pn[:num_owned] = stacked_pn[start:start + num_owned]
        entry['p_norm'] = chunk_pn

    new_state[param_offset] = entry


# ============================================================================
# PHASE 1: Train seed model
# ============================================================================
print0("=" * 60)
print0(f"PHASE 1: Training seed model (depth={args.seed_depth}, n_embd={args.n_embd}, TPP={args.seed_tpp})")
print0("=" * 60)

# Build seed model
seed_model = build_model(args.seed_depth, args.n_embd)
seed_config = seed_model.config
print0(f"Seed model config: {json.dumps(asdict(seed_config), indent=2)}")
seed_model.to_empty(device=device)
seed_model.init_weights()
seed_model = setup_fp8(seed_model)

orig_seed_model = seed_model
seed_model = torch.compile(seed_model, dynamic=False)

# Compute training iterations for both phases
temp_target_model = build_model(args.target_depth, args.n_embd)

# Compute seed_iters based on seed model
seed_iters, _, _, _ = compute_training_params(
    seed_model, args.seed_tpp, args.total_batch_size
)

# Compute total target iters
total_target_iters, _, _, _ = compute_training_params(
    temp_target_model, args.target_param_data_ratio, args.total_batch_size
)

# Target phase trains for remaining steps (subtract seed steps)
target_iters = total_target_iters - seed_iters

# Compute Phase 2 hyperparams based on Phase 2's token budget (not total)
phase2_tokens = target_iters * args.total_batch_size
phase2_ratio = phase2_tokens / get_scaling_params(temp_target_model)
_, target_lr_scale, target_matrix_lr, target_wd = compute_training_params(
    temp_target_model, phase2_ratio, args.total_batch_size
)
del temp_target_model

total_iters = total_target_iters  # For logging/progress only
print0(f"Total iterations: {total_iters}")
print0(f"  Phase 1 (seed): {seed_iters} steps (constant LR, multiplier={args.seed_lr_multiplier})")
print0(f"  Phase 2 (target): {target_iters} steps (phase2_ratio={phase2_ratio:.2f})")
print0(f"Phase 2 hyperparameters: lr_scale={target_lr_scale:.4f}, matrix_lr={target_matrix_lr:.4f}, wd={target_wd:.6f}")

# Seed optimizer (using target model hyperparameters!)
seed_optimizer = create_optimizer(seed_model, target_lr_scale, target_matrix_lr, target_wd)
for g in seed_optimizer.param_groups:
    g["initial_lr"] = g["lr"]

# DataLoader
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(train_loader)

# Gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

# Seed LR scheduler
def get_lr_multiplier(it, num_iters, warmup_ratio, warmdown_ratio, final_lr_frac):
    warmup_iters = round(warmup_ratio * num_iters)
    warmdown_iters = round(warmdown_ratio * num_iters)
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    if warmdown_iters <= 0:
        return 1.0
    if it <= num_iters - warmdown_iters:
        return 1.0
    progress = (num_iters - it) / warmdown_iters
    return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Calculate FLOPs for seed model (for MFU logging)
num_flops_per_token = orig_seed_model.estimate_flops()

# Evaluate at step 0 (before training)
print0("Evaluating at step 0...")
seed_model.eval()
val_loader = build_val_loader()
eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
with disable_fp8(seed_model), autocast_ctx:
    val_bpb_step0 = evaluate_bpb(seed_model, val_loader, eval_steps, token_bytes)
print0(f"Step 0 | Validation bpb: {val_bpb_step0:.6f}")
wandb_run.log({"step": 0, "val/bpb": val_bpb_step0})
seed_model.train()

# Seed training loop
print0(f"Starting seed training for {seed_iters} steps...")
seed_total_time = 0
smooth_train_loss = 0
ema_beta = 0.9

for step in range(seed_iters):
    global_step = step  # Phase 1: global_step = 0 to seed_iters-1
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = seed_model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Phase 1: constant LR
    lrm_adam = args.seed_lr_multiplier
    lrm_matrix = args.seed_lr_multiplier
    for group in seed_optimizer.param_groups:
        if group['kind'] in {'muon', 'hyperball'}:
            group["lr"] = group["initial_lr"] * lrm_matrix
            group["momentum"] = get_muon_momentum(step)
        else:
            group["lr"] = group["initial_lr"] * lrm_adam
        if group['kind'] == 'muon':
            group["weight_decay"] = target_wd
    seed_optimizer.step()
    seed_model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()

    synchronize()
    dt = time.time() - t0
    if step > 5:
        seed_total_time += dt

    # EMA smoothing for train loss
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

    # Compute MFU
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    # Print progress every step
    print0(f"  Phase 1 step {step:04d}/{seed_iters} | global_step {global_step:05d}/{total_iters} | loss: {debiased_smooth_loss:.6f} | lrm_adam={lrm_adam:.2f}, lrm_matrix={lrm_matrix:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}")

    # Log train loss to wandb every 100 global steps
    if global_step % 100 == 0:
        wandb_run.log({"step": global_step, "train/loss": debiased_smooth_loss,
                       "train/lrm_adam": lrm_adam, "train/lrm_matrix": lrm_matrix,
                       "train/dt": dt, "train/tok_per_sec": tok_per_sec, "train/mfu": mfu})

    # Validation every 125 global steps
    if global_step > 0 and global_step % 250 == 0:
        seed_model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(seed_model), autocast_ctx:
            val_bpb = evaluate_bpb(seed_model, val_loader, eval_steps, token_bytes)
        print0(f"  Global step {global_step:05d} | Validation bpb: {val_bpb:.6f}")
        wandb_run.log({"step": global_step, "val/bpb": val_bpb})
        seed_model.train()

    # CORE metrics every 1000 global steps
    if global_step > 0 and global_step % 1000 == 0:
        seed_model.eval()
        with disable_fp8(orig_seed_model), autocast_ctx:
            results = evaluate_core(orig_seed_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"  Global step {global_step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": global_step, "core_metric": results["core_metric"],
                       "centered_results": results["centered_results"]})
        seed_model.train()

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()

# Evaluate seed model
seed_model.eval()
val_loader = build_val_loader()
eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
with disable_fp8(seed_model), autocast_ctx:
    seed_val_bpb = evaluate_bpb(seed_model, val_loader, eval_steps, token_bytes)
print0(f"Seed model val BPB: {seed_val_bpb:.6f}")
seed_model.train()

# ============================================================================
# STACKING: In-memory model + optimizer state transfer
# ============================================================================
print0("")
print0("=" * 60)
print0(f"STACKING: {args.seed_depth} layers -> {args.target_depth} layers")
print0("=" * 60)

# 1) Gather seed optimizer states from all ranks
print0("Gathering optimizer states from all ranks...")
gc.enable(); gc.collect()
all_seed_optim_states = gather_all_optimizer_states(seed_optimizer)

# 2) Get seed model state dict and create stacked version
print0("Creating stacked model state...")
seed_state_dict = orig_seed_model.state_dict()

stacked_model_state = create_stacked_model_state(
    seed_state_dict, args.seed_depth, args.target_depth
)

# 3) Build target model
target_model = build_model(args.target_depth, args.n_embd)
target_config = target_model.config
target_config_kwargs = asdict(target_config)
print0(f"Target model config: {json.dumps(target_config_kwargs, indent=2)}")
target_model.to_empty(device=device)
target_model.init_weights()
target_model.load_state_dict(stacked_model_state, strict=True, assign=True)
del stacked_model_state

# 4) Setup FP8, compile, optimizer for target model
target_model = setup_fp8(target_model)
orig_target_model = target_model
target_model = torch.compile(target_model, dynamic=False)

# Use same hyperparameters as seed phase (already computed from target model)
print0(f"Target training: {target_iters} iterations (using unified hyperparameters)")

target_optimizer = create_optimizer(target_model, target_lr_scale, target_matrix_lr, target_wd)
for g in target_optimizer.param_groups:
    g["initial_lr"] = g["lr"]

# 5) Build stacked optimizer state and load it
print0("Creating stacked optimizer state...")
stacked_optim_state = create_stacked_optimizer_state(
    all_seed_optim_states, orig_seed_model, orig_target_model,
    args.seed_depth, args.target_depth, ddp_rank, ddp_world_size,
)
target_optimizer.load_state_dict(stacked_optim_state)
print0("Stacked optimizer state loaded successfully")

# Free seed model and optimizer memory
del seed_model, orig_seed_model, seed_optimizer, all_seed_optim_states, stacked_optim_state, seed_state_dict
gc.collect()

# ============================================================================
# PHASE 2: Continue training the stacked model
# ============================================================================
print0("")
print0("=" * 60)
print0(f"PHASE 2: Continue training stacked model (depth={args.target_depth}, {target_iters} iterations)")
print0("=" * 60)

# Scaling params for logging
param_counts = target_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = target_model.estimate_flops()
num_scaling_params = get_scaling_params(target_model)

# Weight decay scheduler for Phase 2
def get_weight_decay(it, total_it, wd):
    return wd * (1 - it / total_it)

# Loop state
step = 0
val_bpb = seed_val_bpb
min_val_bpb = float("inf")
smooth_train_loss = 0
total_training_time = seed_total_time
total_batch_size = args.total_batch_size

# Training loop
while True:
    last_step = step == target_iters
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Compute global_step for this iteration (needed for logging)
    current_global_step = seed_iters + step

    # Eval val BPB every 125 global steps
    if current_global_step > 0 and current_global_step % 250 == 0:
        target_model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(target_model), autocast_ctx:
            val_bpb = evaluate_bpb(target_model, val_loader, eval_steps, token_bytes)
        print0(f"Global step {current_global_step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": current_global_step, "total_training_flops": flops_so_far,
                       "total_training_time": total_training_time, "val/bpb": val_bpb})
        target_model.train()

    # CORE metric every 1000 global steps
    results = {}
    if current_global_step > 0 and current_global_step % 1000 == 0:
        target_model.eval()
        with disable_fp8(orig_target_model), autocast_ctx:
            results = evaluate_core(orig_target_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Global step {current_global_step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": current_global_step, "total_training_flops": flops_so_far,
                       "core_metric": results["core_metric"], "centered_results": results["centered_results"]})
        target_model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        target_model.eval()
        prompts = ["The capital of France is", "The chemical symbol of gold is",
                   "If yesterday was Friday, then tomorrow will be"]
        engine = Engine(orig_target_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_target_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        target_model.train()

    # Save checkpoint
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step, orig_target_model.state_dict(), target_optimizer.state_dict(),
            {"step": step, "val_bpb": val_bpb, "model_config": target_config_kwargs,
             "user_config": user_config, "device_batch_size": args.device_batch_size,
             "max_seq_len": args.max_seq_len, "dataloader_state_dict": dataloader_state_dict,
             "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss,
                           "total_training_time": total_training_time}},
            rank=ddp_rank,
        )

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = target_model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Phase 2: local schedule over target_iters
    global_step = seed_iters + step

    lrm_adam = get_lr_multiplier(step, target_iters, args.warmup_ratio, args.warmdown_ratio, args.final_lr_frac)
    lrm_matrix = get_lr_multiplier(step, target_iters, args.matrix_warmup_ratio, args.matrix_warmdown_ratio, args.final_lr_frac)
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(step, target_iters, target_wd)
    for group in target_optimizer.param_groups:
        if group['kind'] in {'muon', 'hyperball'}:
            group["lr"] = group["initial_lr"] * lrm_matrix
            group["momentum"] = muon_momentum
        else:
            group["lr"] = group["initial_lr"] * lrm_adam
        if group['kind'] == 'muon':
            group["weight_decay"] = muon_wd
    target_optimizer.step()
    target_model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * global_step / total_iters
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time = total_training_time / steps_done
        eta_str = f" | eta: {(target_iters - step) * avg_time / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]

    # Print every step with global_step
    print0(f"Phase 2 step {step:05d}/{target_iters:05d} | global_step {global_step:05d}/{total_iters} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm(adam)={lrm_adam:.2f}, lrm(matrix)={lrm_matrix:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    # Log train loss to wandb every 100 global steps
    if global_step % 100 == 0:
        wandb_run.log({"step": global_step, "total_training_flops": flops_so_far,
                       "total_training_time": total_training_time,
                       "train/loss": debiased_smooth_loss, "train/lrm_adam": lrm_adam,
                       "train/lrm_matrix": lrm_matrix, "train/dt": dt,
                       "train/tok_per_sec": tok_per_sec, "train/mfu": mfu, "train/epoch": epoch})

    first_step_of_run = step == 0
    step += 1
    if first_step_of_run:
        gc.collect(); gc.freeze(); gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# Final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

wandb_run.finish()
compute_cleanup()
