"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# Layer stacking utilities

def create_stacked_model_state(seed_state, seed_n_layer, target_n_layer):
    """
    Create a model state dict for a deeper model by repeating seed layers.
    Pattern: [0,1,...,S-1, 0,1,...,S-1, ...] until target_n_layer.
    """
    from nanochat.gpt import has_ve
    new_state = {}
    num_repeats = (target_n_layer + seed_n_layer - 1) // seed_n_layer

    # 1) Copy shared params (embeddings, lm_head) directly
    for key, val in seed_state.items():
        if key.startswith('transformer.wte') or key.startswith('lm_head'):
            new_state[key] = val.clone()

    # 2) Repeat per-layer scalars
    for key in ['resid_lambdas', 'x0_lambdas']:
        new_state[key] = seed_state[key].repeat(num_repeats)[:target_n_layer]

    # 3) Duplicate transformer blocks
    for target_layer in range(target_n_layer):
        source_layer = target_layer % seed_n_layer
        for key in seed_state:
            prefix = f'transformer.h.{source_layer}.'
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                new_key = f'transformer.h.{target_layer}.{suffix}'
                new_state[new_key] = seed_state[key].clone()

    # 4) Duplicate value embeddings
    for target_layer in range(target_n_layer):
        if has_ve(target_layer, target_n_layer):
            source_layer = target_layer % seed_n_layer
            source_key = f'value_embeds.{source_layer}.weight'
            target_key = f'value_embeds.{target_layer}.weight'
            if source_key in seed_state:
                new_state[target_key] = seed_state[source_key].clone()
            else:
                log0(f"Warning: seed layer {source_layer} has no value_embeds, target layer {target_layer} needs one")

    # 5) Zero-init c_proj on bottom layers so they act as pass-through initially.
    #    Bottom layers = all except the topmost seed_n_layer layers.
    num_bottom = target_n_layer - seed_n_layer
    for layer in range(num_bottom):
        for proj_key in [f'transformer.h.{layer}.attn.c_proj.weight',
                         f'transformer.h.{layer}.mlp.c_proj.weight']:
            if proj_key in new_state:
                new_state[proj_key] = torch.zeros_like(new_state[proj_key])
    if num_bottom > 0:
        log0(f"Zero-initialized c_proj weights on bottom {num_bottom} layers (0..{num_bottom-1})")

    return new_state


def create_stacked_optimizer_state(seed_optim_states, seed_n_layer, target_n_layer,
                                   seed_model, target_model, rank, world_size):
    """
    Create a stacked optimizer state dict for a single rank.

    Args:
        seed_optim_states: list of optimizer state dicts, one per rank (loaded from all rank files)
        seed_n_layer: number of layers in seed model
        target_n_layer: number of layers in target model
        seed_model: the seed GPT model (for parameter structure reference)
        target_model: the target GPT model (for parameter structure reference)
        rank: this rank's index
        world_size: total number of ranks
    """
    num_repeats = target_n_layer // seed_n_layer
    # We'll use rank 0's state dict as the structural reference
    seed_sd = seed_optim_states[0]
    seed_groups = seed_sd['param_groups']
    seed_states = seed_sd['state']

    # Build the target optimizer to get param_groups structure
    # We need to know: for each group, how many params, and their shapes
    seed_param_groups_info = _get_param_groups_info(seed_model, seed_n_layer)
    target_param_groups_info = _get_param_groups_info(target_model, target_n_layer)

    new_state = {}
    new_param_groups = []
    param_counter = 0

    for gi, (seed_group, seed_info, target_info) in enumerate(
        zip(seed_groups, seed_param_groups_info, target_param_groups_info)
    ):
        kind = seed_group['kind']
        seed_n_params = seed_info['n_params']
        target_n_params = target_info['n_params']

        # Build new param_group entry (copy hyperparams, update param indices)
        new_group = {k: v for k, v in seed_group.items() if k != 'params'}
        new_param_indices = list(range(param_counter, param_counter + target_n_params))
        new_group['params'] = new_param_indices

        if kind == 'adamw':
            _stack_adamw_group_state(
                new_state, seed_optim_states, seed_states, seed_group,
                seed_info, target_info, param_counter,
                rank, world_size, num_repeats
            )
        elif kind in ('muon', 'hyperball'):
            _stack_matrix_group_state(
                new_state, seed_optim_states, seed_group, kind,
                seed_info, target_info, param_counter,
                rank, world_size, num_repeats
            )

        new_param_groups.append(new_group)
        param_counter += target_n_params

    return {'state': new_state, 'param_groups': new_param_groups}


def _get_param_groups_info(model, n_layer):
    """Extract parameter group structure info from a model (mirrors setup_optimizer ordering)."""
    from nanochat.gpt import has_ve
    block_matrix_params = [p for p in model.transformer.h.parameters() if p.ndim == 2]
    block_1d_params = [p for p in model.transformer.h.parameters() if p.ndim == 1]
    value_embeds_params = list(model.value_embeds.parameters())

    infos = [
        {'name': 'lm_head', 'n_params': 1, 'shapes': [model.lm_head.weight.shape], 'category': 'shared'},
        {'name': 'wte', 'n_params': 1, 'shapes': [model.transformer.wte.weight.shape], 'category': 'shared'},
        {'name': 'value_embeds', 'n_params': len(value_embeds_params),
         'shapes': [p.shape for p in value_embeds_params], 'category': 'per_layer_ve'},
        {'name': 'resid_lambdas', 'n_params': 1, 'shapes': [model.resid_lambdas.shape], 'category': 'scalar_expand'},
        {'name': 'x0_lambdas', 'n_params': 1, 'shapes': [model.x0_lambdas.shape], 'category': 'scalar_expand'},
        {'name': 'block_1d', 'n_params': len(block_1d_params),
         'shapes': [p.shape for p in block_1d_params], 'category': 'per_layer_1d'},
    ]

    # Matrix groups by shape (same ordering as setup_optimizer)
    for shape in sorted({p.shape for p in block_matrix_params}):
        group_params = [p for p in block_matrix_params if p.shape == shape]
        infos.append({
            'name': f'matrix_{shape}', 'n_params': len(group_params),
            'shapes': [shape] * len(group_params), 'category': 'matrix',
            'shape': shape,
        })

    return infos


def _stack_adamw_group_state(new_state, seed_optim_states, seed_states, seed_group,
                              seed_info, target_info, param_offset,
                              rank, world_size, num_repeats):
    """Stack AdamW optimizer state for a parameter group."""
    seed_param_indices = seed_group['params']
    category = seed_info['category']

    if category == 'shared':
        # Shared params (lm_head, wte): copy state directly from this rank
        rank_sd = seed_optim_states[rank]
        for i, seed_idx in enumerate(seed_param_indices):
            if seed_idx in rank_sd['state']:
                state = rank_sd['state'][seed_idx]
                new_state[param_offset + i] = {k: v.clone() for k, v in state.items()}

    elif category == 'scalar_expand':
        # resid_lambdas, x0_lambdas: single param whose shape grows from (seed_n,) to (target_n,)
        # State is replicated (small param), so just use this rank's state
        rank_sd = seed_optim_states[rank]
        for i, seed_idx in enumerate(seed_param_indices):
            if seed_idx in rank_sd['state']:
                state = rank_sd['state'][seed_idx]
                new_entry = {}
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 1:
                        new_entry[k] = v.repeat(num_repeats)[:target_info['shapes'][i][0]]
                    else:
                        new_entry[k] = v
                new_state[param_offset + i] = new_entry

    elif category in ('per_layer_ve', 'per_layer_1d'):
        # Per-layer params that get duplicated: copy state in repeating pattern
        rank_sd = seed_optim_states[rank]
        seed_n = len(seed_param_indices)
        target_n = target_info['n_params']
        for i in range(target_n):
            seed_local_idx = i % seed_n
            seed_idx = seed_param_indices[seed_local_idx]
            if seed_idx in rank_sd['state']:
                state = rank_sd['state'][seed_idx]
                new_state[param_offset + i] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                                 for k, v in state.items()}


def _stack_matrix_group_state(new_state, seed_optim_states, seed_group, kind,
                               seed_info, target_info, param_offset,
                               rank, world_size, num_repeats):
    """Stack Muon/Hyperball optimizer state for a matrix parameter group."""
    seed_n_params = seed_info['n_params']
    target_n_params = target_info['n_params']
    shape = seed_info['shape']

    seed_chunk_size = (seed_n_params + world_size - 1) // world_size
    target_chunk_size = (target_n_params + world_size - 1) // world_size

    # Reconstruct full momentum buffers from all ranks
    def _gather_full_buffer(key):
        chunks = []
        for r in range(world_size):
            r_sd = seed_optim_states[r]
            first_idx = seed_group['params'][0]
            if first_idx in r_sd['state'] and key in r_sd['state'][first_idx]:
                chunks.append(r_sd['state'][first_idx][key])
            else:
                return None
        # Concatenate all rank chunks, trim to actual seed_n_params
        full = torch.cat(chunks, dim=0)[:seed_n_params]
        return full

    full_momentum = _gather_full_buffer('momentum_buffer')
    full_second_momentum = _gather_full_buffer('second_momentum_buffer')
    full_p_norm = _gather_full_buffer('p_norm') if kind == 'hyperball' else None

    # If no state exists yet (seed optimizer not stepped), skip
    if full_momentum is None:
        return

    # Repeat to target size
    stacked_momentum = full_momentum.repeat(num_repeats, *([1] * (full_momentum.ndim - 1)))[:target_n_params]
    stacked_second = full_second_momentum.repeat(num_repeats, *([1] * (full_second_momentum.ndim - 1)))[:target_n_params]

    # Shard for this rank
    start = rank * target_chunk_size
    end = start + target_chunk_size
    chunk_momentum = torch.zeros(target_chunk_size, *stacked_momentum.shape[1:],
                                  dtype=stacked_momentum.dtype, device=stacked_momentum.device)
    chunk_second = torch.zeros(target_chunk_size, *stacked_second.shape[1:],
                                dtype=stacked_second.dtype, device=stacked_second.device)
    num_owned = min(target_chunk_size, max(0, target_n_params - start))
    if num_owned > 0:
        chunk_momentum[:num_owned] = stacked_momentum[start:start + num_owned]
        chunk_second[:num_owned] = stacked_second[start:start + num_owned]

    # Store state at first param index of the group
    first_param_idx = param_offset
    entry = {
        'momentum_buffer': chunk_momentum,
        'second_momentum_buffer': chunk_second,
    }

    if full_p_norm is not None:
        stacked_p_norm = full_p_norm.repeat(num_repeats, *([1] * (full_p_norm.ndim - 1)))[:target_n_params]
        chunk_p_norm = torch.zeros(target_chunk_size, *stacked_p_norm.shape[1:],
                                    dtype=stacked_p_norm.dtype, device=stacked_p_norm.device)
        if num_owned > 0:
            chunk_p_norm[:num_owned] = stacked_p_norm[start:start + num_owned]
        entry['p_norm'] = chunk_p_norm

    new_state[first_param_idx] = entry


# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
