
import torch
import torch.nn as nn
try:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_tensor import Float8Tensor
    from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
    print("TORCHAO_AVAILABLE")
except ImportError:
    print("TORCHAO_UNAVAILABLE")
    exit(0)

def test_reparam():
    # Make a dummy model to convert
    model = nn.Linear(32, 16, bias=False)
    
    # Needs CUDA for FP8 usually, but let's see if we can do this on CPU if not available or just assume CUDA
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        # Can we test Float8 without CUDA? Maybe some parts.
        # Float8Tensor usually requires CUDA capability.
        return

    model = model.cuda().bfloat16()
    
    # Convert to FP8
    config = Float8LinearConfig(enable_fsdp_float8_all_gather=True) 
    # Use simple config
    
    convert_to_float8_training(model, config=config)
    
    linear = model
    print(f"Linear type: {type(linear)}")
    if hasattr(linear, 'weight'):
        w = linear.weight
        print(f"Weight type: {type(w)}")
        print(f"Weight dtype: {w.dtype}")
        
        # Create gamma
        gamma = torch.ones(32, device='cuda', dtype=torch.bfloat16)
        
        # Perform multiplication
        try:
            w_new = w * gamma[None, :]
            print(f"Result type after mult: {type(w_new)}")
            print(f"Result dtype after mult: {w_new.dtype}")
            
            if isinstance(w_new, Float8Tensor):
                print("Multiplication result is Float8Tensor")
            else:
                print("Multiplication result is NOT Float8Tensor")
                
        except Exception as e:
            print(f"Multiplication failed: {e}")

if __name__ == "__main__":
    test_reparam()
