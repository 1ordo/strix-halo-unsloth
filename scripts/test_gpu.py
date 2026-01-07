#!/usr/bin/env python3
"""Quick GPU test for Strix Halo (gfx1151)"""

import sys

def main():
    print("═" * 50)
    print("  Strix Halo GPU Test")
    print("═" * 50)
    
    # Import
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch not installed")
        return 1
    
    # CUDA available
    if not torch.cuda.is_available():
        print("✗ CUDA/ROCm not available")
        return 1
    print(f"✓ CUDA available, {torch.cuda.device_count()} device(s)")
    
    # Device info
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"✓ Device: {name}")
    print(f"  Memory: {props.total_memory / 1e9:.0f} GB")
    
    # Check gfx1151
    archs = torch.cuda.get_arch_list()
    if 'gfx1151' in archs:
        print(f"✓ gfx1151 in arch list")
    else:
        print(f"⚠ gfx1151 not in arch list (may still work)")
    
    # Tensor test
    print("\nRunning tensor tests...")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiply: {x.shape} @ {y.shape} = {z.shape}")
    except Exception as e:
        print(f"✗ Tensor test failed: {e}")
        return 1
    
    # BFloat16
    try:
        x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
        print(f"✓ BFloat16 supported")
    except:
        print(f"⚠ BFloat16 not supported")
    
    # Memory
    print(f"\nMemory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used")
    
    print("\n" + "═" * 50)
    print("✓ All tests passed!")
    print("═" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
