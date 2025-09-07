import torch
import os

def test_gpu_setup():
    """Test GPU setup and PyTorch installation"""
    print("="*60)
    print("GPU SETUP TEST")
    print("="*60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        # Test GPU operations
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"GPU tensor operation successful: {z.shape}")
        
        # Check memory
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        print("✅ GPU test passed!")
        return True
    else:
        print("❌ CUDA not available")
        return False

if __name__ == "__main__":
    test_gpu_setup()
