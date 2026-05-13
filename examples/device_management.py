"""
Example: Device Management in Text4GCN

This example demonstrates how to use Text4GCN with different device configurations
(CPU vs GPU) and how to manage PyTorch devices effectively.
"""

import text4gcn
from text4gcn.utils.device import (
    get_device_manager,
    check_pytorch_installation,
    recommend_installation,
)


def check_system_capabilities():
    """Check system capabilities and PyTorch installation."""
    print("🔍 Checking System Capabilities")
    print("=" * 50)
    
    # Check PyTorch installation
    pytorch_info = check_pytorch_installation()
    print(f"PyTorch Available: {pytorch_info['torch_available']}")
    
    if pytorch_info['torch_available']:
        print(f"PyTorch Version: {pytorch_info['torch_version']}")
        print(f"CUDA Available: {pytorch_info['cuda_available']}")
        
        if pytorch_info['cuda_available']:
            print(f"CUDA Version: {pytorch_info['cuda_version']}")
    
    # Get installation recommendation
    recommendation = recommend_installation()
    print(f"\n💡 Installation Recommendation:")
    print(f"   {recommendation}")
    
    print()


def demonstrate_device_management():
    """Demonstrate device management features."""
    print("🛠️  Device Management Demo")
    print("=" * 50)
    
    # Auto-detect device
    print("1. Auto-detecting best device...")
    device_manager = get_device_manager()
    print(f"   Selected device: {device_manager.device}")
    print(f"   Device type: {device_manager.device.type}")
    
    # Get detailed device information
    device_info = device_manager.get_device_info()
    print(f"\n2. Device Information:")
    for key, value in device_info.items():
        print(f"   {key}: {value}")
    
    # Demonstrate device selection
    print(f"\n3. Manual device selection:")
    
    # Try CPU device
    cpu_manager = get_device_manager("cpu")
    print(f"   CPU device: {cpu_manager.device}")
    print(f"   Is CPU: {cpu_manager.is_cpu}")
    
    # Try CUDA device (if available)
    if device_info.get('cuda_available', False):
        cuda_manager = get_device_manager("cuda")
        print(f"   CUDA device: {cuda_manager.device}")
        print(f"   Is CUDA: {cuda_manager.is_cuda}")
        print(f"   GPU Name: {cuda_manager.get_device_info().get('gpu_name', 'Unknown')}")
        print(f"   GPU Memory: {cuda_manager.get_device_info().get('gpu_memory_gb', 0):.1f} GB")
    else:
        print("   CUDA not available - using CPU")
    
    print()


def demonstrate_model_with_device():
    """Demonstrate using Text4GCN models with device management."""
    print("🤖 Model Training with Device Management")
    print("=" * 50)
    
    # This is a conceptual example - actual implementation would depend on data availability
    print("Example: Training a GCN model with device management")
    print()
    
    # Example 1: Auto-detect device
    print("1. Auto-detect device:")
    print("""
    model = text4gcn.models.GNN(
        dataset_name="R8",
        dataset_path="./data",
        log_dir="./logs",
        builder=text4gcn.builder.Builder.Tfidf,
        device=None  # Auto-detect
    )
    """)
    
    # Example 2: Force CPU
    print("2. Force CPU usage:")
    print("""
    model = text4gcn.models.GNN(
        dataset_name="R8",
        dataset_path="./data", 
        log_dir="./logs",
        builder=text4gcn.builder.Builder.Tfidf,
        device="cpu"  # Force CPU
    )
    """)
    
    # Example 3: Force GPU (if available)
    if device_info.get('cuda_available', False):
        print("3. Force GPU usage:")
        print("""
        model = text4gcn.models.GNN(
            dataset_name="R8",
            dataset_path="./data",
            log_dir="./logs",
            builder=text4gcn.builder.Builder.Tfidf,
            device="cuda",  # Force GPU
        )
        """)
    
    print()


def demonstrate_installation_options():
    """Demonstrate different installation options."""
    print("📦 Installation Options")
    print("=" * 50)
    
    print("1. CPU-only installation:")
    print("   pip install text4gcn[cpu]")
    print()
    
    print("2. GPU installation (CUDA):")
    print("   pip install text4gcn[gpu]")
    print()
    
    print("3. Manual PyTorch installation:")
    print("   # Install Text4GCN without PyTorch")
    print("   pip install text4gcn")
    print()
    print("   # Then install PyTorch based on your system:")
    print("   # For CPU:")
    print("   pip install torch")
    print()
    print("   # For GPU (CUDA 12.1):")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    
    print("4. Development installation:")
    print("   git clone https://github.com/vitormeriat/text4gcn.git")
    print("   cd text4gcn")
    print("   poetry install --extras 'gpu'  # for GPU version")
    print("   poetry install --extras 'cpu'  # for CPU version")
    print()


def main():
    """Main demonstration function."""
    print("🚀 Text4GCN Device Management Example")
    print("=" * 60)
    print()
    
    try:
        # Check system capabilities
        check_system_capabilities()
        
        # Demonstrate device management
        demonstrate_device_management()
        
        # Demonstrate model usage
        demonstrate_model_with_device()
        
        # Show installation options
        demonstrate_installation_options()
        
        print("✅ Example completed successfully!")
        print()
        print("💡 Tips:")
        print("   - Use device=None for auto-detection")
        print("   - Use device='cpu' to force CPU usage")
        print("   - Use device='cuda' to force GPU usage (if available)")
        print("   - Check device_info for detailed system information")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to missing PyTorch installation.")
        print("Please install PyTorch first:")
        print("   pip install text4gcn[cpu]  # for CPU")
        print("   pip install text4gcn[gpu]  # for GPU")


if __name__ == "__main__":
    main()
