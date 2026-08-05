#!/usr/bin/env python3
"""
Example: Device Management in Text4GCN

Demonstrates CPU/GPU detection and how to pass ``device`` into ``GNN``.
"""

from text4gcn import GNN, Builder
from text4gcn.utils.device import (
    get_device_manager,
    check_pytorch_installation,
    recommend_installation,
)


def check_system_capabilities() -> dict:
    print("Checking system capabilities")
    print("=" * 50)

    pytorch_info = check_pytorch_installation()
    print(f"PyTorch Available: {pytorch_info['torch_available']}")

    if pytorch_info["torch_available"]:
        print(f"PyTorch Version: {pytorch_info['torch_version']}")
        print(f"CUDA Available: {pytorch_info['cuda_available']}")
        if pytorch_info["cuda_available"]:
            print(f"CUDA Version: {pytorch_info['cuda_version']}")

    print("\nInstallation recommendation:")
    print(f"  {recommend_installation()}")
    print()
    return pytorch_info


def demonstrate_device_management() -> dict:
    print("Device management demo")
    print("=" * 50)

    device_manager = get_device_manager()
    print(f"1. Auto-selected device: {device_manager.device}")

    device_info = device_manager.get_device_info()
    print("\n2. Device information:")
    for key, value in device_info.items():
        print(f"   {key}: {value}")

    print("\n3. Manual selection:")
    cpu_manager = get_device_manager("cpu")
    print(f"   CPU: {cpu_manager.device} (is_cpu={cpu_manager.is_cpu})")

    if device_info.get("cuda_available", False):
        cuda_manager = get_device_manager("cuda")
        info = cuda_manager.get_device_info()
        print(f"   CUDA: {cuda_manager.device} (is_cuda={cuda_manager.is_cuda})")
        print(f"   GPU name: {info.get('gpu_name', 'Unknown')}")
        print(f"   GPU memory: {info.get('gpu_memory_gb', 0):.1f} GB")
    else:
        print("   CUDA not available — using CPU")
    print()
    return device_info


def demonstrate_model_with_device(device_info: dict) -> None:
    print("GNN device examples (code snippets)")
    print("=" * 50)

    print(
        """
# Auto-detect
result = GNN(
    dataset_name="R8",
    dataset_path="./data",
    builder=Builder.Tfidf,
    device=None,
).fit()

# Force CPU
result = GNN(
    dataset_name="R8",
    dataset_path="./data",
    builder=Builder.Tfidf,
    device="cpu",
).fit()
"""
    )

    if device_info.get("cuda_available", False):
        print(
            """
# Force GPU
result = GNN(
    dataset_name="R8",
    dataset_path="./data",
    builder=Builder.Tfidf,
    device="cuda",
).fit()
"""
        )


def demonstrate_installation_options() -> None:
    print("Installation options")
    print("=" * 50)
    print("  pip install text4gcn[cpu]")
    print("  pip install text4gcn[gpu]")
    print("  pip install text4gcn[syntactic]  # CoreNLP wrapper (optional)")
    print("  poetry install --extras cpu")
    print()


def main() -> None:
    print("Text4GCN Device Management Example")
    print("=" * 60)
    print()

    try:
        check_system_capabilities()
        device_info = demonstrate_device_management()
        demonstrate_model_with_device(device_info)
        demonstrate_installation_options()
        print("Example completed successfully.")
    except Exception as exc:
        print(f"Error: {exc}")
        print("Install PyTorch via: pip install text4gcn[cpu]  or  text4gcn[gpu]")


if __name__ == "__main__":
    main()
