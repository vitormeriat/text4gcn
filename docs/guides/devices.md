# Device Management

Text4GCN uses PyTorch and supports training on CPU, CUDA GPU, and Apple Silicon MPS. Device selection is automatic by default.

---

## Automatic detection

When `device=None` (the default), `GNN` selects the best available device at runtime:

1. CUDA — if `torch.cuda.is_available()`
2. MPS — if `torch.backends.mps.is_available()` (Apple Silicon)
3. CPU — fallback

```python
from text4gcn import GNN

result = GNN(dataset_name="R8", dataset_path="data/").fit()
# Automatically uses CUDA if available
```

---

## Manual device selection

Override the device with the `device` parameter:

```python
# Force CPU
result = GNN(dataset_name="R8", dataset_path="data/", device="cpu").fit()

# Specific CUDA device
result = GNN(dataset_name="R8", dataset_path="data/", device="cuda:0").fit()
result = GNN(dataset_name="R8", dataset_path="data/", device="cuda:1").fit()

# Apple Silicon
result = GNN(dataset_name="R8", dataset_path="data/", device="mps").fit()
```

---

## Checking your PyTorch installation

```python
from text4gcn.utils.device import check_pytorch_installation

info = check_pytorch_installation()
print(info)
# {
#   'torch_version': '2.3.0',
#   'cuda_available': True,
#   'cuda_version': '12.1',
#   'mps_available': False,
#   'device_count': 1,
# }
```

---

## Installation by hardware

Install the PyTorch build that matches your hardware:

=== "CPU only"
    ```bash
    pip install text4gcn[cpu]
    ```

=== "NVIDIA GPU (CUDA)"
    ```bash
    pip install text4gcn[gpu]
    ```

=== "Apple Silicon (MPS)"
    ```bash
    pip install text4gcn[cpu]
    # MPS is part of the standard macOS PyTorch build
    ```

---

## Performance notes

| Corpus size | Recommended device |
|---|---|
| Small (< 10k docs) | CPU is usually fast enough |
| Medium (10k – 100k docs) | GPU provides 3–10× speedup |
| Large (> 100k docs) | GPU strongly recommended |

Text4GCN graphs are sparse, so the GPU advantage scales with graph density. For small benchmark datasets like R8 or MR, CPU training completes in under a minute.

---

## Multi-GPU

Multi-GPU training is not currently supported. Use `device="cuda:N"` to select a specific GPU when multiple are available.
