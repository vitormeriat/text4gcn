# Device Management

Text4GCN uses PyTorch and trains on CPU or on an NVIDIA GPU (CUDA). Device selection is automatic by default.

---

## Automatic detection

When `device=None` (the default), `GNN` uses CUDA if `torch.cuda.is_available()`,
and the CPU otherwise.

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
result = GNN(dataset_name="R8", dataset_path="data/", device="cuda:1").fit()
```

Any PyTorch device string is passed on as is. Devices other than `cpu` and `cuda`,
such as Apple's `mps`, are never selected automatically and are not tested.

---

## Checking your PyTorch installation

```python
from text4gcn.utils.device import check_pytorch_installation

info = check_pytorch_installation()
print(info)
# {'torch_version': '2.3.0', 'cuda_available': True, 'cuda_version': '12.1'}
```

---

## Installation by hardware

`pip install text4gcn` installs PyTorch from PyPI. On Linux that build already
supports CUDA; on Windows and macOS it is CPU-only.

To choose the build yourself, for example a specific CUDA version, or a CPU-only
build on Linux (a much smaller download), install PyTorch first with the command
from [pytorch.org](https://pytorch.org/get-started/locally/), then Text4GCN:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # example: CPU-only
pip install text4gcn
```

---

## Performance notes

The GCN trains on the whole graph at once, with the normalised adjacency held as a
dense matrix: memory grows with the **square** of the number of nodes (documents +
vocabulary words). On 20 Newsgroups with 4 categories (14.5k nodes), one training
run took about 7 minutes on CPU. See
[tutorial 5](../tutorials/05_bring_your_own_corpus.ipynb) to estimate the memory of
your own corpus before training.

Graph building (`Frequency.tfidf()` and the other builders) always runs on the CPU;
a GPU speeds up only the training.

---

## Multi-GPU

Multi-GPU training is not currently supported. Use `device="cuda:N"` to select a specific GPU when multiple are available.
