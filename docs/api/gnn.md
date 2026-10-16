# GNN

```python
from text4gcn import GNN, TrainingResult, Layer
```

---

## GNN

The main entry point for training a Graph Convolutional Network on a text dataset.

```python
class GNN:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        log_dir: str = ".",
        builder: Builder = Builder.Tfidf,
        lr: float = 0.02,
        nhid: int = 200,
        seed: int = 1234,
        epochs: int = 200,
        dropout: float = 0.5,
        layer: Layer = Layer.GCN,
        early_stopping: int = 10,
        device: str | None = None,
        refinement_config: RefinementConfig | None = None,
        verbose: int = 1,
    ) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | `str` | — | Dataset name. Must match the prefix of `.txt`, `.meta`, and preprocessed files on disk. |
| `dataset_path` | `str` | — | Directory containing the dataset files. |
| `log_dir` | `str` | `"."` | Directory where model checkpoints (`.pth`) and training plots are saved. |
| `builder` | `Builder` | `Builder.Tfidf` | Adjacency strategy used to build the graph. Must match the adjacency files already on disk. |
| `lr` | `float` | `0.02` | Learning rate for the Adam optimiser. |
| `nhid` | `int` | `200` | Hidden layer dimension. |
| `seed` | `int` | `1234` | Random seed for PyTorch / NumPy / Python (reproducible training). |
| `epochs` | `int` | `200` | Maximum number of training epochs. |
| `dropout` | `float` | `0.5` | Dropout rate applied between GCN layers. |
| `layer` | `Layer` | `Layer.GCN` | GNN layer type. Only `Layer.GCN` is supported. |
| `early_stopping` | `int` | `10` | Stop when the validation loss rises above its mean over this many previous epochs (the TextGCN criterion); the best validation weights are restored. |
| `device` | `str \| None` | `None` | PyTorch device (`"cpu"`, `"cuda"`, `"cuda:1"`, …). Detected automatically when `None`. |
| `refinement_config` | `RefinementConfig \| None` | `None` | Optional graph refinement applied after loading the corpus. |
| `verbose` | `int` | `1` | `0` silent; `1` a progress bar over the epochs and a summary; `2` one line per epoch instead of the bar (for logs, CI and saved notebooks). |

### Raises

| Exception | When |
|---|---|
| `FileNotFoundError` | Dataset path missing, or node feature / adjacency files missing (includes the command to fix it). |
| `ValueError` | Empty `dataset_name`. |
| `TypeError` | An unsupported `layer` type is requested. |

---

### `fit()`

```python
def fit(self) -> TrainingResult
```

Runs the full pipeline: validates paths → loads corpus → applies optional refinement → trains GCN → evaluates on test set.

Saves the best checkpoint, `metrics-acc.png` and `metrics-loss.png` to `log_dir`. With the default `verbose=1` it prints a progress bar while training, then a summary:

```text
text4gcn · GNN("20NG4") · frequency.tfidf · cpu · seed 1234
  graph     14,537 nodes: 1,973 train · 219 validation · 1,455 test documents + 10,890 words
  model     GCN 14,537 → 200 → 4 · 2.91 M parameters · dropout 0.5 · lr 0.02
  ✓ train     52 epochs (early stop) · best epoch 50 (val_loss 1.3680, val_acc 93.2%)  (6m30s)
  ✓ test      accuracy 86.7% · macro-F1 86.2% · loss 1.2886 · 1,455 documents  (0.6 s)

                     class  precision  recall    f1  support
               alt.atheism      0.805   0.883 0.842      384
                       ...
  model     logs/20261017-000340123.pth
```

Diagnostics (array shapes, model structure) are logged at debug level: see [Diagnostics](index.md#diagnostics).

**Returns** [`TrainingResult`](#trainingresult)

---

### Examples

**Minimal:**

```python
from text4gcn import GNN

result = GNN(dataset_name="R8", dataset_path="data/").fit()
print(result.test_accuracy)
```

**Full configuration:**

```python
from text4gcn import GNN, Builder, Layer

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    log_dir="logs/",
    builder=Builder.Tfidf,
    seed=1234,
    epochs=200,
    lr=0.02,
    nhid=200,
    dropout=0.5,
    early_stopping=10,
    device=None,         # auto-detect
).fit()

print(result)
```

!!! tip "Validation split"
    Reserve validation documents in preprocessing, not in `GNN`:

    ```python
    TextPipeline(dataset_name="R8", dataset_path="data/", seed=1234).execute(
        validation_ratio=0.10,
    )
    ```

**With graph refinement:**

```python
from text4gcn import GNN, Builder, RefinementConfig

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    builder=Builder.Tfidf,
    refinement_config=RefinementConfig(
        min_doc_freq=2,
        edge_weight_threshold=0.1,
        keep_docs=True,
    ),
).fit()
```

**Force CPU:**

```python
result = GNN(dataset_name="R8", dataset_path="data/", device="cpu").fit()
```

---

## TrainingResult

```python
from text4gcn import TrainingResult
```

Structured output of `GNN.fit()`. A frozen dataclass: its fields cannot be changed.

```python
@dataclass(frozen=True)
class TrainingResult:
    test_accuracy: float
    test_loss: float
    model_path: str
    epochs_trained: int
    best_epoch: int
    metrics: dict[str, float]
    history: pandas.DataFrame
    report: pandas.DataFrame
    confusion_matrix: pandas.DataFrame
```

### Fields

| Field | Type | Description |
|---|---|---|
| `test_accuracy` | `float` | Accuracy on the held-out test set (0–1). |
| `test_loss` | `float` | Cross-entropy loss on the held-out test set. |
| `model_path` | `str` | Path of the saved checkpoint (`.pth`) inside `log_dir`, holding the weights with the best validation loss. |
| `epochs_trained` | `int` | Number of epochs that completed before stopping. |
| `best_epoch` | `int` | Epoch with the lowest validation loss; its weights are the ones restored and saved. |
| `metrics` | `dict` | Test `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`. |
| `history` | `DataFrame` | One row per epoch (index `epoch`): `loss`, `accuracy`, `val_loss`, `val_accuracy`, `seconds`. |
| `report` | `DataFrame` | Test `precision`, `recall`, `f1`, `support` per class, indexed by class name. |
| `confusion_matrix` | `DataFrame` | Test confusion matrix: rows are true classes, columns predicted ones. |

### Example

```python
result = GNN(dataset_name="R8", dataset_path="data/").fit()

result                        # TrainingResult(accuracy=0.9712, macro_f1=0.9301, loss=0.0843, epochs=187, best_epoch=177, ...)
result.metrics["macro_f1"]    # 0.9301
result.history[["loss", "val_loss"]].plot()   # learning curves
result.confusion_matrix       # true × predicted, with class names
```

---

## Layer

```python
from text4gcn import Layer
```

Available GNN layer types. Currently only `GCN` is implemented (DCAI 2022 / Yao et al.).

| Value | Description |
|---|---|
| `Layer.GCN` | Graph Convolutional Network (Kipf & Welling, 2017) |
