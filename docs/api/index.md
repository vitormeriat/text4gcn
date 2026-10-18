# API Reference

All public classes and functions are importable directly from the `text4gcn` top-level package.

```python
from text4gcn import (
    GNN,
    TextPipeline,
    Builder,
    Layer,
    GraphBuildStats,
    RefinementConfig,
    TextGCNTrainer,
    TrainingResult,
    apply_refinement,
    compute_adjacency_stats,
    load_adjacency_stats,
    set_verbosity,
)
```

Builder classes live in `text4gcn.builder`:

```python
from text4gcn.builder import Frequency, Semantic, Syntactic
```

---

## Module map

| Class / Function | Import path | Description |
|---|---|---|
| [`GNN`](gnn.md) | `text4gcn` | Train a GCN on a text graph |
| [`TrainingResult`](gnn.md#trainingresult) | `text4gcn` | Structured output of `GNN.fit()` |
| [`TextPipeline`](pipeline.md) | `text4gcn` | Preprocess raw text into node features |
| [`Builder`](builders.md#builder-enum) | `text4gcn` | Enum of available adjacency strategies |
| [`Layer`](gnn.md#layer) | `text4gcn` | Enum of available GNN layer types (`GCN` only) |
| [`TextGCNTrainer`](gnn.md#textgcntrainer) | `text4gcn` | Lower-level training loop behind `GNN.fit()` |
| [`GraphBuildStats`](builders.md#graphbuildstats) | `text4gcn` | Edge/memory stats from adjacency builders |
| [`compute_adjacency_stats`](builders.md#compute_adjacency_stats-load_adjacency_stats) | `text4gcn` | `GraphBuildStats` of an adjacency matrix in memory |
| [`load_adjacency_stats`](builders.md#compute_adjacency_stats-load_adjacency_stats) | `text4gcn` | `GraphBuildStats` of a saved adjacency file |
| [`Frequency`](builders.md#frequency) | `text4gcn.builder` | TF-IDF / PMI adjacency |
| [`Semantic`](builders.md#semantic) | `text4gcn.builder` | Word2Vec / BERT / cosine adjacency |
| [`Syntactic`](builders.md#syntactic) | `text4gcn.builder` | Dependency parsing / LIWC adjacency |
| [`RefinementConfig`](refinement.md#refinementconfig) | `text4gcn` | Graph refinement parameters |
| [`apply_refinement`](refinement.md#apply_refinement) | `text4gcn` | Apply refinement to loaded corpus |
| [`NodeMetadata`](refinement.md#nodemetadata) | `text4gcn.graph` | Per-node metadata returned by refinement |
| [`save_corpus`](datasets.md#save_corpus) | `text4gcn.datasets` | Write your own corpus in the input format |
| [`twenty_newsgroups`](datasets.md#twenty_newsgroups) | `text4gcn.datasets` | Download 20 Newsgroups in the input format |
| [`download`](datasets.md#download) | `text4gcn.datasets` | Download R8, R52, Ohsumed or PubMed |
| [`set_verbosity`](#diagnostics) | `text4gcn` | Show diagnostic logs (shapes, model structure) |

Other public helpers, for advanced use: `GraphRefinement` and `build_node_metadata` (`text4gcn.graph`, what `apply_refinement` uses), `plot_training_history` (`text4gcn.models`, writes the accuracy/loss plots), and the device functions `get_device_manager`, `set_global_device`, `get_global_device`, `check_pytorch_installation` and `DeviceManager` (`text4gcn.utils`, see [Device Management](../guides/devices.md)).

---

## Output and diagnostics

Text4GCN separates three kinds of output, as most current ML libraries do:

| What | Where | Controlled by |
|---|---|---|
| **Progress and summaries** of each step | printed to stdout | `verbose=` of `TextPipeline`, the builders and `GNN`: `0` silent, `1` (default) progress bars and summaries, `2` no progress bars (see below) |
| **Results** | returned objects: `TrainingResult` (metrics, per-class report, confusion matrix, per-epoch history) and `GraphBuildStats` | — |
| **Diagnostics**: array shapes, model structure, file details | the standard `logging` module, logger `"text4gcn"` | `text4gcn.set_verbosity(...)` or your own logging configuration |

`verbose` per class (any other value raises `ValueError`):

| Class | `1` (default) | `2` |
|---|---|---|
| `TextPipeline` | One summary line per stage (no progress bars) | Same as `1` |
| `Frequency` / `Semantic` / `Syntactic` | Header, a progress bar during slow steps, summary line | Header and summary line, no progress bar |
| `GNN` | Progress bar over the epochs, then the summary | One line per epoch instead of the bar, then the summary |

### Diagnostics

The `"text4gcn"` logger has only a `NullHandler`, so it is quiet unless you ask for it:

```python
import text4gcn

text4gcn.set_verbosity("debug")    # "debug", "info", "warning", "error", "critical"
```

`set_verbosity` also accepts a `logging` level number (e.g. `logging.DEBUG`); an unknown name raises `ValueError`.

`set_verbosity` sends the library's messages to stderr; the environment variable
`TEXT4GCN_VERBOSITY=debug` does the same at import time. If your application already
configures `logging`, the `text4gcn` loggers follow that configuration instead.

