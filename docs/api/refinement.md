# Graph Refinement

```python
from text4gcn import RefinementConfig, apply_refinement
```

Graph refinement reduces the size and noise of the word-document graph before training. It filters word nodes by frequency or degree, prunes weak edges, and optionally builds per-node metadata.

---

## RefinementConfig

```python
@dataclass
class RefinementConfig:
    # Node filters
    min_degree: float | None = None
    max_degree: float | None = None
    min_word_degree: float | None = None
    max_word_degree: float | None = None
    min_doc_freq: int | None = None
    max_doc_freq_ratio: float | None = None
    keep_docs: bool = True
    drop_word_indices: set[int] | None = None
    keep_word_indices: set[int] | None = None

    # Edge pruning
    edge_weight_threshold: float | None = None
    top_k_edges_per_node: int | None = None

    # Vocabulary and stopwords
    vocab_path: str | None = None
    stopwords: set[str] | None = None

    # Output
    build_metadata: bool = True
```

### Node filter fields

| Field | Type | Default | Description |
|---|---|---|---|
| `min_degree` | `float \| None` | `None` | Drop any node with total weighted degree below this. |
| `max_degree` | `float \| None` | `None` | Drop any node with total weighted degree above this (hub pruning). |
| `min_word_degree` | `float \| None` | `None` | Drop **word** nodes with degree below this. |
| `max_word_degree` | `float \| None` | `None` | Drop **word** nodes with degree above this. |
| `min_doc_freq` | `int \| None` | `None` | Drop word nodes that appear in fewer than this many documents (equivalent to sklearn's `min_df`). |
| `max_doc_freq_ratio` | `float \| None` | `None` | Drop word nodes that appear in more than this fraction of documents (equivalent to `max_df`). |
| `keep_docs` | `bool` | `True` | When `True`, document nodes are never removed: every node filter, including `min_degree` / `max_degree`, applies to word nodes only. |
| `drop_word_indices` | `set[int] \| None` | `None` | Explicit set of vocabulary indices to drop. |
| `keep_word_indices` | `set[int] \| None` | `None` | When set, only these vocabulary indices are kept (overrides other word filters). |

### Edge pruning fields

| Field | Type | Default | Description |
|---|---|---|---|
| `edge_weight_threshold` | `float \| None` | `None` | Drop edges with weight strictly below this value. |
| `top_k_edges_per_node` | `int \| None` | `None` | Keep only the top-k heaviest edges per node. An edge survives if it is in the top-k of either endpoint, with its original weight. |

### Vocabulary fields

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_path` | `str \| None` | `None` | Path to vocabulary file (one word per line). Required for `stopwords` filtering. If `None`, the vocab is auto-located from `dataset_path`. |
| `stopwords` | `set[str] \| None` | `None` | Set of word strings to drop. Only effective when a vocabulary is available. |

### Output fields

| Field | Type | Default | Description |
|---|---|---|---|
| `build_metadata` | `bool` | `True` | Whether to populate the `metadata` list in the refinement output. |

---

## apply_refinement()

```python
from text4gcn import apply_refinement

def apply_refinement(
    predata: tuple,
    config: RefinementConfig,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    return_metadata: bool = False,
) -> tuple[tuple, dict | None]
```

Applies graph refinement to the output of `LoadData.load_corpus()` (`text4gcn.models.utils.LoadData`). Returns a new `predata` tuple with the same structure.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `predata` | `tuple` | Output of `load_corpus()`: `(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size)`. |
| `config` | `RefinementConfig` | Refinement configuration. |
| `dataset_path` | `str \| None` | Dataset directory, used to auto-locate the vocabulary file. |
| `dataset_name` | `str \| None` | Dataset name, used to auto-locate the vocabulary file. |
| `return_metadata` | `bool` | If `True`, the second return value is a dict with `"metadata"` and `"stats"` keys. |

### Returns

`(predata_refined, info)` where:
- `predata_refined` — same tuple structure as `predata`, with the refined graph.
- `info` — `None` if `return_metadata=False`; otherwise a dict with:
  - `"metadata"` — `list[NodeMetadata]`, one entry per node in the refined graph.
  - `"stats"` — dict with `nodes_original`, `nodes_refined`, `nodes_removed`, `edges_before`, `edges_after`, `train_docs_kept`, `test_docs_kept`.

### Raises

| Exception | When |
|---|---|
| `ValueError` | All nodes (or all train/test docs) were removed by the filters. |

---

## NodeMetadata

```python
from text4gcn.graph import NodeMetadata
```

Metadata for a single node in the (possibly refined) graph.

```python
@dataclass
class NodeMetadata:
    node_type: str          # 'train_doc', 'word', 'test_doc'
    original_index: int     # index in the un-refined graph
    degree: float           # weighted degree in the refined graph
    word_id: int | None     # vocabulary index (word nodes only)
    word: str | None        # word string (word nodes only, if vocab available)
    doc_id: int | None      # document index (doc nodes only)
```

`NodeMetadata.to_dict()` exports the fields as a plain dictionary for serialisation or Pandas ingestion.

---

## Examples

**Basic — drop rare words and weak edges:**

```python
from text4gcn import GNN, Builder, RefinementConfig

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    builder=Builder.Tfidf,
    refinement_config=RefinementConfig(
        min_doc_freq=2,              # drop words in fewer than 2 docs
        edge_weight_threshold=0.1,   # drop very weak edges
        keep_docs=True,
    ),
).fit()
```

**Advanced — inspect refinement stats:**

```python
from text4gcn import RefinementConfig, apply_refinement, Builder
from text4gcn.models.utils import LoadData

predata = LoadData(
    path="data/",
    dataset="R8",
    builder=Builder.Tfidf,
).load_corpus()

config = RefinementConfig(min_doc_freq=3, edge_weight_threshold=0.05)
predata_refined, info = apply_refinement(
    predata,
    config,
    dataset_path="data/",
    dataset_name="R8",
    return_metadata=True,
)

print(info["stats"])
# {'nodes_original': ..., 'nodes_refined': ..., 'nodes_removed': ..., 'edges_before': ..., 'edges_after': ..., ...}
```

**Drop specific words by vocabulary index:**

```python
RefinementConfig(
    drop_word_indices={0, 1, 2},   # drop the first three vocab entries
    keep_docs=True,
)
```

**String-based stopword filtering (requires vocab_path):**

```python
RefinementConfig(
    stopwords={"the", "a", "an", "is", "in"},
    vocab_path="data/R8.shuffled/R8.vocab",
)
```
