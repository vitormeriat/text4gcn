# Getting Started

## Installation

Text4GCN requires Python ≥ 3.10 and PyTorch ≥ 2.0. Install the version that matches your hardware:

=== "CPU"
    ```bash
    pip install text4gcn[cpu]
    ```

=== "GPU (CUDA)"
    ```bash
    pip install text4gcn[gpu]
    ```

=== "From source"
    ```bash
    git clone https://github.com/vitormeriat/text4gcn.git
    cd text4gcn
    pip install -e .[cpu]   # or .[gpu]
    ```

=== "Syntactic (CoreNLP)"
    ```bash
    pip install text4gcn[syntactic]   # Python wrapper; Java CoreNLP jars still required
    ```

=== "Development"
    ```bash
    git clone https://github.com/vitormeriat/text4gcn.git
    cd text4gcn
    poetry install
    ```

### Verify installation

```python
import text4gcn
print(text4gcn.__version__)

from text4gcn.utils.device import check_pytorch_installation
info = check_pytorch_installation()
print(f"PyTorch {info['torch_version']} — CUDA: {info['cuda_available']}")
```

---

## Data format

Each dataset needs two files in the same directory:

**`{dataset_name}.txt`** — one document per line, UTF-8 or Latin-1:
```
computer software engineering systems design
natural language processing machine learning algorithms
```

**`{dataset_name}.meta`** — tab-separated: `index  split  label`:
```
0	train	technology
1	train	science
2	test	technology
```

Supported split values: `train`, `training`, `20news-bydate-train`, anything else is treated as `test`.

### Get a dataset

```python
from text4gcn.datasets import download, save_corpus, twenty_newsgroups

download("R8", "data/")                       # R8, R52, ohsumed or pubmed (SHA-256 verified)
twenty_newsgroups("data/", name="20NG4",      # via scikit-learn, official train/test split
                  categories=["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"])
save_corpus("data/", "mine",                  # your own texts, labels and train/test splits
            texts=["great movie", "boring plot"], labels=["pos", "neg"], splits=["train", "test"])
```

See the [Datasets API](api/datasets.md) and the [Bring your own corpus](tutorials/05_bring_your_own_corpus.ipynb) tutorial.

---

## The three-step pipeline

### Step 1 — Preprocess text

[`TextPipeline`](api/pipeline.md) cleans the corpus, shuffles train/test splits, builds a vocabulary, and extracts node feature matrices. Output is written to disk under `{dataset_path}/`.

```python
from text4gcn import TextPipeline

TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    rare_count=5,       # drop words appearing fewer than 5 times
    language="english", # language for stop-word removal
    seed=1234,
).execute(validation_ratio=0.10)
```


After this step the following directories are created:

```
data/
  R8.cleaned/          ← stop-word and rare-word filtered corpus
  R8.shuffled/         ← train/test splits, vocabulary, word vectors
  R8.node_features/    ← x, y, tx, ty, allx, ally (pickle files)
  log/                 ← pipeline execution log
```

### Step 2 — Build the graph

Choose an adjacency strategy that fits your data and compute time budget.
See [Choosing a Builder](guides/builders.md) for a comparison.

=== "TF-IDF (recommended start)"
    ```python
    from text4gcn.builder import Frequency

    stats = Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)
    print(stats)  # GraphBuildStats
    ```


=== "Word2Vec"
    ```python
    from text4gcn.builder import Semantic

    Semantic(dataset_name="R8", dataset_path="data/").word2vec(
        training_algorithm=1,   # 1 = skip-gram, 0 = CBOW
        embedding_dimension=300,
        num_epochs=20,
        threshold=0.7,
    )
    ```

=== "BERT"
    ```python
    from text4gcn.builder import Semantic

    Semantic(dataset_name="R8", dataset_path="data/").bert(threshold=0.95)
    ```

=== "Cosine similarity"
    ```python
    from text4gcn.builder import Semantic

    Semantic(dataset_name="R8", dataset_path="data/").cosine_similarity(threshold=0.9)
    ```

After this step:

```
data/
  R8.adjacency/
    ind.frequency.tfidf.R8.adj   ← serialised scipy sparse matrix
```

### Step 3 — Train the GCN

```python
from text4gcn import GNN, Builder

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
).fit()

print(result)
# TrainingResult(accuracy=0.9712, loss=0.09843, epochs=187, model_path='logs/20240501-120000.pth')
```

`fit()` prints per-epoch progress to stdout and saves accuracy/loss plots to `log_dir/`.

!!! note "Validation ratio"
    The train/val split is fixed by `TextPipeline.execute(validation_ratio=…)`. `GNN(val_ratio=…)` does not rebuild masks.

---

## Minimal usage

All parameters except `dataset_name` and `dataset_path` have sensible defaults:

```python
from text4gcn import GNN

result = GNN(dataset_name="R8", dataset_path="data/", seed=1234).fit()
print(f"accuracy={result.test_accuracy:.4f}")
```

Defaults: `builder=Builder.Tfidf`, `log_dir="."`, `epochs=200`, `lr=0.02`, `nhid=200`, `dropout=0.5`, `seed=1234`, `device=None` (auto-detect).

---

## With graph refinement

Reduce graph size and noise before training:

```python
from text4gcn import GNN, Builder, RefinementConfig

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    builder=Builder.Tfidf,
    refinement_config=RefinementConfig(
        min_doc_freq=2,              # drop words in fewer than 2 documents
        min_word_degree=1.0,         # drop isolated word nodes
        edge_weight_threshold=0.1,   # drop very weak edges
        keep_docs=True,              # always preserve document nodes
    ),
).fit()
```

See [Graph Refinement](guides/refinement.md) for the full parameter reference.

---

## Error messages

If you call `GNN.fit()` without running a previous step, you get an actionable error:

```
FileNotFoundError: Node feature files not found at 'data/R8.node_features'.

Run the preprocessing step first:

  from text4gcn import TextPipeline
  TextPipeline(
      dataset_name='R8',
      dataset_path='data/',
  ).execute()
```
