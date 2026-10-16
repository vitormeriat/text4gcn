# Builders

```python
from text4gcn.builder import Frequency, Semantic, Syntactic
```

Builders compute the adjacency matrix — the edges of the word-document graph — from the preprocessed corpus. Run a builder after [`TextPipeline`](pipeline.md) and before [`GNN`](gnn.md).

---

## Builder (enum)

```python
from text4gcn import Builder
```

Selects which adjacency file `GNN` loads at training time.

| Value | Adjacency file prefix | Built by |
|---|---|---|
| `Builder.Tfidf` | `frequency.tfidf` | `Frequency(...).tfidf()` |
| `Builder.Word2Vec` | `embedding.word2vec` | `Semantic(...).word2vec()` |
| `Builder.CosineSimilarity` | `cosine.similarity` | `Semantic(...).cosine_similarity()` |
| `Builder.DependencyParsing` | `dependency.parsing` | `Syntactic(...).dependency_parsing()` |
| `Builder.Liwc` | `liwc` | `Syntactic(...).linguistic_inquiry()` |

---

## Frequency

```python
from text4gcn.builder import Frequency
```

Builds a TF-IDF–weighted word-document graph. Fastest option; recommended as a starting point.

```python
class Frequency:
    def __init__(self, dataset_name: str, dataset_path: str, verbose: int = 1) -> None
```

### `tfidf()`

```python
def tfidf(self, window_size: int = 20) -> GraphBuildStats
```

Computes word-word co-occurrence weights via PMI in a sliding window, and word-document weights via TF-IDF. Writes `ind.frequency.tfidf.{dataset_name}.adj` to `{dataset_path}/{dataset_name}.adjacency/`.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | `int` | `20` | Sliding window size for word-word co-occurrence. |

#### Returns

[`GraphBuildStats`](#graphbuildstats) — node/edge counts, density, and optional file size.

#### Example

```python
from text4gcn.builder import Frequency

stats = Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)
print(stats)
```

---

## Semantic

```python
from text4gcn.builder import Semantic
```

Builds edges based on semantic similarity between words (Word2Vec, BERT, cosine similarity).

```python
class Semantic:
    def __init__(self, dataset_name: str, dataset_path: str, verbose: int = 1) -> None
```

### `word2vec()`

```python
def word2vec(
    self,
    training_algorithm: int,
    embedding_dimension: int,
    num_epochs: int,
    language: str = "english",
    threshold: float = 0.7,
    seed: int = 1234,
    workers: int = 1,
) -> GraphBuildStats
```

Trains Word2Vec embeddings on the corpus and connects words whose cosine similarity exceeds `threshold` (DCAI Related Word Filter semantic path).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training_algorithm` | `int` | — | `1` = skip-gram (as in the DCAI 2022 paper), `0` = CBOW. |
| `embedding_dimension` | `int` | — | Embedding size (e.g. 300). |
| `num_epochs` | `int` | — | Word2Vec training epochs (the paper uses 20). |
| `language` | `str` | `"english"` | NLTK stop-word list; stop words get no word–word edges. |
| `threshold` | `float` | `0.7` | Minimum cosine similarity between two adjacent words' embeddings to create an edge. Calibrated on 20 Newsgroups: with 20 epochs, the paper's 0.95 leaves almost no word–word edges. |
| `seed` | `int` | `1234` | Seed of the Word2Vec initialisation. |
| `workers` | `int` | `1` | Word2Vec threads. With `1` the graph is reproducible for a given `seed`; more threads train faster but vary between runs. |

#### Returns

[`GraphBuildStats`](#graphbuildstats)

#### Example

```python
from text4gcn.builder import Semantic

stats = Semantic(dataset_name="R8", dataset_path="data/").word2vec(
    training_algorithm=1,   # skip-gram
    embedding_dimension=300,
    num_epochs=20,
    threshold=0.7,
)
```

---

### `bert()`

```python
def bert(self, language: str = "english", threshold: float = 0.95) -> GraphBuildStats
```

Uses a pre-trained BERT model to embed words and connects similar words.

!!! warning "Experimental"
    `bert()` needs `pip install "text4gcn[bert]"` and always uses `bert-base-uncased`. Its graph is saved, but no `Builder` value selects it, so `GNN` cannot train on it yet.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `str` | `"english"` | NLTK stop-word list. The model is always `bert-base-uncased`. |
| `threshold` | `float` | `0.95` | Minimum cosine similarity to create an edge. |

#### Example

```python
from text4gcn.builder import Semantic

Semantic(dataset_name="R8", dataset_path="data/").bert(threshold=0.95)
```

---

### `cosine_similarity()`

```python
def cosine_similarity(self, threshold: float = 0.95) -> GraphBuildStats
```

Connects words whose WordNet definitions are similar: the cosine similarity of the TF-IDF vectors of their glosses, saved by `TextPipeline`. Words without a WordNet definition get no word–word edges, so the builder is meaningful for English only.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.95` | Minimum similarity to create an edge. |

#### Example

```python
from text4gcn.builder import Semantic

Semantic(dataset_name="R8", dataset_path="data/").cosine_similarity(threshold=0.90)
```

---

## Syntactic

```python
from text4gcn.builder import Syntactic
```

Builds edges from syntactic structure: dependency parsing or LIWC categories.

```python
class Syntactic:
    def __init__(self, dataset_name: str, dataset_path: str, verbose: int = 1) -> None
```

### `dependency_parsing()`

```python
def dependency_parsing(self, core_nlp_path: str, language: str = "english") -> GraphBuildStats
```

Connects words that have a syntactic dependency relation in the corpus, using Stanford CoreNLP.

!!! warning "Experimental"
    Needs `pip install "text4gcn[syntactic]"` and a local Stanford CoreNLP installation (Java jars), which CI does not run. CoreNLP always parses English; `language` only selects the stop words.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `core_nlp_path` | `str` | — | Path to the CoreNLP installation directory. |
| `language` | `str` | `"english"` | NLTK stop-word list. CoreNLP always parses English. |

---

### `linguistic_inquiry()`

```python
def linguistic_inquiry(self, liwc_path: str) -> GraphBuildStats
```

Connects adjacent words that share at least one LIWC (Linguistic Inquiry and Word Count) category, gated by positive PMI. The edge weight is the pair count divided by the largest pair count.

!!! note "LIWC dictionary"
    The LIWC dictionary is proprietary and not distributed with Text4GCN; pass the path to your own `.dic` file.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `liwc_path` | `str` | — | Path to the LIWC dictionary file. |

---

## Output layout

All builders write to the same directory:

```
data/
  R8.adjacency/
    ind.frequency.tfidf.R8.adj        ← Frequency.tfidf()
    ind.embedding.word2vec.R8.adj     ← Semantic.word2vec()
    ind.cosine.similarity.R8.adj      ← Semantic.cosine_similarity()
    ind.dependency.parsing.R8.adj     ← Syntactic.dependency_parsing()
    ind.liwc.R8.adj                   ← Syntactic.linguistic_inquiry()
```

The adjacency file is a serialised `scipy.sparse` matrix. The `Builder` enum value determines which file `GNN` loads at training time.

---

## GraphBuildStats

```python
from text4gcn import GraphBuildStats
```

Frozen dataclass returned by every builder method (`tfidf()`, `word2vec()`, `cosine_similarity()`, `bert()`, `dependency_parsing()`, `linguistic_inquiry()`).

| Field | Type | Description |
|---|---|---|
| `n_nodes` | `int` | Train docs + vocabulary + test docs. |
| `n_edges` | `int` | Number of stored non-zeros (`nnz`). |
| `density` | `float` | `nnz / n_nodes²`. |
| `file_bytes` | `int \| None` | Pickled adjacency size on disk, if known. |
| `builder` | `str \| None` | Builder label (e.g. `frequency.tfidf`). |
| `dataset_name` | `str \| None` | Dataset name. |
| `path` | `str \| None` | Path to the adjacency file. |
| `word_word_edges` | `int \| None` | Stored non-zeros between two word nodes. The builders differ only in these: all keep the same TF-IDF document–word edges. |

`stats.summary()` gives the one-line form that the builders print:

```text
text4gcn · Frequency.tfidf("20NG4") · window 20
  ✓ graph     14,537 nodes · 4.53 M edges (4.26 M word–word) · density 2.15% · 52.0 MiB  (4m00s)
```

`verbose` (`0`, `1` default, `2`) is given to `Frequency`, `Semantic` or `Syntactic`: `0` is silent, `1` adds a progress bar during the slow steps (PMI, word pairs).
