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
    def __init__(self, dataset_name: str, dataset_path: str) -> None
```

### `tfidf()`

```python
def tfidf(self, window_size: int = 20, output_path: str = None) -> GraphBuildStats
```

Computes word-word co-occurrence weights via PMI in a sliding window, and word-document weights via TF-IDF. Writes `ind.frequency.tfidf.{dataset_name}.adj` to `{dataset_path}/{dataset_name}.adjacency/`.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | `int` | `20` | Sliding window size for word-word co-occurrence. |
| `output_path` | `str \| None` | `None` | Override output directory. Uses `{dataset_path}/{dataset_name}.adjacency/` by default. |

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
    def __init__(self, dataset_name: str, dataset_path: str) -> None
```

### `word2vec()`

```python
def word2vec(
    self,
    training_algorithm: int,
    embedding_dimension: int,
    num_epochs: int,
    language: str = "english",
    threshold: float = 0.95,
    output_path: str = None,
) -> GraphBuildStats
```

Trains Word2Vec embeddings on the corpus and connects words whose cosine similarity exceeds `threshold` (DCAI Related Word Filter semantic path).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training_algorithm` | `int` | — | `1` = CBOW, `2` = Skip-Gram. |
| `embedding_dimension` | `int` | — | Embedding size (e.g. 300). |
| `num_epochs` | `int` | — | Training epochs for Word2Vec. |
| `language` | `str` | `"english"` | Language for tokenisation. |
| `threshold` | `float` | `0.95` | Minimum cosine similarity to create an edge. |
| `output_path` | `str \| None` | `None` | Override output directory. |

#### Returns

[`GraphBuildStats`](#graphbuildstats)

#### Example

```python
from text4gcn.builder import Semantic

stats = Semantic(dataset_name="R8", dataset_path="data/").word2vec(
    training_algorithm=1,   # CBOW
    embedding_dimension=300,
    num_epochs=20,
    threshold=0.95,
)
```

---

### `bert()`

```python
def bert(self, language: str = "english", threshold: float = 0.95)
```

Uses a pre-trained BERT model to embed words and connects similar words.

!!! note "Working in Progress"
    BERT adjacency is experimental and may require additional dependencies.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language` | `str` | `"english"` | Language hint for the BERT tokeniser. |
| `threshold` | `float` | `0.95` | Minimum cosine similarity to create an edge. |

#### Example

```python
from text4gcn.builder import Semantic

Semantic(dataset_name="R8", dataset_path="data/").bert(threshold=0.95)
```

---

### `cosine_similarity()`

```python
def cosine_similarity(self, threshold: float = 0.95, output_path: str = None)
```

Connects words using TF-IDF–weighted cosine similarity between document vectors.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.95` | Minimum similarity to create an edge. |
| `output_path` | `str \| None` | `None` | Override output directory. |

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

Builds edges from syntactic structure (dependency parsing, constituency parsing, LIWC categories).

```python
class Syntactic:
    def __init__(self, dataset_name: str, dataset_path: str) -> None
```

### `dependency_parsing()`

```python
def dependency_parsing(self, core_nlp_path: str, language: str = "english")
```

Connects words that have a syntactic dependency relation in the corpus, using Stanford CoreNLP.

!!! note "Optional dependency"
    Install `pip install text4gcn[syntactic]` and provide a local CoreNLP installation (Java jars). The Python wrapper is imported lazily.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `core_nlp_path` | `str` | — | Path to the CoreNLP installation directory. |
| `language` | `str` | `"english"` | Language model to use. |

---

### `linguistic_inquiry()`

```python
def linguistic_inquiry(self, liwc_path: str)
```

Connects words that share LIWC (Linguistic Inquiry and Word Count) categories.

!!! note "Working in Progress"
    Requires access to a LIWC dictionary file.

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

Frozen dataclass returned by `Frequency.tfidf()` and `Semantic.word2vec()` (and other builders that call `build()`).

| Field | Type | Description |
|---|---|---|
| `n_nodes` | `int` | Train docs + vocabulary + test docs. |
| `n_edges` | `int` | Number of stored non-zeros (`nnz`). |
| `density` | `float` | `nnz / n_nodes²`. |
| `file_bytes` | `int \| None` | Pickled adjacency size on disk, if known. |
| `builder` | `str \| None` | Builder label (e.g. `frequency.tfidf`). |
| `dataset_name` | `str \| None` | Dataset name. |
| `path` | `str \| None` | Path to the adjacency file. |
