<div align="center">
  <img width="70%" src="https://raw.githubusercontent.com/vitormeriat/text4gcn/main/imgs/logo.svg">
  <h1 style="margin-bottom:40px; margin-top:20px">Text4GCN</h1>
  <p>A Python framework for text classification using Graph Neural Networks</p>
</div>

-----------------

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/text4gcn.svg)](https://badge.fury.io/py/text4gcn)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/vitormeriat/text4gcn/actions/workflows/ci.yml/badge.svg)](https://github.com/vitormeriat/text4gcn/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-vitormeriat.github.io-blue)](https://vitormeriat.github.io/text4gcn/)

<div id="top"></div>

Text4GCN converts a collection of text documents into a **word-document graph** and trains a Graph Convolutional Network (GCN) on it for document classification. The graph captures vocabulary co-occurrence patterns that traditional bag-of-words models miss.

### Scope and limitations

- **Nicho científico:** construção de *text graphs* e TGCN com **Related Word Filter (RWF)** — menos arestas / memória, accuracy ≈ TextGCN clássico (Pereira & de Castro, DCAI 2022). Não é um substituto de transformers fine-tuned para SOTA em textos curtos.
- **Aprendizado transdutivo:** documentos de teste entram no grafo na construção (como Yao et al., AAAI 2019, e o paper do pacote). Novos documentos exigem reconstruir o grafo e retreinar.
- **Protocolo experimental padrão:** dropout 0.5, lr 0.02, hidden 200, early stopping 10, 10% do train como validação (`TextPipeline.execute(validation_ratio=0.10)`).

Reprodução das Tabelas 2/3 do paper DCAI 2022:

```bash
python scripts/reproduce_dcai2022.py --reference-only
python scripts/reproduce_dcai2022.py --data-dir data/ --dataset R8 --builders frequency,semantic
```

---

## Installation

```bash
pip install text4gcn
pip install "text4gcn[syntactic]"   # optional: Stanford CoreNLP wrapper (Java jars still required)
pip install "text4gcn[bert]"        # optional: transformers, for the experimental BERT builder
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.3, installed from PyPI with the package (CUDA-enabled
on Linux, CPU-only on Windows and macOS). For another build, install PyTorch first from
[pytorch.org](https://pytorch.org/get-started/locally/). The default install does **not**
require Java; `Syntactic.dependency_parsing()` needs the `syntactic` extra plus a local CoreNLP path.

---

## Quick start

```python
from text4gcn import GNN, TextPipeline
from text4gcn.builder import Frequency

# Step 1 — preprocess raw text
TextPipeline(dataset_name="R8", dataset_path="data/").execute()

# Step 2 — build the word-document graph (returns edge/memory stats)
stats = Frequency(dataset_name="R8", dataset_path="data/").tfidf()
print(stats)  # nodes, edges(nnz), density, file size

# Step 3 — train and evaluate
result = GNN(dataset_name="R8", dataset_path="data/").fit()
print(f"Test accuracy: {result.test_accuracy:.4f}")
```

`GNN.fit()` returns a `TrainingResult` dataclass with `test_accuracy`, `test_loss`, `model_path`, and `epochs_trained`.

### Tutorials

Five notebooks, meant to be read in order ([overview](https://vitormeriat.github.io/text4gcn/tutorials/)):

| # | Tutorial | |
|---|---|---|
| 1 | [Text as a graph](docs/tutorials/01_text_as_a_graph.ipynb): what nodes and edges Text4GCN builds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitormeriat/text4gcn/blob/main/docs/tutorials/01_text_as_a_graph.ipynb) |
| 2 | [Your first classifier](docs/tutorials/02_first_classifier.ipynb): 20 Newsgroups, from raw text to accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitormeriat/text4gcn/blob/main/docs/tutorials/02_first_classifier.ipynb) |
| 3 | [Choosing a graph builder](docs/tutorials/03_choosing_a_graph_builder.ipynb): TF-IDF vs. Word2Vec RWF vs. cosine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitormeriat/text4gcn/blob/main/docs/tutorials/03_choosing_a_graph_builder.ipynb) |
| 4 | [Refining the graph](docs/tutorials/04_refining_the_graph.ipynb): fewer edges, same accuracy? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitormeriat/text4gcn/blob/main/docs/tutorials/04_refining_the_graph.ipynb) |
| 5 | [Bring your own corpus](docs/tutorials/05_bring_your_own_corpus.ipynb): your data, any language | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vitormeriat/text4gcn/blob/main/docs/tutorials/05_bring_your_own_corpus.ipynb) |

---

## How it works

```
Raw text documents
      │
      ▼  TextPipeline
Cleaned corpus + node feature matrices
      │
      ▼  Frequency / Semantic / Syntactic builder
Adjacency matrix (word–word and word–doc edges)
      │
      ▼  GNN.fit()
Trained GCN → TrainingResult(accuracy, loss, model_path, epochs)
```

---

## TextPipeline

Cleans the corpus, builds vocabulary, and generates node feature matrices:

```python
from text4gcn import TextPipeline

TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    rare_count=5,        # drop words appearing fewer than 5 times
    language="english",
).execute(
    validation_ratio=0.10,  # fraction of train held out for validation
)
```

---

## Adjacency builders

Choose an edge-weighting strategy:

| Builder class | Method | Enum value |
|---|---|---|
| `Frequency` | `.tfidf()` | `Builder.Tfidf` |
| `Semantic` | `.word2vec()` | `Builder.Word2Vec` |
| `Semantic` | `.cosine_similarity()` | `Builder.CosineSimilarity` |
| `Semantic` | `.bert()` | — (graph is built, but not yet selectable in `GNN`) |
| `Syntactic` | `.dependency_parsing()` | `Builder.DependencyParsing` |
| `Syntactic` | `.linguistic_inquiry()` | `Builder.Liwc` |

```python
from text4gcn.builder import Frequency, Semantic

# TF-IDF (fastest, recommended start) — returns GraphBuildStats
stats = Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)
print(stats.n_edges, stats.file_bytes)

# Or inspect a saved adjacency file later
from text4gcn import load_adjacency_stats
load_adjacency_stats("data/R8.adjacency/ind.frequency.tfidf.R8.adj")

# Word2Vec
Semantic(dataset_name="R8", dataset_path="data/").word2vec(
    training_algorithm=1, embedding_dimension=300, num_epochs=20
)
```

---

## GNN training

```python
from text4gcn import GNN, Builder

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    log_dir="logs/",
    builder=Builder.Tfidf,
    epochs=200,
    lr=0.02,
    nhid=200,
    dropout=0.5,
    val_ratio=0.1,
    early_stopping=10,
    device=None,          # CUDA if available, else CPU
).fit()

print(result.test_accuracy)   # e.g. 0.9712
print(result.epochs_trained)  # e.g. 187
print(result.model_path)      # path to saved .pth file
```

---

## Graph refinement

Reduce graph size and noise before training:

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

---

## Benchmark datasets

| Dataset | Docs | Classes | Train | Test | Get it with |
|---------|------|---------|-------|------|-------------|
| R8      | 7,674 | 8 | 5,485 | 2,189 | `download("R8", path)` |
| R52     | 9,100 | 52 | 6,532 | 2,568 | `download("R52", path)` |
| Ohsumed | 7,400 | 23 | 3,357 | 4,043 | `download("ohsumed", path)` |
| PubMed  | 19,717 | 3 | 13,802 | 5,915 | `download("pubmed", path)` |
| 20NG    | 18,846 | 20 | 11,314 | 7,532 | `twenty_newsgroups(path)` |
| MR      | 10,662 | 2 | 7,108 | 3,554 | your own files, via `save_corpus(...)` |

```python
from text4gcn.datasets import download

download("R8", "data/")  # writes data/R8.txt and data/R8.meta (SHA-256 verified)
```

---

## Documentation

Full documentation at **[vitormeriat.github.io/text4gcn](https://vitormeriat.github.io/text4gcn/)**.

---

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## References

- Pereira & de Castro (2022) — *Natural Language Processing Based on a Text Graph Convolutional Network* (DCAI). See [`papers/`](https://github.com/vitormeriat/text4gcn/tree/main/papers).
- Yao, Mao & Luo (2019) — *Graph Convolutional Networks for Text Classification* (AAAI)
- Kipf & Welling (2017) — *Semi-supervised Classification with Graph Convolutional Networks*

<p align="right">(<a href="#top">back to top</a>)</p>
