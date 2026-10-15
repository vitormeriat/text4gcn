<div align="center">
  <img width="70%" src="https://raw.githubusercontent.com/vitormeriat/text4gcn/main/imgs/logo.svg">
  <h1 style="margin-bottom:40px; margin-top:20px">Text4GCN</h1>
  <p>A Python framework for text classification using Graph Neural Networks</p>
</div>

-----------------

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/text4gcn.svg)](https://badge.fury.io/py/text4gcn)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/vitormeriat/text4gcn/workflows/CI/badge.svg)](https://github.com/vitormeriat/text4gcn/actions)
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
pip install text4gcn[cpu]         # CPU
pip install text4gcn[gpu]         # GPU / CUDA
pip install text4gcn[syntactic]   # optional: Stanford CoreNLP wrapper (Java jars still required)
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. The default install does **not** require Java;
`Syntactic.dependency_parsing()` needs `pip install text4gcn[syntactic]` plus a local CoreNLP path.

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
| `Semantic` | `.bert()` | `Builder.Bert` |
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
    device=None,          # auto-detect CUDA / MPS / CPU
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

| Dataset | Docs | Classes | Train | Test | Avg. length |
|---------|------|---------|-------|------|-------------|
| R8      | 7,674 | 8 | 5,485 | 2,189 | 65.7 |
| R52     | 9,100 | 52 | 6,532 | 2,568 | 69.8 |
| 20NG    | 18,846 | 20 | 11,314 | 7,532 | 221.3 |
| MR      | 10,662 | 2 | 7,108 | 3,554 | 20.4 |
| Ohsumed | 7,400 | 23 | 3,357 | 4,043 | 135.8 |

```python
from text4gcn.datasets import data

data.list()
data.R8(path="data/")
data.R52(path="data/")
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
