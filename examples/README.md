# Text4GCN Examples

This directory contains runnable examples and test utilities for Text4GCN.

---

## Quick start

The full pipeline has three sequential steps. Each step writes files to disk
that the next step reads.

```python
from text4gcn import GNN, TextPipeline, Builder
from text4gcn.builder import Frequency

# Step 1 — preprocess raw text into node features
TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    rare_count=5,
    language="english",
).execute()

# Step 2 — build the word-document adjacency matrix
Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)

# Step 3 — train the GCN and inspect results
result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    log_dir="logs/",
    builder=Builder.Tfidf,
    epochs=200,
    lr=0.02,
    nhid=200,
    dropout=0.5,
).fit()

print(f"Test accuracy: {result.test_accuracy:.4f}")
print(f"Test loss:     {result.test_loss:.5f}")
print(f"Epochs run:    {result.epochs_trained}")
print(f"Model saved:   {result.model_path}")
```

**Minimum viable usage** — defaults to TF-IDF builder, CPU device, current
directory for logs:

```python
from text4gcn import GNN

result = GNN(dataset_name="R8", dataset_path="data/").fit()
print(result)
```

---

## Graph refinement

Filter low-quality nodes and weak edges before training to get a cleaner
graph without rerunning preprocessing:

```python
from text4gcn import GNN, Builder, RefinementConfig

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    log_dir="logs/",
    builder=Builder.Tfidf,
    refinement_config=RefinementConfig(
        min_doc_freq=2,           # drop words appearing in < 2 docs
        min_word_degree=1.0,      # drop words with very low connectivity
        edge_weight_threshold=0.1, # drop weak edges
        keep_docs=True,           # always preserve all document nodes
        build_metadata=True,      # attach node-type info to refined graph
    ),
).fit()

print(result)
```

---

## Choosing an adjacency builder

```python
from text4gcn.builder import Frequency, Semantic, Syntactic

# TF-IDF (PMI) — fast, good general baseline
Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)

# Word2Vec cosine similarity
Semantic(dataset_name="R8", dataset_path="data/").word2vec(
    training_algorithm=1,      # 1 = CBOW, 2 = Skip-Gram
    embedding_dimension=300,
    num_epochs=20,
    threshold=0.95,
)

# BERT sentence similarity
Semantic(dataset_name="R8", dataset_path="data/").bert(
    language="english",
    threshold=0.95,
)

# Cosine similarity on TF-IDF vectors
Semantic(dataset_name="R8", dataset_path="data/").cosine_similarity(threshold=0.9)
```

Pass the matching `Builder` enum value to `GNN`:

| Builder class method | Builder enum value |
|---|---|
| `Frequency.tfidf()` | `Builder.Tfidf` |
| `Semantic.word2vec()` | `Builder.Word2Vec` |
| `Semantic.cosine_similarity()` | `Builder.CosineSimilarity` |
| `Syntactic.dependency_parsing()` | `Builder.DependencyParsing` |
| `Syntactic.linguistic_inquiry()` | `Builder.Liwc` |

---

## Synthetic data for development / testing

Generate synthetic datasets in the same format as benchmark datasets:

```bash
# Small dataset for quick smoke tests
python synthetic_data_generator.py --dataset-name SyntheticR8 --num-documents 500

# Custom train/test split
python synthetic_data_generator.py --dataset-name SyntheticR8 --num-documents 2000 --train-ratio 0.7
```

Run the full pipeline on synthetic data:

```bash
python run_synthetic_test.py
```

---

## Expected file layout

After preprocessing and building, the dataset directory should look like:

```
data/
  R8.txt                          ← raw documents, one per line
  R8.meta                         ← tab-separated: id  split  label
  R8.cleaned/
    R8.txt                        ← stop-word / rare-word filtered
  R8.shuffled/
    R8.txt  R8.meta  R8.train  R8.test  R8.vocab  R8.word_vectors
  R8.node_features/
    ind.R8.x  ind.R8.y  ind.R8.tx  ind.R8.ty  ind.R8.allx  ind.R8.ally
  R8.adjacency/
    ind.frequency.tfidf.R8.adj    ← adjacency matrix (pickle)
  log/
    R8_dataset.txt                ← pipeline run log
```

---

## Files overview

| File | Description |
|---|---|
| `synthetic_data_generator.py` | Generate synthetic text datasets |
| `run_synthetic_test.py` | Quick end-to-end test with synthetic data |
| `test_textpipeline_synthetic.py` | Detailed pipeline test |
| `simple_synthetic_test.py` | Minimal smoke test |
| `graph_refinement_example.py` | Node filtering and edge pruning demo |
| `device_management.py` | CPU/GPU device selection examples |
| `cons.py` | Full pipeline reference script |

---

## Troubleshooting

**`FileNotFoundError: Node feature files not found`**
— You skipped `TextPipeline.execute()`. Run Step 1 first.

**`FileNotFoundError: Adjacency files not found`**
— You skipped the builder step. Run `Frequency(...).tfidf()` (or another
builder) after preprocessing.

**`ModuleNotFoundError`**
— Make sure Text4GCN is installed: `pip install text4gcn[cpu]` (or `[gpu]`).
