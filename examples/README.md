# Text4GCN Examples

Runnable guides for the published API. Prefer an editable install:

```bash
pip install -e ".[cpu]"
```

---

## Recommended entry points

| Resource | What it shows |
|---|---|
| [`../notebooks/text4gcn.ipynb`](../notebooks/text4gcn.ipynb) | Benchmark walkthrough (R8): download → pipeline → TF-IDF → GNN |
| [`synthetic_data_demo.ipynb`](synthetic_data_demo.ipynb) | Offline synthetic notebook (full short train) |
| [`end_to_end_synthetic.py`](end_to_end_synthetic.py) | Canonical CLI smoke: generate → pipeline → TF-IDF → train |
| [`cons.py`](cons.py) | Toggleable full-pipeline reference (`TEXT4GCN_RUN=…`) |

```bash
# Offline end-to-end (no R8 download)
python examples/end_to_end_synthetic.py --docs 400 --epochs 40

# Optional steps on real data already under examples/
TEXT4GCN_RUN=pipe,tfidf,train TEXT4GCN_PATH=examples python examples/cons.py
```

---

## Quick start (API)

```python
from text4gcn import GNN, TextPipeline, Builder
from text4gcn.builder import Frequency

TextPipeline(
    dataset_name="R8",
    dataset_path="data/",
    rare_count=5,
    language="english",
    seed=1234,
).execute(validation_ratio=0.10)

stats = Frequency(dataset_name="R8", dataset_path="data/").tfidf(window_size=20)
print(stats)  # GraphBuildStats

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    log_dir="logs/",
    builder=Builder.Tfidf,
    seed=1234,
    epochs=200,
).fit()

print(result.test_accuracy)
```

Notes:

- Validation split is set in `TextPipeline.execute(validation_ratio=…)`. `GNN(val_ratio=…)` does not rebuild masks.
- Learning is **transductive** (test docs are in the graph at build time).

---

## Graph refinement

```python
from text4gcn import GNN, Builder, RefinementConfig

result = GNN(
    dataset_name="R8",
    dataset_path="data/",
    builder=Builder.Tfidf,
    refinement_config=RefinementConfig(
        min_doc_freq=2,
        min_word_degree=1.0,
        edge_weight_threshold=0.1,
        keep_docs=True,
    ),
).fit()
```

Script: `python examples/graph_refinement_example.py` (requires preprocessed + TF-IDF data).

---

## Choosing an adjacency builder

| Method | `Builder` enum |
|---|---|
| `Frequency.tfidf()` | `Builder.Tfidf` |
| `Semantic.word2vec()` | `Builder.Word2Vec` |
| `Semantic.cosine_similarity()` | `Builder.CosineSimilarity` |
| `Syntactic.dependency_parsing()` | `Builder.DependencyParsing` (`pip install text4gcn[syntactic]`) |
| `Syntactic.linguistic_inquiry()` | `Builder.Liwc` |

---

## Files overview

| File | Description |
|---|---|
| `end_to_end_synthetic.py` | Full synthetic tutorial CLI |
| `synthetic_data_generator.py` | Write `{name}.txt` / `{name}.meta` under an output dir |
| `run_synthetic_test.py` | Wrapper around pipeline synthetic test |
| `test_textpipeline_synthetic.py` | Detailed pipeline checks |
| `simple_synthetic_test.py` | Minimal pipeline smoke |
| `graph_refinement_example.py` | Refinement before / with `GNN` |
| `device_management.py` | CPU/GPU selection |
| `demo_python_versions.py` | Import smoke for Python 3.10+ |
| `cons.py` | Flag / env-driven reference pipeline |
| `synthetic_data_demo.ipynb` | Notebook twin of the synthetic flow |

---

## Troubleshooting

**`FileNotFoundError: Node feature files not found`** — run `TextPipeline.execute()` first.

**`FileNotFoundError: Adjacency files not found`** — run `Frequency(...).tfidf()` (or another builder).

**Import hangs / HuggingFace errors on `import text4gcn`** — BERT weights load only when `Semantic.bert()` is called; upgrade to a build with lazy BERT import.

**`ModuleNotFoundError`** — `pip install -e ".[cpu]"` from the package root.
