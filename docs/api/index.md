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
    TrainingResult,
    apply_refinement,
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
| [`GraphBuildStats`](builders.md#graphbuildstats) | `text4gcn` | Edge/memory stats from adjacency builders |
| [`Frequency`](builders.md#frequency) | `text4gcn.builder` | TF-IDF / PMI adjacency |
| [`Semantic`](builders.md#semantic) | `text4gcn.builder` | Word2Vec / BERT / cosine adjacency |
| [`Syntactic`](builders.md#syntactic) | `text4gcn.builder` | Dependency parsing / LIWC adjacency |
| [`RefinementConfig`](refinement.md#refinementconfig) | `text4gcn` | Graph refinement parameters |
| [`apply_refinement`](refinement.md#apply_refinement) | `text4gcn` | Apply refinement to loaded corpus |
| [`save_corpus`](datasets.md#save_corpus) | `text4gcn.datasets` | Write your own corpus in the input format |
| [`twenty_newsgroups`](datasets.md#twenty_newsgroups) | `text4gcn.datasets` | Download 20 Newsgroups in the input format |
| [`download`](datasets.md#download) | `text4gcn.datasets` | Download R8, R52, Ohsumed or PubMed |
