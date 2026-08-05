# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `GraphBuildStats` returned by Frequency / Word2Vec builders (nodes, nnz, density, file size)
- `TextPipeline(seed=…)` for reproducible corpus shuffle
- Optional `text4gcn[syntactic]` extra (Stanford CoreNLP Python wrapper)
- `scripts/reproduce_dcai2022.py` for DCAI 2022 Tables 2/3 reference + local comparison
- MkDocs + Material theme documentation; CI builds with `mkdocs build --strict`
- Graph refinement (`RefinementConfig`, `apply_refinement`, `NodeMetadata`)
- `GNN.fit()` returns `TrainingResult`; early-stop restores best validation weights

### Changed
- Training seed is deterministic (`GNN(seed=…)` seeds torch / NumPy / Python); no random offset
- Dropout from `GNN(dropout=…)` is passed into the GCN; `model.train()` after each eval epoch
- RWF word–word weights use Max–Min normalisation (DCAI Eq. 6); PMI is a gate only
- `Layer` is `str, Enum` with **only** `Layer.GCN` (GAT removed from the public API)
- CoreNLP import is lazy; syntactic builders require the optional extra
- Documentation CI uses MkDocs (replacing Sphinx)

### Deprecated
- `GNN(val_ratio=…)` does **not** rebuild train/val masks — use `TextPipeline.execute(validation_ratio=…)`

### Removed
- Dead packages (`di/`, `optimization/`, `services/`) and orphan trainers
- AsyncProcessor dead path

### Fixed
- `apply_refinement` CSR conversion before fancy indexing (`dia_matrix` safe)
- Early path validation when `dataset_path` / `dataset_name` are missing or empty
- `np.bool` → `bool` (NumPy ≥ 1.24)

---

## [1.0.0] — 2023-xx-xx

### Added
- Initial public release
- `TextPipeline` for text preprocessing (cleaning, shuffling, node feature extraction)
- `Frequency`, `Semantic`, `Syntactic` builder classes for adjacency construction
- `GNN` for GCN training and evaluation
- Support for R8, R52, 20NG, MR, Ohsumed benchmark datasets
- TF-IDF, Word2Vec, BERT, cosine similarity, dependency parsing, LIWC adjacency strategies
