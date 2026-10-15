# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Tutorials: five notebooks in `docs/tutorials/` (text as a graph → first classifier on 20 Newsgroups → comparing graph builders → refining the graph → your own corpus), rendered in the docs with `mkdocs-jupyter`
- `text4gcn.datasets.save_corpus()` writes any labelled corpus in the `.txt` / `.meta` input format, with validation
- `text4gcn.datasets.twenty_newsgroups()` downloads 20 Newsgroups (scikit-learn) in the input format
- `dataset_path` accepts `pathlib.Path` as well as `str` (`TextPipeline`, builders, `GNN`, datasets)
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
- Corpus is read as UTF-8 by default (was Latin-1); all intermediate files are written/read as UTF-8 regardless of OS locale
- `datasets.download(name, path)` replaces `data.R8()` / `R52()` / `AG_NEWS()`: R8, R52, Ohsumed and PubMed from the `datasets-v1` GitHub release, with SHA-256 verification (the old Azure storage is gone)

### Migrating from 1.x
- No `argparse` is involved anymore: `GNN` works in notebooks/Colab (fixes #3)
- `GNN(dataset=…, path=…, epoches=…)` → `GNN(dataset_name=…, dataset_path=…, epochs=…)`
- `builder` is optional (default `Builder.Tfidf`) (fixes #2); `Builder.Embedding` / `Builder.Frequency` → `Builder.Word2Vec` / `Builder.Tfidf`
- Import from the package root: `from text4gcn import GNN, TextPipeline, Builder, Layer`

### Deprecated
- `GNN(val_ratio=…)` does **not** rebuild train/val masks — use `TextPipeline.execute(validation_ratio=…)`

### Removed
- `data.AG_NEWS()` (no longer hosted)
- Unused `models.utils.get_train_test()` / `read_file()`
- Dead packages (`di/`, `optimization/`, `services/`) and orphan trainers
- AsyncProcessor dead path

### Fixed
- Ohsumed's `training` split (and `20news-bydate-train`) was dropped by the shuffle step, which only accepted values ending in `train`; split handling now follows the documented rule (`train`, `training`, `20news-bydate-train` → train, anything else → test)
- `RefinementConfig(min_doc_freq=…)` / `max_doc_freq_ratio` counted 0 documents for every word on CSR graphs, which removed all words (chance-level accuracy)
- `Semantic.cosine_similarity()` compares word vectors in vectorised chunks (seconds instead of hours on a 10k-word vocabulary) and returns `GraphBuildStats` like the other builders
- Training on CUDA: adjacency, masks and predictions now follow the selected device
- Non-ASCII corpora: `TextPipeline(encoding=…, ascii_only=False)` keeps words of any script, e.g. Arabic with diacritics (fixes #1)
- Dataset downloads no longer save HTTP error pages as data
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
