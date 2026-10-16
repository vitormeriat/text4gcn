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
- Optional `text4gcn[syntactic]` extra (Stanford CoreNLP Python wrapper) and `text4gcn[bert]` extra (transformers, for the experimental BERT builder)
- `scripts/reproduce_dcai2022.py` for DCAI 2022 Tables 2/3 reference + local comparison
- MkDocs + Material theme documentation; CI builds with `mkdocs build --strict`
- Graph refinement (`RefinementConfig`, `apply_refinement`, `NodeMetadata`)
- `GNN.fit()` returns `TrainingResult`; early-stop restores best validation weights

### Changed
- Requires PyTorch ≥ 2.3 (earlier builds cannot run with NumPy 2)
- `transformers` is no longer installed by default: it is only needed by `Semantic.bert()`, via the `bert` extra
- Training seed is deterministic (`GNN(seed=…)` seeds torch / NumPy / Python); no random offset
- Dropout from `GNN(dropout=…)` is passed into the GCN; `model.train()` after each eval epoch
- `Semantic.word2vec(threshold=…)` defaults to 0.7 instead of 0.95: with the 20 epochs now actually applied, 0.95 left almost no word–word edges (79 on 20 Newsgroups, 4 categories)
- RWF word–word weights are the pair count divided by the largest pair count, in (0, 1]; PMI is a gate only. (A Max–Min scaling gave the least frequent pairs weight 0, which deleted their edges.)
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
- The empty `cpu` and `gpu` extras: `pip install text4gcn` installs PyTorch from PyPI; for another build install PyTorch first from pytorch.org
- `utils.recommend_installation()`, and the `torch_available` key of `check_pytorch_installation()` (PyTorch is a required dependency)
- `data.AG_NEWS()` (no longer hosted)
- Unused `models.utils.get_train_test()` / `read_file()`
- Dead packages (`di/`, `optimization/`, `services/`) and orphan trainers
- AsyncProcessor dead path

### Fixed
- `Semantic.word2vec(num_epochs=…)` was never passed to gensim, which always trained 5 epochs. Training is now also reproducible: `seed` (default 1234) and `workers` (default 1) are new parameters, and vector initialisation no longer depends on `PYTHONHASHSEED`. The code behind the DCAI 2022 paper had the same bug, so its SEMANTIC results were obtained with 5 epochs
- `Syntactic.linguistic_inquiry()` looked words up in the LIWC dictionary letter by letter, so real words never matched; it now connects adjacent words that share at least one LIWC category. The code behind the DCAI 2022 paper had the same bug, so its LINGUISTIC results do not reflect LIWC categories
- `Semantic.cosine_similarity()`: words without a WordNet definition all received the same placeholder vector and formed a clique (99% of the word–word edges on 20 Newsgroups); they now get no word–word edges
- `RefinementConfig(keep_docs=True)` did not protect documents from `min_degree` / `max_degree`
- `RefinementConfig(keep_word_indices=…)` now overrides the other word filters, as documented
- `RefinementConfig(top_k_edges_per_node=…)` doubled the weight of edges kept by both endpoints
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
