# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0] — unreleased

### Added
- Tutorials: five notebooks in `docs/tutorials/` (text as a graph → first classifier on 20 Newsgroups → comparing graph builders → refining the graph → your own corpus), rendered in the docs with `mkdocs-jupyter`
- `text4gcn.datasets.save_corpus()` writes any labelled corpus in the `.txt` / `.meta` input format, with validation
- `text4gcn.datasets.twenty_newsgroups()` downloads 20 Newsgroups (scikit-learn) in the input format
- `dataset_path` accepts `pathlib.Path` as well as `str` (`TextPipeline`, builders, `GNN`, datasets)
- `GraphBuildStats` returned by every builder method (nodes, nnz, density, file size)
- `TextPipeline(seed=…)` for reproducible corpus shuffle
- Optional `text4gcn[syntactic]` extra (Stanford CoreNLP Python wrapper) and `text4gcn[bert]` extra (transformers, for the experimental BERT builder)
- `scripts/reproduce_dcai2022.py` for DCAI 2022 Tables 2/3 reference + local comparison
- MkDocs + Material theme documentation; CI builds with `mkdocs build --strict`
- Graph refinement (`RefinementConfig`, `apply_refinement`, `NodeMetadata`)
- `GNN.fit()` returns `TrainingResult`; early-stop restores best validation weights

### Changed
- `TrainingResult` is a frozen dataclass and `model_path` is always set; `Semantic.cosine_similarity()` and its builder class share the 0.95 default
- `TextPipeline` downloads the NLTK resources only when they are missing, quietly
- Requires PyTorch ≥ 2.3 (earlier builds cannot run with NumPy 2)
- `transformers` is no longer installed by default: it is only needed by `Semantic.bert()`, via the `bert` extra
- Training seed is deterministic (`GNN(seed=…)` seeds torch / NumPy / Python); no random offset
- Dropout from `GNN(dropout=…)` is passed into the GCN; `model.train()` after each eval epoch
- `Semantic.word2vec(threshold=…)` defaults to 0.7 instead of 0.95: with the 20 epochs now actually applied, 0.95 left almost no word–word edges (79 on 20 Newsgroups, 4 categories)
- RWF word–word weights are the pair count divided by the largest pair count, in (0, 1]; PMI is a gate only. (A Max–Min scaling gave the least frequent pairs weight 0, which deleted their edges.)
- `Layer` is `str, Enum` with **only** `Layer.GCN` (GAT removed from the public API)
- CoreNLP import is lazy; syntactic builders require the optional extra
- Documentation CI uses MkDocs (replacing Sphinx)
- Development tooling: Ruff replaces black / isort / flake8; one CI workflow lints, tests on Python 3.10–3.13 and builds the package; publishing to PyPI checks that the release tag matches the package version and runs the tests first
- Corpus is read as UTF-8 by default (was Latin-1); all intermediate files are written/read as UTF-8 regardless of OS locale
- `datasets.download(name, path)` replaces `data.R8()` / `R52()` / `AG_NEWS()`: R8, R52, Ohsumed and PubMed from the `datasets-v1` GitHub release, with SHA-256 verification (the old Azure storage is gone)

### Migrating from 1.x
- Requires Python ≥ 3.10 (1.x targeted Python 3.8)
- No `argparse` is involved anymore: `GNN` works in notebooks/Colab (fixes #3)
- `GNN(dataset=…, path=…, epoches=…)` → `GNN(dataset_name=…, dataset_path=…, epochs=…)`
- `builder` is optional (default `Builder.Tfidf`) (fixes #2); `Builder.Embedding` / `Builder.Frequency` → `Builder.Word2Vec` / `Builder.Tfidf`
- Import from the package root: `from text4gcn import GNN, TextPipeline, Builder, Layer`
- `Builder` values changed (`'frequency'` → `'frequency.tfidf'`, `'embedding'` → `'embedding.word2vec'`, `'cosine'` → `'cosine.similarity'`, `'dependency'` → `'dependency.parsing'`), and so did the adjacency file names: graphs built with 1.x are not found; build them again
- Graph files are no longer interchangeable with 1.x even for the same builder: Word2Vec, LIWC, cosine-similarity and RWF weights changed (see *Fixed*)

### Removed
- `output_path` of `tfidf()`, `word2vec()` and `cosine_similarity()`, and `TextPipeline.execute(use_predefined_word_vectors=…)`: they were accepted but ignored
- `GNN(val_ratio=…)`: it never changed the split; set it with `TextPipeline.execute(validation_ratio=…)`
- `Syntactic.constituency_parsing()`, `Semantic.llm()` and `Builder.ConstituencyParsing`, which did nothing
- `builder.list_available_adjacency_methods()` / `list_available_builders()` and the string constants `builder.Tfidf`, `builder.Word2Vec`, …: use the `Builder` enum. `text4gcn.builder.Builder` is now the same enum as `text4gcn.Builder` (they were two distinct classes); `rich` is no longer a dependency
- Unused internal modules (`preprocess.DataShuffler`, `NodeFeatureBuilder`, `WordProcessor`, `modules.utils`, and empty or unreachable files); the `TextPipeline` covers what they did
- The empty `cpu` and `gpu` extras: `pip install text4gcn` installs PyTorch from PyPI; for another build install PyTorch first from pytorch.org
- `utils.recommend_installation()`, and the `torch_available` key of `check_pytorch_installation()` (PyTorch is a required dependency)
- `data.AG_NEWS()` (no longer hosted)
- Unused `models.utils.get_train_test()` / `read_file()`
- The unfinished baseline models (`models/baseline`: TF-IDF, LSTM)

### Fixed
- After graph refinement, the train / test block sizes counted only the training mask, so validation documents were reported as words
- A document left empty by cleaning produced NaN node features (and a `RuntimeWarning`); it now gets a zero vector
- The training log's "Data statistics" table labelled the node count as "Edges"
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

## [1.0.0] — 2022-08-14

### Added
- Initial public release
- `TextPipeline` for text preprocessing (cleaning, shuffling, node feature extraction)
- `Frequency`, `Semantic`, `Syntactic` builder classes for adjacency construction
- `GNN` for GCN training and evaluation
- Support for R8, R52, 20NG, MR, Ohsumed benchmark datasets
- TF-IDF, Word2Vec, BERT, cosine similarity, dependency parsing, LIWC adjacency strategies
