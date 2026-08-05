#!/usr/bin/env python3
"""
Full-pipeline reference script for Text4GCN.

Toggle the flags below to run preprocessing, adjacency builders, and training.
Defaults keep everything off except listing available datasets/builders so the
script is safe to import / run without downloading models.

Usage::

    # From repo root (editable install recommended)
    python examples/cons.py

    # Enable steps by editing the flags, or::
    TEXT4GCN_RUN=pipe,tfidf,train python examples/cons.py
"""

from __future__ import annotations

import os
from pathlib import Path

from text4gcn import GNN, TextPipeline, Builder, Layer
from text4gcn.builder import Frequency, Semantic, Syntactic, list_available_builders
from text4gcn.datasets import data

# ---------------------------------------------------------------------------
# Flags — set True or pass TEXT4GCN_RUN=pipe,tfidf,train
# ---------------------------------------------------------------------------
_RUN = {x.strip().lower() for x in os.environ.get("TEXT4GCN_RUN", "").split(",") if x.strip()}

PIPE = "pipe" in _RUN
TFIDF = "tfidf" in _RUN
WORD2VEC = "word2vec" in _RUN
COSIM = "cosim" in _RUN
BERT = "bert" in _RUN
LIWC = "liwc" in _RUN
DEP = "dep" in _RUN
TRAIN = "train" in _RUN

DATASET = os.environ.get("TEXT4GCN_DATASET", "R8")
PATH = os.environ.get("TEXT4GCN_PATH", str(Path(__file__).resolve().parent))
LOG_DIR = os.environ.get("TEXT4GCN_LOG_DIR", str(Path(PATH) / "log"))
CORE_NLP = os.environ.get("TEXT4GCN_CORENLP", "")
LIWC_PATH = os.environ.get("TEXT4GCN_LIWC", "")


def main() -> None:
    print(f"dataset={DATASET} path={PATH}")
    data.list()
    list_available_builders()

    frequency = Frequency(dataset_name=DATASET, dataset_path=PATH)
    syntactic = Syntactic(dataset_name=DATASET, dataset_path=PATH)
    semantic = Semantic(dataset_name=DATASET, dataset_path=PATH)

    if PIPE:
        if not (Path(PATH) / f"{DATASET}.txt").exists():
            getattr(data, DATASET)(path=PATH)
        TextPipeline(
            dataset_name=DATASET,
            dataset_path=PATH,
            rare_count=5,
            language="english",
            seed=1234,
        ).execute(validation_ratio=0.10)

    if TFIDF:
        print(frequency.tfidf(window_size=20))

    if WORD2VEC:
        print(
            semantic.word2vec(
                training_algorithm=1,
                embedding_dimension=300,
                num_epochs=20,
                threshold=0.95,
                language="english",
            )
        )

    if COSIM:
        print(semantic.cosine_similarity(threshold=0.9))

    if BERT:
        print(semantic.bert(language="english", threshold=0.95))

    if LIWC:
        if not LIWC_PATH:
            raise SystemExit("Set TEXT4GCN_LIWC to a LIWC dictionary path")
        print(syntactic.linguistic_inquiry(liwc_path=LIWC_PATH))

    if DEP:
        if not CORE_NLP:
            raise SystemExit("Set TEXT4GCN_CORENLP to a Stanford CoreNLP directory")
        print(syntactic.dependency_parsing(core_nlp_path=CORE_NLP, language="english"))

    if TRAIN:
        result = GNN(
            dataset_name=DATASET,
            dataset_path=PATH,
            log_dir=LOG_DIR,
            builder=Builder.Tfidf,
            layer=Layer.GCN,
            seed=1234,
            epochs=200,
            dropout=0.5,
            early_stopping=10,
            lr=0.02,
            nhid=200,
        ).fit()
        print(result)

    if not any([PIPE, TFIDF, WORD2VEC, COSIM, BERT, LIWC, DEP, TRAIN]):
        print(
            "\nNo steps enabled. Example:\n"
            "  TEXT4GCN_RUN=pipe,tfidf,train TEXT4GCN_PATH=examples python examples/cons.py"
        )


if __name__ == "__main__":
    main()
