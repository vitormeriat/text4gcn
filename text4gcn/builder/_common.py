"""Shared steps of the graph builders: saving the graph and reporting it."""

from __future__ import annotations

import pickle

import scipy.sparse as sp

from text4gcn.graph.stats import GraphBuildStats, compute_adjacency_stats
from text4gcn.utils.reporting import Reporter


def log_path(dataset_path, dataset_name: str) -> str:
    """The dataset's processing log, shared by the pipeline and every builder."""
    return f"{dataset_path}/log/{dataset_name}_dataset.txt"


def save_graph(adj: sp.spmatrix, path: str, builder: str, dataset_name: str,
               word_word_edges: int | None) -> GraphBuildStats:
    """Pickle the adjacency matrix and return its statistics."""
    with open(path, "wb") as f:
        pickle.dump(adj, f)
    return compute_adjacency_stats(adj, path=path, builder=builder, dataset_name=dataset_name,
                                   word_word_edges=word_word_edges)


def run_build(build, header: str, verbose: int, dataset_path, dataset_name: str) -> GraphBuildStats:
    """Run ``build(report)`` under a header and one summary line, then log it."""
    report = Reporter(verbose)
    report.header(header)
    with report.step("graph") as step:
        stats = build(report)
        step.summary = stats.summary()
    report.save(log_path(dataset_path, dataset_name))
    return stats


def build_rwf_graph(dataset_path, dataset_name: str, relation_pairs, builder: str) -> GraphBuildStats:
    """Build a Related Word Filter graph (DCAI 2022).

    ``relation_pairs(docs_of_words)`` returns the RWF pair counts ``{"wi,wj": count}``;
    the PMI gate, the edge weights and the TF-IDF document–word edges are the same
    for every RWF builder. The graph is saved as ``ind.{builder}.{dataset_name}.adj``.
    """
    from pathlib import Path

    from scipy.sparse import csr_matrix

    from text4gcn.modules import adjacency as adjcy

    corpus_path = f"{dataset_path}/{dataset_name}"
    Path(f"{corpus_path}.adjacency").mkdir(parents=True, exist_ok=True)
    with open(f"{corpus_path}.shuffled/{dataset_name}.txt", encoding="utf-8") as f:
        docs_of_words = [line.split() for line in f]
    with open(f"{corpus_path}.shuffled/{dataset_name}.vocab", encoding="utf-8") as f:
        vocab = f.read().splitlines()
    with open(f"{corpus_path}.shuffled/{dataset_name}.train", encoding="utf-8") as f:
        train_size = len(f.readlines())
    with open(f"{corpus_path}.shuffled/{dataset_name}.test", encoding="utf-8") as f:
        test_size = len(f.readlines())
    word_id_map = {word: i for i, word in enumerate(vocab)}

    rela_pair_count_str = relation_pairs(docs_of_words)

    row, col = [], []
    weight = adjcy.compute_weights_with_PMI(docs_of_words, rela_pair_count_str,
                                            adjcy.build_word_window_freq(docs_of_words),
                                            train_size, word_id_map, row, col)
    word_word_edges = len(weight)
    weight += adjcy.get_weight_tfidf(docs_of_words, word_id_map,
                                     adjcy.get_doc_word_freq(docs_of_words, word_id_map),
                                     train_size, len(vocab), adjcy.define_word_doc_freq(docs_of_words),
                                     vocab, row, col)

    n_nodes = train_size + len(vocab) + test_size
    adj = csr_matrix((weight, (row, col)), shape=(n_nodes, n_nodes))
    return save_graph(adj, f"{corpus_path}.adjacency/ind.{builder}.{dataset_name}.adj", builder,
                      dataset_name, word_word_edges)
