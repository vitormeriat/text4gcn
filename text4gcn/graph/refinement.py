"""
Graph refinement: filter nodes, prune edges, attach metadata.

Refines the doc-word graph to reduce size and noise while preserving
structure useful for classification. Compatible with existing LoadData
and trainer flow.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Set, Dict, Any
import numpy as np
import scipy.sparse as sp

from .metadata import NodeMetadata


@dataclass
class RefinementConfig:
    """
    Configuration for graph refinement.

    All filters are optional; applied in order: node filters then edge pruning.
    """

    # Node filters
    min_degree: Optional[float] = None
    """Drop nodes with total degree < this (after symmetrization)."""
    max_degree: Optional[float] = None
    """Drop nodes with total degree > this (e.g. hub pruning)."""
    min_word_degree: Optional[float] = None
    """For word nodes only: drop if degree < this."""
    max_word_degree: Optional[float] = None
    """For word nodes only: drop if degree > this."""
    min_doc_freq: Optional[int] = None
    """For word nodes only: drop if number of document neighbors < this (min_df)."""
    max_doc_freq_ratio: Optional[float] = None
    """For word nodes only: drop if doc_freq / num_docs > this (max_df ratio)."""
    keep_docs: bool = True
    """If True, always keep all document nodes (train + test). Only filter words."""
    drop_word_indices: Optional[Set[int]] = None
    """Word indices (in vocabulary) to drop, e.g. stopwords."""
    keep_word_indices: Optional[Set[int]] = None
    """If set, only these word indices are kept (overrides other word filters)."""

    # Edge pruning (applied to the subgraph after node filtering)
    edge_weight_threshold: Optional[float] = None
    """Drop edges with weight below this."""
    top_k_edges_per_node: Optional[int] = None
    """Keep only top-k heaviest edges per node (symmetric)."""

    # Metadata and vocab
    vocab_path: Optional[str] = None
    """Path to vocabulary file (one word per line) for word metadata and drop_word_indices by string."""
    stopwords: Optional[Set[str]] = None
    """If set and vocab_path is set, drop word nodes whose word is in this set."""

    # Output
    build_metadata: bool = True
    """Whether to build and return node metadata for the refined graph."""


def _node_type_and_word_id(
    i: int, train_size: int, vocab_size: int
) -> Tuple[str, Optional[int]]:
    if i < train_size:
        return "train_doc", None
    if i < train_size + vocab_size:
        return "word", i - train_size
    return "test_doc", None


def _word_doc_degree(adj: sp.spmatrix, train_size: int, test_size: int) -> np.ndarray:
    """For each node, number of document neighbors (doc-word edges)."""
    n = adj.shape[0]
    vocab_size = n - train_size - test_size
    # Document indices: 0..train_size-1 and (train_size+vocab_size)..(n-1)
    is_doc = np.zeros(n, dtype=bool)
    is_doc[:train_size] = True
    is_doc[train_size + vocab_size:] = True
    doc_columns = sp.csr_matrix(adj)[:, is_doc]
    return np.asarray((doc_columns != 0).sum(axis=1), dtype=float).ravel()


class GraphRefinement:
    """
    Refine a doc-word graph: filter nodes, prune edges, optional metadata.

    Convention: nodes 0..train_size-1 = train docs,
    train_size..train_size+vocab_size-1 = words,
    train_size+vocab_size..N-1 = test docs.
    """

    def __init__(self, config: RefinementConfig):
        self.config = config

    def _vocab_from_path(self, path: Optional[str]) -> Optional[List[str]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return None

    def _compute_keep_mask(
        self,
        adj: sp.spmatrix,
        train_size: int,
        test_size: int,
        vocab: Optional[List[str]],
    ) -> np.ndarray:
        n = adj.shape[0]
        vocab_size = n - train_size - test_size
        num_docs = train_size + test_size
        keep = np.ones(n, dtype=bool)

        try:
            degree = np.array(adj.sum(axis=1)).flatten()
        except Exception:
            degree = np.array(adj.sum(axis=0)).flatten()

        doc_degree_word = None
        if (
            self.config.min_doc_freq is not None
            or self.config.max_doc_freq_ratio is not None
        ):
            doc_degree_word = _word_doc_degree(adj, train_size, test_size)

        for i in range(n):
            node_type, word_id = _node_type_and_word_id(i, train_size, vocab_size)
            deg = degree[i]

            if self.config.min_degree is not None and deg < self.config.min_degree:
                keep[i] = False
                continue
            if self.config.max_degree is not None and deg > self.config.max_degree:
                keep[i] = False
                continue

            if node_type == "word":
                if self.config.min_word_degree is not None and deg < self.config.min_word_degree:
                    keep[i] = False
                    continue
                if self.config.max_word_degree is not None and deg > self.config.max_word_degree:
                    keep[i] = False
                    continue
                if self.config.min_doc_freq is not None and doc_degree_word is not None:
                    if doc_degree_word[i] < self.config.min_doc_freq:
                        keep[i] = False
                        continue
                if self.config.max_doc_freq_ratio is not None and doc_degree_word is not None and num_docs > 0:
                    if doc_degree_word[i] / num_docs > self.config.max_doc_freq_ratio:
                        keep[i] = False
                        continue
                if self.config.keep_word_indices is not None and word_id is not None:
                    if word_id not in self.config.keep_word_indices:
                        keep[i] = False
                        continue
                if self.config.drop_word_indices is not None and word_id is not None:
                    if word_id in self.config.drop_word_indices:
                        keep[i] = False
                        continue
                if self.config.stopwords and vocab and word_id is not None and 0 <= word_id < len(vocab):
                    if vocab[word_id] in self.config.stopwords:
                        keep[i] = False
                        continue
            elif self.config.keep_docs:
                # keep all docs by default; no change to keep[i]
                pass
            # else: could add doc filters here

        return keep

    def _prune_edges(
        self,
        adj: sp.spmatrix,
        weight_threshold: Optional[float],
        top_k: Optional[int],
    ) -> sp.spmatrix:
        """Prune edges: threshold and/or top-k per node. Expects adj to be the subgraph."""
        sub = adj.copy() if sp.issparse(adj) else sp.csr_matrix(adj)

        if weight_threshold is not None:
            sub = sub.multiply(sub >= weight_threshold)

        if top_k is not None and top_k > 0:
            sub = sub.tocsr()
            n = sub.shape[0]
            rows, cols, data = [], [], []
            for i in range(n):
                row = sub.getrow(i)
                if row.nnz == 0:
                    continue
                c = row.indices
                w = row.data
                if len(w) > top_k:
                    idx = np.argsort(w)[-top_k:]
                    c = c[idx]
                    w = w[idx]
                for j, v in zip(c, w):
                    if v != 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(v)
            if rows:
                sub = sp.csr_matrix(
                    (data, (rows, cols)), shape=(n, n)
                )
            sub = sub + sub.T
            sub = sub.multiply(sub > 0)  # keep symmetric, drop duplicates that became 0

        return sub

    def apply(
        self,
        adj: sp.spmatrix,
        features,
        y_train,
        y_val,
        y_test,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        train_size: int,
        test_size: int,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Tuple[Tuple, Optional[List[NodeMetadata]], Dict[str, Any]]:
        """
        Apply refinement and return (predata_tuple, metadata_list, stats).

        Returns
        -------
        predata_tuple : same as load_corpus() (adj, features, y_train, y_val, y_test,
                        train_mask, val_mask, test_mask, train_size, test_size)
        metadata_list : list of NodeMetadata for refined nodes, or None
        stats : dict with keys like 'nodes_removed', 'edges_before', 'edges_after'
        """
        n_orig = adj.shape[0]
        vocab_size_orig = n_orig - train_size - test_size
        vocab = self._vocab_from_path(self.config.vocab_path)
        if vocab is None and dataset_path and dataset_name:
            try:
                path = f"{dataset_path}/{dataset_name}.shuffled/{dataset_name}.vocab"
                vocab = self._vocab_from_path(path)
            except Exception:
                pass

        keep_mask = self._compute_keep_mask(adj, train_size, test_size, vocab)
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) == 0:
            raise ValueError("Refinement removed all nodes.")

        # Check we still have train and test docs
        train_doc_orig = np.arange(train_size)
        test_doc_orig = np.arange(train_size + vocab_size_orig, n_orig)
        kept_train = np.intersect1d(keep_indices, train_doc_orig)
        kept_test = np.intersect1d(keep_indices, test_doc_orig)
        if len(kept_train) == 0:
            raise ValueError("Refinement removed all training document nodes.")
        if len(kept_test) == 0:
            raise ValueError("Refinement removed all test document nodes.")

        # Subgraph: adj — force CSR so fancy indexing works for all scipy formats
        if sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        adj_sub = adj[np.ix_(keep_indices, keep_indices)].copy()
        if not sp.issparse(adj_sub):
            adj_sub = sp.csr_matrix(adj_sub)

        edges_before = int(adj_sub.nnz)
        adj_sub = self._prune_edges(
            adj_sub,
            self.config.edge_weight_threshold,
            self.config.top_k_edges_per_node,
        )
        edges_after = int(adj_sub.nnz)

        # Features and labels: keep rows for keep_indices
        # dia_matrix / coo_matrix are sparse but not reliably subscriptable
        if sp.issparse(features):
            features_sub = sp.csr_matrix(features)[keep_indices, :]
        else:
            features_sub = np.asarray(features)[keep_indices, :]

        y_train_sub = np.asarray(y_train)[keep_indices, :]
        y_val_sub = np.asarray(y_val)[keep_indices, :]
        y_test_sub = np.asarray(y_test)[keep_indices, :]

        # Masks: new mask[i] = original mask at keep_indices[i]
        train_mask_sub = train_mask[keep_indices]
        val_mask_sub = val_mask[keep_indices]
        test_mask_sub = test_mask[keep_indices]

        # train_size and test_size: count of train/test docs in the *original* indexing
        # that are still present. For the trainer, train_size is used for indexing;
        # after refinement the number of nodes is len(keep_indices). The trainer
        # uses train_mask, val_mask, test_mask to know which rows are train/val/test.
        # So we keep train_size and test_size as the *counts* of train and test samples
        # in the refined set (so that metrics make sense).
        train_size_sub = int(train_mask_sub.sum())
        test_size_sub = int(test_mask_sub.sum())

        predata = (
            adj_sub,
            features_sub,
            y_train_sub,
            y_val_sub,
            y_test_sub,
            train_mask_sub,
            val_mask_sub,
            test_mask_sub,
            train_size_sub,
            test_size_sub,
        )

        metadata_list = None
        if self.config.build_metadata:
            # Refined graph node order = order of keep_indices; types come from original indices
            degree_sub = np.array(adj_sub.sum(axis=1)).flatten()
            metadata_list = []
            for k in range(len(keep_indices)):
                orig = int(keep_indices[k])
                node_type, word_id = _node_type_and_word_id(
                    orig, train_size, vocab_size_orig
                )
                word = None
                if vocab and word_id is not None and 0 <= word_id < len(vocab):
                    word = vocab[word_id]
                doc_id = orig if node_type == "train_doc" else (
                    orig - (train_size + vocab_size_orig) if node_type == "test_doc" else None
                )
                metadata_list.append(
                    NodeMetadata(
                        node_type=node_type,
                        original_index=orig,
                        degree=float(degree_sub[k]),
                        word_id=word_id,
                        word=word,
                        doc_id=doc_id,
                    )
                )

        stats = {
            "nodes_original": n_orig,
            "nodes_refined": len(keep_indices),
            "nodes_removed": n_orig - len(keep_indices),
            "edges_before": edges_before,
            "edges_after": edges_after,
            "train_docs_kept": int(train_mask_sub.sum()),
            "test_docs_kept": int(test_mask_sub.sum()),
        }

        return predata, metadata_list, stats


def apply_refinement(
    predata: Tuple,
    config: RefinementConfig,
    dataset_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    return_metadata: bool = False,
) -> Tuple[Tuple, Optional[Dict[str, Any]]]:
    """
    Apply graph refinement to the output of LoadData.load_corpus().

    Parameters
    ----------
    predata : tuple
        (adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
         train_size, test_size) as returned by load_corpus().
    config : RefinementConfig
        Refinement options.
    dataset_path : str, optional
        Dataset path (for loading vocab).
    dataset_name : str, optional
        Dataset name (for loading vocab).
    return_metadata : bool
        If True, return (predata_refined, info_dict) where info_dict contains
        'metadata' and 'stats'. Otherwise return (predata_refined, None).

    Returns
    -------
    predata_refined : tuple
        Same structure as predata, with refined graph.
    info : dict or None
        If return_metadata, dict with 'metadata' (list of NodeMetadata) and 'stats'.
    """
    (adj, features, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, train_size, test_size) = predata

    refiner = GraphRefinement(config)
    predata_new, metadata_list, stats = refiner.apply(
        adj, features, y_train, y_val, y_test,
        train_mask, val_mask, test_mask, train_size, test_size,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
    )

    if return_metadata:
        info = {"metadata": metadata_list, "stats": stats}
        return predata_new, info
    return predata_new, None
