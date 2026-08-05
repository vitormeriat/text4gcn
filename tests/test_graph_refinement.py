"""Tests for graph refinement (node filtering, edge pruning, metadata)."""

import pytest
import numpy as np
import scipy.sparse as sp

from text4gcn.graph import (
    GraphRefinement,
    RefinementConfig,
    apply_refinement,
    NodeMetadata,
    build_node_metadata,
)


def _make_toy_graph():
    """Small doc-word graph: 2 train docs, 3 words, 1 test doc. Nodes 0,1 = train; 2,3,4 = words; 5 = test."""
    train_size, vocab_size, test_size = 2, 3, 1
    n = train_size + vocab_size + test_size
    row = [0, 0, 1, 1, 2, 3, 4, 5, 5]  # doc0-word2, doc0-word3, doc1-word2, doc1-word3, word-word, test-word
    col = [2, 3, 2, 3, 3, 4, 3, 2, 4]
    data = [1.0, 0.5, 1.0, 0.5, 0.2, 0.2, 0.2, 1.0, 0.5]
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    adj = adj + adj.T
    adj = adj.multiply(adj > 0)
    features = sp.identity(n)
    y = np.eye(2)  # 2 classes
    y_train = np.zeros((n, 2))
    y_train[0, 0] = 1
    y_train[1, 1] = 1
    y_val = np.zeros((n, 2))
    y_test = np.zeros((n, 2))
    y_test[5, 0] = 1
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0] = True
    val_mask = np.zeros(n, dtype=bool)
    val_mask[1] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[5] = True
    return (
        adj, features, y_train, np.zeros_like(y_val), y_test,
        train_mask, val_mask, test_mask,
        train_size, test_size,
    )


class TestGraphRefinement:
    """Test GraphRefinement and apply_refinement."""

    def test_apply_refinement_no_op(self):
        """Refinement with no filters should return same sizes (or allow all nodes)."""
        predata = _make_toy_graph()
        config = RefinementConfig(
            min_degree=None,
            build_metadata=False,
        )
        refiner = GraphRefinement(config)
        predata_new, meta, stats = refiner.apply(
            *predata,
            dataset_path=None,
            dataset_name=None,
        )
        assert predata_new[0].shape[0] == predata[0].shape[0]
        assert stats["nodes_original"] == stats["nodes_refined"]

    def test_apply_refinement_min_degree(self):
        """Filtering by min_degree should remove low-degree nodes."""
        predata = _make_toy_graph()
        config = RefinementConfig(
            min_degree=0.5,
            keep_docs=True,
            build_metadata=False,
        )
        refiner = GraphRefinement(config)
        predata_new, _, stats = refiner.apply(
            *predata,
            dataset_path=None,
            dataset_name=None,
        )
        assert predata_new[0].shape[0] <= predata[0].shape[0]
        assert stats["nodes_removed"] >= 0

    def test_apply_refinement_returns_tuple(self):
        """Refined predata must be a tuple of 10 elements."""
        predata = _make_toy_graph()
        config = RefinementConfig(build_metadata=False)
        out, _ = apply_refinement(predata, config, return_metadata=False)
        assert len(out) == 10
        adj, features, y_train, y_val, y_test, tm, vm, tsm, tr_sz, te_sz = out
        assert adj.shape[0] == features.shape[0]
        assert y_train.shape[0] == adj.shape[0]
        assert tm.shape[0] == adj.shape[0]

    def test_apply_refinement_with_metadata(self):
        """When return_metadata=True, info contains 'metadata' and 'stats'."""
        predata = _make_toy_graph()
        config = RefinementConfig(build_metadata=True)
        out, info = apply_refinement(
            predata, config,
            dataset_path=None, dataset_name=None,
            return_metadata=True,
        )
        assert info is not None
        assert "stats" in info
        assert "metadata" in info
        assert len(info["metadata"]) == out[0].shape[0]

    def test_apply_refinement_with_dia_matrix_features(self):
        """Identity features as dia_matrix must not raise (plan item #1)."""
        predata = list(_make_toy_graph())
        n = predata[0].shape[0]
        predata[1] = sp.identity(n, format="dia")
        config = RefinementConfig(min_degree=None, build_metadata=False)
        predata_new, _, stats = GraphRefinement(config).apply(
            *predata,
            dataset_path=None,
            dataset_name=None,
        )
        assert predata_new[0].shape[0] == n
        assert stats["nodes_original"] == stats["nodes_refined"]


class TestNodeMetadata:
    """Test NodeMetadata and build_node_metadata."""

    def test_build_node_metadata(self):
        """Metadata list length equals number of nodes."""
        predata = _make_toy_graph()
        adj = predata[0]
        train_size, test_size = predata[8], predata[9]
        meta = build_node_metadata(adj, train_size, test_size, vocab=None)
        assert len(meta) == adj.shape[0]
        assert meta[0].node_type == "train_doc"
        assert meta[2].node_type == "word"
        assert meta[2].word_id == 0
        assert meta[5].node_type == "test_doc"

    def test_node_metadata_to_dict(self):
        """NodeMetadata.to_dict() returns a dict."""
        m = NodeMetadata(
            node_type="word",
            original_index=2,
            degree=1.5,
            word_id=0,
            word="test",
            doc_id=None,
        )
        d = m.to_dict()
        assert d["node_type"] == "word"
        assert d["word"] == "test"
        assert d["degree"] == 1.5
