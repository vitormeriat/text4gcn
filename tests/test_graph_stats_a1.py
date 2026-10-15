"""Tests for A1 — graph stats API and DCAI reproduction helpers."""

from __future__ import annotations

import pickle
import runpy
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from text4gcn.graph import (
    GraphBuildStats,
    compute_adjacency_stats,
    load_adjacency_stats,
)


class TestGraphBuildStats:
    def test_compute_adjacency_stats(self, tmp_path: Path):
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        path = tmp_path / "toy.adj"
        with path.open("wb") as f:
            pickle.dump(adj, f)

        stats = compute_adjacency_stats(
            adj, path=path, builder="frequency.tfidf", dataset_name="toy"
        )
        assert isinstance(stats, GraphBuildStats)
        assert stats.n_nodes == 3
        assert stats.n_edges == 4
        assert stats.file_bytes == path.stat().st_size
        assert "frequency.tfidf" in str(stats)

        loaded = load_adjacency_stats(path, builder="frequency.tfidf", dataset_name="toy")
        assert loaded.n_edges == stats.n_edges


class TestReproduceScript:
    def test_reference_only_exits_zero(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_dcai2022.py"
        ns = runpy.run_path(str(script))
        assert ns["order_of_magnitude_ok"](3_000_000, 3_165_430)
        assert not ns["order_of_magnitude_ok"](1_000, 3_165_430)
        assert ns["TABLE2_EDGES"]["R8"]["tgcn"] > ns["TABLE2_EDGES"]["R8"]["semantic"]
        assert ns["main"](["--reference-only"]) == 0


def test_vectorised_cosine_word_edges_match_pairwise_loop(tmp_path):
    """The chunked implementation keeps the edges and weights of the original pairwise loop."""
    import itertools
    import pickle

    import numpy as np
    from scipy.spatial.distance import cosine

    from text4gcn.modules.adjacency import extract_cosine_similarity_word_weights

    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(40)]
    vectors = {w: rng.normal(size=8) for w in vocab[:-2]}
    vectors[vocab[0]] = vectors[vocab[1]] * 3            # identical direction
    vectors[vocab[2]] = np.zeros(8)                       # no direction: no edges
    path = tmp_path / "wv.pkl"                            # vocab[-2:] have no vector
    path.write_bytes(pickle.dumps(vectors))

    expected = []
    for (i, wi), (j, wj) in itertools.product(enumerate(vocab), enumerate(vocab)):
        if wi in vectors and wj in vectors and np.any(vectors[wi]) and np.any(vectors[wj]):
            sim = 1.0 - cosine(vectors[wi], vectors[wj])
            if sim > 0.3:
                expected.append((5 + i, 5 + j, sim))

    rows, cols, weights = extract_cosine_similarity_word_weights(
        vocab, train_size=5, word_vec_path=str(path), threshold=0.3, chunk_size=7)

    assert list(zip(rows, cols)) == [(r, c) for r, c, _ in expected]
    np.testing.assert_allclose(weights, [w for _, _, w in expected])
    assert (5, 6) in zip(rows, cols)
