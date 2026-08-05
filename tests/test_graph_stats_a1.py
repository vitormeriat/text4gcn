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
