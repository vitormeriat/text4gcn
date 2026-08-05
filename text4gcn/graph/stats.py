"""Graph construction statistics for memory / edge-count reporting (DCAI 2022 Tab. 2)."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import scipy.sparse as sp


@dataclass(frozen=True)
class GraphBuildStats:
    """Summary of a built word–document adjacency matrix.

    Attributes
    ----------
    n_nodes :
        Number of nodes (train docs + vocab + test docs).
    n_edges :
        Number of stored non-zeros (``nnz``). For unsymmetric construction
        this is the raw edge list size; after symmetrisation at load time the
        trainer may count more.
    density :
        ``nnz / n_nodes²``.
    file_bytes :
        Size of the pickled adjacency file on disk, if known.
    builder :
        Builder label (e.g. ``frequency.tfidf``, ``embedding.word2vec``).
    dataset_name :
        Dataset name used when building.
    path :
        Path to the adjacency file, if known.
    """

    n_nodes: int
    n_edges: int
    density: float
    file_bytes: Optional[int] = None
    builder: Optional[str] = None
    dataset_name: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        size = (
            f", file={self.file_bytes / (1024 * 1024):.2f} MiB"
            if self.file_bytes is not None
            else ""
        )
        label = self.builder or "adjacency"
        return (
            f"GraphBuildStats({label}: nodes={self.n_nodes}, "
            f"edges(nnz)={self.n_edges}, density={self.density:.6e}{size})"
        )


def compute_adjacency_stats(
    adj: sp.spmatrix,
    *,
    path: Optional[Union[str, Path]] = None,
    builder: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> GraphBuildStats:
    """Compute :class:`GraphBuildStats` from a sparse adjacency matrix."""
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = sp.csr_matrix(adj)

    n_nodes = int(adj.shape[0])
    n_edges = int(adj.nnz)
    density = float(n_edges) / float(n_nodes * n_nodes) if n_nodes else 0.0
    file_bytes = None
    path_str = None
    if path is not None:
        path_obj = Path(path)
        path_str = str(path_obj)
        if path_obj.is_file():
            file_bytes = path_obj.stat().st_size

    return GraphBuildStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        density=density,
        file_bytes=file_bytes,
        builder=builder,
        dataset_name=dataset_name,
        path=path_str,
    )


def load_adjacency_stats(
    path: Union[str, Path],
    *,
    builder: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> GraphBuildStats:
    """Load a pickled adjacency matrix and return :class:`GraphBuildStats`."""
    path_obj = Path(path)
    with path_obj.open("rb") as f:
        adj = pickle.load(f)
    return compute_adjacency_stats(
        adj,
        path=path_obj,
        builder=builder,
        dataset_name=dataset_name,
    )


def report_stats(stats: GraphBuildStats, logger: Any = None) -> GraphBuildStats:
    """Print or log graph stats (claim-facing #edges / memory)."""
    message = str(stats)
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)
    else:
        print(message)
    return stats
