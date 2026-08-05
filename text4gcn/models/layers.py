from enum import Enum


class Layer(str, Enum):
    """Supported GNN layer types for :class:`~text4gcn.models.gnn.GNN`.

    Only ``GCN`` is implemented (DCAI 2022 / Yao et al. protocol). GAT was
    previously advertised but never wired into ``GNN.fit``; it will be
    reintroduced only when a real implementation and experiments exist.
    """

    GCN = "GCN"
