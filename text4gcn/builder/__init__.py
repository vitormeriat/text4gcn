"""
Builder module for Text4GCN.

Classes that build the word-document adjacency matrix with frequency-based,
semantic or syntactic word-word edges. ``Builder`` selects which saved graph
``GNN`` trains on.
"""

from text4gcn.models.builder import Builder

from .pipeline import Frequency, Semantic, Syntactic

__all__ = ["Frequency", "Semantic", "Syntactic", "Builder"]
