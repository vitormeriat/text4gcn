"""
Graph refinement and metadata for TextGCN.

This package provides:
- Node filtering (by degree, type, vocabulary)
- Edge pruning (by weight, top-k)
- Node metadata (type, degree, word/doc id) for analysis and LLM-guided weighting
"""

from .refinement import GraphRefinement, RefinementConfig, apply_refinement
from .metadata import NodeMetadata, build_node_metadata

__all__ = [
    "GraphRefinement",
    "RefinementConfig",
    "apply_refinement",
    "NodeMetadata",
    "build_node_metadata",
]
