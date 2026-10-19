"""
Text4GCN: A Python framework for text classification using Graph Neural Networks.

Quick start::

    from text4gcn import GNN, TextPipeline, Builder

    # Step 1 — preprocess raw text
    TextPipeline(dataset_name="R8", dataset_path="data/").execute()

    # Step 2 — build word-document graph
    from text4gcn.builder import Frequency
    Frequency(dataset_name="R8", dataset_path="data/").tfidf()

    # Step 3 — train GCN and inspect results
    result = GNN(dataset_name="R8", dataset_path="data/").fit()
    print(result.test_accuracy)
"""

__version__ = "2.0.0"
__author__ = "Vitor Meriat"
__email__ = "vitormeriat@gmail.com"

# Public API — import these directly from `text4gcn`
from .models import GNN, Builder, Layer, TextGCNTrainer, TrainingResult
from .preprocess import TextPipeline
from .graph import (
    RefinementConfig,
    apply_refinement,
    GraphBuildStats,
    compute_adjacency_stats,
    load_adjacency_stats,
)

from .utils.logging import _configure_library_logger, set_verbosity

# Submodules kept for power users
from . import datasets, builder, models, preprocess, modules, utils, graph

_configure_library_logger()

__all__ = [
    # Core classes
    "GNN",
    "Builder",
    "Layer",
    "TextGCNTrainer",
    "TrainingResult",
    "TextPipeline",
    "RefinementConfig",
    "apply_refinement",
    "GraphBuildStats",
    "compute_adjacency_stats",
    "load_adjacency_stats",
    "set_verbosity",
    # Submodules
    "datasets",
    "builder",
    "models",
    "preprocess",
    "modules",
    "utils",
    "graph",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
