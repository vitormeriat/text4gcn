"""
Text4GCN: A Python framework for text classification using Graph Neural Networks.

Quick start::

    from text4gcn import GNN, TextPipeline, Builder

    TextPipeline(dataset_name="20ng", dataset_path="data/").execute()

    GNN(dataset="20ng", path="data/", log_dir="logs/", builder=Builder.Tfidf).fit()
"""

__version__ = "2.0.0"
__author__ = "Vitor Meriat"
__email__ = "vitormeriat@gmail.com"

# Public API — import these directly from `text4gcn`
from .models import GNN, Builder, Layer, TextGCNTrainer
from .preprocess import TextPipeline
from .graph import RefinementConfig, apply_refinement

# Submodules kept for power users
from . import datasets, builder, models, preprocess, modules, utils, graph

__all__ = [
    # Core classes
    "GNN",
    "Builder",
    "Layer",
    "TextGCNTrainer",
    "TextPipeline",
    "RefinementConfig",
    "apply_refinement",
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
