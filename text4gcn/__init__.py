"""
Text4GCN: A Python framework for text classification using Graph Neural Networks.

This package provides tools for building text-based graphs and applying
Graph Convolutional Networks (GCNs) for document classification tasks.
"""

__version__ = "2.0.0"
__author__ = "Vitor Meriat"
__email__ = "vitormeriat@gmail.com"

# Import main modules
from . import datasets
from . import builder
from . import models
from . import preprocess
from . import modules
from . import utils

__all__ = [
    "datasets",
    "builder", 
    "models",
    "preprocess",
    "modules",
    "utils",
    "__version__",
    "__author__",
    "__email__",
]
