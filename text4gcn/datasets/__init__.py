"""
Datasets module for Text4GCN.

This module provides access to various text classification datasets
commonly used in research and benchmarking.
"""

from .data import download, save_corpus, twenty_newsgroups

__all__ = ["download", "save_corpus", "twenty_newsgroups"]
