"""
Builder module for Text4GCN.

This module provides classes and functions for building adjacency matrices
from text data using various methods including semantic, syntactic, and
frequency-based approaches.
"""

from enum import Enum
from typing import List

from rich.markdown import Markdown
from rich.console import Console

from .pipeline import Frequency, Semantic, Syntactic

# Legacy constants for backward compatibility
Liwc = 'liwc'
CosineSimilarity = 'cosine.similarity'
DependencyParsing = 'dependency.parsing'
ConstituencyParsing = 'constituency.parsing'
Word2Vec = 'embedding.word2vec'
Tfidf = 'frequency.tfidf'

# Documentation strings
_MARKDOWN_ADJ = """
# Text4GCN - ADJACENCY

Methods available in the Adjacency Matrix **`Builder`** 

01. Semantic
  - builder.Semantic().llm()
  - builder.Semantic().bert()
  - builder.Semantic().word2vec()
  - builder.Semantic().cosine_similarity()
02. Syntactic
  - builder.Syntactic().constituency_parsing()
  - builder.Syntactic().dependency_parsing()
03. Frequency
  - builder.Frequency().tfidf()
"""

_MARKDOWN_BLD = """
# Text4GCN - BUILDERS

List of Builders **`Builder`** 

01. Semantic
  - LLM
  - BERT
  - Word2Vec
  - CosineSimilarity
02. Syntactic
  - ConstituencyParsing
  - DependencyParsing
  - Liwc
03. Frequency
  - Tfidf
"""


def list_available_adjacency_methods() -> None:
    """List of Methods available in the Adjacency Matrix `Builder`."""
    console = Console()
    console.print(Markdown(_MARKDOWN_ADJ))


def list_available_builders() -> None:
    """List of Methods available in the Adjacency Matrix `Builder`."""
    console = Console()
    console.print(Markdown(_MARKDOWN_BLD))


class Builder(str, Enum):
    """Enumeration of available builder types."""
    
    Liwc = 'liwc'
    CosineSimilarity = 'cosine.similarity'
    DependencyParsing = 'dependency.parsing'
    ConstituencyParsing = 'constituency.parsing'
    Word2Vec = 'embedding.word2vec'
    Tfidf = 'frequency.tfidf'


# Public API
__all__ = [
    "Frequency",
    "Semantic", 
    "Syntactic",
    "Builder",
    "list_available_adjacency_methods",
    "list_available_builders",
    # Legacy constants
    "Liwc",
    "CosineSimilarity", 
    "DependencyParsing",
    "ConstituencyParsing",
    "Word2Vec",
    "Tfidf",
]
