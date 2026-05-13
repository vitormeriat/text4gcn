"""
Node metadata for the text graph.

Provides structured metadata per node (type, degree, word/doc id)
for analysis, visualization, and downstream LLM-guided weighting.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class NodeMetadata:
    """
    Metadata for a single node in the doc-word graph.

    Attributes
    ----------
    node_type : str
        One of 'train_doc', 'word', 'test_doc'.
    original_index : int
        Index in the original (unrefined) graph.
    degree : float
        Node degree (sum of edge weights or count of edges).
    word_id : Optional[int]
        Vocabulary index if node is a word; None for docs.
    word : Optional[str]
        Word string if node is a word and vocab was provided; None otherwise.
    doc_id : Optional[int]
        Document index (0..train_size-1 or test offset) if node is a doc; None for words.
    """

    node_type: str
    original_index: int
    degree: float
    word_id: Optional[int] = None
    word: Optional[str] = None
    doc_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for serialization or DataFrame."""
        return {
            "node_type": self.node_type,
            "original_index": self.original_index,
            "degree": self.degree,
            "word_id": self.word_id,
            "word": self.word,
            "doc_id": self.doc_id,
        }


def build_node_metadata(
    adj,
    train_size: int,
    test_size: int,
    vocab: Optional[List[str]] = None,
    keep_indices: Optional[np.ndarray] = None,
) -> List[NodeMetadata]:
    """
    Build metadata for each node in the graph.

    Convention: nodes 0..train_size-1 = train docs,
    train_size..train_size+vocab_size-1 = words,
    train_size+vocab_size..N-1 = test docs.

    Parameters
    ----------
    adj : scipy.sparse matrix
        Adjacency matrix (any format with .getrow() or row sums).
    train_size : int
        Number of training documents.
    test_size : int
        Number of test documents.
    vocab : list of str, optional
        Vocabulary list; used to set .word for word nodes.
    keep_indices : array-like, optional
        If provided, metadata is built only for these node indices (e.g. after refinement).

    Returns
    -------
    list of NodeMetadata
        One entry per node (or per index in keep_indices if provided).
    """
    try:
        degree_vec = np.array(adj.sum(axis=1)).flatten()
    except Exception:
        degree_vec = np.array(adj.sum(axis=0)).flatten()

    n_nodes = adj.shape[0]
    vocab_size = n_nodes - train_size - test_size

    indices = (
        np.arange(n_nodes)
        if keep_indices is None
        else np.asarray(keep_indices, dtype=int)
    )
    out = []
    for idx in indices:
        i = int(idx)
        deg = float(degree_vec[i])
        if i < train_size:
            node_type = "train_doc"
            word_id, word, doc_id = None, None, i
        elif i < train_size + vocab_size:
            node_type = "word"
            wid = i - train_size
            word_id = wid
            word = vocab[wid] if vocab is not None and 0 <= wid < len(vocab) else None
            doc_id = None
        else:
            node_type = "test_doc"
            word_id, word = None, None
            doc_id = i - (train_size + vocab_size)
        out.append(
            NodeMetadata(
                node_type=node_type,
                original_index=i,
                degree=deg,
                word_id=word_id,
                word=word,
                doc_id=doc_id,
            )
        )
    return out
