# from .frequency_adj import FrequencyAdjacency
# from .embedding_adj import EmbeddingAdjacency
# from .linguistic_inquiry_adj import LiwcAdjacency
# from .cosine_similarity_adj import CosineSimilarityAdjacency
# from .dependency_parsing_adj import DependencyParsingAdjacency
# from .constituency_parsing_adj import ConstituencyParsingAdjacency
# from .word2vec_adj import Word2VecAdjacency

from .pipeline import Frequency, Semantic, Syntactic

from rich.markdown import Markdown
from rich.console import Console

from enum import Enum


Liwc = 'liwc'
CosineSimilarity = 'cosine.similarity'
DependencyParsing = 'dependency.parsing'
ConstituencyParsing = 'constituency.parsing'
Word2Vec = 'embedding.word2vec'
Tfidf = 'frequency.tfidf'


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


def list_avaiable_ajacency_methods():
    """List of Methods available in the Adjacency Matrix `Builder` 
    """
    console = Console()
    console.print(Markdown(_MARKDOWN_ADJ))

def list_avaiable_builders():
    """List of Methods available in the Adjacency Matrix `Builder` 
    """
    console = Console()
    console.print(Markdown(_MARKDOWN_BLD))


class Builder(str, Enum):
    Liwc = 'liwc'
    # Frequency = 'frequency'
    # Embedding = 'embedding'
    CosineSimilarity = 'cosine.similarity'
    DependencyParsing = 'dependency.parsing'
    ConstituencyParsing = 'constituency.parsing'
    Word2Vec = 'embedding.word2vec'
    Tfidf = 'frequency.tfidf'
