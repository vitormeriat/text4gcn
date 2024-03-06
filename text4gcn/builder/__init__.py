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


MARKDOWN = """
# Text4GCN

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


def list_avaiable_methods():
    """List of Methods available in the Adjacency Matrix `Builder` 
    """
    console = Console()
    console.print(Markdown(MARKDOWN))
