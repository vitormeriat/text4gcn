from enum import Enum


class Builder(str, Enum):
    """Saved graph that ``GNN`` trains on; each value is the adjacency file's builder name."""

    Liwc = "liwc"
    CosineSimilarity = "cosine.similarity"
    DependencyParsing = "dependency.parsing"
    Word2Vec = "embedding.word2vec"
    Bert = "embedding.bert"
    Tfidf = "frequency.tfidf"
