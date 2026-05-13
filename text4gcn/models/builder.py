from enum import Enum


class Builder(str, Enum):
    Liwc = "liwc"
    CosineSimilarity = "cosine.similarity"
    DependencyParsing = "dependency.parsing"
    ConstituencyParsing = "constituency.parsing"
    Word2Vec = "embedding.word2vec"
    Tfidf = "frequency.tfidf"
