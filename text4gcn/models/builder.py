# class Builder:
#     Liwc = 'liwc'
#     Frequency = 'frequency'
#     Embedding = 'embedding'
#     CosineSimilarity = 'cosine'    
#     DependencyParsing = 'dependency'
#     ConstituencyParsing = 'constituency'
    
# import enum
# from enum import StrEnum

# @enum.unique
# class Builder(StrEnum):
#     Liwc = 'liwc'
#     Frequency = 'frequency'
#     Embedding = 'embedding'
#     CosineSimilarity = 'cosine'    
#     DependencyParsing = 'dependency'
#     ConstituencyParsing = 'constituency'

from enum import Enum

class Builder(str, Enum):
    Liwc = 'liwc'
    # Frequency = 'frequency'
    # Embedding = 'embedding'
    CosineSimilarity = 'cosine.similarity'
    DependencyParsing = 'dependency.parsing'
    ConstituencyParsing = 'constituency.parsing'
    Word2Vec = 'embedding.word2vec'
    Tfidf = 'frequency.tfidf'
    
