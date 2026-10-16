from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Iterable
from collections import OrderedDict
import numpy as np


def word_to_vectors(vocabulary, word_vectors):
    return OrderedDict((word, vec.tolist()) for word, vec in zip(vocabulary, word_vectors))


def extract_vocabulary(docs_of_words: Iterable[List[str]]) -> List[str]:
    vocabulary = OrderedDict()
    for words in docs_of_words:
        vocabulary.update((word, None) for word in words)
    return list(vocabulary.keys())


def extract_tf_idf_word_vectors(word_definitions: List[str], max_features: int) -> np.ndarray:
    """TF-IDF vectors of the word definitions, one row per word.

    Words without a definition (empty string) get an all-zero row, so similarity
    builders give them no edges instead of treating them all as identical.
    """
    defined = [i for i, definition in enumerate(word_definitions) if definition]
    if not defined:                     # e.g. a non-English corpus: WordNet knows no word
        return np.zeros((len(word_definitions), 1))
    tf_idf = TfidfVectorizer(max_features=max_features).fit_transform(
        [word_definitions[i] for i in defined]).toarray()
    vectors = np.zeros((len(word_definitions), tf_idf.shape[1]))
    vectors[defined] = tf_idf
    return vectors


def extract_word_definitions(vocabulary: List[str]) -> List[str]:
    """All WordNet definitions of each word, joined; an empty string when WordNet
    does not know the word."""
    from nltk.corpus import wordnet

    return [' '.join(synset.definition() for synset in wordnet.synsets(word.strip()))
            for word in vocabulary]
