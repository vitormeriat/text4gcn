from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Iterable, Tuple
from collections import OrderedDict
import numpy as np
import pickle as pkl
from math import log


def word_to_vectors(vocabulary, word_vectors):
    return OrderedDict((word, vec.tolist()) for word, vec in zip(vocabulary, word_vectors))


def extract_vocabulary(docs_of_words: Iterable[List[str]]) -> List[str]:
    vocabulary = OrderedDict()
    for words in docs_of_words:
        vocabulary.update((word, None) for word in words)
    return list(vocabulary.keys())


def extract_tf_idf_word_vectors(word_definitions: List[str], max_features: int) -> List[np.ndarray]:
    tf_idf_vectorizer = TfidfVectorizer(max_features=max_features)
    return tf_idf_vectorizer.fit_transform(word_definitions).toarray()


def extract_word_definitions(vocabulary: List[str]) -> List[str]:
    from nltk.corpus import wordnet
    #from nltk import download
    #temporary_nltk_folder = 'venv/nltk_data/'
    #download(info_or_id='wordnet', download_dir=temporary_nltk_folder)
    #download(info_or_id='wordnet')
    #download(info_or_id='omw-1.4')

    merged_definitions_of_words = []
    for word in vocabulary:
        syn_sets_of_word = wordnet.synsets(word.strip())
        word_definitions = [syn_set.definition()
                            for syn_set in syn_sets_of_word]
        merged_definitions_of_word = ' '.join(word_definitions) if word_definitions else '<PAD>'

        merged_definitions_of_words.append(merged_definitions_of_word)
    # rmtree(temporary_nltk_folder)
    return merged_definitions_of_words


def build_word_doc_list(docs_of_words):
    word_doc_list = {}
    for i in range(len(docs_of_words)):
        words = docs_of_words[i]
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    return word_doc_list

def define_word_doc_freq(docs_of_words):
    word_doc_list = build_word_doc_list(docs_of_words)
    return {
        word: len(doc_list) for word, doc_list in word_doc_list.items()
    }

def build_word_window_freq(docs_of_words):
    word_window_freq = {}
    for window in docs_of_words:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    return word_window_freq

def relation_pair_statitcs(rela_pair_count_str):
    max_count1 = 0.0
    min_count1 = 0.0
    count1 = []
    for key, value in rela_pair_count_str.items():
        if rela_pair_count_str[key] > max_count1:
            max_count1 = rela_pair_count_str[key]
        if value < min_count1:
            min_count1 = rela_pair_count_str[key]
        count1.append(rela_pair_count_str[key])

    count_mean1 = np.mean(count1)
    count_std1 = np.std(count1, ddof=1)
    return min_count1, max_count1, count_mean1, count_std1

def get_weight_tfidf(docs_of_words, word_id_map, doc_word_freq, train_size, vocab_size, word_doc_freq, vocab, row, col):
    weight_tfidf = []
    for i in range(len(docs_of_words)):
        words = docs_of_words[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = f'{str(i)},{str(j)}'
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(docs_of_words) / word_doc_freq[vocab[j]])
            weight_tfidf.append(freq * idf)
            doc_word_set.add(word)
    return weight_tfidf

def get_doc_word_freq(docs_of_words, word_id_map):
    doc_word_freq = {}
    for doc_id in range(len(docs_of_words)):
        words = docs_of_words[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = f'{str(doc_id)},{str(word_id)}'
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1
    return doc_word_freq