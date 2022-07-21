from typing import List, Dict, Tuple, Iterable, Any
from scipy.spatial.distance import cosine
from collections import Counter
from tqdm import tqdm
from math import log
import numpy as np
import itertools
import pickle


def flatten_nested_iterables(iterables_of_iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [item for sublist in iterables_of_iterables for item in sublist]


def extract_word_to_doc_ids(docs_of_words: List[List[str]]) -> Dict[str, List[int]]:
    """Extracted the document ids where unique words appeared."""
    word_to_doc_ids = {}
    for doc_id, words in enumerate(docs_of_words):
        appeared_words = set()
        for word in words:
            if word not in appeared_words:
                if word in word_to_doc_ids:
                    word_to_doc_ids[word].append(doc_id)
                else:
                    word_to_doc_ids[word] = [doc_id]
                appeared_words.add(word)
    return word_to_doc_ids


def extract_word_to_doc_counts(word_to_doc_ids: Dict[str, List[int]]) -> Dict[str, int]:
    return {word: len(doc_ids) for word, doc_ids in word_to_doc_ids.items()}


def extract_windows(docs_of_words: List[List[str]], window_size: int) -> List[List[str]]:
    """Word co-occurrence with context windows"""
    windows = []
    for doc_words in docs_of_words:
        doc_len = len(doc_words)
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            windows.extend(
                doc_words[j: j + window_size]
                for j in range(doc_len - window_size + 1)
            )

    return windows


def extract_word_counts_in_windows(windows_of_words: List[List[str]]) -> Dict[str, int]:
    """Find the total count of unique words in each window, each window is bag-of-words"""
    bags_of_words = map(set, windows_of_words)
    return Counter(flatten_nested_iterables(bags_of_words))


def extract_word_ids_pair_to_counts(windows_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Dict[str, int]:
    word_ids_pair_to_counts = Counter()
    for window in windows_of_words:
        for i in range(1, len(window)):
            try:
                word_id_i = word_to_id[window[i]]
                for j in range(i):
                    word_id_j = word_to_id[window[j]]
                    if word_id_i != word_id_j:
                        word_ids_pair_to_counts.update(
                            [
                                f'{word_id_i},{word_id_j}',
                                f'{word_id_j},{word_id_i}',
                            ]
                        )
            except:
                pass

    return dict(word_ids_pair_to_counts)


def extract_pmi_word_weights(windows_of_words: List[List[str]], word_to_id: Dict[str, int], vocab: List[str],
                             train_size: int) -> Tuple[List[int], List[int], List[float]]:
    """Calculate PMI as weights"""
    weight_rows = []  # type: List[int]
    weight_cols = []  # type: List[int]
    pmi_weights = []  # type: List[float]

    num_windows = len(windows_of_words)
    word_counts_in_windows = extract_word_counts_in_windows(
        windows_of_words=windows_of_words)
    word_ids_pair_to_counts = extract_word_ids_pair_to_counts(
        windows_of_words, word_to_id)

    for word_id_pair, count in word_ids_pair_to_counts.items():
        word_ids_in_str = word_id_pair.split(',')
        word_id_i, word_id_j = int(word_ids_in_str[0]), int(word_ids_in_str[1])
        word_i, word_j = vocab[word_id_i], vocab[word_id_j]
        word_freq_i, word_freq_j = word_counts_in_windows[word_i], word_counts_in_windows[word_j]

        #pmi_score = log((1.0 * count / num_windows) / (1.0 * word_freq_i * word_freq_j / (num_windows * num_windows)))
        pmi_score = log(
            1.0
            * count
            / num_windows
            / (1.0 * word_freq_i * word_freq_j / num_windows ** 2)
        )

        if pmi_score > 0.0:
            weight_rows.append(train_size + word_id_i)
            weight_cols.append(train_size + word_id_j)
            pmi_weights.append(pmi_score)
    return weight_rows, weight_cols, pmi_weights


def extract_cosine_similarity_word_weights(vocab: List[str], train_size: int,
                                           word_vec_path: str) -> Tuple[List[int], List[int], List[float]]:
    """Calculate Cosine Similarity of Word Vectors as weights"""
    word_vectors = pickle.load(
        file=open(word_vec_path, 'rb'))  # type: Dict[str,List[float]]

    weight_rows = []  # type: List[int]
    weight_cols = []  # type: List[int]
    cos_sim_weights = []  # type: List[float]

    #len_vocab = len(vocab)
    #pbar = tqdm(len_vocab)

    for (i, word_i), (j, word_j) in tqdm(itertools.product(enumerate(vocab), enumerate(vocab))):
        # pbar.update(1)
        if word_i in word_vectors and word_j in word_vectors:
            vector_i = np.array(word_vectors[word_i])
            vector_j = np.array(word_vectors[word_j])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                #print(word_i, word_j, similarity)
                weight_rows.append(train_size + i)
                weight_cols.append(train_size + j)
                cos_sim_weights.append(similarity)
    # pbar.close()
    return weight_rows, weight_cols, cos_sim_weights


def extract_doc_word_ids_pair_to_counts(docs_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Dict[str, int]:
    doc_word_freq = Counter()
    for doc_id, doc_words in enumerate(docs_of_words):
        for word in doc_words:
            try:
                word_id = word_to_id[word]
                doc_word_freq.update([f'{str(doc_id)},{str(word_id)}'])
            except:
                pass
    return dict(doc_word_freq)


def extract_tf_idf_doc_word_weights(
        adj_rows: List[int], adj_cols: List[int], adj_weights: List[float], vocab: List[str], train_size: int,
        docs_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Tuple[List[int], List[int], List[float]]:
    """Extract Doc-Word weights with TF-IDF"""
    doc_word_ids_pair_to_counts = extract_doc_word_ids_pair_to_counts(
        docs_of_words, word_to_id)
    word_to_doc_ids = extract_word_to_doc_ids(docs_of_words=docs_of_words)
    word_to_doc_counts = extract_word_to_doc_counts(
        word_to_doc_ids=word_to_doc_ids)

    vocab_len = len(vocab)
    num_docs = len(docs_of_words)
    # percorro a lista de documentos
    for doc_id, doc_words in enumerate(docs_of_words):
        doc_word_set = set()
        # percorro as palavras do documento
        for word in doc_words:
            # se a palavra ainda não foi calculada...
            if word not in doc_word_set:
                word_id = word_to_id[word]
                word_ids_pair_count = doc_word_ids_pair_to_counts[str(
                    doc_id) + ',' + str(word_id)]

                adj_rows.append(doc_id if doc_id <
                                train_size else doc_id + vocab_len)
                adj_cols.append(train_size + word_id)

                doc_word_idf = log(
                    1.0 * num_docs / word_to_doc_counts[vocab[word_id]])
                adj_weights.append(word_ids_pair_count * doc_word_idf)
                doc_word_set.add(word)
    return adj_rows, adj_cols, adj_weights


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


def compute_weights_with_PMI(docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, min_count1, max_count1, count_mean1, count_std1, row, col):
    weight = []
    errors = 0
    num_window = len(docs_of_words)
    for key, count in rela_pair_count_str.items():
        try:
            temp = key.split(',')
            i = temp[0]
            j = temp[1]

            if i in word_window_freq and j in word_window_freq:
                word_freq_i = word_window_freq[i]
                word_freq_j = word_window_freq[j]
                pmi = log(
                    1.0
                    * count
                    / num_window
                    / (1.0 * word_freq_i * word_freq_j / num_window ** 2)
                )
                if pmi <= 0:
                    continue

                row.append(train_size + word_id_map[i])
                col.append(train_size + word_id_map[j])
                if key in rela_pair_count_str:
                    wei = (rela_pair_count_str[key] -
                           min_count1) / (max_count1 - min_count1)
                    wei = (rela_pair_count_str[key]-count_mean1) / count_std1
                    weight.append(wei)
        except:
            errors += 1

    print(f'[INFO] Error in Compute Weights: {errors}')
    return weight
