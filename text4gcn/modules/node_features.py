from typing import List, Dict, MutableMapping, Tuple, Union
from collections import OrderedDict
from scipy.sparse import csr_matrix
#from ..modules import logger
from text4gcn.modules import logger
import pickle as pkl
import numpy as np


#from preprocessors.configs import PreProcessingConfigs
#from utils.file_ops import check_paths, create_dir
#from utils.common import check_data_set
#from utils.logger import PrintLog
#from time import time
#from math import ceil


# word -> word-vector, words are ordered with respect to vocabulary
WORD_VECTORS_TYPE = MutableMapping[str, np.ndarray]


class NodeFeatures():

    def __init__(self, logger: logger.PrintLog):
        self.logger = logger

    def extract_doc_labels(self, ds_corpus_meta_file: str) -> List[str]:
        with open(ds_corpus_meta_file) as ds_corpus_meta:
            doc_labels = list(OrderedDict.fromkeys(
                doc_meta.split()[2] for doc_meta in ds_corpus_meta))
        return doc_labels

    def compute_x(self, docs_of_words: List[List[str]], tr_size: int, emb_dim: int, w_vectors: WORD_VECTORS_TYPE) -> csr_matrix:
        """ x: feature vectors of training docs, no initial features """
        data_x = []
        for i in range(tr_size):
            doc_vec = np.zeros(emb_dim, dtype=float)  # Initialize
            words = docs_of_words[i]
            for word in words:
                if word in w_vectors:
                    doc_vec += w_vectors[word]

            mean_doc_vec = (doc_vec / len(words)).tolist()
            data_x.extend(mean_doc_vec)

        row_indexes = np.array(
            [[i] * emb_dim for i in range(tr_size)]).flatten().tolist()
        col_indexes = list(range(emb_dim)) * tr_size
        return csr_matrix((data_x, (row_indexes, col_indexes)), shape=(tr_size, emb_dim))

    def compute_y(self, doc_meta_list: List[str], train_size: int, doc_labels: List[str]) -> np.ndarray:
        y = []
        for i in range(train_size):
            doc_meta = doc_meta_list[i]
            one_hot_encoded_label = [0] * len(doc_labels)

            label = doc_meta.split('\t')[2]
            label_index = doc_labels.index(label)
            one_hot_encoded_label[label_index] = 1
            y.append(one_hot_encoded_label)
        return np.array(y)

    def compute_tx(self, docs_of_words: List[List[str]], test_size: int, real_train_size: int, word_emb_dim: int,
                   w_vectors: WORD_VECTORS_TYPE) -> csr_matrix:
        """ 
        tx: feature vectors of test docs, no initial features 
        """
        data_tx = []
        for i in range(test_size):
            doc_vec = np.zeros(word_emb_dim, dtype=float)  # Initialize
            words = docs_of_words[i + real_train_size]
            for word in words:
                if word in w_vectors:
                    doc_vec += w_vectors[word]

            mean_doc_vec = (doc_vec / len(words)).tolist()
            data_tx.extend(mean_doc_vec)

        row_indexes = np.array(
            [[i] * word_emb_dim for i in range(test_size)]).flatten().tolist()
        col_indexes = list(range(word_emb_dim)) * test_size
        return csr_matrix((data_tx, (row_indexes, col_indexes)), shape=(test_size, word_emb_dim))

    def compute_ty(self, doc_meta_list: List[str], test_size: int, real_train_size: int, doc_labels: List[str]) -> np.ndarray:
        ty = []
        for i in range(test_size):
            doc_meta = doc_meta_list[i + real_train_size]
            one_hot_encoded_label = [0] * len(doc_labels)

            label = doc_meta.split('\t')[2]
            label_index = doc_labels.index(label)
            one_hot_encoded_label[label_index] = 1
            ty.append(one_hot_encoded_label)
        return np.array(ty)

    def compute_allx(self, docs_of_words: List[List[str]], real_train_size: int, vocab: List[str],
                     word_vectors: WORD_VECTORS_TYPE, emb_dim: int) -> csr_matrix:
        """
        allx: A superset of x, the feature vectors of both labeled and words (unlabeled training instances)
        """
        word_vectors_arr = self.extract_word_vectors_arr(
            word_vectors, vocab, emb_dim=emb_dim)
        data_allx = []
        row_size = real_train_size + len(vocab)

        for i in range(real_train_size):
            doc_vec = np.zeros(emb_dim, dtype=float)  # Initialize
            words = docs_of_words[i]
            for word in words:
                if word in word_vectors:
                    doc_vec += word_vectors[word]

            mean_doc_vec = (doc_vec / len(words)).tolist()
            data_allx.extend(mean_doc_vec)

        data_allx.extend(word_vectors_arr.flatten())
        data_allx = np.array(data_allx)
        row_indexes = np.array(
            [[i] * emb_dim for i in range(row_size)]).flatten()
        col_indexes = np.array(list(range(emb_dim)) * row_size)

        return csr_matrix((data_allx, (row_indexes, col_indexes)), shape=(row_size, emb_dim))

    def compute_ally(self, doc_meta_list: List[str], real_train_size: int, doc_labels: List[str], vocab_size: int) -> np.ndarray:
        ally = []
        for doc_meta in doc_meta_list[:real_train_size]:
            label = doc_meta.split('\t')[2]
            one_hot_encoded_label = [0] * len(doc_labels)
            label_index = doc_labels.index(label)
            one_hot_encoded_label[label_index] = 1
            ally.append(one_hot_encoded_label)

        zero_filled_one_hot_for_words = [
            np.zeros(len(doc_labels), dtype=int)] * vocab_size
        ally.extend(zero_filled_one_hot_for_words)
        return np.array(ally)

    def load_word_to_word_vectors(self, path: str) -> Tuple[WORD_VECTORS_TYPE, int]:
        word_vectors_as_list = pkl.load(file=open(path, 'rb'))
        word_vectors = OrderedDict((word, np.array(vec_lst))
                                   for word, vec_lst in word_vectors_as_list.items())
        word_embedding_dimension = len(next(iter(word_vectors.values())))
        return word_vectors, word_embedding_dimension

    def extract_word_vectors_arr(self, w_vectors: WORD_VECTORS_TYPE, vocab: List[str], emb_dim: int) -> np.ndarray:
        np.random.seed(0)  # For reproducibility
        word_vectors_arr = np.random.uniform(-0.01,
                                             0.01, (len(vocab), emb_dim))
        if len(w_vectors) != 0:
            for i, word in enumerate(vocab):
                if word in w_vectors:
                    word_vectors_arr[i] = w_vectors[word]
        return word_vectors_arr

    def dump_node_features(self, directory: str, ds: str, node_features_dict: Dict[str, Union[np.ndarray, csr_matrix]]):
        for name, node_feature_matrix in node_features_dict.items():
            with open(f"{directory}/ind.{ds}.{name}", 'wb') as file:
                pkl.dump(node_feature_matrix, file)
