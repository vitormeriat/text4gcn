from collections import OrderedDict
from datetime import datetime, timezone
import scipy.sparse as sp
import pickle as pkl
import numpy as np

from text4gcn.utils.logging import get_logger

logger = get_logger(__name__)


def load_class_names(path, dataset: str) -> list:
    """Class labels in the order of the one-hot label columns (first appearance in the shuffled meta)."""
    with open(f"{path}/{dataset}.shuffled/{dataset}.meta", encoding="utf-8") as meta:
        return list(OrderedDict.fromkeys(line.strip().split('\t')[2] for line in meta if line.strip()))


class LoadData:

    def __init__(self, path, dataset, builder, refinement_config=None):
        self.path = path
        self.dataset = dataset
        # str(Enum) is "Builder.Tfidf" on Python 3.11+; use .value for file names
        self.builder = getattr(builder, "value", builder)
        self.refinement_config = refinement_config

    def parse_index_file(self, filename):
        """Parse index file."""
        return [int(line.strip()) for line in open(filename, encoding="utf-8")]

    def sample_mask(self, idx, row_length: int):
        """Create mask."""
        mask = np.zeros(row_length)
        mask[idx] = 1
        return np.array(mask, dtype=bool)

    def load_corpus(self):
        """
        Loads input corpus from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.train.index => the indices of training docs in original doc list.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """

        node_feature_names = ['x', 'y', 'tx', 'ty', 'allx', 'ally']
        node_features = []
        for node_feature_name in node_feature_names:
            # with open(f"{self.path}/{self.dataset}.node_features/ind.{self.builder}.{self.dataset}.{node_feature_name}", 'rb') as f:
            with open(f"{self.path}/{self.dataset}.node_features/ind.{self.dataset}.{node_feature_name}", 'rb') as f:
                node_features.append(pkl.load(f, encoding='latin1'))

        with open(f"{self.path}/{self.dataset}.adjacency/ind.{self.builder}.{self.dataset}.adj", 'rb') as f:
            adj = pkl.load(f, encoding='latin1')

        x, y, tx, ty, allx, ally = tuple(node_features)

        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))

        logger.debug("loaded x %s, y %s, tx %s, ty %s, allx %s, ally %s, adj %s",
                     x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, adj.shape)

        train_idx_orig = self.parse_index_file(
            f"{self.path}/{self.dataset}.shuffled/{self.dataset}.train")

        train_size = len(train_idx_orig)
        val_size = train_size - x.shape[0]
        test_size = tx.shape[0]

        # This part of the index corresponds to the value after vstack
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)
        idx_test = range(allx.shape[0], allx.shape[0] + test_size)

        # sample_mask plays the role of dividing the data area, the following three lines divide the doc part
        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        # Numpy bool slice operation
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        # Convert asymmetric adjacency matrix to symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        result = (adj, features, y_train, y_val, y_test,
                  train_mask, val_mask, test_mask, train_size, test_size)

        if self.refinement_config is not None:
            from text4gcn.graph import apply_refinement
            result, _ = apply_refinement(
                result,
                self.refinement_config,
                dataset_path=self.path,
                dataset_name=self.dataset,
                return_metadata=False,
            )

        return result


def unique_name():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S%f')[:-3]


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.toarray()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.toarray()
