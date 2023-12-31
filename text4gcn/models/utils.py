from datetime import timedelta, datetime
from tabulate import tabulate
import scipy.sparse as sp
import pickle as pkl
import torch as th
import numpy as np
import random
import time


class LoadData:

    def __init__(self, path, dataset, builder):
        self.path = path
        self.dataset = dataset
        self.builder = builder

    def parse_index_file(self, filename):
        """Parse index file."""
        return [int(line.strip()) for line in open(filename)]

    def sample_mask(self, idx, row_length: int):
        """Create mask."""
        mask = np.zeros(row_length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

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

        print("\nShow info about load data:\n")

        print(tabulate([
            ["x", str(x.shape)],
            ["y", str(y.shape)],
            ["tx", str(tx.shape)],
            ["ty", str(ty.shape)],
            ["allx", str(allx.shape)],
            ["ally", str(ally.shape)],
            ["adj", str(adj.shape)],
            ["Features", str(features.shape)],
            ["Labels", str(labels.shape)],
        ], ["Data", "Shape"]))

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

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def unique_name():
    return datetime.utcnow().strftime('%Y%m%d-%H%M%S%f')[:-3]


def get_train_test(target_fn):
    train_lst = []
    test_lst = []
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)


def return_seed(nums=10):
    return random.sample(range(100000), nums)


def get_time_dif(start_time):
    """Obter tempo decorrido"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def macro_f1(pred, targ, num_classes=None):
    pred = th.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        # A previsão é i, e o rótulo é de fato i
        tp = ((pred == i) & (targ == i)).sum().item()
        # A previsão é i, mas o rótulo não é i
        fp = ((pred == i) & (targ != i)).sum().item()
        # A previsão não é i, mas o rótulo é i
        fn = ((pred != i) & (targ == i)).sum().item()
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def accuracy(pred, targ):
    pred = th.max(pred, 1)[1]
    return ((pred == targ).float()).sum().item() / targ.size()[0]


def sparse_to_tensor(sparse_matrix):
    values = sparse_matrix.data
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    i = th.LongTensor(indices)
    v = th.FloatTensor(values)
    shape = sparse_matrix.shape
    return th.sparse.FloatTensor(i, v, th.Size(shape))


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.A


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
    return adj_normalized.A
