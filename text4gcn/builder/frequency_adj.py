from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules import adjacency as adj
from text4gcn.modules.logger import Process
from scipy.sparse import csr_matrix
import pickle


class FrequencyAdjacency():

    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    def save_history(self, hist):
        self.flop.create_dir(
            dir_path=f'{self.dataset_path}/log', overwrite=False)
        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'a') as my_file:
            my_file.writelines(hist)

    def build(self):
        self.builds()
        hist = self.logger.log_history()
        self.save_history(hist)

    @Process.log("EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix.")
    def builds(self):

        corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        self.flop.create_dir(
            dir_path=f'{corpus_path}.adjacency', overwrite=False)

        docs_of_words = [line.split() for line in open(file=ds_corpus)]
        # Extract Vocabulary.
        vocab = open(ds_corpus_vocabulary).read().splitlines()
        # Word to its id.
        word_to_id = {word: i for i, word in enumerate(vocab)}
        # Real train-size, not adjusted.
        train_size = len(open(ds_corpus_train_idx).readlines())
        # Real test-size.
        test_size = len(open(ds_corpus_test_idx).readlines())

        windows_of_words = adj.extract_windows(
            docs_of_words=docs_of_words, window_size=20)

        self.logger.info("Calculating PMI")
        # Extract word-word weights
        rows, cols, weights = adj.extract_pmi_word_weights(
            windows_of_words, word_to_id, vocab, train_size)

        self.logger.info("Calculating TF-IDF")
        # Extract word-doc weights
        rows, cols, weights = adj.extract_tf_idf_doc_word_weights(
            rows, cols, weights, vocab, train_size, docs_of_words, word_to_id)

        adjacency_len = train_size + len(vocab) + test_size

        self.logger.info(
            f"[INFO] ({len(weights)}, ({len(rows)}, {len(cols)})), shape=({adjacency_len}, {adjacency_len})")

        adjacency_matrix = csr_matrix(
            (weights, (rows, cols)), shape=(adjacency_len, adjacency_len))

        # Dump Adjacency Matrix
        with open(f"{corpus_path}.adjacency/ind.frequency.{self.dataset_name}.adj", 'wb') as f:
            pickle.dump(adjacency_matrix, f)
