from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process
from scipy.sparse import csr_matrix
from text4gcn.modules import adjacency as adj
import pickle


class CosineSimilarityAdjacency():

    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    @Process.log("EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix.")
    def build(self):
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

        self.logger.info("Calculating Cosine Similarity of Word Vectors")

        # As an alternative, use cosine similarity of word vectors as weights:
        ds_corpus_word_vectors = f'{corpus_path}.shuffled/{self.dataset_name}.word_vectors'
        rows, cols, weights = adj.extract_cosine_similarity_word_weights(
            vocab, train_size, ds_corpus_word_vectors)

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
        with open(f"{corpus_path}.adjacency/ind.cosine.{self.dataset_name}.adj", 'wb') as f:
            pickle.dump(adjacency_matrix, f)
