from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules import adjacency as adj
from text4gcn.modules.logger import Process
from text4gcn.graph.stats import compute_adjacency_stats, report_stats
from scipy.sparse import csr_matrix
import pickle


class CosineSimilarityAdjacency():

    def __init__(self, dataset_name: str, dataset_path: str, threshold: float = 0.95):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    @Process.log("EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix.")
    def build(self):
        corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        self.flop.create_dir(dir_path=f'{corpus_path}.adjacency',
                             overwrite=False)

        docs_of_words = [line.split() for line in open(file=ds_corpus, encoding="utf-8")]
        # Extract Vocabulary.
        vocab = open(ds_corpus_vocabulary, encoding="utf-8").read().splitlines()
        # Word to its id.
        word_to_id = {word: i for i, word in enumerate(vocab)}
        # Real train-size, not adjusted.
        train_size = len(open(ds_corpus_train_idx, encoding="utf-8").readlines())
        # Real test-size.
        test_size = len(open(ds_corpus_test_idx, encoding="utf-8").readlines())

        self.logger.info("Calculating Cosine Similarity of Word Vectors")

        # As an alternative, use cosine similarity of word vectors as weights:
        ds_corpus_word_vectors = f'{corpus_path}.shuffled/{self.dataset_name}.word_vectors'
        rows, cols, weights = adj.extract_cosine_similarity_word_weights(vocab=vocab,
                                                                         train_size=train_size,
                                                                         word_vec_path=ds_corpus_word_vectors,
                                                                         threshold=self.threshold)

        self.logger.info("Calculating TF-IDF")
        # Extract word-doc weights
        rows, cols, weights = adj.extract_tf_idf_doc_word_weights(rows,
                                                                  cols,
                                                                  weights,
                                                                  vocab,
                                                                  train_size,
                                                                  docs_of_words,
                                                                  word_to_id)

        adjacency_len = train_size + len(vocab) + test_size

        self.logger.info(
            f"[INFO] ({len(weights)}, ({len(rows)}, {len(cols)})), shape=({adjacency_len}, {adjacency_len})")

        adjacency_matrix = csr_matrix((weights, (rows, cols)),
                                      shape=(adjacency_len, adjacency_len))

        # Dump Adjacency Matrix
        adj_path = f"{corpus_path}.adjacency/ind.cosine.similarity.{self.dataset_name}.adj"
        with open(adj_path, 'wb') as f:
            pickle.dump(adjacency_matrix, f)

        stats = compute_adjacency_stats(
            adjacency_matrix,
            path=adj_path,
            builder="cosine.similarity",
            dataset_name=self.dataset_name,
        )
        report_stats(stats, self.logger)
        return stats
