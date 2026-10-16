from text4gcn.builder._common import run_build, save_graph
from text4gcn.modules import file_ops as flop
from text4gcn.modules import adjacency as adj
from text4gcn.graph.stats import GraphBuildStats
from text4gcn.utils.reporting import Reporter, check_verbose
from scipy.sparse import csr_matrix


class FrequencyAdjacency():
    """
    Creates an heterogenous `doc-word` adjacency matrix based on TF-IDF.

    This is the classic method to create a adjacency matrix.

    ...

    Attributes
    ----------
    dataset_name : str
        name of the dataset
    dataset_path : str
        path to files

    Methods
    -------
    build():
        Builds the adjacency matrix and returns :class:`GraphBuildStats`.
    """

    def __init__(self, dataset_name: str, dataset_path: str, window_size: int = 20, verbose: int = 1):
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.verbose = check_verbose(verbose)
        self.flop = flop.FileOps()

    def _build(self, report: Reporter) -> GraphBuildStats:

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

        windows_of_words = adj.extract_windows(docs_of_words=docs_of_words,
                                               window_size=self.window_size)

        # Extract word-word weights
        rows, cols, weights = adj.extract_pmi_word_weights(windows_of_words,
                                                           word_to_id,
                                                           vocab,
                                                           train_size,
                                                           progress=report.progress)
        word_word_edges = len(weights)

        # Extract word-doc weights
        rows, cols, weights = adj.extract_tf_idf_doc_word_weights(rows,
                                                                  cols,
                                                                  weights,
                                                                  vocab,
                                                                  train_size,
                                                                  docs_of_words,
                                                                  word_to_id)

        adjacency_len = train_size + len(vocab) + test_size
        adjacency_matrix = csr_matrix((weights, (rows, cols)),
                                      shape=(adjacency_len, adjacency_len))

        adj_path = f"{corpus_path}.adjacency/ind.frequency.tfidf.{self.dataset_name}.adj"
        return save_graph(adjacency_matrix, adj_path, "frequency.tfidf", self.dataset_name, word_word_edges)

    def build(self) -> GraphBuildStats:
        """Builds the adjacency matrix and returns edge/memory stats."""
        return run_build(self._build, f'Frequency.tfidf("{self.dataset_name}") · window {self.window_size}',
                         self.verbose, self.dataset_path, self.dataset_name)
