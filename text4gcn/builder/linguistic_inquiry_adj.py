from text4gcn.modules.textvec.liwc import Liwc
from text4gcn.modules import adjacency as adjcy
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.graph.stats import GraphBuildStats, compute_adjacency_stats, report_stats
from nltk.corpus import stopwords
import scipy.sparse as sp
import pickle as pkl
from tqdm import tqdm


class LiwcAdjacency():

    def __init__(self, dataset_name, dataset_path, liwc_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.liwc_path = liwc_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    def build_relation_pair(self, docs_of_words):
        """Count adjacent word pairs that share at least one LIWC category."""
        stop_words = set(stopwords.words('english'))
        liwc = Liwc(self.liwc_path)

        rela_pair_count_str = {}
        for docs_of_word in tqdm(docs_of_words):
            categories = [set(liwc.search(word)) for word in docs_of_word]
            for i in range(len(docs_of_word) - 1):
                word_i, word_j = docs_of_word[i], docs_of_word[i + 1]
                if word_i == word_j or word_i in stop_words or word_j in stop_words:
                    continue
                if not categories[i] & categories[i + 1]:
                    continue
                for pair in (f'{word_i},{word_j}', f'{word_j},{word_i}'):
                    rela_pair_count_str[pair] = rela_pair_count_str.get(pair, 0) + 1
        return rela_pair_count_str

    def compute_weights_with_PMI(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, row, col):
        """Delegate to shared RWF weight helper (DCAI 2022 Eq. 6–8)."""
        return adjcy.compute_weights_with_PMI(
            docs_of_words,
            rela_pair_count_str,
            word_window_freq,
            train_size,
            word_id_map,
            row,
            col,
        )

    def compute_weights(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, vocab_size, word_doc_freq, vocab):
        row = []
        col = []

        # compute weights PMI
        weight = self.compute_weights_with_PMI(docs_of_words, rela_pair_count_str, word_window_freq,
                                               train_size, word_id_map, row, col)

        # doc word frequency
        doc_word_freq = adjcy.get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = adjcy.get_weight_tfidf(
            docs_of_words, word_id_map, doc_word_freq, train_size, vocab_size, word_doc_freq, vocab, row, col)

        return row, col, weight, weight_tfidf

    def build(self) -> GraphBuildStats:

        # corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        # ds_corpus = f'{corpus_path}_clean/{self.dataset_name}.txt'
        # ds_corpus_vocabulary = f'{corpus_path}_clean/{self.dataset_name}.vocab'
        # ds_corpus_train_idx = f'{corpus_path}_clean/{self.dataset_name}.train'
        # ds_corpus_test_idx = f'{corpus_path}_clean/{self.dataset_name}.test'

        corpus_path = f"{self.dataset_path}/{self.dataset_name}"

        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        # checkers
        # check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus, ds_corpus_vocabulary, ds_corpus_train_idx, ds_corpus_test_idx)

        # create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir + "/linguistic_inquiry", overwrite=False)
        self.flop.create_dir(
            dir_path=f'{corpus_path}.adjacency', overwrite=False)

        docs_of_words = [line.split() for line in open(file=ds_corpus, encoding="utf-8")]
        # Extract Vocabulary.
        vocab = open(ds_corpus_vocabulary, encoding="utf-8").read().splitlines()
        # Real train-size, not adjusted.
        train_size = len(open(ds_corpus_train_idx, encoding="utf-8").readlines())
        # Real test-size.
        test_size = len(open(ds_corpus_test_idx, encoding="utf-8").readlines())

        word_doc_freq = adjcy.define_word_doc_freq(docs_of_words)

        vocab_size = len(vocab)
        word_id_map = {}
        id_word_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i
            id_word_map[i] = vocab[i]

        word_window_freq = adjcy.build_word_window_freq(docs_of_words)

        rela_pair_count_str = self.build_relation_pair(docs_of_words)

        row, col, weight, weight_tfidf = self.compute_weights(docs_of_words, rela_pair_count_str,
                                                              word_window_freq, train_size, word_id_map,
                                                              vocab_size, word_doc_freq, vocab)

        # =============================================================
        weight += weight_tfidf
        node_size = train_size + vocab_size + test_size

        self.logger.info(
            f"({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")

        adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        # =============================================================
        adj_path = f"{corpus_path}.adjacency/ind.liwc.{self.dataset_name}.adj"
        with open(adj_path, 'wb') as f:
            pkl.dump(adj, f)

        stats = compute_adjacency_stats(adj, path=adj_path, builder="liwc", dataset_name=self.dataset_name)
        report_stats(stats, self.logger)
        return stats
