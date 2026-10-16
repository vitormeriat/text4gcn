from text4gcn.modules.textvec.word_features import train_word2vec
from text4gcn.modules import adjacency as adjcy
from text4gcn.modules import logger as logger
from text4gcn.modules import file_ops as flop
from text4gcn.graph.stats import GraphBuildStats, compute_adjacency_stats, report_stats
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm


class Word2VecAdjacency():

    def __init__(self,
                 dataset_name,
                 dataset_path,
                 training_regime,
                 embedding_dimension,
                 num_epochs,
                 language: str = "english",
                 threshold: float = 0.7,
                 output_path: str = None,
                 seed: int = 1234,
                 workers: int = 1):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.training_regime = training_regime
        self.embedding_dimension = embedding_dimension
        self.num_epochs = num_epochs
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
        self.language = language
        self.threshold = threshold
        self.output_path = output_path
        self.seed = seed
        self.workers = workers

    def build_relation_pair(self, docs_of_words):
        stop_words = set(stopwords.words(self.language))

        word2vec_model = train_word2vec(
            save_dir=None,
            document_list=docs_of_words,
            num_epochs=self.num_epochs,
            embedding_dimension=self.embedding_dimension,
            training_regime=self.training_regime,
            seed=self.seed,
            workers=self.workers,
        )

        errors = 0
        rela_pair_count_str = {}
        for docs_of_word in tqdm(docs_of_words):
            docs_of_word_len = len(docs_of_word)
            for i in range(docs_of_word_len):
                if i+1 == docs_of_word_len:
                    continue

                if docs_of_word[i] == docs_of_word[i+1]:
                    continue

                if docs_of_word[i] in stop_words or docs_of_word[i+1] in stop_words:
                    continue

                cosine_similarity = word2vec_model.wv.similarity(docs_of_word[i],
                                                                 docs_of_word[i+1])
                if cosine_similarity < self.threshold:
                    errors += 1
                    continue

                word_pair_str = f'{docs_of_word[i]},{docs_of_word[i+1]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # two orders
                word_pair_str = f'{docs_of_word[i+1]},{docs_of_word[i]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1

        print(f'[INFO] Words without similarity: {errors}')
        return rela_pair_count_str

    def compute_weights(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, vocab_size, word_doc_freq, vocab):
        row = []
        col = []

        # compute weights PMI
        weight = adjcy.compute_weights_with_PMI(
            docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, row, col)

        # doc word frequency
        doc_word_freq = adjcy.get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = adjcy.get_weight_tfidf(
            docs_of_words, word_id_map, doc_word_freq, train_size, vocab_size, word_doc_freq, vocab, row, col)

        return row, col, weight, weight_tfidf

    def _save_history(self, hist):
        self.flop.create_dir(
            dir_path=f'{self.dataset_path}/log', overwrite=False)
        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'a', encoding="utf-8") as my_file:
            my_file.writelines(hist)

    def build(self) -> GraphBuildStats:

        corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        # checkers
        # check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus, ds_corpus_vocabulary, ds_corpus_train_idx, ds_corpus_test_idx)
        # create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir + "/semantic", overwrite=False)
        self.flop.create_dir(dir_path=f'{corpus_path}.adjacency',
                             overwrite=False)

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

        row, col, weight, weight_tfidf = self.compute_weights(docs_of_words,
                                                              rela_pair_count_str,
                                                              word_window_freq,
                                                              train_size,
                                                              word_id_map,
                                                              vocab_size,
                                                              word_doc_freq,
                                                              vocab)

        # =============================================================
        weight += weight_tfidf
        node_size = train_size + vocab_size + test_size

        self.logger.info(
            f"({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")

        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        # =============================================================
        adj_path = f"{corpus_path}.adjacency/ind.embedding.word2vec.{self.dataset_name}.adj"
        # Dump Adjacency Matrix
        with open(adj_path, 'wb') as f:
            pickle.dump(adj, f)

        stats = compute_adjacency_stats(
            adj,
            path=adj_path,
            builder="embedding.word2vec",
            dataset_name=self.dataset_name,
        )
        report_stats(stats, self.logger)

        hist = self.logger.log_history()
        self._save_history(hist)
        return stats
