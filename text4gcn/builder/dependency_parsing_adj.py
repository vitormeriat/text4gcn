from text4gcn.modules import adjacency as adjcy
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process
from text4gcn.graph.stats import GraphBuildStats, compute_adjacency_stats, report_stats
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle


def _import_stanford_corenlp():
    try:
        from stanfordcorenlp import StanfordCoreNLP
    except ImportError as exc:
        raise ImportError(
            "stanfordcorenlp is required for Syntactic.dependency_parsing().\n"
            "Install with: pip install 'text4gcn[syntactic]'\n"
            "Also download the Stanford CoreNLP jars and pass core_nlp_path."
        ) from exc
    return StanfordCoreNLP


class DependencyParsingAdjacency():

    def __init__(self, dataset_name, dataset_path, core_nlp_path, language: str='english'):
        self.language = language
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.core_nlp_path = core_nlp_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    def build_relation_pair(self, docs_of_words):
        StanfordCoreNLP = _import_stanford_corenlp()
        stop_words = set(stopwords.words(self.language))
        nlp = StanfordCoreNLP(self.core_nlp_path, lang='en')

        errors = 0
        rela_pair_count_str = {}
        for docs_of_word in tqdm(docs_of_words):
            words = docs_of_word
            sentence = ' '.join(words)

            try:
                res = nlp.dependency_parse(sentence)
                tokenized = nlp.word_tokenize(sentence)
            except Exception:
                #print(f'{sentence} = {e}')
                errors += 1
                res = []

            for pair in list(res):
                #pair=pair.split(", ")
                if pair[0] == 'ROOT' or pair[1] == 'ROOT':
                    continue
                if pair[0] == pair[1]:
                    continue
                # if pair[0] in string.punctuation or pair[1] in string.punctuation:
                #    continue
                if pair[0] in stop_words or pair[1] in stop_words:
                    continue

                word_pair_str = f'{tokenized[pair[2]-1]},{tokenized[pair[1]-1]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # two orders
                word_pair_str = f'{tokenized[pair[1]-1]},{tokenized[pair[2]-1]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
        print(f'[INFO] Error in Dependency Parse: {errors}')
        nlp.close()
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
        weight = self.compute_weights_with_PMI(
            docs_of_words,
            rela_pair_count_str,
            word_window_freq,
            train_size,
            word_id_map,
            row,
            col
        )

        # doc word frequency
        doc_word_freq = adjcy.get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = adjcy.get_weight_tfidf(
            docs_of_words,
            word_id_map,
            doc_word_freq,
            train_size,
            vocab_size,
            word_doc_freq,
            vocab,
            row,
            col
        )

        return row, col, weight, weight_tfidf

    def _save_history(self, hist):
        self.flop.create_dir(
            dir_path=f'{self.dataset_path}/log', overwrite=False)
        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'a', encoding="utf-8") as my_file:
            my_file.writelines(hist)

    @Process.log("EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix.")
    def build(self) -> GraphBuildStats:

        corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        self.flop.create_dir(
            dir_path=f'{corpus_path}.adjacency',
            overwrite=False
        )

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

        row, col, weight, weight_tfidf = self.compute_weights(
            docs_of_words,
            rela_pair_count_str,
            word_window_freq,
            train_size,
            word_id_map,
            vocab_size,
            word_doc_freq,
            vocab
        )

        # =============================================================
        weight += weight_tfidf
        node_size = train_size + vocab_size + test_size

        self.logger.info(
            f"({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")

        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        # =============================================================

        adj_path = f"{corpus_path}.adjacency/ind.dependency.parsing.{self.dataset_name}.adj"
        with open(adj_path, 'wb') as f:
            pickle.dump(adj, f)

        stats = compute_adjacency_stats(adj, path=adj_path, builder="dependency.parsing", dataset_name=self.dataset_name)
        report_stats(stats, self.logger)

        hist = self.logger.log_history()
        self._save_history(hist)
        return stats
