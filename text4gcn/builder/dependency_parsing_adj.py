from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from math import log
import numpy as np
from ..modules import file_ops as flop
from ..modules import logger as logger
from ..modules.logger import Process
from scipy.sparse import csr_matrix
import pickle
from ..modules.word_processor import * #define_word_doc_freq, build_word_window_freq, relation_pair_statitcs, get_weight_tfidf, get_doc_word_freq


class DependencyParsingAdjacency():

    def __init__(self, dataset_name, dataset_path, core_nlp_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.core_nlp_path = core_nlp_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)

    def build_relation_pair(self, docs_of_words):
        stop_words = set(stopwords.words('english'))
        nlp = StanfordCoreNLP(self.core_nlp_path, lang='en')

        errors = 0
        rela_pair_count_str = {}
        for docs_of_word in docs_of_words:
            words = docs_of_word
            sentence = ' '.join(words)

            try:
                res = nlp.dependency_parse(sentence)
                tokenized = nlp.word_tokenize(sentence)
            except Exception as e:
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

    # def relation_pair_statitcs(self, rela_pair_count_str):
    #     max_count1 = 0.0
    #     min_count1 = 0.0
    #     count1 = []
    #     for key, value in rela_pair_count_str.items():
    #         if rela_pair_count_str[key] > max_count1:
    #             max_count1 = rela_pair_count_str[key]
    #         if value < min_count1:
    #             min_count1 = rela_pair_count_str[key]
    #         count1.append(rela_pair_count_str[key])

    #     count_mean1 = np.mean(count1)
    #     count_std1 = np.std(count1, ddof=1)
    #     return min_count1, max_count1, count_mean1, count_std1

    # def get_doc_word_freq(self, docs_of_words, word_id_map):
    #     doc_word_freq = {}
    #     for doc_id in range(len(docs_of_words)):
    #         words = docs_of_words[doc_id]
    #         for word in words:
    #             word_id = word_id_map[word]
    #             doc_word_str = f'{str(doc_id)},{str(word_id)}'
    #             if doc_word_str in doc_word_freq:
    #                 doc_word_freq[doc_word_str] += 1
    #             else:
    #                 doc_word_freq[doc_word_str] = 1
    #     return doc_word_freq

    # def get_weight_tfidf(self, docs_of_words, word_id_map, doc_word_freq, train_size, vocab_size, word_doc_freq, vocab, row, col):
    #     weight_tfidf = []
    #     for i in range(len(docs_of_words)):
    #         words = docs_of_words[i]
    #         doc_word_set = set()
    #         for word in words:
    #             if word in doc_word_set:
    #                 continue
    #             j = word_id_map[word]
    #             key = f'{str(i)},{str(j)}'
    #             freq = doc_word_freq[key]
    #             if i < train_size:
    #                 row.append(i)
    #             else:
    #                 row.append(i + vocab_size)
    #             col.append(train_size + j)
    #             idf = log(1.0 * len(docs_of_words) / word_doc_freq[vocab[j]])
    #             weight_tfidf.append(freq * idf)
    #             doc_word_set.add(word)
    #     return weight_tfidf

    def compute_weights_with_PMI(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, min_count1, max_count1, count_mean1, count_std1, row, col):
        weight = []
        errors = 0
        num_window = len(docs_of_words)
        for key, count in rela_pair_count_str.items():
            try:
                temp = key.split(',')
                i = temp[0]
                j = temp[1]

                if i in word_window_freq and j in word_window_freq:
                    word_freq_i = word_window_freq[i]
                    word_freq_j = word_window_freq[j]
                    pmi = log(
                        1.0
                        * count
                        / num_window
                        / (1.0 * word_freq_i * word_freq_j / num_window ** 2)
                    )
                    if pmi <= 0:
                        continue

                    row.append(train_size + word_id_map[i])
                    col.append(train_size + word_id_map[j])
                    if key in rela_pair_count_str:
                        wei = (rela_pair_count_str[key] -
                               min_count1) / (max_count1 - min_count1)
                        wei = (
                            rela_pair_count_str[key]-count_mean1) / count_std1
                        weight.append(wei)
            except:
                errors += 1

        print(f'[INFO] Error in Compute Weights: {errors}')
        return weight

    def compute_weights(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, vocab_size, word_doc_freq, vocab):
        row = []
        col = []

        min_count1, max_count1, count_mean1, count_std1 = relation_pair_statitcs(
            rela_pair_count_str)

        # compute weights PMI
        weight = self.compute_weights_with_PMI(
            docs_of_words, rela_pair_count_str, word_window_freq, train_size,
            word_id_map, min_count1, max_count1, count_mean1, count_std1, row, col)

        # doc word frequency
        doc_word_freq = get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = get_weight_tfidf(
            docs_of_words, word_id_map, doc_word_freq, train_size,
            vocab_size, word_doc_freq, vocab, row, col)

        return row, col, weight, weight_tfidf

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
        # Real train-size, not adjusted.
        train_size = len(open(ds_corpus_train_idx).readlines())
        # Real test-size.
        test_size = len(open(ds_corpus_test_idx).readlines())

        word_doc_freq = define_word_doc_freq(docs_of_words)

        vocab_size = len(vocab)
        word_id_map = {}
        id_word_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i
            id_word_map[i] = vocab[i]

        word_window_freq = build_word_window_freq(docs_of_words)

        rela_pair_count_str = self.build_relation_pair(docs_of_words)

        row, col, weight, weight_tfidf = self.compute_weights(docs_of_words, rela_pair_count_str,
                                                              word_window_freq, train_size, word_id_map,
                                                              vocab_size, word_doc_freq, vocab)

        # =============================================================
        weight += weight_tfidf
        node_size = train_size + vocab_size + test_size

        #pl.print_log(f"[INFO] ({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")
        self.logger.info(
            f"({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")

        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        # =============================================================

        # Dump Adjacency Matrix
        # with open(cfg.corpus_shuffled_adjacency_dir + "/syntactic_dependency/ind.{}.adj".format(ds_name), 'wb') as f:
        #     pkl.dump(adj, f)
        # Dump Adjacency Matrix
        with open(f"{corpus_path}.adjacency/ind.dep.{self.dataset_name}.adj", 'wb') as f:
            pickle.dump(adj, f)

        # =============================================================
        #elapsed = time() - t1
        #pl.print_log("[INFO] Adjacency Dir='{}'".format(cfg.corpus_shuffled_adjacency_dir))
        #pl.print_log("[INFO] Elapsed time is %f seconds." % elapsed)
        #pl.print_log("[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========")
