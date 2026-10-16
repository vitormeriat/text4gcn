from text4gcn.modules import adjacency as adjcy
from text4gcn.modules import logger as logger
from text4gcn.modules import file_ops as flop
from text4gcn.graph.stats import GraphBuildStats, compute_adjacency_stats, report_stats
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import numpy as np
import pickle
from tqdm import tqdm

# Lazily loaded — importing text4gcn.builder must not hit HuggingFace.
_tokenizer = None
_model = None


def _get_bert():
    """Load BERT tokenizer/model on first use (requires transformers + network/cache)."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Semantic.bert().\n"
                "Install with: pip install 'text4gcn[bert]'"
            ) from exc

        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model = BertModel.from_pretrained("bert-base-uncased")
    return _tokenizer, _model


def get_embeddings(words_batch):
    """CLS embedding of each word, encoded on its own (loads BERT on first call)."""
    import torch

    tokenizer, model = _get_bert()
    inputs = tokenizer(words_batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()


class BERTAdjacency():

    def __init__(self,
                 dataset_name,
                 dataset_path,
                 language: str = "english",
                 threshold: float = 0.95):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
        self.language = language
        self.threshold = threshold

    def build_relation_pair(self, docs_of_words, batch_size=256):
        """Count adjacent word pairs, within a document, whose BERT embeddings have
        cosine similarity >= threshold. Each vocabulary word is embedded once."""
        stop_words = set(stopwords.words(self.language))
        vocab = sorted({word for words in docs_of_words for word in words})
        index = {word: i for i, word in enumerate(vocab)}
        vectors = np.vstack([get_embeddings(vocab[start:start + batch_size])
                             for start in tqdm(range(0, len(vocab), batch_size))])
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

        rela_pair_count_str = {}
        for words in docs_of_words:
            for word_i, word_j in zip(words, words[1:]):
                if word_i == word_j or word_i in stop_words or word_j in stop_words:
                    continue
                if vectors[index[word_i]] @ vectors[index[word_j]] < self.threshold:
                    continue
                for pair in (f'{word_i},{word_j}', f'{word_j},{word_i}'):
                    rela_pair_count_str[pair] = rela_pair_count_str.get(pair, 0) + 1
        return rela_pair_count_str

    def compute_weights(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, vocab_size, word_doc_freq, vocab):
        row = []
        col = []

        # compute weights PMI
        weight = adjcy.compute_weights_with_PMI(docs_of_words,
                                                rela_pair_count_str,
                                                word_window_freq,
                                                train_size,
                                                word_id_map,
                                                row,
                                                col)

        # doc word frequency
        doc_word_freq = adjcy.get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = adjcy.get_weight_tfidf(docs_of_words,
                                              word_id_map,
                                              doc_word_freq,
                                              train_size,
                                              vocab_size,
                                              word_doc_freq,
                                              vocab,
                                              row,
                                              col)

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
        adj_path = f"{corpus_path}.adjacency/ind.embedding.bert.{self.dataset_name}.adj"
        with open(adj_path, 'wb') as f:
            pickle.dump(adj, f)

        stats = compute_adjacency_stats(
            adj,
            path=adj_path,
            builder="embedding.bert",
            dataset_name=self.dataset_name,
        )
        report_stats(stats, self.logger)

        hist = self.logger.log_history()
        self._save_history(hist)
        return stats
