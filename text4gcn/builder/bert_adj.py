from text4gcn.builder._common import build_rwf_graph, run_build
from text4gcn.graph.stats import GraphBuildStats
from text4gcn.modules.adjacency import no_progress
from text4gcn.utils.reporting import Reporter, check_verbose
from nltk.corpus import stopwords
import numpy as np

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
                 threshold: float = 0.95,
                 verbose: int = 1):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.verbose = check_verbose(verbose)
        self.language = language
        self.threshold = threshold

    def build_relation_pair(self, docs_of_words, progress=no_progress, batch_size=256):
        """Count adjacent word pairs, within a document, whose BERT embeddings have
        cosine similarity >= threshold. Each vocabulary word is embedded once."""
        stop_words = set(stopwords.words(self.language))
        vocab = sorted({word for words in docs_of_words for word in words})
        index = {word: i for i, word in enumerate(vocab)}
        starts = range(0, len(vocab), batch_size)
        vectors = np.vstack([get_embeddings(vocab[start:start + batch_size])
                             for start in progress(starts, desc="BERT", total=len(starts), unit="batch")])
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

    def build(self) -> GraphBuildStats:
        def build(report: Reporter) -> GraphBuildStats:
            return build_rwf_graph(self.dataset_path, self.dataset_name,
                                   lambda docs: self.build_relation_pair(docs, report.progress),
                                   "embedding.bert")

        return run_build(build, f'Semantic.bert("{self.dataset_name}") · threshold {self.threshold}',
                         self.verbose, self.dataset_path, self.dataset_name)
