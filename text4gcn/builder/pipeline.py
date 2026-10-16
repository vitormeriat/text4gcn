from __future__ import annotations

import os

from text4gcn.builder.cosine_similarity_adj import CosineSimilarityAdjacency
from text4gcn.builder.dependency_parsing_adj import DependencyParsingAdjacency
from text4gcn.builder.word2vec_adj import Word2VecAdjacency
from text4gcn.builder.frequency_adj import FrequencyAdjacency
from text4gcn.builder.linguistic_inquiry_adj import LiwcAdjacency
from text4gcn.graph.stats import GraphBuildStats


class Adjacency:
    def __init__(self, dataset_name: str, dataset_path: str | os.PathLike[str]) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path


class Frequency(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str | os.PathLike[str]) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def tfidf(self, window_size: int = 20) -> GraphBuildStats:
        """Build TF-IDF + PMI adjacency (Yao / Frequency baseline)."""
        adj = FrequencyAdjacency(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            window_size=window_size,
        )
        return adj.build()


class Syntactic(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str | os.PathLike[str]) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def dependency_parsing(self, core_nlp_path: str, language: str = "english") -> GraphBuildStats:
        """Build adjacency via Stanford CoreNLP dependency pairs (RWF syntactic).

        Requires ``pip install text4gcn[syntactic]`` and a local CoreNLP jar path.
        """
        adj = DependencyParsingAdjacency(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            core_nlp_path=core_nlp_path,
            language=language,
        )
        return adj.build()

    def linguistic_inquiry(self, liwc_path: str) -> GraphBuildStats:
        """Build adjacency via LIWC lexicon pairs (RWF linguistic)."""
        adj = LiwcAdjacency(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            liwc_path=liwc_path,
        )
        return adj.build()


class Semantic(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str | os.PathLike[str]) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def word2vec(
        self,
        training_algorithm: int,
        embedding_dimension: int,
        num_epochs: int,
        language: str = "english",
        threshold: float = 0.7,
        seed: int = 1234,
        workers: int = 1,
    ) -> GraphBuildStats:
        """Build Word2Vec RWF adjacency (DCAI 2022 semantic filter).

        The default ``threshold`` (0.7) was calibrated on 20 Newsgroups with the
        paper's 20 epochs; the paper's 0.95 leaves almost no word–word edges once
        the epochs are actually applied. ``seed`` fixes the Word2Vec initialisation. With ``workers=1`` (default)
        the graph is reproducible; more workers train faster but vary between runs.
        """
        adj = Word2VecAdjacency(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            training_regime=training_algorithm,
            embedding_dimension=embedding_dimension,
            num_epochs=num_epochs,
            language=language,
            threshold=threshold,
            seed=seed,
            workers=workers,
        )
        return adj.build()

    def bert(
        self, language: str = "english", threshold: float = 0.95
    ) -> GraphBuildStats:
        """Build adjacency from BERT pairwise similarities (edge construction).

        Lazily imports ``BERTAdjacency`` so HuggingFace models are only loaded
        when this method is called.
        """
        from text4gcn.builder.bert_adj import BERTAdjacency

        adj = BERTAdjacency(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            language=language,
            threshold=threshold,
        )
        return adj.build()

    def cosine_similarity(self, threshold: float = 0.95) -> GraphBuildStats:
        """Build adjacency from precomputed word-vector cosine similarity."""
        adj = CosineSimilarityAdjacency(
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            threshold=threshold,
        )
        return adj.build()
