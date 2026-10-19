from text4gcn.builder._common import build_rwf_graph, run_build
from text4gcn.graph.stats import GraphBuildStats
from text4gcn.modules.adjacency import no_progress
from text4gcn.modules.textvec.word_features import train_word2vec
from text4gcn.utils.reporting import Reporter, check_verbose
from nltk.corpus import stopwords


class Word2VecAdjacency():

    def __init__(self,
                 dataset_name,
                 dataset_path,
                 training_regime,
                 embedding_dimension,
                 num_epochs,
                 language: str = "english",
                 threshold: float = 0.7,
                 seed: int = 1234,
                 workers: int = 1,
                 verbose: int = 1):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.training_regime = training_regime
        self.embedding_dimension = embedding_dimension
        self.num_epochs = num_epochs
        self.verbose = check_verbose(verbose)
        self.language = language
        self.threshold = threshold
        self.seed = seed
        self.workers = workers

    def build_relation_pair(self, docs_of_words, progress=no_progress):
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

        rela_pair_count_str = {}
        for docs_of_word in progress(docs_of_words, desc="word pairs", total=len(docs_of_words), unit="doc"):
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

        return rela_pair_count_str

    def build(self) -> GraphBuildStats:
        def build(report: Reporter) -> GraphBuildStats:
            return build_rwf_graph(self.dataset_path, self.dataset_name,
                                   lambda docs: self.build_relation_pair(docs, report.progress),
                                   "embedding.word2vec")

        algorithm = "skip-gram" if self.training_regime == 1 else "CBOW"
        return run_build(build, f'Semantic.word2vec("{self.dataset_name}") · {algorithm} · '
                         f'{self.embedding_dimension} dims · {self.num_epochs} epochs · threshold {self.threshold}',
                         self.verbose, self.dataset_path, self.dataset_name)
