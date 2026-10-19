from text4gcn.builder._common import build_rwf_graph, run_build
from text4gcn.graph.stats import GraphBuildStats
from text4gcn.modules.adjacency import no_progress
from text4gcn.modules.textvec.liwc import Liwc
from text4gcn.utils.reporting import Reporter, check_verbose
from nltk.corpus import stopwords


class LiwcAdjacency():

    def __init__(self, dataset_name, dataset_path, liwc_path, verbose: int = 1):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.liwc_path = liwc_path
        self.verbose = check_verbose(verbose)

    def build_relation_pair(self, docs_of_words, progress=no_progress):
        """Count adjacent word pairs that share at least one LIWC category."""
        stop_words = set(stopwords.words('english'))
        liwc = Liwc(self.liwc_path)

        rela_pair_count_str = {}
        for docs_of_word in progress(docs_of_words, desc="word pairs", total=len(docs_of_words), unit="doc"):
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

    def build(self) -> GraphBuildStats:
        def build(report: Reporter) -> GraphBuildStats:
            return build_rwf_graph(self.dataset_path, self.dataset_name,
                                   lambda docs: self.build_relation_pair(docs, report.progress), "liwc")

        return run_build(build, f'Syntactic.linguistic_inquiry("{self.dataset_name}")',
                         self.verbose, self.dataset_path, self.dataset_name)
