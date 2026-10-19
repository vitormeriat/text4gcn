from text4gcn.builder._common import build_rwf_graph, run_build
from text4gcn.graph.stats import GraphBuildStats
from text4gcn.modules.adjacency import no_progress
from text4gcn.utils.logging import get_logger
from text4gcn.utils.reporting import Reporter, check_verbose
from nltk.corpus import stopwords

logger = get_logger(__name__)


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

    def __init__(self, dataset_name, dataset_path, core_nlp_path, language: str = 'english', verbose: int = 1):
        self.language = language
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.core_nlp_path = core_nlp_path
        self.verbose = check_verbose(verbose)

    def build_relation_pair(self, docs_of_words, progress=no_progress):
        StanfordCoreNLP = _import_stanford_corenlp()
        stop_words = set(stopwords.words(self.language))
        nlp = StanfordCoreNLP(self.core_nlp_path, lang='en')

        errors = 0
        rela_pair_count_str = {}
        for docs_of_word in progress(docs_of_words, desc="parsing", total=len(docs_of_words), unit="doc"):
            words = docs_of_word
            sentence = ' '.join(words)

            try:
                res = nlp.dependency_parse(sentence)
                tokenized = nlp.word_tokenize(sentence)
            except Exception:
                errors += 1
                res = []

            # Each relation is (label, governor index, dependent index); indices are
            # 1-based into the tokens and governor 0 is the ROOT.
            for relation, governor, dependent in res:
                if relation == 'ROOT' or governor == 0:
                    continue
                word_i, word_j = tokenized[governor - 1], tokenized[dependent - 1]
                if word_i == word_j or word_i in stop_words or word_j in stop_words:
                    continue
                for pair in (f'{word_i},{word_j}', f'{word_j},{word_i}'):
                    rela_pair_count_str[pair] = rela_pair_count_str.get(pair, 0) + 1
        if errors:
            logger.warning("CoreNLP could not parse %d documents; they get no word-word edges.", errors)
        nlp.close()
        return rela_pair_count_str

    def build(self) -> GraphBuildStats:
        def build(report: Reporter) -> GraphBuildStats:
            return build_rwf_graph(self.dataset_path, self.dataset_name,
                                   lambda docs: self.build_relation_pair(docs, report.progress),
                                   "dependency.parsing")

        return run_build(build, f'Syntactic.dependency_parsing("{self.dataset_name}")',
                         self.verbose, self.dataset_path, self.dataset_name)
