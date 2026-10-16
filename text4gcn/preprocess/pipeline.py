from text4gcn.modules.node_features import NodeFeatures
from text4gcn.modules import word_processor as wdprc
from text4gcn.modules import clean_data as clean
from text4gcn.modules import file_ops as flop
from text4gcn.utils.logging import get_logger
from text4gcn.utils.reporting import Reporter, check_verbose
from collections import OrderedDict
from math import ceil
import random
import os

logger = get_logger(__name__)


class TextPipeline():
    """
    Create a folder structure to store the files representing the
    processed text and the node resources to be used to create the
    text graph.

    ...

    Attributes
    ----------
    dataset_name : str
        name of the dataset
    rare_count : int
        sets a value to consider a `rare` term
    dataset_path : str
        path to files
    language : str
        `language` to retrieve stop words
    seed : int
        Random seed for train/test document shuffle (reproducibility).
    encoding : str
        Encoding of the input `{dataset_name}.txt` corpus (default ``"utf-8"``).
        Use ``"latin1"`` for legacy corpora that are not valid UTF-8.
    ascii_only : bool
        If True (default), cleaning keeps only ASCII letters and digits, as in
        the original TextGCN preprocessing. Set to False for corpora in other
        scripts (e.g. Arabic, Cyrillic, accented Latin) so their words are kept.
    verbose : int
        ``0`` silent, ``1`` (default) or ``2``: one summary line per stage.

    Methods
    -------
    execute(additional=""):
        Run the pipeline
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | os.PathLike[str],
        rare_count: int = 5,
        language: str = "english",
        seed: int = 1234,
        encoding: str = "utf-8",
        ascii_only: bool = True,
        verbose: int = 1,
    ):
        from pathlib import Path

        self.flop = flop.FileOps()
        self.verbose = check_verbose(verbose)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.rare_count = rare_count
        self.language = language
        self.seed = seed
        self.encoding = encoding
        self.ascii_only = ascii_only

        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name must be a non-empty string.")
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(
                f"dataset_path does not exist: {dataset_path}\n\n"
                "Create the directory and place "
                f"'{dataset_name}.txt' / '{dataset_name}.meta' before running execute()."
            )

    def clean_data(self) -> str:

        corpus_path = f"{self.dataset_path}/{self.dataset_name}.txt"
        corpus_cleaned = f"{self.dataset_path}/{self.dataset_name}.cleaned"

        # Checkers
        # flop.check_data_set(data_set_name=self.dataset_name, all_data_set_names=self.config.data_sets)
        self.flop.check_paths(self.dataset_path)

        self.flop.create_dir(
            dir_path=corpus_cleaned,
            overwrite=False)

        try:
            with open(corpus_path, 'r', encoding=self.encoding) as corpus:
                docs_of_words = [
                    clean.clean_str(line.strip(), ascii_only=self.ascii_only).split()
                    for line in corpus]
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding, e.object, e.start, e.end,
                f"{e.reason}. '{corpus_path}' is not valid {self.encoding}; "
                "pass TextPipeline(encoding=...) with the corpus encoding, "
                "e.g. encoding='latin1'") from e

        word_counts = clean.extract_word_counts(docs_of_words=docs_of_words)

        stop_words = clean.retrieve_stop_words(language=self.language)

        docs_of_words = clean.remove_stop_words(
            docs_of_words,
            stop_words=stop_words
        )

        docs_of_words = clean.remove_rare_words(
            docs_of_words,
            word_counts=word_counts,
            rare_count=self.rare_count
        )

        n_words = len({word for words in docs_of_words for word in words})
        docs_of_words = clean.glue_lines(
            lines_of_words=docs_of_words,
            glue_str=' ',
            with_strip=True
        )

        self.flop.write_iterable_to_file(
            an_iterable=docs_of_words,
            file_path=f'{corpus_cleaned}/{self.dataset_name}.txt',
            file_mode='w'
        )

        return f"{len(docs_of_words):,} documents · {n_words:,} words kept (rare_count={self.rare_count})"

    def shuffle_data(self) -> str:

        ds_corpus = f"{self.dataset_path}/{self.dataset_name}.cleaned/{self.dataset_name}.txt"
        ds_corpus_meta = f"{self.dataset_path}/{self.dataset_name}.meta"

        ds_corpus_shuffled = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"
        ds_corpus_shuffled_train_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.train"
        ds_corpus_shuffled_test_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.test"
        ds_corpus_shuffled_meta = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.meta"

        # Checkers
        # check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus_meta, corpus_cleaned)

        # Create dirs if not exist
        self.flop.create_dir(
            dir_path=f"{self.dataset_path}/{self.dataset_name}.shuffled/",
            overwrite=False
        )

        all_doc_meta_list, train_doc_meta_list, test_doc_meta_list = self.flop.load_corpus_meta(
            corpus_meta_path=ds_corpus_meta
        )

        cleaned_doc_lines = [line.strip() for line in open(ds_corpus, 'r', encoding="utf-8")]

        random.seed(self.seed)

        # Shuffle train ids and write to file
        train_doc_meta_ids = [all_doc_meta_list.index(
            train_doc_meta) for train_doc_meta in train_doc_meta_list]

        random.shuffle(train_doc_meta_ids)

        self.flop.write_iterable_to_file(
            an_iterable=train_doc_meta_ids,
            file_path=ds_corpus_shuffled_train_idx,
            file_mode='w'
        )

        # Shuffle test ids and write to file
        test_doc_meta_ids = [all_doc_meta_list.index(
            test_doc_meta) for test_doc_meta in test_doc_meta_list]

        random.shuffle(test_doc_meta_ids)

        self.flop.write_iterable_to_file(
            an_iterable=test_doc_meta_ids,
            file_path=ds_corpus_shuffled_test_idx,
            file_mode='w'
        )

        all_doc_meta_ids = train_doc_meta_ids + test_doc_meta_ids
        # Write shuffled meta to file
        shuffled_doc_meta_list = [all_doc_meta_list[all_doc_meta_id]
                                  for all_doc_meta_id in all_doc_meta_ids]

        self.flop.write_iterable_to_file(
            an_iterable=shuffled_doc_meta_list,
            file_path=ds_corpus_shuffled_meta,
            file_mode='w'
        )

        # Write shuffled document files to file
        shuffled_doc_lines = [cleaned_doc_lines[all_doc_meta_id]
                              for all_doc_meta_id in all_doc_meta_ids]

        self.flop.write_iterable_to_file(
            an_iterable=shuffled_doc_lines,
            file_path=ds_corpus_shuffled,
            file_mode='w'
        )
        return f"train {len(train_doc_meta_ids):,} · test {len(test_doc_meta_ids):,} documents (seed {self.seed})"

    def prepare_words(self) -> str:

        ds_corpus = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"

        # Checkers
        # check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus)
        self.flop.check_paths(self.dataset_path)

        # Create output directories
        # create_dir(dir_path=cfg.corpus_shuffled_vocab_dir, overwrite=False)
        # create_dir(dir_path=cfg.corpus_shuffled_word_vectors_dir, overwrite=False)

        ds_corpus_vocabulary = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.vocab"
        ds_corpus_word_vectors = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.word_vectors"

        # ###################################################

        # Build vocabulary
        docs_of_words_generator = (line.split() for line in open(ds_corpus, encoding="utf-8"))

        vocabulary = wdprc.extract_vocabulary(
            docs_of_words=docs_of_words_generator
        )

        self.flop.write_iterable_to_file(
            an_iterable=vocabulary,
            file_path=ds_corpus_vocabulary,
            file_mode='w'
        )

        # # Extract word definitions
        word_definitions = wdprc.extract_word_definitions(
            vocabulary=vocabulary
        )

        # # Extract & Dump word vectors
        word_vectors = wdprc.extract_tf_idf_word_vectors(
            word_definitions=word_definitions, max_features=1000
        )

        word_to_word_vectors_dict = wdprc.word_to_vectors(
            vocabulary, word_vectors
        )

        self.flop.write_picke(
            word_to_word_vectors_dict,
            ds_corpus_word_vectors
        )
        n_defined = sum(1 for definition in word_definitions if definition)
        return f"{len(vocabulary):,} words · {n_defined:,} with a WordNet definition"

    def build_node_features(self, validation_ratio: float) -> str:

        ds_corpus = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"
        ds_corpus_meta = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.meta"
        ds_corpus_vocabulary = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.vocab"
        ds_corpus_train_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.train"
        ds_corpus_test_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.test"

        # output directory of node features
        dir_corpus_node_features = f"{self.dataset_path}/{self.dataset_name}.node_features"

        # checkers
        self.flop.check_paths(ds_corpus, ds_corpus_meta, ds_corpus_vocabulary)
        self.flop.check_paths(ds_corpus_train_idx, ds_corpus_train_idx)

        # Create output directory of node features
        self.flop.create_dir(
            dir_path=dir_corpus_node_features,
            overwrite=False
        )

        # Adjust train size, for different training rates, for example: use 90% of training set
        real_train_size = len(open(ds_corpus_train_idx, encoding="utf-8").readlines())
        adjusted_train_size = ceil(real_train_size * (1.0 - validation_ratio))
        test_size = len(open(ds_corpus_test_idx, encoding="utf-8").readlines())

        nfeatures = NodeFeatures()

        word_vectors, word_emb_dim = OrderedDict(), 300

        # Extract Vocabulary
        vocabulary = open(ds_corpus_vocabulary, encoding="utf-8").read().splitlines()
        # Extract Meta List
        doc_meta_list = open(file=ds_corpus_meta, mode='r', encoding="utf-8").read().splitlines()
        # Extract Document Labels
        doc_labels = nfeatures.extract_doc_labels(
            ds_corpus_meta_file=ds_corpus_meta)
        # Extract Documents of Words
        docs_of_words = [line.split() for line in open(file=ds_corpus, encoding="utf-8")]

        # Extract mean document word vectors and one hot labels of train-set
        # The feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        x = nfeatures.compute_x(
            docs_of_words,
            adjusted_train_size,
            word_emb_dim,
            w_vectors=word_vectors
        )

        # Extract mean document word vectors and one hot labels of test-set
        # The one-hot labels of the labeled training instances as numpy.ndarray object;
        y = nfeatures.compute_y(
            doc_meta_list,
            train_size=adjusted_train_size,
            doc_labels=doc_labels
        )

        # The feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        tx = nfeatures.compute_tx(
            docs_of_words,
            test_size,
            real_train_size,
            word_emb_dim,
            w_vectors=word_vectors
        )

        # The one-hot labels of the test instances as numpy.ndarray object;
        ty = nfeatures.compute_ty(
            doc_meta_list,
            test_size=test_size,
            real_train_size=real_train_size,
            doc_labels=doc_labels
        )

        # Extract doc_features + word_features
        # The feature vectors of both labeled and unlabeled training instances (a
        # superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        allx = nfeatures.compute_allx(
            docs_of_words,
            real_train_size,
            vocabulary,
            word_vectors,
            emb_dim=word_emb_dim
        )

        # The labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ally = nfeatures.compute_ally(
            doc_meta_list,
            real_train_size,
            doc_labels,
            vocab_size=len(vocabulary)
        )

        # Dump node features matrices to files
        node_feature_matrices = {
            "x": x,
            "y": y,
            "tx": tx,
            "ty": ty,
            "allx": allx,
            "ally": ally
        }

        nfeatures.dump_node_features(
            directory=dir_corpus_node_features,
            ds=self.dataset_name,
            node_features_dict=node_feature_matrices
        )

        logger.debug("x %s, y %s, tx %s, ty %s, allx %s, ally %s",
                     x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
        val_size = real_train_size - adjusted_train_size
        return (f"train {adjusted_train_size:,} · validation {val_size:,} · test {test_size:,} documents "
                f"· {len(doc_labels)} classes")

    def execute(
        self,
        validation_ratio: float = 0.10,
        seed: int | None = None,
    ) -> None:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        validation_ratio : float
            Fraction of training data reserved for validation. This is the
            **only** place the train/val split is defined (DCAI 2022 / Yao et
            al. use ``0.10``). Default ``0.10``.
        seed : int or None
            If provided, overrides :attr:`seed` for this run (document shuffle).
        """
        if seed is not None:
            self.seed = seed

        report = Reporter(self.verbose)
        report.header(f'TextPipeline("{self.dataset_name}") · language {self.language} '
                      f'· validation {validation_ratio:.0%}')
        clean.config_nltk()

        with report.step("clean") as step:
            step.summary = self.clean_data()
        with report.step("shuffle") as step:
            step.summary = self.shuffle_data()
        with report.step("words") as step:
            step.summary = self.prepare_words()
        with report.step("features") as step:
            step.summary = self.build_node_features(validation_ratio=validation_ratio)

        report.save(f"{self.dataset_path}/log/{self.dataset_name}_dataset.txt", mode="w")
