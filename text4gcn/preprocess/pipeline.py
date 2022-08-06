# from ..modules.node_features import NodeFeatures
# from ..modules import word_processor as wdprc
# from ..modules import clean_data as clean
# from ..modules import file_ops as flop
# from ..modules import logger as logger
# from ..modules.logger import Process
# from collections import OrderedDict
# from math import ceil
# import random

from text4gcn.modules.node_features import NodeFeatures
from text4gcn.modules import word_processor as wdprc
from text4gcn.modules import clean_data as clean
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process
from collections import OrderedDict
from math import ceil
import random


class TextPipeline():
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(
        self,
        dataset_name: str,
        rare_count: int,
        dataset_path: str,
        language: str
    ):
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.rare_count = rare_count
        self.language = language

    def save_history(self, hist):
        self.flop.create_dir(
            dir_path=f'{self.dataset_path}/log',
            overwrite=False
        )

        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'w') as my_file:
            my_file.writelines(hist)

    def execute(self):
        self.logger.info(clean.create_title("NLTK Configuration"))
        clean.config_nltk()
        self.logger.info()

        self.clean_data()
        self.shuffle_data()
        self.prepare_words()
        self.build_node_features(
            validation_ratio=0.10,
            use_predefined_word_vectors=False
        )

        hist = self.logger.log_history()
        self.save_history(hist)

    @Process.log("CLEANED DATA: Removed rare & stop-words.")
    def clean_data(self):

        corpus_path = f"{self.dataset_path}/{self.dataset_name}.txt"
        corpus_cleaned = f"{self.dataset_path}/{self.dataset_name}.cleaned"

        # Checkers
        # flop.check_data_set(data_set_name=self.dataset_name, all_data_set_names=self.config.data_sets)
        self.flop.check_paths(self.dataset_path)

        self.flop.create_dir(
            dir_path=corpus_cleaned,
            overwrite=False)

        docs_of_words = [clean.clean_str(line.strip().decode(
            'latin1')).split() for line in open(corpus_path, 'rb')]

        word_counts = clean.extract_word_counts(docs_of_words=docs_of_words)

        stop_words = clean.retrieve_stop_words(language=self.language)

        # TODO: IF
        docs_of_words = clean.remove_stop_words(
            docs_of_words,
            stop_words=stop_words
        )

        # TODO: IF
        docs_of_words = clean.remove_rare_words(
            docs_of_words,
            word_counts=word_counts,
            rare_count=self.rare_count
        )

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

        self.logger.info(f"Rare-Count = <{self.rare_count}>")

    @Process.log("SHUFFLED DATA: Corpus documents shuffled.")
    def shuffle_data(self):

        ds_corpus = f"{self.dataset_path}/{self.dataset_name}.cleaned/{self.dataset_name}.txt"
        ds_corpus_meta = f"{self.dataset_path}/{self.dataset_name}.meta"

        ds_corpus_shuffled = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"
        ds_corpus_shuffled_train_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.train"
        ds_corpus_shuffled_test_idx = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.test"
        ds_corpus_shuffled_meta = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.meta"

        # Checkers
        #check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        #check_paths(ds_corpus_meta, corpus_cleaned)

        # Create dirs if not exist
        self.flop.create_dir(
            dir_path=f"{self.dataset_path}/{self.dataset_name}.shuffled/",
            overwrite=False
        )

        all_doc_meta_list, train_doc_meta_list, test_doc_meta_list = self.flop.load_corpus_meta(
            corpus_meta_path=ds_corpus_meta
        )

        cleaned_doc_lines = [line.strip() for line in open(ds_corpus, 'r')]

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

    @Process.log("PREPARED WORDS: Vocabulary & word-vectors extracted.")
    def prepare_words(self):

        ds_corpus = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.txt"

        # Checkers
        #check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus)
        self.flop.check_paths(self.dataset_path)

        # Create output directories
        #create_dir(dir_path=cfg.corpus_shuffled_vocab_dir, overwrite=False)
        #create_dir(dir_path=cfg.corpus_shuffled_word_vectors_dir, overwrite=False)

        ds_corpus_vocabulary = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.vocab"
        ds_corpus_word_vectors = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.word_vectors"

        # ###################################################

        # Build vocabulary
        docs_of_words_generator = (line.split() for line in open(ds_corpus))

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

    @Process.log("EXTRACTED NODE FEATURES: x, y, tx, ty, allx, ally.")
    def build_node_features(self, validation_ratio: float, use_predefined_word_vectors: bool):

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
        real_train_size = len(open(ds_corpus_train_idx).readlines())
        adjusted_train_size = ceil(real_train_size * (1.0 - validation_ratio))
        test_size = len(open(ds_corpus_test_idx).readlines())

        nfeatures = NodeFeatures(logger=self.logger)

        # =============================================================== TESTE
        # # Extract word_vectors and word_embedding_dimension
        # if use_predefined_word_vectors:
        #     #ds_corpus_word_vectors = cfg.corpus_shuffled_word_vectors_dir + ds_name + '.word_vectors'
        #     ds_corpus_word_vectors = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.word_vectors"
        #     # ds_corpus_word_vectors =  'glove.6B.300d.txt'  # Alternatively, you can use GLOVE word-embeddings
        #     word_vectors, word_emb_dim = nfeatures.load_word_to_word_vectors(
        #         path=ds_corpus_word_vectors)
        # else:
        #     word_vectors, word_emb_dim = OrderedDict(), 300
        # ===============================================================

        # TODO: parametrize
        word_vectors, word_emb_dim = OrderedDict(), 300

        # Extract Vocabulary
        vocabulary = open(ds_corpus_vocabulary).read().splitlines()
        # Extract Meta List
        doc_meta_list = open(file=ds_corpus_meta, mode='r').read().splitlines()
        # Extract Document Labels
        doc_labels = nfeatures.extract_doc_labels(
            ds_corpus_meta_file=ds_corpus_meta)
        # Extract Documents of Words
        docs_of_words = [line.split() for line in open(file=ds_corpus)]

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

        self.logger.info(f"x.shape={x.shape}, y.shape={y.shape}")
        self.logger.info(f"tx.shape={tx.shape}, ty.shape={ty.shape}")
        self.logger.info(f"allx.shape={allx.shape}, ally.shape={ally.shape}")
