from os.path import isabs
from os import getcwd


class PreProcessingConfig():

    def __init__(self):

        self.dataset_name = None
        self.rare_count = None
        self.dataset_path = None
        self.language = None


        # # List of Valid Data-sets
        # self.data_sets = None
        # # List of Valid Adjacency-sets
        # self.adjacency_sets = None
        # # Extension of data-sets, e.g. "txt"
        # self.data_set_extension = None
        # # Original Corpus Directory
        # self.corpus_dir = None
        # # Cleaned Corpus Directory
        # self.corpus_cleaned_dir = None
        # # Original Meta Directory of Corpus
        # self.corpus_meta_dir = None
        # # Shuffled Corpus Directory
        # self.corpus_shuffled_dir = None
        # # Train and Test Index of Shuffled Corpus
        # self.corpus_shuffled_split_index_dir = None
        # # Meta Directory of Shuffled Corpus
        # self.corpus_shuffled_meta_dir = None
        # # Vocabulary of Shuffled Corpus
        # self.corpus_shuffled_vocab_dir = None
        # # Word-Vectors of Shuffled Corpus
        # self.corpus_shuffled_word_vectors_dir = None
        # # Node Features (x,y,tx,ty,allx) of Shuffled Corpus
        # self.corpus_shuffled_node_features_dir = None
        # # Adjacency Matrix (adj) of Shuffled Corpus
        # self.corpus_shuffled_adjacency_dir = None
        # self.core_nlp_path = None
        # self.liwc_path = None

    

    # def make_path_absolute(self, a_path: str) -> str:

    #     if isabs(a_path):
    #         print(f'[WARN] Path:{a_path} is already absolute.')
    #         return a_path

    #     current_working_dir = getcwd()

    #     return f'{current_working_dir}/{a_path}'

    # def build(self) -> 'PreProcessingConfig':

    #     self.corpus_dir = self.make_path_absolute(
    #         self.corpus_dir)

    #     self.corpus_cleaned_dir = self.make_path_absolute(
    #         self.corpus_cleaned_dir)

    #     self.corpus_meta_dir = self.make_path_absolute(
    #         self.corpus_meta_dir)

    #     self.corpus_shuffled_dir = self.make_path_absolute(
    #         self.corpus_shuffled_dir)

    #     self.corpus_shuffled_split_index_dir = self.make_path_absolute(
    #         self.corpus_shuffled_split_index_dir)

    #     self.corpus_shuffled_meta_dir = self.make_path_absolute(
    #         self.corpus_shuffled_meta_dir)

    #     self.corpus_shuffled_vocab_dir = self.make_path_absolute(
    #         self.corpus_shuffled_vocab_dir)

    #     self.corpus_shuffled_word_vectors_dir = self.make_path_absolute(
    #         self.corpus_shuffled_word_vectors_dir)

    #     self.corpus_shuffled_node_features_dir = self.make_path_absolute(
    #         self.corpus_shuffled_node_features_dir)

    #     self.corpus_shuffled_adjacency_dir = self.make_path_absolute(
    #         self.corpus_shuffled_adjacency_dir)

    #     self.core_nlp_path = self.make_path_absolute(
    #         self.core_nlp_path)

    #     return self
