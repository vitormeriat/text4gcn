"""
Word Processor - Single Responsibility: Process words and vocabulary.

This class is responsible for:
- Extracting vocabulary
- Processing word definitions
- Creating word vectors
- Managing word embeddings
"""

from typing import List, Dict, Any
from text4gcn.modules import word_processor as wdprc
from text4gcn.modules import file_ops as flop
from text4gcn.modules import logger as logger
from text4gcn.modules.logger import Process


class WordProcessor:
    """Responsible for word processing and vocabulary management."""
    
    def __init__(self, dataset_name: str, dataset_path: str):
        """
        Initialize WordProcessor.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset files
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        
        # Initialize dependencies
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
    
    @Process.log("PREPARED WORDS: Vocabulary & word-vectors extracted.")
    def prepare_words(self, shuffled_data_path: str) -> str:
        """
        Extract vocabulary and create word vectors.
        
        Args:
            shuffled_data_path: Path to shuffled data
            
        Returns:
            Path to word vectors file
        """
        ds_corpus = f"{shuffled_data_path}/{self.dataset_name}.txt"
        ds_corpus_vocabulary = f"{shuffled_data_path}/{self.dataset_name}.vocab"
        ds_corpus_word_vectors = f"{shuffled_data_path}/{self.dataset_name}.word_vectors"
        
        # Check paths
        self.flop.check_paths(self.dataset_path)
        
        # Build vocabulary
        docs_of_words_generator = (line.split() for line in open(ds_corpus))
        
        vocabulary = wdprc.extract_vocabulary(docs_of_words=docs_of_words_generator)
        
        # Write vocabulary
        self.flop.write_iterable_to_file(
            an_iterable=vocabulary,
            file_path=ds_corpus_vocabulary,
            file_mode='w'
        )
        
        # Extract word definitions
        word_definitions = wdprc.extract_word_definitions(vocabulary=vocabulary)
        
        # Extract word vectors
        word_vectors = wdprc.extract_tf_idf_word_vectors(
            word_definitions=word_definitions, 
            max_features=1000
        )
        
        # Create word to vectors mapping
        word_to_word_vectors_dict = wdprc.word_to_vectors(vocabulary, word_vectors)
        
        # Save word vectors
        self.flop.write_picke(word_to_word_vectors_dict, ds_corpus_word_vectors)
        
        return ds_corpus_word_vectors
    
    def get_vocabulary_path(self) -> str:
        """
        Get path to vocabulary file.
        
        Returns:
            Path to vocabulary file
        """
        return f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.vocab"
    
    def get_word_vectors_path(self) -> str:
        """
        Get path to word vectors file.
        
        Returns:
            Path to word vectors file
        """
        return f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.word_vectors"


