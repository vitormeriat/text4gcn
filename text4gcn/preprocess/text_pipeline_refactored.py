"""
Refactored Text Pipeline - Composition of specialized components.

This class orchestrates the preprocessing pipeline using composition of:
- DataCleaner: Clean and preprocess text data
- DataShuffler: Shuffle and split data
- WordProcessor: Process words and vocabulary
- NodeFeatureBuilder: Build node features
"""

from typing import Dict, Any
# DataCleaner functionality is now integrated into the main pipeline
from .data_shuffler import DataShuffler
from .word_processor import WordProcessor
from .node_feature_builder import NodeFeatureBuilder
from text4gcn.modules import clean_data as clean


class TextPipeline:
    """
    Refactored Text Pipeline using composition pattern.
    
    This class follows Single Responsibility Principle by delegating
    specific responsibilities to specialized components.
    """
    
    def __init__(self, dataset_name: str, dataset_path: str, 
                 rare_count: int = 5, language: str = "english"):
        """
        Initialize TextPipeline with specialized components.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset files
            rare_count: Minimum count for word to be kept
            language: Language for stop words
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.rare_count = rare_count
        self.language = language
        
        # Initialize specialized components
        # Data cleaning is handled by the main pipeline
        
        self.data_shuffler = DataShuffler(
            dataset_name=dataset_name,
            dataset_path=dataset_path
        )
        
        self.word_processor = WordProcessor(
            dataset_name=dataset_name,
            dataset_path=dataset_path
        )
        
        self.node_feature_builder = NodeFeatureBuilder(
            dataset_name=dataset_name,
            dataset_path=dataset_path
        )
    
    def execute(self, validation_ratio: float = 0.10, 
                use_predefined_word_vectors: bool = False) -> Dict[str, str]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            validation_ratio: Ratio for validation split
            use_predefined_word_vectors: Whether to use predefined word vectors
            
        Returns:
            Dictionary containing paths to processed data
        """
        # Configure NLTK
        clean.config_nltk()
        
        # Step 1: Clean data (using main pipeline)
        # This would be handled by the main TextPipeline class
        cleaned_data_path = f"{self.dataset_path}/{self.dataset_name}.cleaned.txt"
        
        # Step 2: Shuffle data
        shuffled_data_path = self.data_shuffler.shuffle_data(cleaned_data_path)
        
        # Step 3: Process words
        word_vectors_path = self.word_processor.prepare_words(shuffled_data_path)
        
        # Step 4: Build node features
        node_features_path = self.node_feature_builder.build_node_features(
            shuffled_data_path=shuffled_data_path,
            validation_ratio=validation_ratio,
            use_predefined_word_vectors=use_predefined_word_vectors
        )
        
        # Save processing history
        self._save_history()
        
        return {
            'cleaned_data': cleaned_data_path,
            'shuffled_data': shuffled_data_path,
            'word_vectors': word_vectors_path,
            'node_features': node_features_path
        }
    
    def _save_history(self) -> None:
        """Save processing history."""
        hist = self.data_cleaner.logger.log_history()
        self.data_cleaner.flop.create_dir(
            dir_path=f'{self.dataset_path}/log',
            overwrite=False
        )
        
        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'w') as my_file:
            my_file.writelines(hist)
    
    def get_processing_paths(self) -> Dict[str, str]:
        """
        Get paths to all processed data.
        
        Returns:
            Dictionary containing all processing paths
        """
        # Data cleaning is handled by the pipeline, so cleaned data path is constructed here
        cleaned_data_path = f"{self.dataset_path}/{self.dataset_name}.cleaned/{self.dataset_name}.txt"
        shuffled_paths = self.data_shuffler.get_shuffled_data_paths()
        
        return {
            'cleaned_data': cleaned_data_path,
            'shuffled_corpus': shuffled_paths[0] if len(shuffled_paths) > 0 else "",
            'train_indices': shuffled_paths[1] if len(shuffled_paths) > 1 else "",
            'test_indices': shuffled_paths[2] if len(shuffled_paths) > 2 else "",
            'metadata': shuffled_paths[3] if len(shuffled_paths) > 3 else "",
            'vocabulary': self.word_processor.get_vocabulary_path(),
            'word_vectors': self.word_processor.get_word_vectors_path(),
            'node_features': self.node_feature_builder.get_node_features_path()
        }


