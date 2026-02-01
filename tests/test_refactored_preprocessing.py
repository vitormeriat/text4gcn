"""
Unit tests for refactored preprocessing components following SOLID principles.

This module contains comprehensive unit tests for the refactored
preprocessing components to ensure they work correctly and maintain
the same functionality as the original implementation.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import refactored components
from text4gcn.preprocess.data_shuffler import DataShuffler
from text4gcn.preprocess.word_processor import WordProcessor
from text4gcn.preprocess.node_feature_builder import NodeFeatureBuilder
from text4gcn.preprocess.text_pipeline_refactored import TextPipeline

# Note: DataCleaner import is missing - this test may need to be updated
# when DataCleaner is implemented as a separate class or if it's integrated elsewhere
try:
    from text4gcn.preprocess.data_cleaner import DataCleaner
except ImportError:
    # If DataCleaner doesn't exist, we'll skip these tests
    DataCleaner = None


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Skip tests if DataCleaner is not available
        if DataCleaner is None:
            pytest.skip("DataCleaner class not available")
        
        self.dataset_name = "test_dataset"
        self.dataset_path = tempfile.mkdtemp()
        self.rare_count = 5
        self.language = "english"
        
        # Initialize DataCleaner with test parameters
        self.cleaner = DataCleaner(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            rare_count=self.rare_count,
            language=self.language
        )
    
    def test_initialization(self):
        """Test DataCleaner initialization."""
        # Verify all initialization parameters are set correctly
        assert self.cleaner.dataset_name == self.dataset_name
        assert self.cleaner.dataset_path == self.dataset_path
        assert self.cleaner.rare_count == self.rare_count
        assert self.cleaner.language == self.language
        # Verify that logger and file operations are initialized
        assert self.cleaner.logger is not None
        assert self.cleaner.flop is not None
    
    @patch('text4gcn.modules.clean_data.clean_str')
    @patch('text4gcn.modules.clean_data.extract_word_counts')
    @patch('text4gcn.modules.clean_data.retrieve_stop_words')
    @patch('text4gcn.modules.clean_data.remove_stop_words')
    @patch('text4gcn.modules.clean_data.remove_rare_words')
    @patch('text4gcn.modules.clean_data.glue_lines')
    def test_clean_data(self, mock_glue, mock_remove_rare, mock_remove_stop,
                       mock_retrieve_stop, mock_extract_counts, mock_clean_str):
        """Test data cleaning process."""
        # Create mock input file
        input_file = os.path.join(self.dataset_path, f"{self.dataset_name}.txt")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, 'w') as f:
            f.write("test document 1\n")
            f.write("test document 2\n")
        
        # Mock dependencies
        mock_clean_str.return_value = "cleaned text"
        mock_extract_counts.return_value = {"word": 10}
        mock_retrieve_stop.return_value = {"the", "a", "an"}
        mock_remove_stop.return_value = [["cleaned", "text"]]
        mock_remove_rare.return_value = [["cleaned", "text"]]
        mock_glue.return_value = ["cleaned text"]
        
        # Mock file operations
        with patch.object(self.cleaner.flop, 'check_paths'), \
             patch.object(self.cleaner.flop, 'create_dir'), \
             patch.object(self.cleaner.flop, 'write_iterable_to_file'):
            
            result = self.cleaner.clean_data()
        
        assert result == f"{self.dataset_path}/{self.dataset_name}.cleaned"
    
    def test_get_cleaned_data_path(self):
        """Test getting cleaned data path."""
        # Get the path where cleaned data should be stored
        path = self.cleaner.get_cleaned_data_path()
        # Verify the path follows the expected structure
        expected = f"{self.dataset_path}/{self.dataset_name}.cleaned/{self.dataset_name}.txt"
        assert path == expected


class TestDataShuffler:
    """Test cases for DataShuffler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_name = "test_dataset"
        self.dataset_path = tempfile.mkdtemp()
        
        self.shuffler = DataShuffler(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path
        )
    
    def test_initialization(self):
        """Test DataShuffler initialization."""
        assert self.shuffler.dataset_name == self.dataset_name
        assert self.shuffler.dataset_path == self.dataset_path
        assert self.shuffler.logger is not None
        assert self.shuffler.flop is not None
    
    @patch('text4gcn.modules.file_ops.FileOps.load_corpus_meta')
    @patch('text4gcn.modules.file_ops.FileOps.create_dir')
    @patch('text4gcn.modules.file_ops.FileOps.write_iterable_to_file')
    def test_shuffle_data(self, mock_write, mock_create_dir, mock_load_meta):
        """Test data shuffling process."""
        # Mock metadata loading
        mock_load_meta.return_value = (
            ["doc1", "doc2", "doc3", "doc4"],  # all_doc_meta_list
            ["doc1", "doc2"],  # train_doc_meta_list
            ["doc3", "doc4"]   # test_doc_meta_list
        )
        
        # Create mock cleaned data file
        cleaned_data_path = os.path.join(self.dataset_path, "cleaned_data.txt")
        with open(cleaned_data_path, 'w') as f:
            f.write("document 1\n")
            f.write("document 2\n")
            f.write("document 3\n")
            f.write("document 4\n")
        
        result = self.shuffler.shuffle_data(cleaned_data_path)
        
        expected = f"{self.dataset_path}/{self.dataset_name}.shuffled"
        assert result == expected
    
    def test_get_shuffled_data_paths(self):
        """Test getting shuffled data paths."""
        # Get all paths for shuffled data files
        paths = self.shuffler.get_shuffled_data_paths()
        
        # Verify that exactly 4 paths are returned (txt, train, test, meta)
        assert len(paths) == 4
        # Verify that all paths have expected file extensions
        assert all(path.endswith('.txt') or path.endswith('.train') or 
                  path.endswith('.test') or path.endswith('.meta') for path in paths)


class TestWordProcessor:
    """Test cases for WordProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_name = "test_dataset"
        self.dataset_path = tempfile.mkdtemp()
        
        self.processor = WordProcessor(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path
        )
    
    def test_initialization(self):
        """Test WordProcessor initialization."""
        assert self.processor.dataset_name == self.dataset_name
        assert self.processor.dataset_path == self.dataset_path
        assert self.processor.logger is not None
        assert self.processor.flop is not None
    
    @patch('text4gcn.modules.word_processor.extract_vocabulary')
    @patch('text4gcn.modules.word_processor.extract_word_definitions')
    @patch('text4gcn.modules.word_processor.extract_tf_idf_word_vectors')
    @patch('text4gcn.modules.word_processor.word_to_vectors')
    def test_prepare_words(self, mock_word_to_vectors, mock_extract_vectors,
                          mock_extract_definitions, mock_extract_vocab):
        """Test word processing."""
        # Mock dependencies
        mock_extract_vocab.return_value = ["word1", "word2", "word3"]
        mock_extract_definitions.return_value = ["def1", "def2", "def3"]
        mock_extract_vectors.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_word_to_vectors.return_value = {"word1": [0.1, 0.2], "word2": [0.3, 0.4]}
        
        # Create mock shuffled data
        shuffled_data_path = os.path.join(self.dataset_path, "shuffled")
        os.makedirs(shuffled_data_path, exist_ok=True)
        corpus_file = os.path.join(shuffled_data_path, f"{self.dataset_name}.txt")
        with open(corpus_file, 'w') as f:
            f.write("word1 word2\n")
            f.write("word2 word3\n")
        
        # Mock file operations
        with patch.object(self.processor.flop, 'check_paths'), \
             patch.object(self.processor.flop, 'write_iterable_to_file'), \
             patch.object(self.processor.flop, 'write_picke'):
            
            result = self.processor.prepare_words(shuffled_data_path)
        
        expected = f"{shuffled_data_path}/{self.dataset_name}.word_vectors"
        assert result == expected
    
    def test_get_vocabulary_path(self):
        """Test getting vocabulary path."""
        path = self.processor.get_vocabulary_path()
        expected = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.vocab"
        assert path == expected
    
    def test_get_word_vectors_path(self):
        """Test getting word vectors path."""
        path = self.processor.get_word_vectors_path()
        expected = f"{self.dataset_path}/{self.dataset_name}.shuffled/{self.dataset_name}.word_vectors"
        assert path == expected


class TestNodeFeatureBuilder:
    """Test cases for NodeFeatureBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_name = "test_dataset"
        self.dataset_path = tempfile.mkdtemp()
        
        self.builder = NodeFeatureBuilder(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path
        )
    
    def test_initialization(self):
        """Test NodeFeatureBuilder initialization."""
        assert self.builder.dataset_name == self.dataset_name
        assert self.builder.dataset_path == self.dataset_path
        assert self.builder.logger is not None
        assert self.builder.flop is not None
        assert self.builder.nfeatures is not None
    
    @patch('text4gcn.modules.node_features.NodeFeatures.extract_doc_labels')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_x')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_y')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_tx')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_ty')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_allx')
    @patch('text4gcn.modules.node_features.NodeFeatures.compute_ally')
    @patch('text4gcn.modules.node_features.NodeFeatures.dump_node_features')
    def test_build_node_features(self, mock_dump, mock_compute_ally, mock_compute_allx,
                                mock_compute_ty, mock_compute_tx, mock_compute_y,
                                mock_compute_x, mock_extract_labels):
        """Test node feature building."""
        # Create mock files
        shuffled_data_path = os.path.join(self.dataset_path, "shuffled")
        os.makedirs(shuffled_data_path, exist_ok=True)
        
        # Create required files
        files = [
            f"{self.dataset_name}.txt",
            f"{self.dataset_name}.meta",
            f"{self.dataset_name}.vocab",
            f"{self.dataset_name}.train",
            f"{self.dataset_name}.test"
        ]
        
        for filename in files:
            filepath = os.path.join(shuffled_data_path, filename)
            with open(filepath, 'w') as f:
                f.write("test data\n")
        
        # Mock dependencies
        mock_extract_labels.return_value = ["label1", "label2"]
        mock_compute_x.return_value = Mock()
        mock_compute_y.return_value = Mock()
        mock_compute_tx.return_value = Mock()
        mock_compute_ty.return_value = Mock()
        mock_compute_allx.return_value = Mock()
        mock_compute_ally.return_value = Mock()
        
        # Mock file operations
        with patch.object(self.builder.flop, 'check_paths'), \
             patch.object(self.builder.flop, 'create_dir'):
            
            result = self.builder.build_node_features(shuffled_data_path)
        
        expected = f"{self.dataset_path}/{self.dataset_name}.node_features"
        assert result == expected
    
    def test_get_node_features_path(self):
        """Test getting node features path."""
        path = self.builder.get_node_features_path()
        expected = f"{self.dataset_path}/{self.dataset_name}.node_features"
        assert path == expected


class TestTextPipeline:
    """Test cases for refactored TextPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_name = "test_dataset"
        self.dataset_path = tempfile.mkdtemp()
        self.rare_count = 5
        self.language = "english"
        
        self.pipeline = TextPipeline(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            rare_count=self.rare_count,
            language=self.language
        )
    
    def test_initialization(self):
        """Test TextPipeline initialization."""
        assert self.pipeline.dataset_name == self.dataset_name
        assert self.pipeline.dataset_path == self.dataset_path
        assert self.pipeline.rare_count == self.rare_count
        assert self.pipeline.language == self.language
        
        # Check that all components are initialized
        assert isinstance(self.pipeline.data_cleaner, DataCleaner)
        assert isinstance(self.pipeline.data_shuffler, DataShuffler)
        assert isinstance(self.pipeline.word_processor, WordProcessor)
        assert isinstance(self.pipeline.node_feature_builder, NodeFeatureBuilder)
    
    @patch('text4gcn.modules.clean_data.config_nltk')
    @patch.object(DataCleaner, 'clean_data')
    @patch.object(DataShuffler, 'shuffle_data')
    @patch.object(WordProcessor, 'prepare_words')
    @patch.object(NodeFeatureBuilder, 'build_node_features')
    def test_execute(self, mock_build_features, mock_prepare_words, 
                    mock_shuffle_data, mock_clean_data, mock_config_nltk):
        """Test complete pipeline execution."""
        # Mock component methods
        mock_clean_data.return_value = "cleaned_path"
        mock_shuffle_data.return_value = "shuffled_path"
        mock_prepare_words.return_value = "word_vectors_path"
        mock_build_features.return_value = "node_features_path"
        
        # Mock file operations
        with patch.object(self.pipeline.data_cleaner.flop, 'create_dir'), \
             patch.object(self.pipeline.data_cleaner.flop, 'write_iterable_to_file'):
            
            result = self.pipeline.execute()
        
        # Check that all components were called
        mock_config_nltk.assert_called_once()
        mock_clean_data.assert_called_once()
        mock_shuffle_data.assert_called_once_with("cleaned_path")
        mock_prepare_words.assert_called_once_with("shuffled_path")
        mock_build_features.assert_called_once()
        
        # Check result structure
        assert 'cleaned_data' in result
        assert 'shuffled_data' in result
        assert 'word_vectors' in result
        assert 'node_features' in result
    
    def test_get_processing_paths(self):
        """Test getting processing paths."""
        paths = self.pipeline.get_processing_paths()
        
        expected_keys = [
            'cleaned_data', 'shuffled_corpus', 'train_indices', 
            'test_indices', 'metadata', 'vocabulary', 'word_vectors', 'node_features'
        ]
        
        for key in expected_keys:
            assert key in paths
            assert isinstance(paths[key], str)


# Integration tests
class TestRefactoredPreprocessingIntegration:
    """Integration tests for refactored preprocessing components."""
    
    def test_data_cleaner_integration(self):
        """Test DataCleaner integration."""
        cleaner = DataCleaner(
            dataset_name="test",
            dataset_path=tempfile.mkdtemp(),
            rare_count=5,
            language="english"
        )
        
        # Test initialization
        assert cleaner.dataset_name == "test"
        assert cleaner.rare_count == 5
        assert cleaner.language == "english"
    
    def test_data_shuffler_integration(self):
        """Test DataShuffler integration."""
        shuffler = DataShuffler(
            dataset_name="test",
            dataset_path=tempfile.mkdtemp()
        )
        
        # Test initialization
        assert shuffler.dataset_name == "test"
        assert shuffler.logger is not None
        assert shuffler.flop is not None
    
    def test_word_processor_integration(self):
        """Test WordProcessor integration."""
        processor = WordProcessor(
            dataset_name="test",
            dataset_path=tempfile.mkdtemp()
        )
        
        # Test initialization
        assert processor.dataset_name == "test"
        assert processor.logger is not None
        assert processor.flop is not None
    
    def test_node_feature_builder_integration(self):
        """Test NodeFeatureBuilder integration."""
        builder = NodeFeatureBuilder(
            dataset_name="test",
            dataset_path=tempfile.mkdtemp()
        )
        
        # Test initialization
        assert builder.dataset_name == "test"
        assert builder.logger is not None
        assert builder.flop is not None
        assert builder.nfeatures is not None
    
    def test_text_pipeline_integration(self):
        """Test TextPipeline integration."""
        pipeline = TextPipeline(
            dataset_name="test",
            dataset_path=tempfile.mkdtemp(),
            rare_count=5,
            language="english"
        )
        
        # Test initialization
        assert pipeline.dataset_name == "test"
        assert pipeline.rare_count == 5
        assert pipeline.language == "english"
        
        # Test component initialization
        assert isinstance(pipeline.data_cleaner, DataCleaner)
        assert isinstance(pipeline.data_shuffler, DataShuffler)
        assert isinstance(pipeline.word_processor, WordProcessor)
        assert isinstance(pipeline.node_feature_builder, NodeFeatureBuilder)


if __name__ == '__main__':
    pytest.main([__file__])


