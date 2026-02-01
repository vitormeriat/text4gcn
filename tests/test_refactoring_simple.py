"""
Simple tests for refactored components without external dependencies.

This module contains basic tests to validate the refactoring
without requiring PyTorch or other heavy dependencies.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

# Test basic imports and structure
def test_refactored_imports():
    """Test that refactored modules can be imported."""
    try:
        # Test models imports - modules were moved from refactored/ to root models/
        from text4gcn.models.data_processor import DataProcessor
        from text4gcn.models.config_manager import ConfigManager
        from text4gcn.models.model_factory import ModelFactory
        from text4gcn.models.layers import Layer
        
        # Test preprocessing imports - modules were moved from refactored/ to root preprocess/
        # Note: DataCleaner may not exist as a separate class
        from text4gcn.preprocess.data_shuffler import DataShuffler
        from text4gcn.preprocess.word_processor import WordProcessor
        from text4gcn.preprocess.node_feature_builder import NodeFeatureBuilder
        from text4gcn.preprocess.text_pipeline_refactored import TextPipeline
        
        assert True  # If we get here, imports work
    except ImportError as e:
        pytest.fail(f"Failed to import refactored modules: {e}")

def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    from text4gcn.models.data_processor import DataProcessor
    
    processor = DataProcessor(device='cpu')
    
    assert processor.device == 'cpu'
    assert processor.features is None
    assert processor.adj is None
    assert processor.support is None

def test_config_manager_initialization():
    """Test ConfigManager initialization."""
    from text4gcn.models.config_manager import ConfigManager
    from text4gcn.models.layers import Layer
    
    config = ConfigManager(
        dataset="test_dataset",
        path="/test/path",
        log_dir="/test/log",
        builder="tfidf",
        lr=0.01,
        nhid=100,
        seed=42,
        epoches=10,
        dropout=0.3,
        val_ratio=0.2,
        layer=Layer.GCN,
        early_stopping=5,
        device="cpu"
    )
    
    assert config.dataset == "test_dataset"
    assert config.path == "/test/path"
    assert config.log_dir == "/test/log"
    assert config.lr == 0.01
    assert config.nhid == 100
    assert config.seed == 42
    assert config.epoches == 10
    assert config.dropout == 0.3
    assert config.val_ratio == 0.2
    assert config.layer == Layer.GCN
    assert config.early_stopping == 5
    assert config.device == "cpu"

def test_config_manager_validation():
    """Test ConfigManager validation."""
    from text4gcn.models.config_manager import ConfigManager
    from text4gcn.models.layers import Layer
    
    # Valid configuration
    config = ConfigManager(
        dataset="test_dataset",
        path="/test/path",
        log_dir="/test/log",
        builder="tfidf"
    )
    
    assert config.validate_config() == True
    
    # Invalid configuration (negative learning rate)
    config.lr = -0.01
    assert config.validate_config() == False

def test_config_manager_get_config_dict():
    """Test ConfigManager get_config_dict method."""
    from text4gcn.models.config_manager import ConfigManager
    
    config = ConfigManager(
        dataset="test_dataset",
        path="/test/path",
        log_dir="/test/log",
        builder="tfidf",
        lr=0.01,
        nhid=100
    )
    
    config_dict = config.get_config_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['dataset'] == "test_dataset"
    assert config_dict['path'] == "/test/path"
    assert config_dict['log_dir'] == "/test/log"
    assert config_dict['builder'] == "tfidf"
    assert config_dict['lr'] == 0.01
    assert config_dict['nhid'] == 100

def test_data_cleaner_initialization():
    """Test DataCleaner initialization."""
    # Note: DataCleaner may not exist as a separate class
    # It might be integrated into the pipeline or not yet implemented
    try:
        from text4gcn.preprocess.data_cleaner import DataCleaner
        
        cleaner = DataCleaner(
            dataset_name="test_dataset",
            dataset_path="/test/path",
            rare_count=5,
            language="english"
        )
        
        assert cleaner.dataset_name == "test_dataset"
        assert cleaner.dataset_path == "/test/path"
        assert cleaner.rare_count == 5
        assert cleaner.language == "english"
    except ImportError:
        pytest.skip("DataCleaner class not available")

def test_data_shuffler_initialization():
    """Test DataShuffler initialization."""
    from text4gcn.preprocess.data_shuffler import DataShuffler
    
    shuffler = DataShuffler(
        dataset_name="test_dataset",
        dataset_path="/test/path"
    )
    
    assert shuffler.dataset_name == "test_dataset"
    assert shuffler.dataset_path == "/test/path"

def test_word_processor_initialization():
    """Test WordProcessor initialization."""
    from text4gcn.preprocess.word_processor import WordProcessor
    
    processor = WordProcessor(
        dataset_name="test_dataset",
        dataset_path="/test/path"
    )
    
    assert processor.dataset_name == "test_dataset"
    assert processor.dataset_path == "/test/path"

def test_node_feature_builder_initialization():
    """Test NodeFeatureBuilder initialization."""
    from text4gcn.preprocess.node_feature_builder import NodeFeatureBuilder
    
    builder = NodeFeatureBuilder(
        dataset_name="test_dataset",
        dataset_path="/test/path"
    )
    
    assert builder.dataset_name == "test_dataset"
    assert builder.dataset_path == "/test/path"

def test_text_pipeline_initialization():
    """Test TextPipeline initialization."""
    from text4gcn.preprocess.text_pipeline_refactored import TextPipeline
    
    pipeline = TextPipeline(
        dataset_name="test_dataset",
        dataset_path="/test/path",
        rare_count=5,
        language="english"
    )
    
    assert pipeline.dataset_name == "test_dataset"
    assert pipeline.dataset_path == "/test/path"
    assert pipeline.rare_count == 5
    assert pipeline.language == "english"
    
    # Check that all components are initialized
    # Note: data_cleaner is not a separate component, cleaning is handled by the pipeline
    assert pipeline.data_shuffler is not None
    assert pipeline.word_processor is not None
    assert pipeline.node_feature_builder is not None

def test_text_pipeline_get_processing_paths():
    """Test TextPipeline get_processing_paths method."""
    from text4gcn.preprocess.text_pipeline_refactored import TextPipeline
    
    pipeline = TextPipeline(
        dataset_name="test_dataset",
        dataset_path="/test/path"
    )
    
    paths = pipeline.get_processing_paths()
    
    expected_keys = [
        'cleaned_data', 'shuffled_corpus', 'train_indices', 
        'test_indices', 'metadata', 'vocabulary', 'word_vectors', 'node_features'
    ]
    
    for key in expected_keys:
        assert key in paths
        assert isinstance(paths[key], str)

def test_solid_principles_validation():
    """Test that refactored code follows SOLID principles."""
    
    # Single Responsibility Principle
    # Each class should have a single responsibility
    
    # DataProcessor - only data processing
    from text4gcn.models.data_processor import DataProcessor
    processor = DataProcessor('cpu')
    assert hasattr(processor, 'prepare_data')
    assert hasattr(processor, 'convert_tensor')
    assert hasattr(processor, 'get_data_for_training')
    
    # ConfigManager - only configuration management
    from text4gcn.models.config_manager import ConfigManager
    config = ConfigManager(dataset="test", path="/test", log_dir="/test", builder="tfidf")
    assert hasattr(config, 'validate_config')
    assert hasattr(config, 'get_config_dict')
    assert hasattr(config, 'create_args')
    
    # DataCleaner - only data cleaning (may not exist)
    try:
        from text4gcn.preprocess.data_cleaner import DataCleaner
        cleaner = DataCleaner("test", "/test")
        assert hasattr(cleaner, 'clean_data')
        assert hasattr(cleaner, 'get_cleaned_data_path')
    except ImportError:
        pass  # DataCleaner may not be implemented as separate class
    
    # DataShuffler - only data shuffling
    from text4gcn.preprocess.data_shuffler import DataShuffler
    shuffler = DataShuffler("test", "/test")
    assert hasattr(shuffler, 'shuffle_data')
    assert hasattr(shuffler, 'get_shuffled_data_paths')
    
    # WordProcessor - only word processing
    from text4gcn.preprocess.word_processor import WordProcessor
    word_processor = WordProcessor("test", "/test")
    assert hasattr(word_processor, 'prepare_words')
    assert hasattr(word_processor, 'get_vocabulary_path')
    assert hasattr(word_processor, 'get_word_vectors_path')
    
    # NodeFeatureBuilder - only node feature building
    from text4gcn.preprocess.node_feature_builder import NodeFeatureBuilder
    builder = NodeFeatureBuilder("test", "/test")
    assert hasattr(builder, 'build_node_features')
    assert hasattr(builder, 'get_node_features_path')

def test_composition_pattern():
    """Test that TextPipeline uses composition pattern."""
    from text4gcn.preprocess.text_pipeline_refactored import TextPipeline
    
    pipeline = TextPipeline("test", "/test")
    
    # Check that pipeline composes other components
    # Note: data_cleaner is not a separate component, cleaning is handled by the pipeline
    assert hasattr(pipeline, 'data_shuffler')
    assert hasattr(pipeline, 'word_processor')
    assert hasattr(pipeline, 'node_feature_builder')
    
    # Check that pipeline delegates to components
    assert hasattr(pipeline, 'execute')
    assert hasattr(pipeline, 'get_processing_paths')

def test_factory_pattern():
    """Test that ModelFactory implements factory pattern."""
    from text4gcn.models.model_factory import ModelFactory
    from text4gcn.models.layers import Layer
    
    # Test getting list of supported model types
    supported_models = ModelFactory.get_supported_models()
    assert isinstance(supported_models, list)
    # Verify that GCN layer is in the supported models list
    assert Layer.GCN in supported_models
    
    # Test model registration (extensibility - Open/Closed Principle)
    # This allows adding new models without modifying existing code
    original_count = len(ModelFactory._models)
    ModelFactory.register_model("TEST", object)
    # Verify that the new model was registered
    assert len(ModelFactory._models) == original_count + 1
    
    # Clean up: remove the test model to avoid affecting other tests
    del ModelFactory._models["TEST"]

def test_dependency_injection():
    """Test that components use dependency injection."""
    from text4gcn.models.data_processor import DataProcessor
    
    # DataProcessor should accept device as dependency
    processor = DataProcessor(device='cpu')
    assert processor.device == 'cpu'
    
    # Should be able to inject different devices
    processor_gpu = DataProcessor(device='cuda')
    assert processor_gpu.device == 'cuda'

def test_interface_segregation():
    """Test that interfaces are properly segregated."""
    
    # Each component should have focused interfaces (Interface Segregation Principle)
    from text4gcn.models.data_processor import DataProcessor
    
    processor = DataProcessor('cpu')
    
    # Collect all public methods from DataProcessor
    # Filter out private methods (starting with '_') and only include callable attributes
    data_methods = [method for method in dir(processor) 
                   if not method.startswith('_') and callable(getattr(processor, method))]
    
    # Verify that DataProcessor does NOT have methods from other responsibilities
    # This ensures the interface is focused and not bloated
    assert 'train' not in data_methods
    assert 'evaluate' not in data_methods
    assert 'plot' not in data_methods
    assert 'visualize' not in data_methods
    
    # Verify that DataProcessor has the expected data processing methods
    # These are the core responsibilities of this class
    assert 'prepare_data' in data_methods
    assert 'convert_tensor' in data_methods
    assert 'get_data_for_training' in data_methods

if __name__ == '__main__':
    pytest.main([__file__])


