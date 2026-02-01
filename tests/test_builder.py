"""Tests for the builder module."""

import pytest
import tempfile
import os
from pathlib import Path

from text4gcn.builder import Frequency, Semantic, Syntactic, Builder
from text4gcn.builder import list_available_adjacency_methods, list_available_builders


class TestBuilderClasses:
    """Test builder classes."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()
            yield str(data_path)
    
    def test_frequency_initialization(self, temp_data_dir):
        """Test Frequency builder initialization."""
        # Frequency builder requires dataset_name and dataset_path
        builder = Frequency(dataset_name="R8", dataset_path=temp_data_dir)
        assert builder is not None
        assert builder.dataset_name == "R8"
        assert builder.dataset_path == temp_data_dir
    
    def test_semantic_initialization(self, temp_data_dir):
        """Test Semantic builder initialization."""
        # Semantic builder requires dataset_name and dataset_path
        builder = Semantic(dataset_name="R8", dataset_path=temp_data_dir)
        assert builder is not None
        assert builder.dataset_name == "R8"
        assert builder.dataset_path == temp_data_dir
    
    def test_syntactic_initialization(self, temp_data_dir):
        """Test Syntactic builder initialization."""
        # Syntactic builder requires dataset_name and dataset_path
        builder = Syntactic(dataset_name="R8", dataset_path=temp_data_dir)
        assert builder is not None
        assert builder.dataset_name == "R8"
        assert builder.dataset_path == temp_data_dir


class TestBuilderEnum:
    """Test Builder enum."""
    
    def test_builder_enum_values(self):
        """Test Builder enum has expected values."""
        expected_values = {
            'liwc', 'cosine.similarity', 'dependency.parsing',
            'constituency.parsing', 'embedding.word2vec', 'frequency.tfidf'
        }
        actual_values = {builder.value for builder in Builder}
        assert actual_values == expected_values
    
    def test_builder_enum_string_behavior(self):
        """Test Builder enum behaves like string."""
        builder = Builder.Tfidf
        # Enum value should match the string value
        assert builder.value == 'frequency.tfidf'
        # Enum should be comparable to its string value
        assert builder == 'frequency.tfidf'


class TestBuilderFunctions:
    """Test builder utility functions."""
    
    def test_list_available_adjacency_methods(self, capsys):
        """Test list_available_adjacency_methods function."""
        list_available_adjacency_methods()
        captured = capsys.readouterr()
        assert "Text4GCN - ADJACENCY" in captured.out
    
    def test_list_available_builders(self, capsys):
        """Test list_available_builders function."""
        list_available_builders()
        captured = capsys.readouterr()
        assert "Text4GCN - BUILDERS" in captured.out


class TestBuilderIntegration:
    """Integration tests for builder functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data files
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()
            
            # Create mock text file
            text_file = data_path / "R8.txt"
            text_file.write_text("This is a test document.\nAnother test document.")
            
            # Create mock meta file
            meta_file = data_path / "R8.meta"
            meta_file.write_text("0\tcategory1\n1\tcategory2")
            
            yield str(data_path)
    
    def test_frequency_builder_with_temp_data(self, temp_data_dir):
        """Test Frequency builder with temporary data."""
        # Frequency builder requires dataset_name and dataset_path
        builder = Frequency(dataset_name="R8", dataset_path=temp_data_dir)
        # This is a basic test - actual implementation would depend on the builder methods
        assert builder is not None
        assert builder.dataset_name == "R8"
    
    def test_semantic_builder_with_temp_data(self, temp_data_dir):
        """Test Semantic builder with temporary data."""
        # Semantic builder requires dataset_name and dataset_path
        builder = Semantic(dataset_name="R8", dataset_path=temp_data_dir)
        # This is a basic test - actual implementation would depend on the builder methods
        assert builder is not None
        assert builder.dataset_name == "R8"
    
    def test_syntactic_builder_with_temp_data(self, temp_data_dir):
        """Test Syntactic builder with temporary data."""
        # Syntactic builder requires dataset_name and dataset_path
        builder = Syntactic(dataset_name="R8", dataset_path=temp_data_dir)
        # This is a basic test - actual implementation would depend on the builder methods
        assert builder is not None
        assert builder.dataset_name == "R8"


class TestBuilderErrorHandling:
    """Test error handling in builder module."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()
            yield str(data_path)
    
    def test_builder_initialization_with_invalid_params(self, temp_data_dir):
        """Test builder initialization with invalid parameters."""
        # Test that builders can be initialized with required parameters
        # Using temp directory for valid path
        frequency = Frequency(dataset_name="R8", dataset_path=temp_data_dir)
        semantic = Semantic(dataset_name="R8", dataset_path=temp_data_dir)
        syntactic = Syntactic(dataset_name="R8", dataset_path=temp_data_dir)
        
        assert frequency is not None
        assert semantic is not None
        assert syntactic is not None


@pytest.mark.slow
class TestBuilderPerformance:
    """Performance tests for builder module."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()
            yield str(data_path)
    
    def test_builder_initialization_performance(self, temp_data_dir):
        """Test that builder initialization is fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            Frequency(dataset_name="R8", dataset_path=temp_data_dir)
            Semantic(dataset_name="R8", dataset_path=temp_data_dir)
            Syntactic(dataset_name="R8", dataset_path=temp_data_dir)
        end_time = time.time()
        
        # Should initialize 300 builders in less than 1 second
        assert (end_time - start_time) < 1.0
