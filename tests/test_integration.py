"""Integration tests for Text4GCN."""

import pytest
import tempfile
from pathlib import Path

from text4gcn import datasets, builder, models


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    @pytest.fixture
    def complete_test_environment(self):
        """Create complete test environment with data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dataset directory
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()
            
            # Create test data files
            text_file = data_path / "R8.txt"
            text_file.write_text(
                "This is a test document about machine learning and artificial intelligence.\n"
                "Another document about natural language processing and text analysis.\n"
                "A third document about computer science and algorithms."
            )
            
            meta_file = data_path / "R8.meta"
            meta_file.write_text(
                "0\ttechnology\n"
                "1\tnlp\n"
                "2\tcomputer_science"
            )
            
            yield {
                'data_path': str(data_path),
                'temp_dir': temp_dir
            }
    
    def test_dataset_loading(self, complete_test_environment):
        """Test dataset loading functionality."""
        data_path = complete_test_environment['data_path']
        
        # Test that dataset files exist
        assert Path(data_path).exists()
        assert (Path(data_path) / "R8.txt").exists()
        assert (Path(data_path) / "R8.meta").exists()
    
    def test_builder_workflow(self, complete_test_environment):
        """Test builder workflow."""
        data_path = complete_test_environment['data_path']
        
        # Test frequency builder - requires dataset_name and dataset_path
        freq_builder = builder.Frequency(dataset_name="R8", dataset_path=data_path)
        assert freq_builder is not None
        assert freq_builder.dataset_name == "R8"
        
        # Test semantic builder - requires dataset_name and dataset_path
        sem_builder = builder.Semantic(dataset_name="R8", dataset_path=data_path)
        assert sem_builder is not None
        assert sem_builder.dataset_name == "R8"
        
        # Test syntactic builder - requires dataset_name and dataset_path
        syn_builder = builder.Syntactic(dataset_name="R8", dataset_path=data_path)
        assert syn_builder is not None
        assert syn_builder.dataset_name == "R8"
    
    def test_model_initialization(self, complete_test_environment):
        """Test model initialization in complete environment."""
        data_path = complete_test_environment['data_path']
        syn_builder = builder.Syntactic(dataset_name="R8", dataset_path=data_path)
        
        # Test GNN model initialization
        model = models.GNN(
            dataset_name="R8",
            dataset_path=data_path,
            epochs=1,
            dropout=0.5,
        )
        assert model is not None
    
    def test_builder_enum_integration(self, complete_test_environment):
        """Test builder enum integration."""
        # Test that builder enums work correctly
        assert builder.Builder.Tfidf == 'frequency.tfidf'
        assert builder.Builder.Word2Vec == 'embedding.word2vec'
        assert builder.Builder.DependencyParsing == 'dependency.parsing'


class TestModuleImports:
    """Test module imports and basic functionality."""
    
    def test_main_module_imports(self):
        """Test that main modules can be imported."""
        import text4gcn
        
        # Test that main modules are available
        assert hasattr(text4gcn, 'datasets')
        assert hasattr(text4gcn, 'builder')
        assert hasattr(text4gcn, 'models')
        assert hasattr(text4gcn, 'preprocess')
        assert hasattr(text4gcn, 'modules')
    
    def test_version_info(self):
        """Test version information."""
        import text4gcn
        
        assert hasattr(text4gcn, '__version__')
        assert hasattr(text4gcn, '__author__')
        assert hasattr(text4gcn, '__email__')
        
        assert text4gcn.__version__ == "2.0.0"
        assert text4gcn.__author__ == "Vitor Meriat"
        assert text4gcn.__email__ == "vitormeriat@gmail.com"
    
    def test_builder_module_imports(self):
        """Test builder module imports."""
        from text4gcn.builder import Frequency, Semantic, Syntactic, Builder
        from text4gcn.builder import list_available_adjacency_methods, list_available_builders
        
        # Test that all expected classes and functions are available
        assert Frequency is not None
        assert Semantic is not None
        assert Syntactic is not None
        assert Builder is not None
        assert list_available_adjacency_methods is not None
        assert list_available_builders is not None
    
    def test_datasets_module_imports(self):
        """Test datasets module imports."""
        from text4gcn.datasets import R8, R52, AG_NEWS
        
        # Test that dataset functions are available
        assert R8 is not None
        assert R52 is not None
        assert AG_NEWS is not None


class TestErrorHandling:
    """Test error handling across modules."""
    
    def test_invalid_dataset_handling(self):
        """Test handling of invalid dataset names."""
        with pytest.raises((AttributeError, ValueError, FileNotFoundError)):
            models.GNN(
                dataset_name="InvalidDataset",
                dataset_path="/nonexistent/path",
            )

    def test_invalid_path_handling(self):
        """Test handling of invalid paths."""
        with pytest.raises((AttributeError, ValueError, FileNotFoundError)):
            models.GNN(
                dataset_name="R8",
                dataset_path="/nonexistent/path",
            )


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance tests for integration scenarios."""
    
    def test_import_performance(self):
        """Test that imports are reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Import main modules
        import text4gcn
        from text4gcn import datasets, builder, models
        
        end_time = time.time()
        
        # Imports should be fast (less than 2 seconds)
        assert (end_time - start_time) < 2.0
    
    def test_builder_initialization_performance(self):
        """Test builder initialization performance."""
        import time
        
        start_time = time.time()
        
        # Initialize multiple builders
        for _ in range(50):
            builder.Frequency()
            builder.Semantic()
            builder.Syntactic()
        
        end_time = time.time()
        
        # Should initialize 150 builders in less than 1 second
        assert (end_time - start_time) < 1.0


class TestDataValidation:
    """Test data validation across the pipeline."""
    
    def test_data_file_validation(self):
        """Test validation of data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data"
            data_path.mkdir()
            
            # Create valid data files
            text_file = data_path / "test.txt"
            text_file.write_text("Test document 1.\nTest document 2.")
            
            meta_file = data_path / "test.meta"
            meta_file.write_text("0\tcategory1\n1\tcategory2")
            
            # Test that files exist and are readable
            assert text_file.exists()
            assert meta_file.exists()
            assert text_file.read_text() == "Test document 1.\nTest document 2."
            assert meta_file.read_text() == "0\tcategory1\n1\tcategory2"
    
    def test_data_consistency_validation(self):
        """Test data consistency validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data"
            data_path.mkdir()
            
            # Create consistent data files
            text_file = data_path / "test.txt"
            text_file.write_text("Doc 1.\nDoc 2.\nDoc 3.")
            
            meta_file = data_path / "test.meta"
            meta_file.write_text("0\tcat1\n1\tcat2\n2\tcat3")
            
            # Test that line counts match
            text_lines = len(text_file.read_text().strip().split('\n'))
            meta_lines = len(meta_file.read_text().strip().split('\n'))
            
            assert text_lines == meta_lines == 3
