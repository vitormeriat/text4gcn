"""Tests for the models module."""

import pytest
import tempfile
from pathlib import Path

from text4gcn.models import GNN, Layer, Builder as ModelBuilder


class TestLayerEnum:
    """Test Layer enum."""

    def test_layer_enum_values(self):
        """Test Layer enum has expected values."""
        assert hasattr(Layer, "GCN")
        assert not hasattr(Layer, "GAT")

    def test_layer_enum_string_behavior(self):
        """Test Layer enum behaves like string (str, Enum)."""
        layer = Layer.GCN
        assert layer.value == "GCN"
        assert layer == "GCN"
        assert Layer.GCN == Layer("GCN")


class TestGNNModel:
    """Test GNN model class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()

            # Create mock data files
            text_file = data_path / "R8.txt"
            text_file.write_text("This is a test document.\nAnother test document.")

            meta_file = data_path / "R8.meta"
            meta_file.write_text("0\tcategory1\n1\tcategory2")

            yield str(data_path)

    def test_gnn_initialization(self, temp_data_dir):
        """Test GNN model initialization."""
        # Test basic initialization
        model = GNN(
            dataset_name="R8",
            dataset_path=temp_data_dir,
            layer=Layer.GCN if hasattr(Layer, 'GCN') else None,
            epochs=10,
            dropout=0.5,
        )
        assert model is not None

    def test_gnn_initialization_with_defaults(self, temp_data_dir):
        """Test GNN model initialization with default parameters."""
        model = GNN(
            dataset_name="R8",
            dataset_path=temp_data_dir,
        )
        assert model is not None

    def test_gnn_initialization_with_custom_params(self, temp_data_dir):
        """Test GNN model initialization with custom parameters."""
        model = GNN(
            dataset_name="R8",
            dataset_path=temp_data_dir,
            layer=Layer.GCN if hasattr(Layer, 'GCN') else None,
            epochs=50,
            dropout=0.3,
            lr=0.01,
            nhid=128,
        )
        assert model is not None


class TestModelBuilder:
    """Test ModelBuilder enum."""

    def test_builder_value_used_in_load_data_path(self):
        """Python 3.11+ str(Enum) is 'Builder.Tfidf'; adjacency files use .value."""
        from text4gcn.models.utils import LoadData

        loader = LoadData(path="data", dataset="R8", builder=ModelBuilder.Tfidf)
        assert loader.builder == "frequency.tfidf"
        assert "Builder." not in loader.builder


class TestGNNErrorHandling:
    """Test error handling in GNN model."""

    def test_gnn_initialization_with_invalid_dataset(self):
        """Test GNN initialization with invalid dataset."""
        with pytest.raises((ValueError, FileNotFoundError, AttributeError)):
            GNN(
                dataset_name="InvalidDataset",
                dataset_path="/nonexistent/path",
            )

    def test_gnn_initialization_with_invalid_path(self):
        """Test GNN initialization with invalid path."""
        with pytest.raises((ValueError, FileNotFoundError, AttributeError)):
            GNN(
                dataset_name="R8",
                dataset_path="/nonexistent/path",
            )


@pytest.mark.slow
class TestGNNPerformance:
    """Performance tests for GNN model."""

    def test_gnn_initialization_performance(self):
        """Test that GNN initialization is reasonably fast."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()

            # Create minimal test data
            text_file = data_path / "R8.txt"
            text_file.write_text("Test document.")

            meta_file = data_path / "R8.meta"
            meta_file.write_text("0\tcategory1")

            start_time = time.time()
            for _ in range(10):
                GNN(
                    dataset_name="R8",
                    dataset_path=str(data_path),
                    epochs=1,
                )
            end_time = time.time()

            # Should initialize 10 models in less than 5 seconds
            assert (end_time - start_time) < 5.0


class TestGNNIntegration:
    """Integration tests for GNN model."""

    @pytest.fixture
    def complete_test_data(self):
        """Create complete test dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "R8"
            data_path.mkdir()

            # Create more comprehensive test data
            text_file = data_path / "R8.txt"
            text_file.write_text(
                "This is the first test document about technology.\n"
                "This is the second test document about science.\n"
                "This is the third test document about sports."
            )

            meta_file = data_path / "R8.meta"
            meta_file.write_text(
                "0\ttechnology\n"
                "1\tscience\n"
                "2\tsports"
            )

            yield str(data_path)

    def test_gnn_with_complete_data(self, complete_test_data):
        """Test GNN with complete test data."""
        model = GNN(
            dataset_name="R8",
            dataset_path=complete_test_data,
            epochs=1,
            dropout=0.5,
        )
        assert model is not None

        # Test that model can be created without errors
        # Note: Actual training would require more setup
        try:
            # This would be the actual training call
            # model.fit()
            pass
        except Exception as e:
            # If training fails due to missing dependencies, that's expected in tests
            assert "CUDA" in str(e) or "torch" in str(e) or "device" in str(e)
