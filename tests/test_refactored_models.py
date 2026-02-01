"""
Unit tests for refactored models following SOLID principles.

This module contains comprehensive unit tests for the refactored
model components to ensure they work correctly and maintain
the same functionality as the original implementation.
"""

import pytest
import torch as th
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from scipy.sparse import csr_matrix

# Import refactored components
from text4gcn.models.data_processor import DataProcessor
from text4gcn.models.model_trainer import ModelTrainer
from text4gcn.models.model_evaluator import ModelEvaluator
from text4gcn.models.metrics_visualizer import MetricsVisualizer
from text4gcn.models.model_factory import ModelFactory
from text4gcn.models.config_manager import ConfigManager
from text4gcn.models.textgcn_trainer import TextGCNTrainer
from text4gcn.models.layers import Layer


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.processor = DataProcessor(device=self.device)
        
        # Create mock data
        self.mock_predata = self._create_mock_predata()
    
    def _create_mock_predata(self):
        """Create mock preprocessed data."""
        n_nodes = 100
        n_classes = 5
        
        # Create mock adjacency matrix
        adj = csr_matrix(np.random.rand(n_nodes, n_nodes))
        
        # Create mock features
        features = np.random.rand(n_nodes, 50)
        
        # Create mock labels
        y_train = np.random.rand(60, n_classes)
        y_val = np.random.rand(20, n_classes)
        y_test = np.random.rand(20, n_classes)
        
        # Create mock masks
        train_mask = np.zeros(n_nodes, dtype=bool)
        train_mask[:60] = True
        val_mask = np.zeros(n_nodes, dtype=bool)
        val_mask[60:80] = True
        test_mask = np.zeros(n_nodes, dtype=bool)
        test_mask[80:] = True
        
        return (adj, features, y_train, y_val, y_test, 
                train_mask, val_mask, test_mask, 60, 20)
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        assert self.processor.device == self.device
        assert self.processor.features is None
        assert self.processor.adj is None
    
    def test_prepare_data(self):
        """Test data preparation."""
        self.processor.prepare_data(self.mock_predata)
        
        assert self.processor.features is not None
        assert self.processor.adj is not None
        assert self.processor.support is not None
        assert self.processor.train_mask is not None
        assert self.processor.val_mask is not None
        assert self.processor.test_mask is not None
    
    def test_convert_tensor(self):
        """Test tensor conversion."""
        self.processor.prepare_data(self.mock_predata)
        self.processor.convert_tensor()
        
        # Check that all data is converted to tensors
        assert isinstance(self.processor.features, th.Tensor)
        assert isinstance(self.processor.y_train, th.Tensor)
        assert isinstance(self.processor.y_val, th.Tensor)
        assert isinstance(self.processor.y_test, th.Tensor)
        assert isinstance(self.processor.train_mask, th.Tensor)
        assert isinstance(self.processor.val_mask, th.Tensor)
        assert isinstance(self.processor.test_mask, th.Tensor)
    
    def test_get_data_for_training(self):
        """Test getting data for training."""
        self.processor.prepare_data(self.mock_predata)
        self.processor.convert_tensor()
        
        data = self.processor.get_data_for_training()
        
        assert 'features' in data
        assert 'support' in data
        assert 'train_mask' in data
        assert 'val_mask' in data
        assert 'test_mask' in data
        assert 'y_train' in data
        assert 'y_val' in data
        assert 'y_test' in data


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = Mock()
        self.criterion = Mock()
        self.optimizer = Mock()
        self.log_dir = tempfile.mkdtemp()
        
        self.trainer = ModelTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            max_epochs=2,
            early_stopping=1,
            log_dir=self.log_dir
        )
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.criterion == self.criterion
        assert self.trainer.optimizer == self.optimizer
        assert self.trainer.device == self.device
        assert self.trainer.max_epochs == 2
        assert self.trainer.early_stopping == 1
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Mock model and data
        features = th.randn(10, 5)
        y_train = th.randn(10, 3)
        train_mask = th.ones(10)
        tm_train_mask = th.ones(10, 3)
        
        # Mock model forward pass
        self.model.return_value = th.randn(10, 3)
        self.criterion.return_value = th.tensor(0.5)
        
        loss, acc = self.trainer.train_epoch(features, y_train, train_mask, tm_train_mask)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_save_model(self):
        """Test model saving."""
        model_path = self.trainer.save_model()
        assert os.path.exists(model_path)
        assert model_path.endswith('.pth')


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.evaluator = ModelEvaluator(device=self.device)
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        assert self.evaluator.device == self.device
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Mock model and data
        model = Mock()
        criterion = Mock()
        features = th.randn(10, 5)
        labels = th.randn(10, 3)
        mask = np.ones(10, dtype=bool)
        
        # Mock model forward pass
        model.eval.return_value = None
        model.return_value = th.randn(10, 3)
        criterion.return_value = th.tensor(0.5)
        
        with patch('torch.no_grad'):
            loss, acc, pred, labels_out, duration = self.evaluator.evaluate_model(
                model, criterion, features, labels, mask
            )
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(pred, np.ndarray)
        assert isinstance(labels_out, np.ndarray)
        assert isinstance(duration, float)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        test_labels = [0, 1, 2, 0, 1]
        test_pred = [0, 1, 2, 0, 1]
        
        metrics = self.evaluator.compute_metrics(test_labels, test_pred)
        
        assert 'f1_macro' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics


class TestMetricsVisualizer:
    """Test cases for MetricsVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_dir = tempfile.mkdtemp()
        self.visualizer = MetricsVisualizer(log_dir=self.log_dir)
    
    def test_initialization(self):
        """Test MetricsVisualizer initialization."""
        assert self.visualizer.log_dir == self.log_dir
    
    def test_plot_accuracy(self):
        """Test accuracy plotting."""
        train_acc = [0.5, 0.6, 0.7]
        eval_acc = [0.4, 0.5, 0.6]
        train_loss = [1.0, 0.8, 0.6]
        eval_loss = [1.1, 0.9, 0.7]
        
        self.visualizer.plot_accuracy(train_acc, eval_acc, train_loss, eval_loss)
        
        # Check that files were created
        assert os.path.exists(f'{self.log_dir}/metrics-acc.png')
        assert os.path.exists(f'{self.log_dir}/metrics-loss.png')
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        cm = np.array([[3, 1], [1, 2]])
        class_names = ['Class A', 'Class B']
        
        self.visualizer.plot_confusion_matrix(cm, class_names)
        
        assert os.path.exists(f'{self.log_dir}/confusion-matrix.png')
    
    def test_plot_metrics_summary(self):
        """Test metrics summary plotting."""
        metrics_dict = {
            'f1_macro': 0.8,
            'precision_macro': 0.75,
            'recall_macro': 0.85,
            'accuracy': 0.8
        }
        
        self.visualizer.plot_metrics_summary(metrics_dict)
        
        assert os.path.exists(f'{self.log_dir}/metrics-summary.png')


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def test_create_model_gcn(self):
        """Test GCN model creation."""
        input_dim = 10
        output_dim = 5
        support = [th.randn(10, 10)]
        num_classes = 3
        
        model = ModelFactory.create_model(
            layer_type=Layer.GCN,
            input_dim=input_dim,
            output_dim=output_dim,
            support=support,
            num_classes=num_classes
        )
        
        assert isinstance(model, th.nn.Module)
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
    
    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        with pytest.raises(ValueError):
            ModelFactory.create_model(
                layer_type="UNSUPPORTED",
                input_dim=10,
                output_dim=5,
                support=[th.randn(10, 10)],
                num_classes=3
            )
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = Mock()
        model.parameters.return_value = [th.randn(10, 5)]
        
        optimizer = ModelFactory.create_optimizer(model, lr=0.01)
        
        assert isinstance(optimizer, th.optim.Optimizer)
    
    def test_create_criterion(self):
        """Test criterion creation."""
        criterion = ModelFactory.create_criterion()
        
        assert isinstance(criterion, th.nn.Module)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ConfigManager(
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
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        assert self.config.dataset == "test_dataset"
        assert self.config.path == "/test/path"
        assert self.config.log_dir == "/test/log"
        assert self.config.lr == 0.01
        assert self.config.nhid == 100
        assert self.config.seed == 42
        assert self.config.epoches == 10
        assert self.config.dropout == 0.3
        assert self.config.val_ratio == 0.2
        assert self.config.layer == Layer.GCN
        assert self.config.early_stopping == 5
        assert self.config.device == "cpu"
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        assert self.config.validate_config() == True
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        # Test invalid learning rate
        self.config.lr = -0.01
        assert self.config.validate_config() == False
        
        # Test invalid dropout
        self.config.lr = 0.01
        self.config.dropout = 1.5
        assert self.config.validate_config() == False
    
    def test_create_args(self):
        """Test args creation."""
        args = self.config.create_args()
        
        assert args.dataset == "test_dataset"
        assert args.lr == 0.01
        assert args.nhid == 100
        assert args.epoches == 10
        assert args.dropout == 0.3
        assert args.val_ratio == 0.2
        assert args.layer == Layer.GCN
        assert args.early_stopping == 5
        assert args.device == "cpu"
    
    def test_get_config_dict(self):
        """Test getting configuration dictionary."""
        config_dict = self.config.get_config_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['dataset'] == "test_dataset"
        assert config_dict['lr'] == 0.01
        assert config_dict['nhid'] == 100
    
    def test_update_config(self):
        """Test configuration update."""
        self.config.update_config(lr=0.02, nhid=200)
        
        assert self.config.lr == 0.02
        assert self.config.nhid == 200


class TestTextGCNTrainer:
    """Test cases for refactored TextGCNTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_dir = tempfile.mkdtemp()
        self.config = ConfigManager(
            dataset="test_dataset",
            path="/test/path",
            log_dir=self.log_dir,
            builder="tfidf",
            lr=0.01,
            nhid=50,
            seed=42,
            epoches=2,
            dropout=0.3,
            val_ratio=0.2,
            layer=Layer.GCN,
            early_stopping=1,
            device="cpu"
        )
        
        self.trainer = TextGCNTrainer(config=self.config)
    
    def test_initialization(self):
        """Test TextGCNTrainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.device == "cpu"
        assert isinstance(self.trainer.data_processor, DataProcessor)
        assert isinstance(self.trainer.model_evaluator, ModelEvaluator)
        assert isinstance(self.trainer.metrics_visualizer, MetricsVisualizer)
    
    @patch('text4gcn.models.refactored.textgcn_trainer.LoadData')
    def test_fit(self, mock_load_data):
        """Test model fitting."""
        # Mock LoadData
        mock_loader = Mock()
        mock_loader.load_corpus.return_value = self._create_mock_predata()
        mock_load_data.return_value = mock_loader
        
        # Mock model factory
        with patch('text4gcn.models.refactored.textgcn_trainer.ModelFactory') as mock_factory:
            mock_model = Mock()
            mock_factory.create_model.return_value = mock_model
            mock_factory.create_optimizer.return_value = Mock()
            mock_factory.create_criterion.return_value = Mock()
            
            # Mock model trainer
            with patch('text4gcn.models.refactored.textgcn_trainer.ModelTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer.train.return_value = {'train_loss': [0.5, 0.4], 'train_acc': [0.6, 0.7]}
                mock_trainer.save_model.return_value = '/test/model.pth'
                mock_trainer_class.return_value = mock_trainer
                
                # Mock evaluator
                with patch.object(self.trainer.model_evaluator, 'test_model') as mock_test:
                    mock_test.return_value = (0.3, 0.8, {'accuracy': 0.8})
                    
                    # Mock visualizer
                    with patch.object(self.trainer.metrics_visualizer, 'create_training_report'):
                        result = self.trainer.fit(self._create_mock_predata())
        
        assert 'training_history' in result
        assert 'test_loss' in result
        assert 'test_accuracy' in result
        assert 'metrics' in result
        assert 'model_path' in result
    
    def _create_mock_predata(self):
        """Create mock preprocessed data for testing."""
        n_nodes = 50
        n_classes = 3
        
        # Create mock data
        adj = csr_matrix(np.random.rand(n_nodes, n_nodes))
        features = np.random.rand(n_nodes, 20)
        y_train = np.random.rand(30, n_classes)
        y_val = np.random.rand(10, n_classes)
        y_test = np.random.rand(10, n_classes)
        
        train_mask = np.zeros(n_nodes, dtype=bool)
        train_mask[:30] = True
        val_mask = np.zeros(n_nodes, dtype=bool)
        val_mask[30:40] = True
        test_mask = np.zeros(n_nodes, dtype=bool)
        test_mask[40:] = True
        
        return (adj, features, y_train, y_val, y_test,
                train_mask, val_mask, test_mask, 30, 10)


# Integration tests
class TestRefactoredIntegration:
    """Integration tests for refactored components."""
    
    def test_data_processor_integration(self):
        """Test DataProcessor integration."""
        processor = DataProcessor(device='cpu')
        
        # Create mock data
        n_nodes = 20
        adj = csr_matrix(np.random.rand(n_nodes, n_nodes))
        features = np.random.rand(n_nodes, 10)
        y_train = np.random.rand(10, 3)
        y_val = np.random.rand(5, 3)
        y_test = np.random.rand(5, 3)
        
        train_mask = np.zeros(n_nodes, dtype=bool)
        train_mask[:10] = True
        val_mask = np.zeros(n_nodes, dtype=bool)
        val_mask[10:15] = True
        test_mask = np.zeros(n_nodes, dtype=bool)
        test_mask[15:] = True
        
        predata = (adj, features, y_train, y_val, y_test,
                  train_mask, val_mask, test_mask, 10, 5)
        
        # Test full pipeline
        processor.prepare_data(predata)
        processor.convert_tensor()
        data = processor.get_data_for_training()
        
        assert 'features' in data
        assert 'support' in data
        assert isinstance(data['features'], th.Tensor)
        assert isinstance(data['support'], list)
    
    def test_model_factory_integration(self):
        """Test ModelFactory integration."""
        # Test GCN model creation
        model = ModelFactory.create_model(
            layer_type=Layer.GCN,
            input_dim=10,
            output_dim=5,
            support=[th.randn(10, 10)],
            num_classes=3,
            dropout_rate=0.5
        )
        
        # Test forward pass
        x = th.randn(10, 10)
        output = model(x)
        
        assert output.shape == (10, 3)  # num_classes
    
    def test_config_manager_integration(self):
        """Test ConfigManager integration."""
        config = ConfigManager(
            dataset="test",
            path="/test",
            log_dir="/test/log",
            builder="tfidf"
        )
        
        # Test validation
        assert config.validate_config()
        
        # Test args creation
        args = config.create_args()
        assert hasattr(args, 'dataset')
        assert hasattr(args, 'lr')
        assert hasattr(args, 'nhid')


if __name__ == '__main__':
    pytest.main([__file__])

