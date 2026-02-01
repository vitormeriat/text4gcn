"""
Refactored TextGCN Trainer - Composition of specialized components.

This class orchestrates the training process using composition of:
- DataProcessor: Data processing and preparation
- ModelTrainer: Model training
- ModelEvaluator: Model evaluation
- MetricsVisualizer: Metrics visualization
- ModelFactory: Model creation
- ConfigManager: Configuration management
"""

import torch as th
from typing import Dict, Any, Tuple
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .metrics_visualizer import MetricsVisualizer
from .model_factory import ModelFactory
from .config_manager import ConfigManager
from text4gcn.models.utils import LoadData


class TextGCNTrainer:
    """
    Refactored TextGCN Trainer using composition pattern.
    
    This class follows Single Responsibility Principle by delegating
    specific responsibilities to specialized components.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize TextGCNTrainer with configuration.
        
        Args:
            config: ConfigManager instance containing all configuration
        """
        self.config = config
        self.device = config.device
        
        # Initialize components
        self.data_processor = DataProcessor(device=self.device)
        self.model_evaluator = ModelEvaluator(device=self.device)
        self.metrics_visualizer = MetricsVisualizer(log_dir=config.log_dir)
        
        # Will be initialized after data processing
        self.model = None
        self.model_trainer = None
        self.data = None
    
    def fit(self, predata: Tuple) -> Dict[str, Any]:
        """
        Train the model using the refactored approach.
        
        Args:
            predata: Preprocessed data tuple
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Validate configuration
        if not self.config.validate_config():
            raise ValueError("Invalid configuration")
        
        # Print configuration
        self.config.print_config()
        
        # Process data
        self.data_processor.prepare_data(predata)
        self.data_processor.convert_tensor()
        self.data = self.data_processor.get_data_for_training()
        
        # Create model
        self.model = ModelFactory.create_model(
            layer_type=self.config.layer,
            input_dim=self.data['features'].shape[0],
            output_dim=self.config.nhid,
            support=self.data['support'],
            num_classes=self.data['y_train'].shape[1],
            dropout_rate=self.config.dropout
        ).to(self.device)
        
        # Create optimizer and criterion
        optimizer = ModelFactory.create_optimizer(self.model, lr=self.config.lr)
        criterion = ModelFactory.create_criterion()
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            max_epochs=self.config.epoches,
            early_stopping=self.config.early_stopping,
            log_dir=self.config.log_dir
        )
        
        # Print model structure
        print("\nModel structure:\n\n", self.model, "\n")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}")
        
        # Train model
        training_history = self.model_trainer.train(self.data, self.model_evaluator)
        
        # Test model
        test_loss, test_acc, metrics = self.model_evaluator.test_model(
            self.model, criterion, self.data
        )
        
        # Create visualizations
        self.metrics_visualizer.create_training_report(training_history, metrics)
        
        return {
            'training_history': training_history,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'metrics': metrics,
            'model_path': self.model_trainer.save_model()
        }
    
    def get_model(self) -> th.nn.Module:
        """
        Get the trained model.
        
        Returns:
            Trained PyTorch model
        """
        return self.model
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get processed data.
        
        Returns:
            Dictionary containing processed data
        """
        return self.data
