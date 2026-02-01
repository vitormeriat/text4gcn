"""
Service Interfaces for Text4GCN.

This module defines service interfaces following the Interface Segregation Principle.
Each interface represents a specific responsibility and can be implemented independently.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union
import torch as th
import numpy as np
from pathlib import Path


class IDataProcessor(ABC):
    """Interface for data processing services."""
    
    @abstractmethod
    def prepare_data(self, predata: Tuple) -> None:
        """
        Prepare data from preprocessed data tuple.
        
        Args:
            predata: Tuple containing preprocessed data
        """
        pass
    
    @abstractmethod
    def convert_tensor(self) -> None:
        """Convert data to PyTorch tensors."""
        pass
    
    @abstractmethod
    def get_data_for_training(self) -> Dict[str, Any]:
        """
        Get processed data for training.
        
        Returns:
            Dictionary containing processed data
        """
        pass


class IModelTrainer(ABC):
    """Interface for model training services."""
    
    @abstractmethod
    def train_epoch(self, features: th.Tensor, y_train: th.Tensor, 
                   train_mask: th.Tensor, tm_train_mask: th.Tensor) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            features: Input features
            y_train: Training labels
            train_mask: Training mask
            tm_train_mask: Training mask for loss calculation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        pass
    
    @abstractmethod
    def train(self, data: Dict[str, Any], evaluator: 'IModelEvaluator') -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            data: Dictionary containing training data
            evaluator: Model evaluator instance
            
        Returns:
            Dictionary containing training history
        """
        pass
    
    @abstractmethod
    def save_model(self) -> str:
        """
        Save the trained model.
        
        Returns:
            Path to saved model
        """
        pass


class IModelEvaluator(ABC):
    """Interface for model evaluation services."""
    
    @abstractmethod
    def evaluate_model(self, model: th.nn.Module, criterion: th.nn.Module, 
                      features: th.Tensor, labels: th.Tensor, mask: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
        """
        Evaluate model on given data.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            features: Input features
            labels: True labels
            mask: Evaluation mask
            
        Returns:
            Tuple of (loss, accuracy, predictions, labels, duration)
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, test_labels: List[int], test_pred: List[int]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            test_labels: True labels
            test_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def test_model(self, model: th.nn.Module, criterion: th.nn.Module, 
                  data: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
        """
        Test model on test set.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            data: Dictionary containing test data
            
        Returns:
            Tuple of (test_loss, test_accuracy, metrics_dict)
        """
        pass


class IMetricsVisualizer(ABC):
    """Interface for metrics visualization services."""
    
    @abstractmethod
    def plot_accuracy(self, train_acc_hist: List[float], eval_acc_hist: List[float], 
                     train_loss_hist: List[float], eval_loss_hist: List[float]) -> None:
        """
        Plot accuracy and loss curves.
        
        Args:
            train_acc_hist: Training accuracy history
            eval_acc_hist: Validation accuracy history
            train_loss_hist: Training loss history
            eval_loss_hist: Validation loss history
        """
        pass
    
    @abstractmethod
    def plot_confusion_matrix(self, confusion_matrix: Any, class_names: List[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names for labels
        """
        pass
    
    @abstractmethod
    def create_training_report(self, history: Dict[str, List[float]], 
                            metrics: Dict[str, float]) -> None:
        """
        Create comprehensive training report.
        
        Args:
            history: Training history dictionary
            metrics: Final metrics dictionary
        """
        pass


class IConfigurationService(ABC):
    """Interface for configuration services."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None, type_hint: Optional[type] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            type_hint: Expected type
            
        Returns:
            Configuration value
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        pass
    
    @abstractmethod
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            schema: Validation schema
            
        Returns:
            List of validation errors
        """
        pass


class ILoggingService(ABC):
    """Interface for logging services."""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """
        Log info message.
        
        Args:
            message: Message to log
        """
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """
        Log warning message.
        
        Args:
            message: Message to log
        """
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """
        Log error message.
        
        Args:
            message: Message to log
        """
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """
        Log debug message.
        
        Args:
            message: Message to log
        """
        pass


class IFileService(ABC):
    """Interface for file operations services."""
    
    @abstractmethod
    def create_directory(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Create directory.
        
        Args:
            path: Directory path
            overwrite: Whether to overwrite existing directory
        """
        pass
    
    @abstractmethod
    def write_file(self, path: Union[str, Path], content: str, mode: str = 'w') -> None:
        """
        Write content to file.
        
        Args:
            path: File path
            content: Content to write
            mode: File mode
        """
        pass
    
    @abstractmethod
    def read_file(self, path: Union[str, Path]) -> str:
        """
        Read file content.
        
        Args:
            path: File path
            
        Returns:
            File content
        """
        pass
    
    @abstractmethod
    def file_exists(self, path: Union[str, Path]) -> bool:
        """
        Check if file exists.
        
        Args:
            path: File path
            
        Returns:
            True if file exists
        """
        pass


