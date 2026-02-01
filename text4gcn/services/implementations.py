"""
Service Implementations for Text4GCN with Dependency Injection.

This module contains concrete implementations of service interfaces
using dependency injection for loose coupling.
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional, Union
from pathlib import Path
import os
import time
from sklearn import metrics

from .interfaces import (
    IDataProcessor, IModelTrainer, IModelEvaluator, IMetricsVisualizer,
    IConfigurationService, ILoggingService, IFileService
)
from text4gcn.models.utils import preprocess_features, preprocess_adj, unique_name


class DataProcessorService(IDataProcessor):
    """Data processor service implementation."""
    
    def __init__(self, device: str, logging_service: ILoggingService):
        """
        Initialize data processor service.
        
        Args:
            device: Device to process data on
            logging_service: Logging service
        """
        self.device = device
        self.logging_service = logging_service
        self.features = None
        self.adj = None
        self.support = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train_size = None
        self.test_size = None
        self.tm_train_mask = None
    
    def prepare_data(self, predata: Tuple) -> None:
        """Prepare data from preprocessed data tuple."""
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = predata
        
        # Featureless approach - use identity matrix
        self.features = sp.identity(features.shape[0])
        # Preprocess features
        self.features = preprocess_features(self.features)
        # Adjacency matrix
        self.adj = adj
        self.support = [preprocess_adj(adj)]
        
        # Store masks and labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_size = train_size
        self.test_size = test_size
        
        self.logging_service.info("Data prepared successfully")
    
    def convert_tensor(self) -> None:
        """Convert data to PyTorch tensors and move to device."""
        self.logging_service.info("Converting data to tensors")
        
        # Convert to tensors and move to device
        self.features = th.from_numpy(self.features).float().to(self.device)
        self.y_train = th.from_numpy(self.y_train).float().to(self.device)
        self.y_val = th.from_numpy(self.y_val).float().to(self.device)
        self.y_test = th.from_numpy(self.y_test).float().to(self.device)
        self.train_mask = th.from_numpy(self.train_mask).float().to(self.device)
        self.val_mask = th.from_numpy(self.val_mask).float().to(self.device)
        self.test_mask = th.from_numpy(self.test_mask).float().to(self.device)
        
        # Create training mask for loss calculation
        self.tm_train_mask = th.transpose(th.unsqueeze(self.train_mask, 0), 1, 0).repeat(1, self.y_train.shape[1])
        
        # Convert support to tensors
        self.support = [th.Tensor(self.support[i]) for i in range(len(self.support))]
        
        self.logging_service.info("Data converted to tensors successfully")
    
    def get_data_for_training(self) -> Dict[str, Any]:
        """Get processed data for training."""
        return {
            'features': self.features,
            'support': self.support,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'tm_train_mask': self.tm_train_mask,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'adj': self.adj
        }


class ModelTrainerService(IModelTrainer):
    """Model trainer service implementation."""
    
    def __init__(self, device: str, logging_service: ILoggingService, file_service: IFileService):
        """
        Initialize model trainer service.
        
        Args:
            device: Device to train on
            logging_service: Logging service
            file_service: File service
        """
        self.device = device
        self.logging_service = logging_service
        self.file_service = file_service
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.max_epochs = None
        self.early_stopping = None
        self.log_dir = None
        
        # Training history
        self.train_loss_hist = []
        self.eval_loss_hist = []
        self.train_acc_hist = []
        self.eval_acc_hist = []
        self.val_losses = []
    
    def set_training_config(self, model: th.nn.Module, criterion: th.nn.Module, 
                           optimizer: th.optim.Optimizer, max_epochs: int, 
                           early_stopping: int, log_dir: str):
        """Set training configuration."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.log_dir = log_dir
    
    def train_epoch(self, features: th.Tensor, y_train: th.Tensor, 
                   train_mask: th.Tensor, tm_train_mask: th.Tensor) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        logits = self.model(features)
        
        # Compute loss
        loss = self.criterion(logits * tm_train_mask, th.max(y_train, 1)[1])
        
        # Compute accuracy
        try:
            acc = ((th.max(logits, 1)[1] == th.max(y_train, 1)[1]).float() * train_mask).sum().item() / train_mask.sum().item()
        except ZeroDivisionError:
            acc = 0.0
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc
    
    def train(self, data: Dict[str, Any], evaluator: IModelEvaluator) -> Dict[str, List[float]]:
        """Train the model."""
        t1 = time()
        
        self.logging_service.info("Starting model training")
        
        for epoch in range(self.max_epochs):
            epoch_start_time = time()
            
            # Training step
            loss, acc = self.train_epoch(
                data['features'], data['y_train'], 
                data['train_mask'], data['tm_train_mask']
            )
            
            # Validation
            val_loss, val_acc, _, _, duration = evaluator.evaluate_model(
                self.model, self.criterion, data['features'], 
                data['y_val'], data['val_mask']
            )
            
            self.val_losses.append(val_loss)
            
            # Print progress
            self.logging_service.info(f"Epoch {epoch + 1}: train_loss={loss:.5f}, train_acc={acc:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}")
            
            # Early stopping check
            if epoch > self.early_stopping and self.val_losses[-1] > np.mean(self.val_losses[-(self.early_stopping + 1):-1]):
                self.logging_service.info(f'Early stopping at epoch {epoch + 1}')
                self.save_model()
                break
            
            # Store history
            self.train_loss_hist.append(float(loss))
            self.train_acc_hist.append(float(acc))
            self.eval_loss_hist.append(float(val_loss))
            self.eval_acc_hist.append(float(val_acc))
        
        self.logging_service.info("Model training completed")
        
        return {
            'train_loss': self.train_loss_hist,
            'train_acc': self.train_acc_hist,
            'eval_loss': self.eval_loss_hist,
            'eval_acc': self.eval_acc_hist
        }
    
    def save_model(self) -> str:
        """Save the trained model."""
        name = unique_name()
        path = f"{self.log_dir}/{name}.pth"
        
        # Ensure directory exists
        self.file_service.create_directory(Path(self.log_dir), overwrite=False)
        
        # Save model
        th.save(self.model.state_dict(), path)
        
        self.logging_service.info(f"Model saved to {path}")
        return path


class ModelEvaluatorService(IModelEvaluator):
    """Model evaluator service implementation."""
    
    def __init__(self, device: str, logging_service: ILoggingService):
        """
        Initialize model evaluator service.
        
        Args:
            device: Device to evaluate on
            logging_service: Logging service
        """
        self.device = device
        self.logging_service = logging_service
    
    def evaluate_model(self, model: th.nn.Module, criterion: th.nn.Module, 
                      features: th.Tensor, labels: th.Tensor, mask: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
        """Evaluate model on given data."""
        t_test = time()
        
        model.eval()
        with th.no_grad():
            logits = model(features)
            t_mask = th.from_numpy(np.array(mask * 1., dtype=np.float32))
            tm_mask = th.transpose(th.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = criterion(logits * tm_mask, th.max(labels, 1)[1])
            pred = th.max(logits, 1)[1]
            
            try:
                acc = ((pred == th.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
            except ZeroDivisionError:
                acc = 0.0
        
        return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time() - t_test)
    
    def compute_metrics(self, test_labels: List[int], test_pred: List[int]) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        self.logging_service.info("Computing evaluation metrics")
        
        # Compute metrics
        f1_macro = metrics.f1_score(test_labels, test_pred, average="macro")
        precision_macro = metrics.precision_score(test_labels, test_pred, average="macro")
        recall_macro = metrics.recall_score(test_labels, test_pred, average="macro")
        accuracy = metrics.accuracy_score(test_labels, test_pred)
        
        self.logging_service.info(f"F1-Score (Macro): {f1_macro:.4f}")
        self.logging_service.info(f"Precision-Score (Macro): {precision_macro:.4f}")
        self.logging_service.info(f"Recall-Score (Macro): {recall_macro:.4f}")
        self.logging_service.info(f"Accuracy-Score: {accuracy:.4f}")
        
        return {
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'accuracy': accuracy,
            'confusion_matrix': metrics.confusion_matrix(test_labels, test_pred),
            'macro_metrics': metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'),
            'micro_metrics': metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'),
            'weighted_metrics': metrics.precision_recall_fscore_support(test_labels, test_pred, average='weighted')
        }
    
    def test_model(self, model: th.nn.Module, criterion: th.nn.Module, 
                  data: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
        """Test model on test set."""
        # Evaluate on test set
        test_loss, test_acc, pred, labels, test_duration = self.evaluate_model(
            model, criterion, data['features'], data['y_test'], data['test_mask']
        )
        
        self.logging_service.info(f"Test set results: loss={test_loss:.5f}, accuracy={test_acc:.5f}, time={test_duration:.5f}")
        
        # Extract test predictions and labels
        test_pred = []
        test_labels = []
        for i in range(len(data['test_mask'])):
            if data['test_mask'][i]:
                test_pred.append(pred[i])
                test_labels.append(np.argmax(labels[i]))
        
        # Compute metrics
        metrics_dict = self.compute_metrics(test_labels, test_pred)
        
        return test_loss, test_acc, metrics_dict


class MetricsVisualizerService(IMetricsVisualizer):
    """Metrics visualizer service implementation."""
    
    def __init__(self, log_dir: str, file_service: IFileService, logging_service: ILoggingService):
        """
        Initialize metrics visualizer service.
        
        Args:
            log_dir: Directory to save visualizations
            file_service: File service
            logging_service: Logging service
        """
        self.log_dir = log_dir
        self.file_service = file_service
        self.logging_service = logging_service
    
    def plot_accuracy(self, train_acc_hist: List[float], eval_acc_hist: List[float], 
                     train_loss_hist: List[float], eval_loss_hist: List[float]) -> None:
        """Plot accuracy and loss curves."""
        self.logging_service.info("Creating accuracy and loss plots")
        
        epochs = range(1, len(train_acc_hist) + 1)
        
        # Plot accuracy
        plt.figure(figsize=(16, 10))
        plt.plot(epochs, train_acc_hist, 'b', label='Training acc')
        plt.plot(epochs, eval_acc_hist, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        acc_path = f'{self.log_dir}/metrics-acc.png'
        plt.savefig(acc_path, dpi=250, bbox_inches='tight')
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(16, 10))
        plt.plot(epochs, train_loss_hist, 'b', label='Training loss')
        plt.plot(epochs, eval_loss_hist, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        loss_path = f'{self.log_dir}/metrics-loss.png'
        plt.savefig(loss_path, dpi=250, bbox_inches='tight')
        plt.close()
        
        self.logging_service.info(f"Plots saved to {self.log_dir}")
    
    def plot_confusion_matrix(self, confusion_matrix: Any, class_names: List[str] = None) -> None:
        """Plot confusion matrix."""
        self.logging_service.info("Creating confusion matrix plot")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2.
        for i, j in np.ndindex(confusion_matrix.shape):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        cm_path = f'{self.log_dir}/confusion-matrix.png'
        plt.savefig(cm_path, dpi=250, bbox_inches='tight')
        plt.close()
        
        self.logging_service.info(f"Confusion matrix saved to {cm_path}")
    
    def create_training_report(self, history: Dict[str, List[float]], 
                            metrics: Dict[str, float]) -> None:
        """Create comprehensive training report."""
        self.logging_service.info("Creating training report")
        
        # Plot all visualizations
        self.plot_accuracy(
            history['train_acc'], history['eval_acc'],
            history['train_loss'], history['eval_loss']
        )
        
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        self.logging_service.info(f"Training report created in {self.log_dir}")


class ConfigurationService(IConfigurationService):
    """Configuration service implementation."""
    
    def __init__(self, config_manager):
        """
        Initialize configuration service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
    
    def get(self, key: str, default: Any = None, type_hint: Optional[type] = None) -> Any:
        """Get configuration value."""
        return self.config_manager.get(key, default, type_hint)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config_manager.set(key, value)
    
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        return self.config_manager.validate(schema)


class LoggingService(ILoggingService):
    """Logging service implementation."""
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize logging service.
        
        Args:
            log_level: Logging level
        """
        self.log_level = log_level
    
    def info(self, message: str) -> None:
        """Log info message."""
        print(f"[INFO] {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        print(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        print(f"[ERROR] {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        if self.log_level == "DEBUG":
            print(f"[DEBUG] {message}")


class FileService(IFileService):
    """File service implementation."""
    
    def __init__(self, logging_service: ILoggingService):
        """
        Initialize file service.
        
        Args:
            logging_service: Logging service
        """
        self.logging_service = logging_service
    
    def create_directory(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """Create directory."""
        path = Path(path)
        if path.exists() and not overwrite:
            self.logging_service.warning(f"Directory {path} already exists")
            return
        
        path.mkdir(parents=True, exist_ok=True)
        self.logging_service.info(f"Directory created: {path}")
    
    def write_file(self, path: Union[str, Path], content: str, mode: str = 'w') -> None:
        """Write content to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, mode) as f:
            f.write(content)
        
        self.logging_service.info(f"File written: {path}")
    
    def read_file(self, path: Union[str, Path]) -> str:
        """Read file content."""
        path = Path(path)
        with open(path, 'r') as f:
            content = f.read()
        
        self.logging_service.info(f"File read: {path}")
        return content
    
    def file_exists(self, path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return Path(path).exists()


