"""
Metrics Visualizer - Single Responsibility: Visualize training metrics.

This class is responsible for:
- Plotting training/validation curves
- Saving visualization files
- Metrics visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


class MetricsVisualizer:
    """Responsible for visualizing training metrics and results."""
    
    def __init__(self, log_dir: str):
        """
        Initialize MetricsVisualizer.
        
        Args:
            log_dir: Directory to save visualizations
        """
        self.log_dir = log_dir
    
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
        plt.savefig(f'{self.log_dir}/metrics-acc.png', dpi=250, bbox_inches='tight')
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
        plt.savefig(f'{self.log_dir}/metrics-loss.png', dpi=250, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix: Any, class_names: List[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names for labels
        """
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
        plt.savefig(f'{self.log_dir}/confusion-matrix.png', dpi=250, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_summary(self, metrics_dict: Dict[str, float]) -> None:
        """
        Plot metrics summary.
        
        Args:
            metrics_dict: Dictionary containing various metrics
        """
        # Extract key metrics
        metrics_names = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']
        metrics_values = [metrics_dict.get(name, 0.0) for name in metrics_names]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/metrics-summary.png', dpi=250, bbox_inches='tight')
        plt.close()
    
    def create_training_report(self, history: Dict[str, List[float]], 
                            metrics: Dict[str, float]) -> None:
        """
        Create comprehensive training report.
        
        Args:
            history: Training history dictionary
            metrics: Final metrics dictionary
        """
        # Plot all visualizations
        self.plot_accuracy(
            history['train_acc'], history['eval_acc'],
            history['train_loss'], history['eval_loss']
        )
        
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        self.plot_metrics_summary(metrics)
        
        print(f"All visualizations saved to {self.log_dir}/")
