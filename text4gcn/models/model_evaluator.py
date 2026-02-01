"""
Model Evaluator - Single Responsibility: Evaluate model performance.

This class is responsible for:
- Evaluating model on validation/test sets
- Computing metrics
- Performance analysis
"""

import torch as th
import numpy as np
from time import time
from typing import Tuple, List, Dict, Any
from sklearn import metrics


class ModelEvaluator:
    """Responsible for model evaluation and metrics computation."""
    
    def __init__(self, device: str):
        """
        Initialize ModelEvaluator.
        
        Args:
            device: Device to evaluate on (cpu/cuda)
        """
        self.device = device
    
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
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            test_labels: True labels
            test_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        print("\nTest Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        
        print("\nMacro average Test Precision, Recall and F1-Score...")
        macro_metrics = metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro')
        print(macro_metrics)
        
        print("\nMicro average Test Precision, Recall and F1-Score...")
        micro_metrics = metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro')
        print(micro_metrics)
        
        print("\nWeighted average Test Precision, Recall and F1-Score...")
        weighted_metrics = metrics.precision_recall_fscore_support(test_labels, test_pred, average='weighted')
        print(weighted_metrics)
        
        print("\nConfusion Matrix...")
        cm = metrics.confusion_matrix(test_labels, test_pred)
        print(cm)
        
        # Individual metrics
        f1_macro = metrics.f1_score(test_labels, test_pred, average="macro")
        precision_macro = metrics.precision_score(test_labels, test_pred, average="macro")
        recall_macro = metrics.recall_score(test_labels, test_pred, average="macro")
        accuracy = metrics.accuracy_score(test_labels, test_pred)
        
        print(f"\nF1-Score (Macro): {f1_macro:.4f}")
        print(f"Precision-Score (Macro): {precision_macro:.4f}")
        print(f"Recall-Score (Macro): {recall_macro:.4f}")
        print(f"Accuracy-Score: {accuracy:.4f}")
        
        return {
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'macro_metrics': macro_metrics,
            'micro_metrics': micro_metrics,
            'weighted_metrics': weighted_metrics
        }
    
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
        # Evaluate on test set
        test_loss, test_acc, pred, labels, test_duration = self.evaluate_model(
            model, criterion, data['features'], data['y_test'], data['test_mask']
        )
        
        print("\nTest set results: loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(
            test_loss, test_acc, test_duration))
        
        # Extract test predictions and labels
        test_pred = []
        test_labels = []
        for i in range(len(data['test_mask'])):
            if data['test_mask'][i]:
                test_pred.append(pred[i])
                test_labels.append(np.argmax(labels[i]))
        
        # Compute metrics
        metrics_dict = self.compute_metrics(test_labels, test_pred)
        
        # Print embeddings info
        tmp = model.layer1.embedding.numpy()
        word_embeddings = tmp[data['train_size']: data['adj'].shape[0] - data['test_size']]
        train_doc_embeddings = tmp[:data['train_size']]
        test_doc_embeddings = tmp[data['adj'].shape[0] - data['test_size']:]
        
        print('\nEmbeddings:')
        print(f'Word_embeddings: {len(word_embeddings)}')
        print(f'Train_doc_embeddings: {len(train_doc_embeddings)}')
        print(f'Test_doc_embeddings: {len(test_doc_embeddings)}')
        
        return test_loss, test_acc, metrics_dict
