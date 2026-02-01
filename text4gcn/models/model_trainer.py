"""
Model Trainer - Single Responsibility: Train neural network models.

This class is responsible for:
- Training the model
- Managing training loop
- Early stopping
- Model saving
"""

import torch as th
import numpy as np
from time import time
from typing import Dict, Any, List, Tuple
from text4gcn.models.utils import unique_name


class ModelTrainer:
    """Responsible for training neural network models."""
    
    def __init__(self, model: th.nn.Module, criterion: th.nn.Module, 
                 optimizer: th.optim.Optimizer, device: str, 
                 max_epochs: int, early_stopping: int, log_dir: str):
        """
        Initialize ModelTrainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            max_epochs: Maximum number of epochs
            early_stopping: Early stopping patience
            log_dir: Directory for logging
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.log_dir = log_dir
        
        # Training history
        self.train_loss_hist = []
        self.eval_loss_hist = []
        self.train_acc_hist = []
        self.eval_acc_hist = []
        self.val_losses = []
    
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
    
    def train(self, data: Dict[str, Any], evaluator: 'ModelEvaluator') -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            data: Dictionary containing training data
            evaluator: ModelEvaluator instance for validation
            
        Returns:
            Dictionary containing training history
        """
        t1 = time()
        
        print(f"\n{'∎'*20} Starting Training")
        
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
            print("Epoch:{:04d}, train_loss={:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}".format(
                epoch + 1, loss, acc, val_loss, val_acc, time() - epoch_start_time))
            
            # Early stopping check
            if epoch > self.early_stopping and self.val_losses[-1] > np.mean(self.val_losses[-(self.early_stopping + 1):-1]):
                print(f'Early stopping... Validation loss decreased ({val_loss:.6f}). Saving model...')
                self.save_model()
                break
            
            # Store history
            self.train_loss_hist.append(float(loss))
            self.train_acc_hist.append(float(acc))
            self.eval_loss_hist.append(float(val_loss))
            self.eval_acc_hist.append(float(val_acc))
        
        print('\nCompleted training batch', epoch+1, 'Training Loss is: %.4f' % loss, 
              'Validation Loss is: %.4f' % val_loss, 'Accuracy is %.4f' % acc)
        
        print("\nOptimization Finished!")
        
        return {
            'train_loss': self.train_loss_hist,
            'train_acc': self.train_acc_hist,
            'eval_loss': self.eval_loss_hist,
            'eval_acc': self.eval_acc_hist
        }
    
    def save_model(self) -> str:
        """
        Save the trained model.
        
        Returns:
            Path to saved model
        """
        name = unique_name()
        path = f"{self.log_dir}/{name}.pth"
        th.save(self.model.state_dict(), path)
        return path
