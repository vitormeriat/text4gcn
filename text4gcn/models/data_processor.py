"""
Data Processor - Single Responsibility: Process and prepare data for training.

This class is responsible for:
- Loading and preprocessing data
- Converting data to tensors
- Preparing features and labels
"""

import scipy.sparse as sp
import numpy as np
import torch as th
from tabulate import tabulate
from typing import Tuple, Dict, Any
from text4gcn.models.utils import preprocess_features, preprocess_adj


class DataProcessor:
    """Responsible for data processing and preparation."""
    
    def __init__(self, device: str):
        """
        Initialize DataProcessor.
        
        Args:
            device: Device to process data on (cpu/cuda)
        """
        self.device = device
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
        """
        Prepare data from preprocessed data tuple.
        
        Args:
            predata: Tuple containing (adj, features, y_train, y_val, y_test, 
                     train_mask, val_mask, test_mask, train_size, test_size)
        """
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
    
    def convert_tensor(self) -> None:
        """Convert data to PyTorch tensors and move to device."""
        print("\nData statistics\n")
        print(tabulate([
            ["Edges", "Classes", "Train samples", "Val samples", "Test samples"],
            [self.features.shape[0], self.y_train.shape[1], 
             self.train_mask.sum(), self.val_mask.sum(), self.test_mask.sum()]],
            headers="firstrow"))
        
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
    
    def get_data_for_training(self) -> Dict[str, Any]:
        """
        Get processed data for training.
        
        Returns:
            Dictionary containing all processed data
        """
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
