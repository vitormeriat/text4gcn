"""
Config Manager - Single Responsibility: Manage configuration.

This class is responsible for:
- Managing model configuration
- Validating configuration parameters
- Configuration serialization/deserialization
"""

import argparse
import numpy as np
from typing import Dict, Any, Optional
from text4gcn.models.layers import Layer


class ConfigManager:
    """Responsible for managing model configuration."""
    
    def __init__(self, dataset: str, path: str, log_dir: str, builder: str,
                 lr: float = 0.02, nhid: int = 200, seed: int = 1234,
                 epoches: int = 200, dropout: float = 0.5, val_ratio: float = 0.1,
                 layer: Layer = Layer.GCN, early_stopping: int = 10,
                 device: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            dataset: Dataset name
            path: Dataset path
            log_dir: Log directory
            builder: Builder type
            lr: Learning rate
            nhid: Hidden dimension
            seed: Random seed
            epoches: Number of epochs
            dropout: Dropout rate
            val_ratio: Validation ratio
            layer: Layer type
            early_stopping: Early stopping patience
            device: Device (cpu/cuda)
        """
        self.dataset = dataset
        self.path = path
        self.log_dir = log_dir
        self.builder = builder
        self.lr = lr
        self.nhid = nhid
        self.seed = seed
        self.epoches = epoches
        self.dropout = dropout
        self.val_ratio = val_ratio
        self.layer = layer
        self.early_stopping = early_stopping
        self.device = device
    
    def create_args(self) -> argparse.Namespace:
        """
        Create argparse.Namespace object for compatibility.
        
        Returns:
            argparse.Namespace object
        """
        parser = argparse.ArgumentParser(description='TextGCN')
        args = parser.parse_args([])  # Empty args
        
        # Set all configuration parameters
        args.early_stopping = self.early_stopping
        args.seed = np.random.randint(1, 100000) if self.seed is None else self.seed
        args.val_ratio = self.val_ratio
        args.max_epoch = self.epoches
        args.dataset = self.dataset
        args.dropout = self.dropout
        args.log_dir = self.log_dir
        args.layer = self.layer
        args.nhid = self.nhid
        args.lr = self.lr
        args.device = self.device
        
        return args
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate numeric parameters
            assert self.lr > 0, "Learning rate must be positive"
            assert self.nhid > 0, "Hidden dimension must be positive"
            assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
            assert 0 < self.val_ratio < 1, "Validation ratio must be between 0 and 1"
            assert self.epoches > 0, "Epochs must be positive"
            assert self.early_stopping >= 0, "Early stopping must be non-negative"
            
            # Validate layer type
            assert self.layer in [Layer.GCN, Layer.GAT], f"Unsupported layer: {self.layer}"
            
            # Validate paths (basic checks)
            assert self.dataset, "Dataset name cannot be empty"
            assert self.path, "Path cannot be empty"
            assert self.log_dir, "Log directory cannot be empty"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'dataset': self.dataset,
            'path': self.path,
            'log_dir': self.log_dir,
            'builder': self.builder,
            'lr': self.lr,
            'nhid': self.nhid,
            'seed': self.seed,
            'epoches': self.epoches,
            'dropout': self.dropout,
            'val_ratio': self.val_ratio,
            'layer': self.layer,
            'early_stopping': self.early_stopping,
            'device': self.device
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def print_config(self) -> None:
        """Print current configuration."""
        print(f"\n{'='*30} MODEL CONFIGURATION\n")
        config_dict = self.get_config_dict()
        for key, value in config_dict.items():
            print(f"{key}: {value}")
        print()
