"""
Model Factory - Single Responsibility: Create model instances.

This class is responsible for:
- Creating model instances based on configuration
- Managing model creation logic
- Factory pattern implementation
"""

import torch as th
from typing import Type, Dict, Any
from text4gcn.models.model import GCN
from text4gcn.models.layers import Layer


class ModelFactory:
    """Factory for creating model instances."""
    
    _models = {
        Layer.GCN: GCN,
        # Layer.GAT: GAT,  # Future implementation
    }
    
    @classmethod
    def create_model(cls, layer_type: Layer, input_dim: int, output_dim: int, 
                    support: list, num_classes: int, dropout_rate: float = 0.5) -> th.nn.Module:
        """
        Create a model instance.
        
        Args:
            layer_type: Type of layer (GCN, GAT, etc.)
            input_dim: Input dimension
            output_dim: Output dimension
            support: Support matrices
            num_classes: Number of classes
            dropout_rate: Dropout rate
            
        Returns:
            PyTorch model instance
            
        Raises:
            ValueError: If layer type is not supported
        """
        if layer_type not in cls._models:
            raise ValueError(f"Unsupported layer type: {layer_type}. "
                           f"Supported types: {list(cls._models.keys())}")
        
        model_class = cls._models[layer_type]
        
        return model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            support=support,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
    
    @classmethod
    def register_model(cls, layer_type: Layer, model_class: Type[th.nn.Module]) -> None:
        """
        Register a new model type.
        
        Args:
            layer_type: Layer type enum
            model_class: Model class to register
        """
        cls._models[layer_type] = model_class
    
    @classmethod
    def get_supported_models(cls) -> list:
        """
        Get list of supported model types.
        
        Returns:
            List of supported layer types
        """
        return list(cls._models.keys())
    
    @classmethod
    def create_optimizer(cls, model: th.nn.Module, lr: float = 0.01, 
                        weight_decay: float = 5e-4) -> th.optim.Optimizer:
        """
        Create optimizer for the model.
        
        Args:
            model: PyTorch model
            lr: Learning rate
            weight_decay: Weight decay
            
        Returns:
            PyTorch optimizer
        """
        return th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    @classmethod
    def create_criterion(cls) -> th.nn.Module:
        """
        Create loss criterion.
        
        Returns:
            PyTorch loss function
        """
        return th.nn.CrossEntropyLoss()
