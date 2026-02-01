"""
Services module for Text4GCN with Dependency Injection.

This module contains service interfaces and implementations that use
dependency injection for loose coupling and testability.
"""

from .interfaces import (
    IDataProcessor,
    IModelTrainer, 
    IModelEvaluator,
    IMetricsVisualizer,
    IConfigurationService,
    ILoggingService,
    IFileService
)

from .implementations import (
    DataProcessorService,
    ModelTrainerService,
    ModelEvaluatorService,
    MetricsVisualizerService,
    ConfigurationService,
    LoggingService,
    FileService
)

from .service_factory import ServiceFactory

__all__ = [
    # Interfaces
    "IDataProcessor",
    "IModelTrainer",
    "IModelEvaluator", 
    "IMetricsVisualizer",
    "IConfigurationService",
    "ILoggingService",
    "IFileService",
    
    # Implementations
    "DataProcessorService",
    "ModelTrainerService",
    "ModelEvaluatorService",
    "MetricsVisualizerService",
    "ConfigurationService",
    "LoggingService",
    "FileService",
    
    # Factory
    "ServiceFactory"
]


