"""
Service Factory for Text4GCN with Dependency Injection.

This module provides a factory for creating and configuring services
with dependency injection support.
"""

from typing import Any, Dict, Type, Optional
from text4gcn.di import DIContainer, ServiceLifetime
from text4gcn.di.configuration import ConfigurationManager
from .interfaces import (
    IDataProcessor, IModelTrainer, IModelEvaluator, IMetricsVisualizer,
    IConfigurationService, ILoggingService, IFileService
)
from .implementations import (
    DataProcessorService, ModelTrainerService, ModelEvaluatorService,
    MetricsVisualizerService, ConfigurationService, LoggingService, FileService
)


class ServiceFactory:
    """
    Factory for creating and configuring services with dependency injection.
    
    This factory follows the Factory Pattern and uses dependency injection
    to create loosely coupled services.
    """
    
    def __init__(self, container: Optional[DIContainer] = None):
        """
        Initialize service factory.
        
        Args:
            container: DI container instance (optional)
        """
        self.container = container or DIContainer()
        self._setup_default_services()
    
    def _setup_default_services(self):
        """Setup default service registrations."""
        # Register core services
        self.container.register_singleton(ILoggingService, LoggingService)
        self.container.register_singleton(IFileService, FileService)
        self.container.register_singleton(IConfigurationService, ConfigurationService)
        
        # Register model services
        self.container.register_transient(IDataProcessor, DataProcessorService)
        self.container.register_transient(IModelTrainer, ModelTrainerService)
        self.container.register_transient(IModelEvaluator, ModelEvaluatorService)
        self.container.register_transient(IMetricsVisualizer, MetricsVisualizerService)
    
    def create_data_processor(self, device: str) -> IDataProcessor:
        """
        Create data processor service.
        
        Args:
            device: Device to process data on
            
        Returns:
            Data processor service instance
        """
        logging_service = self.container.resolve(ILoggingService)
        return DataProcessorService(device, logging_service)
    
    def create_model_trainer(self, device: str) -> IModelTrainer:
        """
        Create model trainer service.
        
        Args:
            device: Device to train on
            
        Returns:
            Model trainer service instance
        """
        logging_service = self.container.resolve(ILoggingService)
        file_service = self.container.resolve(IFileService)
        return ModelTrainerService(device, logging_service, file_service)
    
    def create_model_evaluator(self, device: str) -> IModelEvaluator:
        """
        Create model evaluator service.
        
        Args:
            device: Device to evaluate on
            
        Returns:
            Model evaluator service instance
        """
        logging_service = self.container.resolve(ILoggingService)
        return ModelEvaluatorService(device, logging_service)
    
    def create_metrics_visualizer(self, log_dir: str) -> IMetricsVisualizer:
        """
        Create metrics visualizer service.
        
        Args:
            log_dir: Directory for saving visualizations
            
        Returns:
            Metrics visualizer service instance
        """
        file_service = self.container.resolve(IFileService)
        logging_service = self.container.resolve(ILoggingService)
        return MetricsVisualizerService(log_dir, file_service, logging_service)
    
    def create_configuration_service(self, config_manager: ConfigurationManager) -> IConfigurationService:
        """
        Create configuration service.
        
        Args:
            config_manager: Configuration manager instance
            
        Returns:
            Configuration service instance
        """
        return ConfigurationService(config_manager)
    
    def create_logging_service(self, log_level: str = "INFO") -> ILoggingService:
        """
        Create logging service.
        
        Args:
            log_level: Logging level
            
        Returns:
            Logging service instance
        """
        return LoggingService(log_level)
    
    def create_file_service(self) -> IFileService:
        """
        Create file service.
        
        Returns:
            File service instance
        """
        logging_service = self.container.resolve(ILoggingService)
        return FileService(logging_service)
    
    def register_service(self, service_type: Type, implementation: Type, 
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'ServiceFactory':
        """
        Register a service in the container.
        
        Args:
            service_type: Service interface type
            implementation: Service implementation type
            lifetime: Service lifetime
            
        Returns:
            Self for method chaining
        """
        if lifetime == ServiceLifetime.SINGLETON:
            self.container.register_singleton(service_type, implementation)
        elif lifetime == ServiceLifetime.TRANSIENT:
            self.container.register_transient(service_type, implementation)
        elif lifetime == ServiceLifetime.SCOPED:
            self.container.register_scoped(service_type, implementation)
        
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'ServiceFactory':
        """
        Register a service instance.
        
        Args:
            service_type: Service type
            instance: Service instance
            
        Returns:
            Self for method chaining
        """
        self.container.register_instance(service_type, instance)
        return self
    
    def get_container(self) -> DIContainer:
        """
        Get the DI container.
        
        Returns:
            DI container instance
        """
        return self.container
    
    def create_scope(self, scope_name: str):
        """
        Create a new scope.
        
        Args:
            scope_name: Name of the scope
            
        Returns:
            DI scope instance
        """
        return self.container.create_scope(scope_name)
    
    def resolve(self, service_type: Type) -> Any:
        """
        Resolve a service from the container.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
        """
        return self.container.resolve(service_type)
    
    def is_registered(self, service_type: Type) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_type: Service type
            
        Returns:
            True if service is registered
        """
        return self.container.is_registered(service_type)


class ServiceBuilder:
    """
    Builder for creating service configurations.
    
    This builder follows the Builder Pattern for creating complex
    service configurations with dependency injection.
    """
    
    def __init__(self):
        """Initialize service builder."""
        self._services: Dict[Type, Dict[str, Any]] = {}
        self._dependencies: Dict[Type, list] = {}
    
    def add_service(self, service_type: Type, implementation: Type, 
                   lifetime: str = "transient", **kwargs) -> 'ServiceBuilder':
        """
        Add a service to the configuration.
        
        Args:
            service_type: Service interface type
            implementation: Service implementation type
            lifetime: Service lifetime
            **kwargs: Additional service configuration
            
        Returns:
            Self for method chaining
        """
        self._services[service_type] = {
            'implementation': implementation,
            'lifetime': lifetime,
            **kwargs
        }
        return self
    
    def add_dependency(self, service_type: Type, dependency: Type) -> 'ServiceBuilder':
        """
        Add a dependency to a service.
        
        Args:
            service_type: Service type
            dependency: Dependency type
            
        Returns:
            Self for method chaining
        """
        if service_type not in self._dependencies:
            self._dependencies[service_type] = []
        self._dependencies[service_type].append(dependency)
        return self
    
    def build(self) -> ServiceFactory:
        """
        Build the service factory with configuration.
        
        Returns:
            Configured service factory
        """
        container = DIContainer()
        factory = ServiceFactory(container)
        
        # Register services
        for service_type, config in self._services.items():
            implementation = config['implementation']
            lifetime = ServiceLifetime(config['lifetime'])
            
            factory.register_service(service_type, implementation, lifetime)
        
        return factory
    
    def validate(self) -> list:
        """
        Validate the service configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for circular dependencies
        for service_type, dependencies in self._dependencies.items():
            if service_type in dependencies:
                errors.append(f"Circular dependency detected for {service_type.__name__}")
        
        # Check for missing dependencies
        for service_type, dependencies in self._dependencies.items():
            for dependency in dependencies:
                if dependency not in self._services:
                    errors.append(f"Missing dependency {dependency.__name__} for {service_type.__name__}")
        
        return errors


