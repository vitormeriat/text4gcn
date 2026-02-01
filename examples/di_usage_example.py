#!/usr/bin/env python3
"""
Dependency Injection Usage Example for Text4GCN.

This example demonstrates how to use the dependency injection system
in Text4GCN for creating loosely coupled, testable components.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text4gcn.di import DIContainer, ServiceLifetime, set_container
from text4gcn.di.configuration import ConfigurationManager
from text4gcn.services import ServiceFactory
from text4gcn.services.interfaces import IDataProcessor, IModelTrainer, IModelEvaluator


def example_basic_di_usage():
    """Example of basic dependency injection usage."""
    print("🔧 Basic Dependency Injection Usage")
    print("=" * 50)
    
    # Create DI container
    container = DIContainer()
    
    # Register services
    container.register_singleton(ILoggingService, LoggingService)
    container.register_transient(IDataProcessor, DataProcessorService)
    container.register_transient(IModelTrainer, ModelTrainerService)
    
    # Set global container
    set_container(container)
    
    # Resolve services
    logging_service = container.resolve(ILoggingService)
    data_processor = container.resolve(IDataProcessor)
    
    print(f"✅ Logging service created: {type(logging_service).__name__}")
    print(f"✅ Data processor created: {type(data_processor).__name__}")


def example_service_factory_usage():
    """Example of using ServiceFactory."""
    print("\n🏭 Service Factory Usage")
    print("=" * 50)
    
    # Create service factory
    factory = ServiceFactory()
    
    # Create services with dependencies
    data_processor = factory.create_data_processor(device="cpu")
    model_trainer = factory.create_model_trainer(device="cpu")
    model_evaluator = factory.create_model_evaluator(device="cpu")
    
    print(f"✅ Data processor: {type(data_processor).__name__}")
    print(f"✅ Model trainer: {type(model_trainer).__name__}")
    print(f"✅ Model evaluator: {type(model_evaluator).__name__}")


def example_configuration_usage():
    """Example of configuration management."""
    print("\n⚙️ Configuration Management")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Load configuration from dictionary
    config_dict = {
        "model": {
            "learning_rate": 0.01,
            "epochs": 100,
            "batch_size": 32
        },
        "data": {
            "device": "cpu",
            "validation_split": 0.2
        }
    }
    
    config_manager.load_from_dict(config_dict)
    
    # Get configuration values
    lr = config_manager.get("model.learning_rate", type_hint=float)
    epochs = config_manager.get("model.epochs", type_hint=int)
    device = config_manager.get("data.device", default="cpu")
    
    print(f"✅ Learning rate: {lr}")
    print(f"✅ Epochs: {epochs}")
    print(f"✅ Device: {device}")


def example_scope_usage():
    """Example of scoped services."""
    print("\n🎯 Scoped Services Usage")
    print("=" * 50)
    
    # Create container
    container = DIContainer()
    
    # Register scoped service
    container.register_scoped(IModelTrainer, ModelTrainerService)
    
    # Create scope
    with container.create_scope("training_session") as scope:
        # Resolve service in scope
        trainer1 = container.resolve(IModelTrainer)
        trainer2 = container.resolve(IModelTrainer)
        
        # Should be the same instance within scope
        print(f"✅ Same instance in scope: {trainer1 is trainer2}")
    
    # Create another scope
    with container.create_scope("another_session") as scope:
        trainer3 = container.resolve(IModelTrainer)
        print(f"✅ New instance in new scope: {trainer1 is not trainer3}")


def example_service_builder():
    """Example of using ServiceBuilder."""
    print("\n🔨 Service Builder Usage")
    print("=" * 50)
    
    from text4gcn.services.service_factory import ServiceBuilder
    
    # Create service builder
    builder = ServiceBuilder()
    
    # Add services
    builder.add_service(ILoggingService, LoggingService, "singleton")
    builder.add_service(IFileService, FileService, "singleton")
    builder.add_service(IDataProcessor, DataProcessorService, "transient")
    
    # Add dependencies
    builder.add_dependency(IDataProcessor, ILoggingService)
    builder.add_dependency(IModelTrainer, ILoggingService)
    builder.add_dependency(IModelTrainer, IFileService)
    
    # Validate configuration
    errors = builder.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Build factory
    factory = builder.build()
    print(f"✅ Service factory created with {len(factory.get_container().get_registered_services())} services")


def example_decorators_usage():
    """Example of using DI decorators."""
    print("\n🎭 Decorators Usage")
    print("=" * 50)
    
    from text4gcn.di.decorators import inject, singleton, transient
    
    # Set up container
    container = DIContainer()
    container.register_singleton(ILoggingService, LoggingService)
    set_container(container)
    
    # Use decorators
    @singleton
    class MyService:
        def __init__(self, logging_service: ILoggingService):
            self.logging_service = logging_service
    
    @inject(ILoggingService)
    def my_function(logging_service: ILoggingService):
        logging_service.info("Function called with injected dependency")
        return "Success"
    
    # Test decorated class
    service1 = MyService()
    service2 = MyService()
    print(f"✅ Singleton service: {service1 is service2}")
    
    # Test decorated function
    result = my_function()
    print(f"✅ Function result: {result}")


def example_error_handling():
    """Example of error handling in DI system."""
    print("\n⚠️ Error Handling")
    print("=" * 50)
    
    from text4gcn.di.exceptions import ServiceNotFoundError, CircularDependencyError
    
    container = DIContainer()
    
    try:
        # Try to resolve unregistered service
        container.resolve(IModelTrainer)
    except ServiceNotFoundError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Register services with circular dependency
    class ServiceA:
        def __init__(self, service_b: 'ServiceB'):
            self.service_b = service_b
    
    class ServiceB:
        def __init__(self, service_a: ServiceA):
            self.service_a = service_a
    
    container.register_transient(ServiceA, ServiceA)
    container.register_transient(ServiceB, ServiceB)
    
    try:
        container.resolve(ServiceA)
    except CircularDependencyError as e:
        print(f"✅ Caught circular dependency: {e}")


def main():
    """Run all examples."""
    print("🚀 Text4GCN Dependency Injection Examples")
    print("=" * 60)
    
    try:
        example_basic_di_usage()
        example_service_factory_usage()
        example_configuration_usage()
        example_scope_usage()
        example_service_builder()
        example_decorators_usage()
        example_error_handling()
        
        print("\n🎉 All examples completed successfully!")
        print("✅ Dependency injection system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


