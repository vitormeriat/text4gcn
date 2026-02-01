"""
Unit tests for Dependency Injection system.

This module contains comprehensive tests for the DI system including:
- Container functionality
- Service registration and resolution
- Lifecycle management
- Scope management
- Error handling
- Configuration management
"""

import pytest
import tempfile
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List

# Import DI components
from text4gcn.di import DIContainer, ServiceLifetime, set_container
from text4gcn.di.configuration import ConfigurationManager
from text4gcn.di.exceptions import (
    DIContainerError, ServiceNotFoundError, CircularDependencyError,
    ServiceRegistrationError, ConfigurationError, ScopeError
)
from text4gcn.di.decorators import inject, singleton, transient, scoped
from text4gcn.services import ServiceFactory
from text4gcn.services.interfaces import (
    IDataProcessor, IModelTrainer, IModelEvaluator, IMetricsVisualizer,
    IConfigurationService, ILoggingService, IFileService
)


class TestDIContainer:
    """Test cases for DIContainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = DIContainer()
    
    def test_initialization(self):
        """Test container initialization."""
        assert self.container is not None
        assert len(self.container.get_registered_services()) == 0
    
    def test_register_singleton(self):
        """Test singleton service registration."""
        class TestService:
            pass
        
        self.container.register_singleton(TestService, TestService)
        
        assert self.container.is_registered(TestService)
        service_info = self.container._services[TestService]
        assert service_info.lifetime == ServiceLifetime.SINGLETON
    
    def test_register_transient(self):
        """Test transient service registration."""
        class TestService:
            pass
        
        self.container.register_transient(TestService, TestService)
        
        assert self.container.is_registered(TestService)
        service_info = self.container._services[TestService]
        assert service_info.lifetime == ServiceLifetime.TRANSIENT
    
    def test_register_scoped(self):
        """Test scoped service registration."""
        class TestService:
            pass
        
        self.container.register_scoped(TestService, TestService)
        
        assert self.container.is_registered(TestService)
        service_info = self.container._services[TestService]
        assert service_info.lifetime == ServiceLifetime.SCOPED
    
    def test_register_instance(self):
        """Test service instance registration."""
        class TestService:
            pass
        
        instance = TestService()
        self.container.register_instance(TestService, instance)
        
        assert self.container.is_registered(TestService)
        service_info = self.container._services[TestService]
        assert service_info.instance is instance
    
    def test_register_factory(self):
        """Test factory registration."""
        class TestService:
            pass
        
        def factory():
            return TestService()
        
        self.container.register_factory(TestService, factory)
        
        assert self.container.is_registered(TestService)
        service_info = self.container._services[TestService]
        assert service_info.factory is factory
    
    def test_resolve_singleton(self):
        """Test singleton service resolution."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        self.container.register_singleton(TestService, TestService)
        
        instance1 = self.container.resolve(TestService)
        instance2 = self.container.resolve(TestService)
        
        assert instance1 is instance2
        assert instance1.value == "test"
    
    def test_resolve_transient(self):
        """Test transient service resolution."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        self.container.register_transient(TestService, TestService)
        
        instance1 = self.container.resolve(TestService)
        instance2 = self.container.resolve(TestService)
        
        assert instance1 is not instance2
        assert instance1.value == "test"
        assert instance2.value == "test"
    
    def test_resolve_with_dependencies(self):
        """Test service resolution with dependencies."""
        class Dependency:
            def __init__(self):
                self.value = "dependency"
        
        class Service:
            def __init__(self, dependency: Dependency):
                self.dependency = dependency
        
        self.container.register_singleton(Dependency, Dependency)
        self.container.register_transient(Service, Service)
        
        service = self.container.resolve(Service)
        
        assert service.dependency is not None
        assert service.dependency.value == "dependency"
    
    def test_resolve_unregistered_service(self):
        """Test resolving unregistered service."""
        class UnregisteredService:
            pass
        
        with pytest.raises(ServiceNotFoundError):
            self.container.resolve(UnregisteredService)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
        
        self.container.register_transient(ServiceA, ServiceA)
        self.container.register_transient(ServiceB, ServiceB)
        
        with pytest.raises(CircularDependencyError):
            self.container.resolve(ServiceA)
    
    def test_scope_management(self):
        """Test scope management."""
        class ScopedService:
            def __init__(self):
                self.value = "scoped"
        
        self.container.register_scoped(ScopedService, ScopedService)
        
        # Test without scope
        with pytest.raises(DIContainerError):
            self.container.resolve(ScopedService)
        
        # Test with scope
        with self.container.create_scope("test_scope") as scope:
            instance1 = self.container.resolve(ScopedService)
            instance2 = self.container.resolve(ScopedService)
            
            assert instance1 is instance2
            assert instance1.value == "scoped"
        
        # Test new scope
        with self.container.create_scope("new_scope") as scope:
            instance3 = self.container.resolve(ScopedService)
            assert instance3 is not instance1


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
    
    def test_initialization(self):
        """Test configuration manager initialization."""
        assert self.config_manager is not None
        assert len(self.config_manager.get_all()) == 0
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "model": {
                "learning_rate": 0.01,
                "epochs": 100
            },
            "data": {
                "device": "cpu"
            }
        }
        
        self.config_manager.load_from_dict(config_dict)
        
        assert self.config_manager.get("model.learning_rate") == 0.01
        assert self.config_manager.get("model.epochs") == 100
        assert self.config_manager.get("data.device") == "cpu"
    
    def test_get_with_default(self):
        """Test getting configuration with default value."""
        value = self.config_manager.get("nonexistent.key", default="default_value")
        assert value == "default_value"
    
    def test_get_with_type_hint(self):
        """Test getting configuration with type hint."""
        self.config_manager.set("test.int", "123")
        self.config_manager.set("test.float", "3.14")
        
        int_value = self.config_manager.get("test.int", type_hint=int)
        float_value = self.config_manager.get("test.float", type_hint=float)
        
        assert int_value == 123
        assert float_value == 3.14
    
    def test_set_nested_value(self):
        """Test setting nested configuration values."""
        self.config_manager.set("nested.key", "value")
        
        assert self.config_manager.get("nested.key") == "value"
        assert self.config_manager.get("nested") == {"key": "value"}
    
    def test_validation(self):
        """Test configuration validation."""
        config_dict = {
            "model": {
                "learning_rate": 0.01,
                "epochs": 100
            }
        }
        
        self.config_manager.load_from_dict(config_dict)
        
        schema = {
            "model": {
                "properties": {
                    "learning_rate": {
                        "type": float,
                        "required": True
                    },
                    "epochs": {
                        "type": int,
                        "required": True,
                        "min": 1,
                        "max": 1000
                    }
                }
            }
        }
        
        errors = self.config_manager.validate(schema)
        assert len(errors) == 0
    
    def test_validation_with_errors(self):
        """Test configuration validation with errors."""
        config_dict = {
            "model": {
                "learning_rate": "invalid",
                "epochs": 2000
            }
        }
        
        self.config_manager.load_from_dict(config_dict)
        
        schema = {
            "model": {
                "properties": {
                    "learning_rate": {
                        "type": float,
                        "required": True
                    },
                    "epochs": {
                        "type": int,
                        "required": True,
                        "min": 1,
                        "max": 1000
                    }
                }
            }
        }
        
        errors = self.config_manager.validate(schema)
        assert len(errors) > 0
    
    def test_watchers(self):
        """Test configuration change watchers."""
        changes = []
        
        def watcher(key, value):
            changes.append((key, value))
        
        self.config_manager.add_watcher(watcher)
        
        self.config_manager.set("test.key", "value")
        
        assert len(changes) == 1
        assert changes[0] == ("test.key", "value")


class TestDecorators:
    """Test cases for DI decorators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container = DIContainer()
        set_container(self.container)
    
    def test_inject_decorator_class(self):
        """Test @inject decorator on class."""
        class Dependency:
            def __init__(self):
                self.value = "dependency"
        
        self.container.register_singleton(Dependency, Dependency)
        
        @inject(Dependency)
        class Service:
            def __init__(self, dependency: Dependency):
                self.dependency = dependency
        
        service = Service()
        assert service.dependency.value == "dependency"
    
    def test_inject_decorator_function(self):
        """Test @inject decorator on function."""
        class Dependency:
            def __init__(self):
                self.value = "dependency"
        
        self.container.register_singleton(Dependency, Dependency)
        
        @inject(Dependency)
        def my_function(dependency: Dependency):
            return dependency.value
        
        result = my_function()
        assert result == "dependency"
    
    def test_singleton_decorator(self):
        """Test @singleton decorator."""
        @singleton
        class SingletonService:
            def __init__(self):
                self.value = "singleton"
        
        instance1 = SingletonService()
        instance2 = SingletonService()
        
        assert instance1 is instance2
        assert instance1.value == "singleton"
    
    def test_transient_decorator(self):
        """Test @transient decorator."""
        @transient
        class TransientService:
            def __init__(self):
                self.value = "transient"
        
        instance1 = TransientService()
        instance2 = TransientService()
        
        assert instance1 is not instance2
        assert instance1.value == "transient"
        assert instance2.value == "transient"


class TestServiceFactory:
    """Test cases for ServiceFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = ServiceFactory()
    
    def test_initialization(self):
        """Test service factory initialization."""
        assert self.factory is not None
        assert self.factory.get_container() is not None
    
    def test_create_data_processor(self):
        """Test creating data processor service."""
        processor = self.factory.create_data_processor(device="cpu")
        
        assert processor is not None
        assert processor.device == "cpu"
    
    def test_create_model_trainer(self):
        """Test creating model trainer service."""
        trainer = self.factory.create_model_trainer(device="cpu")
        
        assert trainer is not None
        assert trainer.device == "cpu"
    
    def test_create_model_evaluator(self):
        """Test creating model evaluator service."""
        evaluator = self.factory.create_model_evaluator(device="cpu")
        
        assert evaluator is not None
        assert evaluator.device == "cpu"
    
    def test_register_service(self):
        """Test registering services."""
        class TestService:
            pass
        
        self.factory.register_service(TestService, TestService, ServiceLifetime.SINGLETON)
        
        assert self.factory.is_registered(TestService)
    
    def test_register_instance(self):
        """Test registering service instance."""
        class TestService:
            pass
        
        instance = TestService()
        self.factory.register_instance(TestService, instance)
        
        resolved = self.factory.resolve(TestService)
        assert resolved is instance
    
    def test_create_scope(self):
        """Test creating scope."""
        scope = self.factory.create_scope("test_scope")
        
        assert scope is not None
        assert scope.scope_name == "test_scope"


class TestIntegration:
    """Integration tests for DI system."""
    
    def test_full_workflow(self):
        """Test complete DI workflow."""
        # Create container
        container = DIContainer()
        
        # Register services
        container.register_singleton(ILoggingService, Mock)
        container.register_transient(IDataProcessor, Mock)
        container.register_transient(IModelTrainer, Mock)
        
        # Set global container
        set_container(container)
        
        # Resolve services
        logging_service = container.resolve(ILoggingService)
        data_processor = container.resolve(IDataProcessor)
        model_trainer = container.resolve(IModelTrainer)
        
        assert logging_service is not None
        assert data_processor is not None
        assert model_trainer is not None
    
    def test_service_dependencies(self):
        """Test service with dependencies."""
        class LoggingService:
            def __init__(self):
                self.level = "INFO"
        
        class DataProcessor:
            def __init__(self, logging_service: LoggingService):
                self.logging_service = logging_service
        
        container = DIContainer()
        container.register_singleton(LoggingService, LoggingService)
        container.register_transient(DataProcessor, DataProcessor)
        
        processor = container.resolve(DataProcessor)
        
        assert processor.logging_service is not None
        assert processor.logging_service.level == "INFO"
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        config_manager = ConfigurationManager()
        config_dict = {
            "model": {
                "learning_rate": 0.01,
                "epochs": 100
            }
        }
        config_manager.load_from_dict(config_dict)
        
        factory = ServiceFactory()
        factory.register_instance(ConfigurationManager, config_manager)
        
        # Test that configuration is available
        assert factory.is_registered(ConfigurationManager)


if __name__ == '__main__':
    pytest.main([__file__])


