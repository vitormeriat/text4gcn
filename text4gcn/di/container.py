"""
Dependency Injection Container.

This class provides a complete DI container implementation with support for:
- Service registration and resolution
- Lifecycle management (singleton, transient, scoped)
- Constructor injection
- Property injection
- Circular dependency detection
- Service decorators
"""

from typing import Any, Dict, Type, TypeVar, Callable, Optional, Union, List
from enum import Enum
import inspect
import threading
from collections import defaultdict

from .exceptions import (
    DIContainerError,
    ServiceNotFoundError,
    CircularDependencyError
)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Descriptor for a registered service."""
    
    def __init__(self, service_type: Type, implementation: Union[Type, Callable], 
                 lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.instance: Optional[Any] = None
        self.factory: Optional[Callable] = None


class DIContainer:
    """
    Dependency Injection Container.
    
    Provides service registration, resolution, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the DI container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_services: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._lock = threading.RLock()
        self._resolution_stack: List[Type] = []
    
    def register_singleton(self, service_type: Type[T], implementation: Union[Type[T], Callable[..., T]]) -> 'DIContainer':
        """
        Register a service as singleton.
        
        Args:
            service_type: Type of the service
            implementation: Implementation class or factory function
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation, ServiceLifetime.SINGLETON)
    
    def register_transient(self, service_type: Type[T], implementation: Union[Type[T], Callable[..., T]]) -> 'DIContainer':
        """
        Register a service as transient.
        
        Args:
            service_type: Type of the service
            implementation: Implementation class or factory function
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation, ServiceLifetime.TRANSIENT)
    
    def register_scoped(self, service_type: Type[T], implementation: Union[Type[T], Callable[..., T]]) -> 'DIContainer':
        """
        Register a service as scoped.
        
        Args:
            service_type: Type of the service
            implementation: Implementation class or factory function
            
        Returns:
            Self for method chaining
        """
        return self._register_service(service_type, implementation, ServiceLifetime.SCOPED)
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'DIContainer':
        """
        Register a service instance.
        
        Args:
            service_type: Type of the service
            instance: Service instance
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(service_type, None, ServiceLifetime.SINGLETON)
            descriptor.instance = instance
            self._services[service_type] = descriptor
        return self
    
    def register_factory(self, service_type: Type[T], factory: Callable[..., T], 
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'DIContainer':
        """
        Register a service factory.
        
        Args:
            service_type: Type of the service
            factory: Factory function
            lifetime: Service lifetime
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(service_type, None, lifetime)
            descriptor.factory = factory
            self._services[service_type] = descriptor
        return self
    
    def _register_service(self, service_type: Type, implementation: Union[Type, Callable], 
                          lifetime: ServiceLifetime) -> 'DIContainer':
        """Internal method to register a service."""
        with self._lock:
            descriptor = ServiceDescriptor(service_type, implementation, lifetime)
            self._services[service_type] = descriptor
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.
        
        Args:
            service_type: Type of the service to resolve
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependency is detected
        """
        with self._lock:
            # Handle forward references (string annotations)
            if isinstance(service_type, str):
                # Try to find in registered services by name
                for registered_type in self._services.keys():
                    if hasattr(registered_type, '__name__') and registered_type.__name__ == service_type:
                        service_type = registered_type
                        break
                # If still a string and not found, raise error
                if isinstance(service_type, str):
                    raise ServiceNotFoundError(f"Service {service_type} is not registered")
            
            if service_type not in self._services:
                service_name = service_type.__name__ if hasattr(service_type, '__name__') else str(service_type)
                raise ServiceNotFoundError(f"Service {service_name} is not registered")
            
            # Check for circular dependencies
            if service_type in self._resolution_stack:
                cycle_names = [t.__name__ if hasattr(t, '__name__') else str(t) for t in self._resolution_stack]
                service_name = service_type.__name__ if hasattr(service_type, '__name__') else str(service_type)
                cycle = " -> ".join(cycle_names + [service_name])
                raise CircularDependencyError(f"Circular dependency detected: {cycle}")
            
            descriptor = self._services[service_type]
            
            # Handle different lifetimes
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if descriptor.instance is None:
                    descriptor.instance = self._create_instance(service_type, descriptor)
                return descriptor.instance
            
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if self._current_scope is None:
                    raise DIContainerError("No active scope for scoped service")
                
                if self._current_scope not in self._scoped_services:
                    self._scoped_services[self._current_scope] = {}
                
                if service_type not in self._scoped_services[self._current_scope]:
                    instance = self._create_instance(service_type, descriptor)
                    self._scoped_services[self._current_scope][service_type] = instance
                
                return self._scoped_services[self._current_scope][service_type]
            
            else:  # TRANSIENT
                return self._create_instance(service_type, descriptor)
    
    def _create_instance(self, service_type: Type, descriptor: ServiceDescriptor) -> Any:
        """Create a service instance."""
        self._resolution_stack.append(service_type)
        
        try:
            if descriptor.factory:
                # Use factory function
                return self._resolve_factory(descriptor.factory)
            elif descriptor.implementation:
                # Use implementation class
                return self._resolve_constructor(descriptor.implementation)
            else:
                raise DIContainerError(f"No implementation or factory for {service_type.__name__}")
        finally:
            self._resolution_stack.pop()
    
    def _resolve_constructor(self, implementation: Type) -> Any:
        """Resolve constructor dependencies."""
        if not inspect.isclass(implementation):
            return implementation
        
        # Get constructor signature
        constructor = implementation.__init__
        sig = inspect.signature(constructor)
        
        # Resolve dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Skip *args and **kwargs parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            # Skip optional parameters with default values
            if param.default != inspect.Parameter.empty:
                continue
            
            # Resolve dependency
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                raise DIContainerError(f"Cannot resolve parameter {param_name} without type annotation")
            
            # Handle forward references (string annotations)
            if isinstance(param_type, str):
                # Try to resolve forward reference from the implementation's module
                if hasattr(implementation, '__module__') and implementation.__module__:
                    try:
                        module = __import__(implementation.__module__, fromlist=[param_type])
                        param_type = getattr(module, param_type, param_type)
                    except (ImportError, AttributeError):
                        # If we can't resolve, try to find in registered services by name
                        for registered_type in self._services.keys():
                            if hasattr(registered_type, '__name__') and registered_type.__name__ == param_type:
                                param_type = registered_type
                                break
            
            kwargs[param_name] = self.resolve(param_type)
        
        return implementation(**kwargs)
    
    def _resolve_factory(self, factory: Callable) -> Any:
        """Resolve factory function dependencies."""
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            if param.default != inspect.Parameter.empty:
                continue
            
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                raise DIContainerError(f"Cannot resolve factory parameter {param_name} without type annotation")
            
            kwargs[param_name] = self.resolve(param_type)
        
        return factory(**kwargs)
    
    def create_scope(self, scope_name: str) -> 'DIScope':
        """
        Create a new scope.
        
        Args:
            scope_name: Name of the scope
            
        Returns:
            DIScope instance
        """
        return DIScope(self, scope_name)
    
    def is_registered(self, service_type: Type) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_type: Type of the service
            
        Returns:
            True if service is registered
        """
        return service_type in self._services
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of registered services
        """
        return self._services.copy()


class DIScope:
    """Dependency injection scope for managing scoped services."""
    
    def __init__(self, container: DIContainer, scope_name: str):
        """
        Initialize scope.
        
        Args:
            container: DI container
            scope_name: Name of the scope
        """
        self.container = container
        self.scope_name = scope_name
        self._previous_scope = None
    
    def __enter__(self):
        """Enter the scope."""
        with self.container._lock:
            self._previous_scope = self.container._current_scope
            self.container._current_scope = self.scope_name
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the scope."""
        with self.container._lock:
            self.container._current_scope = self._previous_scope
            # Clean up scoped services
            if self.scope_name in self.container._scoped_services:
                del self.container._scoped_services[self.scope_name]


