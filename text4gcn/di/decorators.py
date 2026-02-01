"""
Dependency Injection Decorators.

This module provides decorators for dependency injection including:
- @inject decorator for automatic dependency injection
- @singleton decorator for singleton services
- @transient decorator for transient services
- @scoped decorator for scoped services
"""

from typing import Any, Callable, Type, TypeVar, Optional, Dict, List
from functools import wraps
import inspect
import threading

T = TypeVar('T')

# Global container instance
_container: Optional['DIContainer'] = None
_container_lock = threading.Lock()


def set_container(container: 'DIContainer'):
    """
    Set the global DI container.
    
    Args:
        container: DI container instance
    """
    global _container
    with _container_lock:
        _container = container


def get_container() -> Optional['DIContainer']:
    """
    Get the global DI container.
    
    Returns:
        DI container instance or None
    """
    return _container


def inject(*dependencies: Type) -> Callable:
    """
    Decorator for automatic dependency injection.
    
    Args:
        *dependencies: Types of dependencies to inject
        
    Returns:
        Decorated function or class
    """
    def decorator(func_or_class: T) -> T:
        if inspect.isclass(func_or_class):
            return _inject_class(func_or_class, dependencies)
        else:
            return _inject_function(func_or_class, dependencies)
    
    return decorator


def _inject_class(cls: Type[T], dependencies: List[Type]) -> Type[T]:
    """Inject dependencies into class constructor."""
    original_init = cls.__init__
    
    # Get parameter names from the constructor signature
    sig = inspect.signature(original_init)
    param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Get container
        container = get_container()
        if container is None:
            raise RuntimeError("DI container not set. Call set_container() first.")
        
        # Resolve dependencies and inject by parameter name
        for i, dep_type in enumerate(dependencies):
            # Skip if already provided as positional argument
            if i < len(args):
                continue
            
            # Find the parameter name for this dependency
            param_name = param_names[i] if i < len(param_names) else f'dependency_{i}'
            
            # Skip if already provided as keyword argument
            if param_name in kwargs:
                continue
            
            try:
                resolved = container.resolve(dep_type)
                kwargs[param_name] = resolved
            except Exception as e:
                dep_name = dep_type.__name__ if hasattr(dep_type, '__name__') else str(dep_type)
                raise RuntimeError(f"Failed to resolve dependency {dep_name}: {e}")
        
        # Call original constructor
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


def _inject_function(func: Callable, dependencies: List[Type]) -> Callable:
    """Inject dependencies into function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get container
        container = get_container()
        if container is None:
            raise RuntimeError("DI container not set. Call set_container() first.")
        
        # Resolve dependencies
        resolved_deps = []
        for dep_type in dependencies:
            try:
                resolved = container.resolve(dep_type)
                resolved_deps.append(resolved)
            except Exception as e:
                raise RuntimeError(f"Failed to resolve dependency {dep_type.__name__}: {e}")
        
        # Call function with resolved dependencies
        return func(*args, *resolved_deps, **kwargs)
    
    return wrapper


def singleton(service_type: Optional[Type] = None) -> Callable:
    """
    Decorator for singleton services.
    
    Args:
        service_type: Service type (optional, defaults to decorated class)
        
    Returns:
        Decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Register as singleton
        container = get_container()
        if container is not None:
            container.register_singleton(cls, cls)
        
        # Store singleton instance as class attribute
        _singleton_instances = {}
        
        original_init = cls.__init__
        original_new = cls.__new__
        
        def __new__(cls, *args, **kwargs):
            if cls not in _singleton_instances:
                if original_new is object.__new__:
                    instance = original_new(cls)
                else:
                    instance = original_new(cls, *args, **kwargs)
                _singleton_instances[cls] = instance
            return _singleton_instances[cls]
        
        cls.__new__ = __new__
        
        return cls
    
    # If called without parentheses, service_type will be the class
    if service_type is not None and inspect.isclass(service_type):
        return decorator(service_type)
    
    return decorator


def transient(service_type: Optional[Type] = None) -> Callable:
    """
    Decorator for transient services.
    
    Args:
        service_type: Service type (optional, defaults to decorated class)
        
    Returns:
        Decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Register as transient
        container = get_container()
        if container is not None:
            container.register_transient(cls, cls)
        
        return cls
    
    # If called without parentheses, service_type will be the class
    if service_type is not None and inspect.isclass(service_type):
        return decorator(service_type)
    
    return decorator


def scoped(service_type: Optional[Type] = None) -> Callable:
    """
    Decorator for scoped services.
    
    Args:
        service_type: Service type (optional, defaults to decorated class)
        
    Returns:
        Decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        nonlocal service_type
        if service_type is None:
            service_type = cls
        
        # Register as scoped
        container = get_container()
        if container is not None:
            container.register_scoped(service_type, cls)
        
        return cls
    
    return decorator


def auto_register(cls: Type[T]) -> Type[T]:
    """
    Decorator for automatic service registration.
    
    Automatically registers the class as a transient service.
    
    Args:
        cls: Class to register
        
    Returns:
        Registered class
    """
    container = get_container()
    if container is not None:
        container.register_transient(cls, cls)
    
    return cls


def inject_property(property_name: str, service_type: Type) -> Callable:
    """
    Decorator for property injection.
    
    Args:
        property_name: Name of the property
        service_type: Type of service to inject
        
    Returns:
        Property decorator
    """
    def decorator(cls: Type[T]) -> Type[T]:
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call original constructor
            original_init(self, *args, **kwargs)
            
            # Inject property
            container = get_container()
            if container is not None:
                try:
                    resolved = container.resolve(service_type)
                    setattr(self, property_name, resolved)
                except Exception as e:
                    raise RuntimeError(f"Failed to inject property {property_name}: {e}")
        
        cls.__init__ = new_init
        return cls
    
    return decorator


class ServiceLocator:
    """Service locator pattern implementation."""
    
    def __init__(self, container: 'DIContainer'):
        """
        Initialize service locator.
        
        Args:
            container: DI container
        """
        self._container = container
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get service instance.
        
        Args:
            service_type: Service type
            
        Returns:
            Service instance
        """
        return self._container.resolve(service_type)
    
    def get_services(self, service_type: Type[T]) -> List[T]:
        """
        Get all services of a type.
        
        Args:
            service_type: Service type
            
        Returns:
            List of service instances
        """
        # This would require multiple implementations of the same interface
        # For now, return single instance
        return [self._container.resolve(service_type)]


def service_locator(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class use service locator pattern.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
    """
    original_init = cls.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Call original constructor
        original_init(self, *args, **kwargs)
        
        # Add service locator
        container = get_container()
        if container is not None:
            self._service_locator = ServiceLocator(container)
    
    cls.__init__ = new_init
    return cls


