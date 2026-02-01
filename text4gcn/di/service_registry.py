"""
Service Registry for Text4GCN.

This module provides service registration and discovery functionality
for the dependency injection system.
"""

from typing import Any, Dict, Type, List, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import inspect


class ServiceType(Enum):
    """Service type enumeration."""
    INTERFACE = "interface"
    IMPLEMENTATION = "implementation"
    FACTORY = "factory"
    INSTANCE = "instance"


class ServiceInfo:
    """Information about a registered service."""
    
    def __init__(self, service_type: Type, implementation: Type, 
                 service_type_enum: ServiceType, lifetime: str, 
                 dependencies: List[Type] = None):
        self.service_type = service_type
        self.implementation = implementation
        self.service_type_enum = service_type_enum
        self.lifetime = lifetime
        self.dependencies = dependencies or []
        self.registered_at = None
        self.metadata: Dict[str, Any] = {}


class ServiceRegistry:
    """
    Service Registry for managing service registrations.
    
    Provides service discovery, validation, and metadata management.
    """
    
    def __init__(self):
        """Initialize the service registry."""
        self._services: Dict[Type, ServiceInfo] = {}
        self._interfaces: Dict[Type, List[Type]] = {}
        self._implementations: Dict[Type, List[Type]] = {}
    
    def register_interface(self, interface: Type, implementation: Type, 
                          lifetime: str = "transient") -> 'ServiceRegistry':
        """
        Register an interface with its implementation.
        
        Args:
            interface: Interface type
            implementation: Implementation type
            lifetime: Service lifetime
            
        Returns:
            Self for method chaining
        """
        dependencies = self._extract_dependencies(implementation)
        
        service_info = ServiceInfo(
            service_type=interface,
            implementation=implementation,
            service_type_enum=ServiceType.INTERFACE,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        self._services[interface] = service_info
        
        # Track interface mappings
        if interface not in self._interfaces:
            self._interfaces[interface] = []
        self._interfaces[interface].append(implementation)
        
        # Track implementation mappings
        if implementation not in self._implementations:
            self._implementations[implementation] = []
        self._implementations[implementation].append(interface)
        
        return self
    
    def register_implementation(self, implementation: Type, 
                               lifetime: str = "transient") -> 'ServiceRegistry':
        """
        Register a concrete implementation.
        
        Args:
            implementation: Implementation type
            lifetime: Service lifetime
            
        Returns:
            Self for method chaining
        """
        dependencies = self._extract_dependencies(implementation)
        
        service_info = ServiceInfo(
            service_type=implementation,
            implementation=implementation,
            service_type_enum=ServiceType.IMPLEMENTATION,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        self._services[implementation] = service_info
        return self
    
    def register_factory(self, service_type: Type, factory: Callable, 
                        lifetime: str = "transient") -> 'ServiceRegistry':
        """
        Register a factory function.
        
        Args:
            service_type: Service type
            factory: Factory function
            lifetime: Service lifetime
            
        Returns:
            Self for method chaining
        """
        dependencies = self._extract_dependencies(factory)
        
        service_info = ServiceInfo(
            service_type=service_type,
            implementation=factory,
            service_type_enum=ServiceType.FACTORY,
            lifetime=lifetime,
            dependencies=dependencies
        )
        
        self._services[service_type] = service_info
        return self
    
    def _extract_dependencies(self, target: Type) -> List[Type]:
        """
        Extract dependencies from a class or function.
        
        Args:
            target: Class or function to analyze
            
        Returns:
            List of dependency types
        """
        dependencies = []
        
        if inspect.isclass(target):
            # Analyze constructor
            constructor = target.__init__
            sig = inspect.signature(constructor)
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.default != inspect.Parameter.empty:
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        
        elif callable(target):
            # Analyze function signature
            sig = inspect.signature(target)
            
            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        
        return dependencies
    
    def get_service_info(self, service_type: Type) -> Optional[ServiceInfo]:
        """
        Get service information.
        
        Args:
            service_type: Service type
            
        Returns:
            Service information or None
        """
        return self._services.get(service_type)
    
    def get_implementations(self, interface: Type) -> List[Type]:
        """
        Get all implementations of an interface.
        
        Args:
            interface: Interface type
            
        Returns:
            List of implementation types
        """
        return self._interfaces.get(interface, [])
    
    def get_interfaces(self, implementation: Type) -> List[Type]:
        """
        Get all interfaces implemented by a class.
        
        Args:
            implementation: Implementation type
            
        Returns:
            List of interface types
        """
        return self._implementations.get(implementation, [])
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate all service dependencies.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for service_type, service_info in self._services.items():
            for dependency in service_info.dependencies:
                if dependency not in self._services:
                    errors.append(f"Service {service_type.__name__} depends on unregistered service {dependency.__name__}")
        
        return errors
    
    def get_dependency_graph(self) -> Dict[Type, List[Type]]:
        """
        Get the dependency graph.
        
        Returns:
            Dictionary mapping services to their dependencies
        """
        return {service_type: service_info.dependencies 
                for service_type, service_info in self._services.items()}
    
    def get_services_by_lifetime(self, lifetime: str) -> List[Type]:
        """
        Get services by lifetime.
        
        Args:
            lifetime: Service lifetime
            
        Returns:
            List of service types
        """
        return [service_type for service_type, service_info in self._services.items() 
                if service_info.lifetime == lifetime]
    
    def get_all_services(self) -> Dict[Type, ServiceInfo]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of all services
        """
        return self._services.copy()
    
    def clear(self):
        """Clear all registered services."""
        self._services.clear()
        self._interfaces.clear()
        self._implementations.clear()


class ServiceDiscovery:
    """Service discovery utilities."""
    
    @staticmethod
    def find_services_by_annotation(module, annotation: Type) -> List[Type]:
        """
        Find services by annotation.
        
        Args:
            module: Module to search
            annotation: Annotation type
            
        Returns:
            List of classes with the annotation
        """
        services = []
        
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj) and hasattr(obj, '__annotations__'):
                if annotation in obj.__annotations__.values():
                    services.append(obj)
        
        return services
    
    @staticmethod
    def find_services_by_base_class(module, base_class: Type) -> List[Type]:
        """
        Find services by base class.
        
        Args:
            module: Module to search
            base_class: Base class to search for
            
        Returns:
            List of classes inheriting from base class
        """
        services = []
        
        for name in dir(module):
            obj = getattr(module, name)
            if (inspect.isclass(obj) and 
                issubclass(obj, base_class) and 
                obj != base_class):
                services.append(obj)
        
        return services


