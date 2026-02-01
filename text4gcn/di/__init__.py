"""
Dependency Injection System for Text4GCN.

This module provides a complete dependency injection system following
the Dependency Inversion Principle and enabling loose coupling between components.
"""

from .container import DIContainer, ServiceLifetime
from .service_registry import ServiceRegistry
from .configuration import ConfigurationManager
from .decorators import inject, singleton, transient, set_container
from .exceptions import DIContainerError, ServiceNotFoundError, CircularDependencyError

__all__ = [
    "DIContainer",
    "ServiceLifetime",
    "ServiceRegistry", 
    "ConfigurationManager",
    "inject",
    "singleton",
    "transient",
    "set_container",
    "DIContainerError",
    "ServiceNotFoundError",
    "CircularDependencyError"
]


