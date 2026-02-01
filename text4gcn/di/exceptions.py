"""
Dependency Injection Exceptions.

This module defines custom exceptions for the dependency injection system.
"""


class DIContainerError(Exception):
    """Base exception for DI container errors."""
    pass


class ServiceNotFoundError(DIContainerError):
    """Exception raised when a service is not found."""
    
    def __init__(self, message: str, service_type: str = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            service_type: Type of service that was not found
        """
        super().__init__(message)
        self.service_type = service_type


class CircularDependencyError(DIContainerError):
    """Exception raised when circular dependency is detected."""
    
    def __init__(self, message: str, dependency_chain: list = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            dependency_chain: Chain of dependencies causing the cycle
        """
        super().__init__(message)
        self.dependency_chain = dependency_chain or []


class ServiceRegistrationError(DIContainerError):
    """Exception raised when service registration fails."""
    
    def __init__(self, message: str, service_type: str = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            service_type: Type of service that failed to register
        """
        super().__init__(message)
        self.service_type = service_type


class ConfigurationError(DIContainerError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
        """
        super().__init__(message)
        self.config_key = config_key


class ScopeError(DIContainerError):
    """Exception raised when scope operations fail."""
    
    def __init__(self, message: str, scope_name: str = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            scope_name: Name of the scope that caused the error
        """
        super().__init__(message)
        self.scope_name = scope_name


