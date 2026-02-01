"""
Configuration Management System.

This module provides a comprehensive configuration system with support for:
- Multiple configuration sources (JSON, YAML, environment variables)
- Configuration validation
- Type conversion
- Nested configuration access
- Configuration inheritance
- Hot reloading
"""

import json
import yaml
import os
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import threading
from dataclasses import dataclass, field
from enum import Enum


T = TypeVar('T')


class ConfigurationSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DICTIONARY = "dictionary"
    DEFAULT = "default"


@dataclass
class ConfigurationValue:
    """Configuration value with metadata."""
    value: Any
    source: ConfigurationSource
    key: str
    type_hint: Optional[Type] = None
    required: bool = False
    default_value: Any = None
    description: str = ""


@dataclass
class ConfigurationSection:
    """Configuration section."""
    name: str
    values: Dict[str, ConfigurationValue] = field(default_factory=dict)
    subsections: Dict[str, 'ConfigurationSection'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigurationValidator:
    """Configuration validator."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type) -> bool:
        """
        Validate value type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            
        Returns:
            True if valid
        """
        if expected_type == Any:
            return True
        
        if expected_type == Union[str, int]:
            return isinstance(value, (str, int))
        
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_required(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """
        Validate required configuration keys.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Returns:
            List of missing keys
        """
        missing = []
        for key in required_keys:
            if key not in config or config[key] is None:
                missing.append(key)
        return missing
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """
        Validate numeric range.
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            True if valid
        """
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True


class ConfigurationManager:
    """
    Configuration Manager for Text4GCN.
    
    Provides centralized configuration management with multiple sources.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config: Dict[str, Any] = {}
        self._sections: Dict[str, ConfigurationSection] = {}
        self._validators: Dict[str, ConfigurationValidator] = {}
        self._sources: List[ConfigurationSource] = []
        self._lock = threading.RLock()
        self._watchers: List[Callable] = []
    
    def load_from_file(self, file_path: Union[str, Path], 
                      source: ConfigurationSource = ConfigurationSource.FILE) -> 'ConfigurationManager':
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            source: Configuration source type
            
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with self._lock:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config = json.load(f)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            
            self._merge_config(config, source)
            self._sources.append(source)
        
        return self
    
    def load_from_environment(self, prefix: str = "TEXT4GCN_", 
                            separator: str = "__") -> 'ConfigurationManager':
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            separator: Separator for nested keys
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            env_config = {}
            
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Remove prefix and convert to nested structure
                    config_key = key[len(prefix):].lower()
                    nested_keys = config_key.split(separator)
                    
                    # Create nested dictionary
                    current = env_config
                    for nested_key in nested_keys[:-1]:
                        if nested_key not in current:
                            current[nested_key] = {}
                        current = current[nested_key]
                    
                    # Set final value
                    current[nested_keys[-1]] = self._convert_env_value(value)
            
            self._merge_config(env_config, ConfigurationSource.ENVIRONMENT)
            self._sources.append(ConfigurationSource.ENVIRONMENT)
        
        return self
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> 'ConfigurationManager':
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            self._merge_config(config_dict, ConfigurationSource.DICTIONARY)
            self._sources.append(ConfigurationSource.DICTIONARY)
        
        return self
    
    def _merge_config(self, new_config: Dict[str, Any], source: ConfigurationSource):
        """Merge new configuration into existing."""
        def merge_dicts(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
        
        merge_dicts(self._config, new_config)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None, type_hint: Optional[Type] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            type_hint: Expected type
            
        Returns:
            Configuration value
        """
        with self._lock:
            value = self._get_nested_value(self._config, key)
            
            if value is None:
                return default
            
            if type_hint and not isinstance(value, type_hint):
                try:
                    return type_hint(value)
                except (ValueError, TypeError):
                    return default
            
            return value
    
    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration section.
        
        Args:
            section_name: Section name
            
        Returns:
            Section configuration or None
        """
        return self.get(section_name)
    
    def set(self, key: str, value: Any) -> 'ConfigurationManager':
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            self._set_nested_value(self._config, key, value)
            self._notify_watchers(key, value)
        
        return self
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested value from configuration."""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested value in configuration."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def validate(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            schema: Validation schema
            
        Returns:
            List of validation errors
        """
        errors = []
        
        def validate_section(config: Dict, schema: Dict, path: str = ""):
            for key, rules in schema.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in config:
                    if rules.get('required', False):
                        errors.append(f"Missing required configuration: {current_path}")
                    continue
                
                value = config[key]
                
                # Type validation
                if 'type' in rules:
                    expected_type = rules['type']
                    if not ConfigurationValidator.validate_type(value, expected_type):
                        errors.append(f"Invalid type for {current_path}: expected {expected_type.__name__}")
                
                # Range validation
                if 'min' in rules or 'max' in rules:
                    if isinstance(value, (int, float)):
                        if not ConfigurationValidator.validate_range(value, rules.get('min'), rules.get('max')):
                            errors.append(f"Value out of range for {current_path}")
                
                # Nested validation
                if 'properties' in rules and isinstance(value, dict):
                    validate_section(value, rules['properties'], current_path)
        
        validate_section(self._config, schema)
        return errors
    
    def add_watcher(self, callback: Callable[[str, Any], None]) -> 'ConfigurationManager':
        """
        Add configuration change watcher.
        
        Args:
            callback: Callback function
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            self._watchers.append(callback)
        return self
    
    def _notify_watchers(self, key: str, value: Any):
        """Notify watchers of configuration changes."""
        for callback in self._watchers:
            try:
                callback(key, value)
            except Exception:
                pass  # Ignore watcher errors
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "json") -> 'ConfigurationManager':
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format (json, yaml)
            
        Returns:
            Self for method chaining
        """
        file_path = Path(file_path)
        
        with self._lock:
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return self
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration.
        
        Returns:
            Complete configuration dictionary
        """
        with self._lock:
            return self._config.copy()
    
    def clear(self):
        """Clear all configuration."""
        with self._lock:
            self._config.clear()
            self._sections.clear()
            self._sources.clear()
            self._watchers.clear()


