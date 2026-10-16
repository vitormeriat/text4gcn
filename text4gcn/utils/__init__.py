"""
Utility modules for Text4GCN.

This package contains utility functions and classes for device management,
logging, and other common functionality.
"""

from .device import (
    DeviceManager,
    get_device_manager,
    check_pytorch_installation,
    set_global_device,
    get_global_device,
)

__all__ = [
    "DeviceManager",
    "get_device_manager",
    "check_pytorch_installation", 
    "set_global_device",
    "get_global_device",
]
