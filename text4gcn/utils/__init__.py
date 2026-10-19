"""
Utility modules for Text4GCN.

This package contains utility functions and classes for device management,
logging, and other common functionality.
"""

from .logging import get_logger, set_verbosity
from .device import (
    DeviceManager,
    get_device_manager,
    check_pytorch_installation,
    set_global_device,
    get_global_device,
)

__all__ = [
    "get_logger",
    "set_verbosity",
    "DeviceManager",
    "get_device_manager",
    "check_pytorch_installation",
    "set_global_device",
    "get_global_device",
]
