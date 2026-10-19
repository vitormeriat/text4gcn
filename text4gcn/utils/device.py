"""
Device management utilities for Text4GCN.

This module provides utilities for detecting and managing PyTorch devices,
supporting both CPU and GPU (CUDA) configurations.
"""

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages PyTorch device selection and configuration."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize device manager.

        Args:
            device: Specific device to use. If None, auto-detects best available device.
        """
        self._device = self._resolve_device(device)
        self._log_device_info()

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve the device to use."""
        if device is None:
            return self._auto_detect_device()

        if isinstance(device, str):
            return torch.device(device)

        return device

    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            return device
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")
            return device

    def _log_device_info(self) -> None:
        """Log device information."""
        if self._device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("Using CPU")

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device.type == "cpu"

    def get_device_info(self) -> dict:
        """Get detailed device information."""
        info = {
            "device": str(self._device),
            "type": self._device.type,
            "is_cuda": self.is_cuda,
            "is_cpu": self.is_cpu,
        }

        if self.is_cuda:
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })

        return info

    def move_to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or model to the current device."""
        return tensor_or_model.to(self._device)

    def set_seed(self, seed: int = 42) -> None:
        """Set random seeds for ``random``, NumPy, and PyTorch (CPU/CUDA)."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.is_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")


def get_device_manager(device: Optional[Union[str, torch.device]] = None) -> DeviceManager:
    """
    Get a device manager instance.

    Args:
        device: Specific device to use. If None, auto-detects best available device.

    Returns:
        DeviceManager instance.
    """
    return DeviceManager(device)


def check_pytorch_installation() -> dict:
    """
    Check PyTorch installation and return information.

    Returns:
        Dictionary with installation information.
    """
    cuda_available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
    }


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def set_global_device(device: Optional[Union[str, torch.device]] = None) -> DeviceManager:
    """
    Set the global device manager.

    Args:
        device: Specific device to use.

    Returns:
        Global device manager instance.
    """
    global _global_device_manager
    _global_device_manager = DeviceManager(device)
    return _global_device_manager


def get_global_device() -> DeviceManager:
    """
    Get the global device manager.

    Returns:
        Global device manager instance.
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager
