"""Tests for device management utilities."""

import pytest
from unittest.mock import patch, MagicMock

from text4gcn.utils.device import (
    DeviceManager,
    get_device_manager,
    check_pytorch_installation,
    set_global_device,
    get_global_device,
)


class TestDeviceManager:
    """Test DeviceManager class."""

    def test_device_manager_initialization_cpu(self):
        """Test DeviceManager initialization with CPU."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            manager = DeviceManager("cpu")
            assert manager.device.type == "cpu"
            assert manager.is_cpu
            assert not manager.is_cuda

    def test_device_manager_initialization_cuda(self):
        """Test DeviceManager initialization with CUDA."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props

            manager = DeviceManager("cuda")
            assert manager.device.type == "cuda"
            assert manager.is_cuda
            assert not manager.is_cpu

    def test_auto_detect_device_cpu(self):
        """Test auto-detection when CUDA is not available."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device

            manager = DeviceManager()
            assert manager.device.type == "cpu"
            assert manager.is_cpu

    def test_auto_detect_device_cuda(self):
        """Test auto-detection when CUDA is available."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            mock_torch.cuda.is_available.return_value = True
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props

            manager = DeviceManager()
            assert manager.device.type == "cuda"
            assert manager.is_cuda

    def test_get_device_info_cpu(self):
        """Test device info for CPU."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            manager = DeviceManager("cpu")
            info = manager.get_device_info()

            # device is returned as string representation
            assert info["device"] == "cpu" or "cpu" in str(info["device"])
            assert info["type"] == "cpu"
            assert info["is_cpu"]
            assert not info["is_cuda"]

    def test_get_device_info_cuda(self):
        """Test device info for CUDA."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props

            manager = DeviceManager("cuda")
            info = manager.get_device_info()

            # device is returned as string representation
            assert info["device"] == "cuda" or "cuda" in str(info["device"])
            assert info["type"] == "cuda"
            assert info["is_cuda"]
            assert not info["is_cpu"]
            assert info["cuda_version"] == "12.1"
            assert info["gpu_name"] == "Test GPU"
            assert info["gpu_memory_gb"] == 8.0

    def test_move_to_device(self):
        """Test moving tensors/models to device."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = "moved_tensor"

            manager = DeviceManager("cuda")
            result = manager.move_to_device(mock_tensor)

            assert result == "moved_tensor"
            # Verify that to() was called with the device object
            mock_tensor.to.assert_called_once()

    def test_set_seed(self):
        """Test setting random seed."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            manager = DeviceManager("cpu")
            manager.set_seed(42)

            mock_torch.manual_seed.assert_called_once_with(42)
            mock_torch.cuda.manual_seed.assert_not_called()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_device_manager(self):
        """Test get_device_manager function."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            manager = get_device_manager("cpu")
            assert isinstance(manager, DeviceManager)
            assert manager.device.type == "cpu"

    def test_check_pytorch_installation_available(self):
        """Test check_pytorch_installation when PyTorch is available."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.__version__ = "2.1.2"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.version.cuda = "12.1"

            info = check_pytorch_installation()

            assert info == {"torch_version": "2.1.2", "cuda_available": True, "cuda_version": "12.1"}

class TestGlobalDeviceManager:
    """Test global device manager functionality."""

    def setup_method(self):
        """Reset global device manager before each test."""
        # Reset global device manager to avoid state leakage between tests
        import text4gcn.utils.device as device_module
        device_module._global_device_manager = None

    def test_set_global_device(self):
        """Test setting global device manager."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props

            manager = set_global_device("cuda")
            assert isinstance(manager, DeviceManager)
            assert manager.device.type == "cuda"

    def test_get_global_device(self):
        """Test getting global device manager."""
        # Reset global device manager before test
        import text4gcn.utils.device as device_module
        device_module._global_device_manager = None

        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            manager = get_global_device()
            assert isinstance(manager, DeviceManager)
            assert manager.device.type == "cpu"


@pytest.mark.slow
class TestDeviceManagerPerformance:
    """Performance tests for device manager."""

    def test_device_manager_initialization_performance(self):
        """Test that device manager initialization is fast."""
        import time

        with patch('text4gcn.utils.device.torch') as mock_torch:
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cpu"
            mock_device.__str__ = lambda self: "cpu"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = False

            start_time = time.time()
            for _ in range(100):
                DeviceManager("cpu")
            end_time = time.time()

            # Should initialize 100 device managers in less than 1 second
            assert (end_time - start_time) < 1.0


class TestDeviceManagerIntegration:
    """Integration tests for device manager."""

    def test_device_manager_with_real_torch(self):
        """Test device manager with real PyTorch (if available)."""
        try:
            import torch
            # Test with real PyTorch if available
            manager = DeviceManager()
            assert isinstance(manager.device, torch.device)
            assert manager.device.type in ["cpu", "cuda"]
        except ImportError:
            pytest.skip("PyTorch not available for integration test")

    def test_device_info_completeness(self):
        """Test that device info contains all expected fields."""
        with patch('text4gcn.utils.device.torch') as mock_torch, \
             patch('text4gcn.utils.device.logger'):
            # Create a mock device object with type attribute
            mock_device = MagicMock()
            mock_device.type = "cuda"
            mock_device.__str__ = lambda self: "cuda"
            mock_torch.device.return_value = mock_device
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_props = MagicMock()
            mock_props.total_memory = 8e9
            mock_torch.cuda.get_device_properties.return_value = mock_props

            manager = DeviceManager("cuda")
            info = manager.get_device_info()

            expected_fields = [
                "device", "type", "is_cuda", "is_cpu",
                "cuda_version", "gpu_name", "gpu_memory", "gpu_memory_gb"
            ]

            for field in expected_fields:
                assert field in info
