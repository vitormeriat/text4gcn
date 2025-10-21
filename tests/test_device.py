"""Tests for device management utilities."""

import pytest
from unittest.mock import patch, MagicMock
import warnings

from text4gcn.utils.device import (
    DeviceManager,
    get_device_manager,
    check_pytorch_installation,
    recommend_installation,
    set_global_device,
    get_global_device,
)


class TestDeviceManager:
    """Test DeviceManager class."""
    
    def test_device_manager_initialization_cpu(self):
        """Test DeviceManager initialization with CPU."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            
            manager = DeviceManager("cpu")
            assert manager.device == "cpu"
            assert manager.is_cpu
            assert not manager.is_cuda
    
    def test_device_manager_initialization_cuda(self):
        """Test DeviceManager initialization with CUDA."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cuda"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8e9
            
            manager = DeviceManager("cuda")
            assert manager.device == "cuda"
            assert manager.is_cuda
            assert not manager.is_cpu
    
    def test_auto_detect_device_cpu(self):
        """Test auto-detection when CUDA is not available."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"
            
            manager = DeviceManager()
            assert manager.device == "cpu"
            assert manager.is_cpu
    
    def test_auto_detect_device_cuda(self):
        """Test auto-detection when CUDA is available."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.return_value = "cuda"
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8e9
            
            manager = DeviceManager()
            assert manager.device == "cuda"
            assert manager.is_cuda
    
    def test_get_device_info_cpu(self):
        """Test device info for CPU."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            
            manager = DeviceManager("cpu")
            info = manager.get_device_info()
            
            assert info["device"] == "cpu"
            assert info["type"] == "cpu"
            assert info["is_cpu"]
            assert not info["is_cuda"]
    
    def test_get_device_info_cuda(self):
        """Test device info for CUDA."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cuda"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8e9
            
            manager = DeviceManager("cuda")
            info = manager.get_device_info()
            
            assert info["device"] == "cuda"
            assert info["type"] == "cuda"
            assert info["is_cuda"]
            assert not info["is_cpu"]
            assert info["cuda_version"] == "12.1"
            assert info["gpu_name"] == "Test GPU"
            assert info["gpu_memory_gb"] == 8.0
    
    def test_move_to_device(self):
        """Test moving tensors/models to device."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cuda"
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = "moved_tensor"
            
            manager = DeviceManager("cuda")
            result = manager.move_to_device(mock_tensor)
            
            assert result == "moved_tensor"
            mock_tensor.to.assert_called_once_with("cuda")
    
    def test_set_seed(self):
        """Test setting random seed."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            
            manager = DeviceManager("cpu")
            manager.set_seed(42)
            
            mock_torch.manual_seed.assert_called_once_with(42)
            mock_torch.cuda.manual_seed.assert_not_called()


class TestDeviceManagerErrorHandling:
    """Test error handling in DeviceManager."""
    
    def test_no_torch_import(self):
        """Test behavior when PyTorch is not available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                DeviceManager()
            
            assert "PyTorch is not installed" in str(exc_info.value)
            assert "pip install text4gcn[cpu]" in str(exc_info.value)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_device_manager(self):
        """Test get_device_manager function."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            
            manager = get_device_manager("cpu")
            assert isinstance(manager, DeviceManager)
            assert manager.device == "cpu"
    
    def test_check_pytorch_installation_available(self):
        """Test check_pytorch_installation when PyTorch is available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', True):
            with patch('text4gcn.utils.device.torch') as mock_torch:
                mock_torch.__version__ = "2.1.2"
                mock_torch.cuda.is_available.return_value = True
                mock_torch.version.cuda = "12.1"
                
                info = check_pytorch_installation()
                
                assert info["torch_available"]
                assert info["torch_version"] == "2.1.2"
                assert info["cuda_available"]
                assert info["cuda_version"] == "12.1"
    
    def test_check_pytorch_installation_unavailable(self):
        """Test check_pytorch_installation when PyTorch is not available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', False):
            info = check_pytorch_installation()
            
            assert not info["torch_available"]
            assert info["torch_version"] is None
            assert not info["cuda_available"]
            assert info["cuda_version"] is None
    
    def test_recommend_installation_no_torch(self):
        """Test installation recommendation when PyTorch is not available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', False):
            recommendation = recommend_installation()
            assert "pip install text4gcn[cpu]" in recommendation
    
    def test_recommend_installation_with_cuda(self):
        """Test installation recommendation when CUDA is available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', True):
            with patch('text4gcn.utils.device.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                
                recommendation = recommend_installation()
                assert "pip install text4gcn[gpu]" in recommendation
    
    def test_recommend_installation_cpu_only(self):
        """Test installation recommendation when only CPU is available."""
        with patch('text4gcn.utils.device.TORCH_AVAILABLE', True):
            with patch('text4gcn.utils.device.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                
                recommendation = recommend_installation()
                assert "pip install text4gcn[cpu]" in recommendation


class TestGlobalDeviceManager:
    """Test global device manager functionality."""
    
    def test_set_global_device(self):
        """Test setting global device manager."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cuda"
            mock_torch.cuda.is_available.return_value = True
            
            manager = set_global_device("cuda")
            assert isinstance(manager, DeviceManager)
            assert manager.device == "cuda"
    
    def test_get_global_device(self):
        """Test getting global device manager."""
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            
            manager = get_global_device()
            assert isinstance(manager, DeviceManager)
            assert manager.device == "cpu"


@pytest.mark.slow
class TestDeviceManagerPerformance:
    """Performance tests for device manager."""
    
    def test_device_manager_initialization_performance(self):
        """Test that device manager initialization is fast."""
        import time
        
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cpu"
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
        with patch('text4gcn.utils.device.torch') as mock_torch:
            mock_torch.device.return_value = "cuda"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "Test GPU"
            mock_torch.version.cuda = "12.1"
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8e9
            
            manager = DeviceManager("cuda")
            info = manager.get_device_info()
            
            expected_fields = [
                "device", "type", "is_cuda", "is_cpu",
                "cuda_version", "gpu_name", "gpu_memory", "gpu_memory_gb"
            ]
            
            for field in expected_fields:
                assert field in info
