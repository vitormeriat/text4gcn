"""
Unit tests for Optimization System.

This module contains comprehensive tests for the optimization system including:
- Memory optimization tests
- Caching system tests
- Performance monitoring tests
- Resource management tests
- Async operations tests
- Optimization utilities tests
"""

import pytest
import time
import threading
import numpy as np
import torch as th
from unittest.mock import Mock, patch
from typing import Any, Dict, List

# Import optimization components
from text4gcn.optimization import (
    MemoryOptimizer, MemoryProfiler, MemoryStrategy,
    CacheManager, CacheStrategy, CacheLevel,
    PerformanceMonitor, MetricsCollector, MetricType, AlertLevel,
    ResourceManager, ResourceType, ResourceState,
    AsyncProcessor, AsyncTrainer, OperationType, OperationStatus,
    OptimizationUtils, OptimizationConfig
)


class TestMemoryOptimizer:
    """Test cases for MemoryOptimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler(enable_tracing=False)
        self.optimizer = MemoryOptimizer(self.profiler)
    
    def test_initialization(self):
        """Test memory optimizer initialization."""
        assert self.optimizer is not None
        assert self.optimizer.profiler is not None
        assert not any(self.optimizer.strategies.values())
    
    def test_enable_strategy(self):
        """Test enabling optimization strategies."""
        self.optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        assert self.optimizer.strategies[MemoryStrategy.LAZY_LOADING]
        
        self.optimizer.enable_strategy(MemoryStrategy.MEMORY_POOLING)
        assert self.optimizer.strategies[MemoryStrategy.MEMORY_POOLING]
    
    def test_disable_strategy(self):
        """Test disabling optimization strategies."""
        self.optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        self.optimizer.disable_strategy(MemoryStrategy.LAZY_LOADING)
        assert not self.optimizer.strategies[MemoryStrategy.LAZY_LOADING]
    
    def test_memory_pooling(self):
        """Test memory pooling functionality."""
        self.optimizer.enable_strategy(MemoryStrategy.MEMORY_POOLING)
        
        # Test getting pooled object
        def factory():
            return [1, 2, 3, 4, 5]
        
        obj1 = self.optimizer.get_pooled_object("test_pool", factory)
        assert obj1 == [1, 2, 3, 4, 5]
        
        # Test returning to pool
        self.optimizer.return_to_pool("test_pool", obj1)
        
        # Should get same object from pool
        obj2 = self.optimizer.get_pooled_object("test_pool", factory)
        assert obj2 == [1, 2, 3, 4, 5]
    
    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        self.optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        
        def loader():
            return "loaded_data"
        
        self.optimizer.register_lazy_loader("test_loader", loader)
        
        lazy_obj = self.optimizer.lazy_load("test_loader")
        assert hasattr(lazy_obj, '__getattr__')
        assert hasattr(lazy_obj, '__call__')
    
    def test_compress_tensor(self):
        """Test tensor compression."""
        self.optimizer.enable_strategy(MemoryStrategy.COMPRESSION)
        
        tensor = th.randn(100, 100, dtype=th.float32)
        compressed = self.optimizer.compress_tensor(tensor)
        
        assert compressed.dtype == th.float16
        assert compressed.shape == tensor.shape
    
    def test_decompress_tensor(self):
        """Test tensor decompression."""
        tensor = th.randn(100, 100, dtype=th.float16)
        decompressed = self.optimizer.decompress_tensor(tensor, th.float32)
        
        assert decompressed.dtype == th.float32
        assert decompressed.shape == tensor.shape
    
    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        results = self.optimizer.optimize_memory_usage()
        
        assert 'before' in results
        assert 'after' in results
        assert 'strategies_applied' in results
        assert 'memory_freed' in results
        assert 'objects_collected' in results
    
    def test_get_memory_report(self):
        """Test getting memory report."""
        report = self.optimizer.get_memory_report()
        
        assert 'current_stats' in report
        assert 'memory_by_type' in report
        assert 'growth_rate' in report
        assert 'active_strategies' in report
        assert 'pool_sizes' in report
        assert 'lazy_loaders' in report


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager()
    
    def test_initialization(self):
        """Test cache manager initialization."""
        assert self.cache_manager is not None
        assert self.cache_manager.l1 is not None
        assert self.cache_manager.l2 is not None
        assert self.cache_manager.l3 is not None
    
    def test_put_and_get(self):
        """Test putting and getting from cache."""
        data = {"key": "value", "number": 42}
        
        # Put in cache
        success = self.cache_manager.put("test_key", data)
        assert success
        
        # Get from cache
        retrieved = self.cache_manager.get("test_key")
        assert retrieved == data
    
    def test_cache_levels(self):
        """Test cache level hierarchy."""
        data = "test_data"
        
        # Put in L1
        self.cache_manager.put("l1_key", data, level=CacheLevel.L1)
        
        # Should be in L1
        result = self.cache_manager.get("l1_key")
        assert result == data
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        data = "expiring_data"
        
        # Put with short TTL
        self.cache_manager.put("ttl_key", data, ttl=0.1)
        
        # Should be available immediately
        result = self.cache_manager.get("ttl_key")
        assert result == data
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        result = self.cache_manager.get("ttl_key")
        assert result is None
    
    def test_invalidation(self):
        """Test cache invalidation."""
        data = "invalidatable_data"
        
        # Put in cache
        self.cache_manager.put("invalid_key", data)
        
        # Should be available
        result = self.cache_manager.get("invalid_key")
        assert result == data
        
        # Invalidate
        self.cache_manager.invalidate("invalid_key")
        
        # Should be gone
        result = self.cache_manager.get("invalid_key")
        assert result is None
    
    def test_pattern_invalidation(self):
        """Test pattern-based invalidation."""
        # Put multiple keys
        self.cache_manager.put("prefix_key1", "data1")
        self.cache_manager.put("prefix_key2", "data2")
        self.cache_manager.put("other_key", "data3")
        
        # Invalidate by pattern
        self.cache_manager.invalidate_pattern("prefix_")
        
        # Prefix keys should be gone
        assert self.cache_manager.get("prefix_key1") is None
        assert self.cache_manager.get("prefix_key2") is None
        
        # Other key should remain
        assert self.cache_manager.get("other_key") == "data3"
    
    def test_compression(self):
        """Test cache compression."""
        self.cache_manager.enable_compression()
        
        data = "compressible_data"
        self.cache_manager.put("compressed_key", data)
        
        result = self.cache_manager.get("compressed_key")
        assert result == data
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Put some data
        self.cache_manager.put("stats_key1", "data1")
        self.cache_manager.put("stats_key2", "data2")
        
        # Get stats
        stats = self.cache_manager.get_combined_stats()
        
        assert stats.hits >= 0
        assert stats.misses >= 0
        assert stats.total_size >= 0
        assert stats.entry_count >= 0
    
    def test_optimize(self):
        """Test cache optimization."""
        # Put some data
        self.cache_manager.put("opt_key1", "data1")
        self.cache_manager.put("opt_key2", "data2")
        
        # Optimize
        results = self.cache_manager.optimize()
        
        assert 'before' in results
        assert 'after' in results
        assert 'optimizations_applied' in results


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
        self.monitor = PerformanceMonitor(self.collector)
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        assert self.monitor is not None
        assert self.monitor.collector is not None
        assert not self.monitor.monitoring
    
    def test_record_metric(self):
        """Test recording metrics."""
        self.monitor.record_metric("test.metric", 42.0, MetricType.CUSTOM, "units")
        
        stats = self.monitor.get_metric_stats("test.metric")
        assert stats is not None
        assert stats.count >= 1
    
    def test_thresholds(self):
        """Test threshold monitoring."""
        # Set threshold
        self.monitor.set_threshold("test.metric", AlertLevel.WARNING, 50.0)
        
        # Record metric below threshold
        self.monitor.record_metric("test.metric", 30.0, MetricType.CUSTOM)
        
        # Should not trigger alert
        alerts = self.monitor.get_alerts()
        assert len(alerts) == 0
        
        # Record metric above threshold
        self.monitor.record_metric("test.metric", 60.0, MetricType.CUSTOM)
        
        # Should trigger alert
        alerts = self.monitor.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].level == AlertLevel.WARNING
    
    def test_alert_callbacks(self):
        """Test alert callbacks."""
        callback_called = []
        
        def callback(alert):
            callback_called.append(alert)
        
        self.monitor.add_alert_callback("test.metric", callback)
        self.monitor.set_threshold("test.metric", AlertLevel.ERROR, 50.0)
        
        # Trigger alert
        self.monitor.record_metric("test.metric", 60.0, MetricType.CUSTOM)
        
        # Callback should be called
        assert len(callback_called) >= 1
    
    def test_measure_execution_time(self):
        """Test execution time measurement."""
        with self.monitor.measure_execution_time("test_operation"):
            time.sleep(0.1)
        
        # Should have recorded execution time
        stats = self.monitor.get_metric_stats("execution_time.test_operation")
        assert stats is not None
        assert stats.count >= 1
    
    def test_benchmark_function(self):
        """Test function benchmarking."""
        def test_func():
            time.sleep(0.01)
            return "result"
        
        benchmark = self.monitor.benchmark_function(test_func)
        
        assert benchmark['function_name'] == 'test_func'
        assert benchmark['success'] is True
        assert benchmark['execution_time'] > 0
        assert benchmark['result'] == "result"
    
    def test_get_performance_report(self):
        """Test getting performance report."""
        # Record some metrics
        self.monitor.record_metric("test.metric1", 10.0, MetricType.CUSTOM)
        self.monitor.record_metric("test.metric2", 20.0, MetricType.CUSTOM)
        
        report = self.monitor.get_performance_report()
        
        assert 'timestamp' in report
        assert 'metrics_count' in report
        assert 'alerts_count' in report
        assert 'metrics' in report
        assert 'alerts' in report


class TestResourceManager:
    """Test cases for ResourceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager()
    
    def test_initialization(self):
        """Test resource manager initialization."""
        assert self.resource_manager is not None
        assert len(self.resource_manager.pools) == 0
        assert len(self.resource_manager.quotas) == 0
    
    def test_create_pool(self):
        """Test creating resource pool."""
        pool = self.resource_manager.create_pool(
            "test_pool", ResourceType.CPU, max_size=10, min_size=2
        )
        
        assert pool.name == "test_pool"
        assert pool.resource_type == ResourceType.CPU
        assert pool.max_size == 10
        assert pool.min_size == 2
    
    def test_add_resource(self):
        """Test adding resource to pool."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU)
        
        success = self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        assert success
        
        # Try to add to non-existent pool
        success = self.resource_manager.add_resource("nonexistent", "resource_2", 1.0)
        assert not success
    
    def test_allocate_resource(self):
        """Test resource allocation."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU)
        self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        
        resource = self.resource_manager.allocate_resource("test_pool", 0.5)
        assert resource is not None
        assert resource.id == "resource_1"
        assert resource.state == ResourceState.ALLOCATED
    
    def test_release_resource(self):
        """Test resource release."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU)
        self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        
        # Allocate resource
        resource = self.resource_manager.allocate_resource("test_pool", 0.5)
        assert resource is not None
        
        # Release resource
        success = self.resource_manager.release_resource(resource.id)
        assert success
        
        # Resource should be available again
        assert resource.state == ResourceState.AVAILABLE
    
    def test_set_quota(self):
        """Test setting resource quota."""
        self.resource_manager.set_quota(ResourceType.CPU, 8.0, 0.8, 0.95)
        
        assert ResourceType.CPU in self.resource_manager.quotas
        quota = self.resource_manager.quotas[ResourceType.CPU]
        assert quota.max_usage == 8.0
        assert quota.soft_limit == 0.8
        assert quota.hard_limit == 0.95
    
    def test_check_quota(self):
        """Test quota checking."""
        self.resource_manager.set_quota(ResourceType.CPU, 8.0, 0.8, 0.95)
        
        # Should allow allocation within quota
        assert self.resource_manager.check_quota(ResourceType.CPU, 4.0)
        
        # Should not allow allocation exceeding quota
        assert not self.resource_manager.check_quota(ResourceType.CPU, 10.0)
    
    def test_resource_context(self):
        """Test resource context manager."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU)
        self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        
        with self.resource_manager.resource_context("test_pool", 0.5) as resource:
            assert resource is not None
            assert resource.state == ResourceState.ALLOCATED
        
        # Resource should be released after context
        assert resource.state == ResourceState.AVAILABLE
    
    def test_get_pool_stats(self):
        """Test getting pool statistics."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU, max_size=5)
        self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        self.resource_manager.add_resource("test_pool", "resource_2", 1.0)
        
        stats = self.resource_manager.get_pool_stats("test_pool")
        assert stats is not None
        assert stats['pool_name'] == "test_pool"
        assert stats['total_resources'] == 2
        assert stats['available_resources'] == 2
    
    def test_optimize_resources(self):
        """Test resource optimization."""
        self.resource_manager.create_pool("test_pool", ResourceType.CPU)
        self.resource_manager.add_resource("test_pool", "resource_1", 1.0)
        
        results = self.resource_manager.optimize_resources()
        
        assert 'before' in results
        assert 'after' in results
        assert 'optimizations_applied' in results


class TestAsyncProcessor:
    """Test cases for AsyncProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AsyncProcessor(max_workers=2)
    
    def test_initialization(self):
        """Test async processor initialization."""
        assert self.processor is not None
        assert self.processor.max_workers == 2
        assert not self.processor.running
    
    def test_start_stop(self):
        """Test starting and stopping processor."""
        self.processor.start()
        assert self.processor.running
        
        self.processor.stop()
        assert not self.processor.running
    
    def test_submit_function(self):
        """Test submitting function for execution."""
        self.processor.start()
        
        def test_func(x):
            return x * 2
        
        task_id = self.processor.submit_function(test_func, 5)
        assert task_id is not None
        
        # Wait for completion
        result = self.processor.wait_for_operation(task_id, timeout=5.0)
        assert result == 10
        
        self.processor.stop()
    
    def test_operation_status(self):
        """Test operation status tracking."""
        self.processor.start()
        
        def slow_func():
            time.sleep(0.1)
            return "done"
        
        task_id = self.processor.submit_function(slow_func)
        
        # Check status
        status = self.processor.get_operation_status(task_id)
        assert status in [OperationStatus.PENDING, OperationStatus.RUNNING]
        
        # Wait for completion
        result = self.processor.wait_for_operation(task_id, timeout=5.0)
        assert result == "done"
        
        # Check final status
        status = self.processor.get_operation_status(task_id)
        assert status == OperationStatus.COMPLETED
        
        self.processor.stop()
    
    def test_cancel_operation(self):
        """Test cancelling operation."""
        self.processor.start()
        
        def slow_func():
            time.sleep(1.0)
            return "done"
        
        task_id = self.processor.submit_function(slow_func)
        
        # Cancel operation
        success = self.processor.cancel_operation(task_id)
        assert success
        
        self.processor.stop()
    
    def test_get_stats(self):
        """Test getting processor statistics."""
        self.processor.start()
        
        def test_func():
            return "result"
        
        task_id = self.processor.submit_function(test_func)
        
        stats = self.processor.get_stats()
        assert 'total_operations' in stats
        assert 'completed_operations' in stats
        assert 'running' in stats
        
        self.processor.stop()


class TestOptimizationUtils:
    """Test cases for OptimizationUtils."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.utils = OptimizationUtils()
    
    def test_initialization(self):
        """Test optimization utils initialization."""
        assert self.utils is not None
        assert self.utils.config is not None
    
    def test_profile_function(self):
        """Test function profiling."""
        def test_func(n):
            return sum(range(n))
        
        profile = self.utils.profile_function(test_func, 1000)
        
        assert profile['function_name'] == 'test_func'
        assert profile['wall_time'] > 0
        assert profile['success'] is True
        assert profile['result'] == sum(range(1000))
    
    def test_benchmark_function(self):
        """Test function benchmarking."""
        def test_func(n):
            return n * n
        
        benchmark = self.utils.benchmark_function(test_func, iterations=3, n=100)
        
        assert benchmark['function_name'] == 'test_func'
        assert benchmark['iterations'] == 3
        assert benchmark['times']['mean'] > 0
        assert benchmark['success_rate'] == 1.0
    
    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        results = self.utils.optimize_memory_usage()
        
        assert 'initial_memory' in results
        assert 'final_memory' in results
        assert 'memory_freed' in results
        assert 'garbage_collected' in results
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        def test_func():
            return "test"
        
        self.utils.profile_function(test_func)
        self.utils.benchmark_function(test_func, iterations=2)
        
        summary = self.utils.get_performance_summary()
        
        assert 'profiled_functions' in summary
        assert 'benchmarked_functions' in summary
        assert 'total_profiles' in summary
        assert 'total_benchmarks' in summary
    
    def test_clear_profiles(self):
        """Test clearing profiles."""
        def test_func():
            return "test"
        
        self.utils.profile_function(test_func)
        self.utils.benchmark_function(test_func)
        
        # Should have profiles
        summary = self.utils.get_performance_summary()
        assert summary['total_profiles'] > 0
        
        # Clear profiles
        self.utils.clear_profiles()
        
        # Should be empty
        summary = self.utils.get_performance_summary()
        assert summary['total_profiles'] == 0


class TestIntegration:
    """Integration tests for optimization system."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create all components
        profiler = MemoryProfiler(enable_tracing=False)
        memory_optimizer = MemoryOptimizer(profiler)
        cache_manager = CacheManager()
        performance_monitor = PerformanceMonitor()
        resource_manager = ResourceManager()
        async_processor = AsyncProcessor(max_workers=2)
        utils = OptimizationUtils()
        
        # Configure optimization
        memory_optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        cache_manager.enable_compression()
        
        # Set up resources
        resource_manager.create_pool("test_pool", ResourceType.CPU, max_size=2)
        resource_manager.add_resource("test_pool", "cpu_1", 1.0)
        
        # Start systems
        performance_monitor.start_monitoring()
        resource_manager.start_monitoring()
        async_processor.start()
        
        # Simulate workflow
        def workflow_function():
            # Cache some data
            data = np.random.rand(100, 100)
            cache_manager.put("workflow_data", data)
            
            # Process with memory optimization
            with memory_optimizer.profiler.track_allocation(data):
                result = np.sum(data)
            
            # Monitor performance
            performance_monitor.record_metric("workflow.processing_time", 0.1, MetricType.EXECUTION_TIME)
            
            return result
        
        # Execute workflow
        result = workflow_function()
        assert result is not None
        
        # Get optimization results
        memory_results = memory_optimizer.optimize_memory_usage()
        cache_stats = cache_manager.get_combined_stats()
        performance_report = performance_monitor.get_performance_report()
        resource_stats = resource_manager.get_quota_stats()
        
        # Verify results
        assert memory_results['memory_freed'] >= 0
        assert cache_stats.hit_ratio >= 0
        assert performance_report['metrics_count'] >= 0
        assert isinstance(resource_stats, dict)
        
        # Cleanup
        performance_monitor.stop_monitoring()
        resource_manager.stop_monitoring()
        async_processor.stop()
        
        # Verify cleanup
        assert not performance_monitor.monitoring
        assert not resource_manager._monitoring
        assert not async_processor.running


if __name__ == '__main__':
    pytest.main([__file__])


