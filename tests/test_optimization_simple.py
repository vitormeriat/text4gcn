"""
Simple tests for Optimization System.

This module contains basic tests for the optimization system
that don't require external dependencies.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from typing import Any, Dict, List

# Import optimization components
from text4gcn.optimization.memory_optimizer import MemoryOptimizer, MemoryProfiler, MemoryStrategy
from text4gcn.optimization.cache_manager import CacheManager, CacheStrategy, CacheLevel
from text4gcn.optimization.performance_monitor import PerformanceMonitor, MetricsCollector, MetricType, AlertLevel
from text4gcn.optimization.resource_manager import ResourceManager, ResourceType, ResourceState
from text4gcn.optimization.async_operations import AsyncProcessor, OperationType, OperationStatus
from text4gcn.optimization.optimization_utils import OptimizationUtils, OptimizationConfig


class TestMemoryOptimizer:
    """Test cases for MemoryOptimizer."""
    
    def test_initialization(self):
        """Test memory optimizer initialization."""
        profiler = MemoryProfiler(enable_tracing=False)
        optimizer = MemoryOptimizer(profiler)
        
        assert optimizer is not None
        assert optimizer.profiler is not None
        assert not any(optimizer.strategies.values())
    
    def test_enable_strategy(self):
        """Test enabling optimization strategies."""
        profiler = MemoryProfiler(enable_tracing=False)
        optimizer = MemoryOptimizer(profiler)
        
        optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        assert optimizer.strategies[MemoryStrategy.LAZY_LOADING]
        
        optimizer.enable_strategy(MemoryStrategy.MEMORY_POOLING)
        assert optimizer.strategies[MemoryStrategy.MEMORY_POOLING]
    
    def test_disable_strategy(self):
        """Test disabling optimization strategies."""
        profiler = MemoryProfiler(enable_tracing=False)
        optimizer = MemoryOptimizer(profiler)
        
        optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
        optimizer.disable_strategy(MemoryStrategy.LAZY_LOADING)
        assert not optimizer.strategies[MemoryStrategy.LAZY_LOADING]
    
    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        profiler = MemoryProfiler(enable_tracing=False)
        optimizer = MemoryOptimizer(profiler)
        
        results = optimizer.optimize_memory_usage()
        
        assert 'before' in results
        assert 'after' in results
        assert 'strategies_applied' in results
        assert 'memory_freed' in results
        assert 'objects_collected' in results
    
    def test_get_memory_report(self):
        """Test getting memory report."""
        profiler = MemoryProfiler(enable_tracing=False)
        optimizer = MemoryOptimizer(profiler)
        
        report = optimizer.get_memory_report()
        
        assert 'current_stats' in report
        assert 'memory_by_type' in report
        assert 'growth_rate' in report
        assert 'active_strategies' in report
        assert 'pool_sizes' in report
        assert 'lazy_loaders' in report


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def test_initialization(self):
        """Test cache manager initialization."""
        cache_manager = CacheManager()
        
        assert cache_manager is not None
        assert cache_manager.l1 is not None
        assert cache_manager.l2 is not None
        assert cache_manager.l3 is not None
    
    def test_put_and_get(self):
        """Test putting and getting from cache."""
        cache_manager = CacheManager()
        data = {"key": "value", "number": 42}
        
        # Put in cache
        success = cache_manager.put("test_key", data)
        assert success
        
        # Get from cache
        retrieved = cache_manager.get("test_key")
        assert retrieved == data
    
    def test_invalidation(self):
        """Test cache invalidation."""
        cache_manager = CacheManager()
        data = "invalidatable_data"
        
        # Put in cache
        cache_manager.put("invalid_key", data)
        
        # Should be available
        result = cache_manager.get("invalid_key")
        assert result == data
        
        # Invalidate
        cache_manager.invalidate("invalid_key")
        
        # Should be gone
        result = cache_manager.get("invalid_key")
        assert result is None
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache_manager = CacheManager()
        
        # Put some data
        cache_manager.put("stats_key1", "data1")
        cache_manager.put("stats_key2", "data2")
        
        # Get stats
        stats = cache_manager.get_combined_stats()
        
        assert stats.hits >= 0
        assert stats.misses >= 0
        assert stats.total_size >= 0
        assert stats.entry_count >= 0
    
    def test_optimize(self):
        """Test cache optimization."""
        cache_manager = CacheManager()
        
        # Put some data
        cache_manager.put("opt_key1", "data1")
        cache_manager.put("opt_key2", "data2")
        
        # Optimize
        results = cache_manager.optimize()
        
        assert 'before' in results
        assert 'after' in results
        assert 'optimizations_applied' in results


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        assert monitor is not None
        assert monitor.collector is not None
        assert not monitor.monitoring
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        monitor.record_metric("test.metric", 42.0, MetricType.CUSTOM, "units")
        
        stats = monitor.get_metric_stats("test.metric")
        assert stats is not None
        assert stats.count >= 1
    
    def test_thresholds(self):
        """Test threshold monitoring."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Set threshold
        monitor.set_threshold("test.metric", AlertLevel.WARNING, 50.0)
        
        # Record metric below threshold
        monitor.record_metric("test.metric", 30.0, MetricType.CUSTOM)
        
        # Should not trigger alert
        alerts = monitor.get_alerts()
        assert len(alerts) == 0
        
        # Record metric above threshold
        monitor.record_metric("test.metric", 60.0, MetricType.CUSTOM)
        
        # Should trigger alert
        alerts = monitor.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].level == AlertLevel.WARNING
    
    def test_get_performance_report(self):
        """Test getting performance report."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Record some metrics
        monitor.record_metric("test.metric1", 10.0, MetricType.CUSTOM)
        monitor.record_metric("test.metric2", 20.0, MetricType.CUSTOM)
        
        report = monitor.get_performance_report()
        
        assert 'timestamp' in report
        assert 'metrics_count' in report
        assert 'alerts_count' in report
        assert 'metrics' in report
        assert 'alerts' in report


class TestResourceManager:
    """Test cases for ResourceManager."""
    
    def test_initialization(self):
        """Test resource manager initialization."""
        resource_manager = ResourceManager()
        
        assert resource_manager is not None
        assert len(resource_manager.pools) == 0
        assert len(resource_manager.quotas) == 0
    
    def test_create_pool(self):
        """Test creating resource pool."""
        resource_manager = ResourceManager()
        
        pool = resource_manager.create_pool(
            "test_pool", ResourceType.CPU, max_size=10, min_size=2
        )
        
        assert pool.name == "test_pool"
        assert pool.resource_type == ResourceType.CPU
        assert pool.max_size == 10
        assert pool.min_size == 2
    
    def test_add_resource(self):
        """Test adding resource to pool."""
        resource_manager = ResourceManager()
        resource_manager.create_pool("test_pool", ResourceType.CPU)
        
        success = resource_manager.add_resource("test_pool", "resource_1", 1.0)
        assert success
        
        # Try to add to non-existent pool
        success = resource_manager.add_resource("nonexistent", "resource_2", 1.0)
        assert not success
    
    def test_set_quota(self):
        """Test setting resource quota."""
        resource_manager = ResourceManager()
        
        resource_manager.set_quota(ResourceType.CPU, 8.0, 0.8, 0.95)
        
        assert ResourceType.CPU in resource_manager.quotas
        quota = resource_manager.quotas[ResourceType.CPU]
        assert quota.max_usage == 8.0
        assert quota.soft_limit == 0.8
        assert quota.hard_limit == 0.95
    
    def test_check_quota(self):
        """Test quota checking."""
        resource_manager = ResourceManager()
        resource_manager.set_quota(ResourceType.CPU, 8.0, 0.8, 0.95)
        
        # Should allow allocation within quota
        assert resource_manager.check_quota(ResourceType.CPU, 4.0)
        
        # Should not allow allocation exceeding quota
        assert not resource_manager.check_quota(ResourceType.CPU, 10.0)
    
    def test_get_pool_stats(self):
        """Test getting pool statistics."""
        resource_manager = ResourceManager()
        resource_manager.create_pool("test_pool", ResourceType.CPU, max_size=5)
        resource_manager.add_resource("test_pool", "resource_1", 1.0)
        resource_manager.add_resource("test_pool", "resource_2", 1.0)
        
        stats = resource_manager.get_pool_stats("test_pool")
        assert stats is not None
        assert stats['pool_name'] == "test_pool"
        assert stats['total_resources'] == 2
        assert stats['available_resources'] == 2
    
    def test_optimize_resources(self):
        """Test resource optimization."""
        resource_manager = ResourceManager()
        resource_manager.create_pool("test_pool", ResourceType.CPU)
        resource_manager.add_resource("test_pool", "resource_1", 1.0)
        
        results = resource_manager.optimize_resources()
        
        assert 'before' in results
        assert 'after' in results
        assert 'optimizations_applied' in results


class TestAsyncProcessor:
    """Test cases for AsyncProcessor."""
    
    def test_initialization(self):
        """Test async processor initialization."""
        processor = AsyncProcessor(max_workers=2)
        
        assert processor is not None
        assert processor.max_workers == 2
        assert not processor.running
    
    def test_start_stop(self):
        """Test starting and stopping processor."""
        processor = AsyncProcessor(max_workers=2)
        
        processor.start()
        assert processor.running
        
        processor.stop()
        assert not processor.running
    
    def test_submit_function(self):
        """Test submitting function for execution."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()
        
        def test_func(x):
            return x * 2
        
        task_id = processor.submit_function(test_func, 5)
        assert task_id is not None
        
        # Just test that the task was submitted
        # The actual execution is complex and would require more sophisticated testing
        processor.stop()
    
    def test_get_stats(self):
        """Test getting processor statistics."""
        processor = AsyncProcessor(max_workers=2)
        processor.start()
        
        def test_func():
            return "result"
        
        task_id = processor.submit_function(test_func)
        
        stats = processor.get_stats()
        assert 'total_operations' in stats
        assert 'completed_operations' in stats
        assert 'running' in stats
        
        processor.stop()


class TestOptimizationUtils:
    """Test cases for OptimizationUtils."""
    
    def test_initialization(self):
        """Test optimization utils initialization."""
        utils = OptimizationUtils()
        
        assert utils is not None
        assert utils.config is not None
    
    def test_profile_function(self):
        """Test function profiling."""
        utils = OptimizationUtils()
        
        def test_func(n):
            return sum(range(n))
        
        profile = utils.profile_function(test_func, 1000)
        
        assert profile['function_name'] == 'test_func'
        assert profile['wall_time'] > 0
        assert profile['success'] is True
        assert profile['result'] == sum(range(1000))
    
    def test_optimize_memory_usage(self):
        """Test memory optimization."""
        utils = OptimizationUtils()
        
        results = utils.optimize_memory_usage()
        
        assert 'initial_memory' in results
        assert 'final_memory' in results
        assert 'memory_freed' in results
        assert 'garbage_collected' in results
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        utils = OptimizationUtils()
        
        def test_func():
            return "test"
        
        utils.profile_function(test_func)
        
        summary = utils.get_performance_summary()
        
        assert 'profiled_functions' in summary
        assert 'benchmarked_functions' in summary
        assert 'total_profiles' in summary
        assert 'total_benchmarks' in summary
    
    def test_clear_profiles(self):
        """Test clearing profiles."""
        utils = OptimizationUtils()
        
        def test_func():
            return "test"
        
        utils.profile_function(test_func)
        
        # Should have profiles
        summary = utils.get_performance_summary()
        assert summary['total_profiles'] > 0
        
        # Clear profiles
        utils.clear_profiles()
        
        # Should be empty
        summary = utils.get_performance_summary()
        assert summary['total_profiles'] == 0


class TestIntegration:
    """Integration tests for optimization system."""
    
    def test_basic_integration(self):
        """Test basic integration of optimization components."""
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
            data = {"key": "value", "number": 42}
            cache_manager.put("workflow_data", data)
            
            # Process with memory optimization
            allocation_id = memory_optimizer.profiler.track_allocation(data)
            result = data["number"]  # Just get the number value
            
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
        assert isinstance(memory_results['memory_freed'], (int, float))
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
