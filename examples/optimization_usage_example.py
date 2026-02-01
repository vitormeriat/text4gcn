#!/usr/bin/env python3
"""
Optimization Usage Example for Text4GCN.

This example demonstrates how to use the optimization system
for memory optimization, caching, performance monitoring, and resource management.
"""

import sys
import time
import numpy as np
import torch as th
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text4gcn.optimization import (
    MemoryOptimizer, MemoryProfiler, MemoryStrategy,
    CacheManager, CacheStrategy, CacheLevel,
    PerformanceMonitor, MetricsCollector, MetricType, AlertLevel,
    ResourceManager, ResourceType, ResourceState,
    AsyncProcessor, AsyncTrainer, OperationType, OperationStatus,
    OptimizationUtils, OptimizationConfig
)


def example_memory_optimization():
    """Example of memory optimization."""
    print("🧠 Memory Optimization Example")
    print("=" * 50)
    
    # Create memory optimizer
    profiler = MemoryProfiler(enable_tracing=True)
    optimizer = MemoryOptimizer(profiler)
    
    # Enable optimization strategies
    optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
    optimizer.enable_strategy(MemoryStrategy.MEMORY_POOLING)
    optimizer.enable_strategy(MemoryStrategy.GARBAGE_COLLECTION)
    
    # Create some data
    data = np.random.rand(1000, 1000)
    tensor = th.randn(1000, 1000)
    
    # Track allocations
    data_id = optimizer.profiler.track_allocation(data)
    tensor_id = optimizer.profiler.track_allocation(tensor)
    
    print(f"✅ Data allocated with ID: {data_id}")
    print(f"✅ Tensor allocated with ID: {tensor_id}")
    
    # Get memory stats
    stats = profiler.get_memory_stats()
    print(f"📊 Memory usage: {stats.used_memory / 1024 / 1024:.2f} MB")
    print(f"📊 Memory percentage: {stats.memory_percentage:.2f}%")
    
    # Optimize memory
    results = optimizer.optimize_memory_usage()
    print(f"🧹 Memory freed: {results['memory_freed'] / 1024 / 1024:.2f} MB")
    print(f"🧹 Objects collected: {results['objects_collected']}")
    
    # Get memory report
    report = optimizer.get_memory_report()
    print(f"📈 Memory by type: {report['memory_by_type']}")
    print(f"📈 Growth rate: {report['growth_rate']:.2f} bytes/sec")


def example_caching_system():
    """Example of caching system."""
    print("\n💾 Caching System Example")
    print("=" * 50)
    
    # Create cache manager
    cache_manager = CacheManager()
    
    # Enable compression
    cache_manager.enable_compression()
    
    # Cache some data
    data1 = np.random.rand(100, 100)
    data2 = {"key": "value", "number": 42}
    data3 = th.randn(50, 50)
    
    # Put data in cache
    cache_manager.put("data1", data1, ttl=60.0)
    cache_manager.put("data2", data2, ttl=120.0)
    cache_manager.put("data3", data3, ttl=30.0)
    
    print("✅ Data cached successfully")
    
    # Retrieve data
    retrieved1 = cache_manager.get("data1")
    retrieved2 = cache_manager.get("data2")
    retrieved3 = cache_manager.get("data3")
    
    print(f"✅ Retrieved data1: {retrieved1 is not None}")
    print(f"✅ Retrieved data2: {retrieved2 is not None}")
    print(f"✅ Retrieved data3: {retrieved3 is not None}")
    
    # Get cache statistics
    stats = cache_manager.get_combined_stats()
    print(f"📊 Cache hits: {stats.hits}")
    print(f"📊 Cache misses: {stats.misses}")
    print(f"📊 Hit ratio: {stats.hit_ratio:.2f}")
    print(f"📊 Total size: {stats.total_size / 1024 / 1024:.2f} MB")
    
    # Optimize cache
    optimization_results = cache_manager.optimize()
    print(f"🔧 Optimizations applied: {optimization_results['optimizations_applied']}")


def example_performance_monitoring():
    """Example of performance monitoring."""
    print("\n📊 Performance Monitoring Example")
    print("=" * 50)
    
    # Create performance monitor
    collector = MetricsCollector()
    monitor = PerformanceMonitor(collector)
    
    # Set up thresholds
    monitor.set_threshold("system.cpu_usage", AlertLevel.WARNING, 80.0)
    monitor.set_threshold("system.memory_usage", AlertLevel.ERROR, 90.0)
    
    # Add alert callback
    def alert_callback(alert):
        print(f"🚨 Alert: {alert.message}")
    
    monitor.add_alert_callback("system.cpu_usage", alert_callback)
    monitor.add_alert_callback("system.memory_usage", alert_callback)
    
    # Start monitoring
    monitor.start_monitoring(interval=1.0)
    
    # Simulate some work
    print("🔄 Simulating work...")
    time.sleep(3)
    
    # Record some custom metrics
    monitor.record_metric("custom.operation_time", 1.5, MetricType.EXECUTION_TIME, "seconds")
    monitor.record_metric("custom.accuracy", 0.95, MetricType.CUSTOM, "%")
    
    # Get performance report
    report = monitor.get_performance_report()
    print(f"📈 Metrics count: {report['metrics_count']}")
    print(f"📈 Alerts count: {report['alerts_count']}")
    print(f"📈 Unresolved alerts: {report['unresolved_alerts']}")
    
    # Stop monitoring
    monitor.stop_monitoring()


def example_resource_management():
    """Example of resource management."""
    print("\n🔧 Resource Management Example")
    print("=" * 50)
    
    # Create resource manager
    resource_manager = ResourceManager()
    
    # Create resource pools
    cpu_pool = resource_manager.create_pool("cpu_pool", ResourceType.CPU, max_size=10)
    memory_pool = resource_manager.create_pool("memory_pool", ResourceType.MEMORY, max_size=5)
    
    # Add resources to pools
    resource_manager.add_resource("cpu_pool", "cpu_1", 1.0)
    resource_manager.add_resource("cpu_pool", "cpu_2", 1.0)
    resource_manager.add_resource("memory_pool", "mem_1", 1024 * 1024 * 1024)  # 1GB
    
    print("✅ Resource pools created")
    
    # Set quotas
    resource_manager.set_quota(ResourceType.CPU, 8.0, soft_limit=0.8, hard_limit=0.95)
    resource_manager.set_quota(ResourceType.MEMORY, 16 * 1024 * 1024 * 1024, 0.8, 0.95)  # 16GB
    
    # Start monitoring
    resource_manager.start_monitoring(interval=2.0)
    
    # Allocate resources
    with resource_manager.resource_context("cpu_pool", 0.5) as cpu_resource:
        print(f"✅ Allocated CPU resource: {cpu_resource.id}")
        time.sleep(1)
    
    with resource_manager.resource_context("memory_pool", 512 * 1024 * 1024) as mem_resource:
        print(f"✅ Allocated memory resource: {mem_resource.id}")
        time.sleep(1)
    
    # Get pool statistics
    cpu_stats = resource_manager.get_pool_stats("cpu_pool")
    memory_stats = resource_manager.get_pool_stats("memory_pool")
    
    print(f"📊 CPU pool utilization: {cpu_stats['utilization']:.2f}%")
    print(f"📊 Memory pool utilization: {memory_stats['utilization']:.2f}%")
    
    # Get quota statistics
    quota_stats = resource_manager.get_quota_stats()
    print(f"📊 CPU quota usage: {quota_stats['cpu']['utilization']:.2f}%")
    print(f"📊 Memory quota usage: {quota_stats['memory']['utilization']:.2f}%")
    
    # Stop monitoring
    resource_manager.stop_monitoring()


def example_async_operations():
    """Example of async operations."""
    print("\n⚡ Async Operations Example")
    print("=" * 50)
    
    # Create async processor
    processor = AsyncProcessor(max_workers=4)
    processor.start()
    
    # Create async trainer
    trainer = AsyncTrainer(processor)
    
    # Define some work functions
    def heavy_computation(n):
        """Simulate heavy computation."""
        time.sleep(2)
        return n * n
    
    def data_processing(data):
        """Simulate data processing."""
        time.sleep(1)
        return len(data)
    
    # Submit tasks
    task1_id = processor.submit_function(heavy_computation, 10, operation_type=OperationType.COMPUTATION)
    task2_id = processor.submit_function(data_processing, [1, 2, 3, 4, 5], operation_type=OperationType.DATA_PROCESSING)
    
    print(f"✅ Task 1 submitted: {task1_id}")
    print(f"✅ Task 2 submitted: {task2_id}")
    
    # Wait for results
    result1 = processor.wait_for_operation(task1_id, timeout=10.0)
    result2 = processor.wait_for_operation(task2_id, timeout=10.0)
    
    print(f"✅ Task 1 result: {result1}")
    print(f"✅ Task 2 result: {result2}")
    
    # Get processor statistics
    stats = processor.get_stats()
    print(f"📊 Total operations: {stats['total_operations']}")
    print(f"📊 Completed operations: {stats['completed_operations']}")
    print(f"📊 Failed operations: {stats['failed_operations']}")
    
    # Stop processor
    processor.stop()


def example_optimization_utils():
    """Example of optimization utilities."""
    print("\n🛠️ Optimization Utils Example")
    print("=" * 50)
    
    # Create optimization utils
    utils = OptimizationUtils()
    
    # Define a function to profile
    def sample_function(n):
        """Sample function for profiling."""
        data = np.random.rand(n, n)
        result = np.sum(data)
        return result
    
    # Profile the function
    profile = utils.profile_function(sample_function, 1000)
    print(f"✅ Function profiled: {profile['function_name']}")
    print(f"📊 Execution time: {profile['wall_time']:.4f}s")
    print(f"📊 Memory used: {profile['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"📊 CPU usage: {profile['cpu_usage']:.2f}%")
    
    # Benchmark the function
    benchmark = utils.benchmark_function(sample_function, iterations=5, n=500)
    print(f"✅ Function benchmarked: {benchmark['function_name']}")
    print(f"📊 Mean time: {benchmark['times']['mean']:.4f}s")
    print(f"📊 Std time: {benchmark['times']['std']:.4f}s")
    print(f"📊 Success rate: {benchmark['success_rate']:.2f}")
    
    # Optimize memory usage
    memory_results = utils.optimize_memory_usage()
    print(f"🧹 Memory freed: {memory_results['memory_freed'] / 1024 / 1024:.2f} MB")
    print(f"🧹 Objects collected: {memory_results['garbage_collected']}")
    
    # Get performance summary
    summary = utils.get_performance_summary()
    print(f"📈 Profiled functions: {summary['profiled_functions']}")
    print(f"📈 Benchmarked functions: {summary['benchmarked_functions']}")


def example_integration():
    """Example of integrated optimization system."""
    print("\n🔗 Integrated Optimization Example")
    print("=" * 50)
    
    # Create all optimization components
    profiler = MemoryProfiler()
    memory_optimizer = MemoryOptimizer(profiler)
    cache_manager = CacheManager()
    performance_monitor = PerformanceMonitor()
    resource_manager = ResourceManager()
    async_processor = AsyncProcessor()
    utils = OptimizationUtils()
    
    # Configure optimization strategies
    memory_optimizer.enable_strategy(MemoryStrategy.LAZY_LOADING)
    memory_optimizer.enable_strategy(MemoryStrategy.MEMORY_POOLING)
    cache_manager.enable_compression()
    
    # Set up resource pools
    resource_manager.create_pool("computation_pool", ResourceType.CPU, max_size=4)
    resource_manager.set_quota(ResourceType.CPU, 8.0)
    
    # Start monitoring
    performance_monitor.start_monitoring()
    resource_manager.start_monitoring()
    async_processor.start()
    
    print("✅ All optimization systems initialized")
    
    # Simulate integrated workflow
    def integrated_workflow():
        """Simulate integrated optimization workflow."""
        # Cache some data
        data = np.random.rand(1000, 1000)
        cache_manager.put("workflow_data", data)
        
        # Process data with memory optimization
        with memory_optimizer.profiler.track_allocation(data):
            result = np.sum(data)
        
        # Monitor performance
        performance_monitor.record_metric("workflow.processing_time", 0.5, MetricType.EXECUTION_TIME)
        
        return result
    
    # Execute workflow
    result = integrated_workflow()
    print(f"✅ Workflow completed with result: {result}")
    
    # Get comprehensive optimization report
    memory_report = memory_optimizer.get_memory_report()
    cache_stats = cache_manager.get_combined_stats()
    performance_report = performance_monitor.get_performance_report()
    resource_stats = resource_manager.get_quota_stats()
    
    print(f"📊 Memory usage: {memory_report['current_stats'].used_memory / 1024 / 1024:.2f} MB")
    print(f"📊 Cache hit ratio: {cache_stats.hit_ratio:.2f}")
    print(f"📊 Performance metrics: {performance_report['metrics_count']}")
    print(f"📊 Resource utilization: {resource_stats}")
    
    # Cleanup
    performance_monitor.stop_monitoring()
    resource_manager.stop_monitoring()
    async_processor.stop()
    
    print("✅ All systems cleaned up")


def main():
    """Run all optimization examples."""
    print("🚀 Text4GCN Optimization Examples")
    print("=" * 60)
    
    try:
        example_memory_optimization()
        example_caching_system()
        example_performance_monitoring()
        example_resource_management()
        example_async_operations()
        example_optimization_utils()
        example_integration()
        
        print("\n🎉 All optimization examples completed successfully!")
        print("✅ Optimization system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
