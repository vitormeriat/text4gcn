"""
Optimization Utilities for Text4GCN.

This module provides utility functions for optimization including:
- Performance profiling utilities
- Memory optimization helpers
- Cache optimization strategies
- Resource optimization tools
- Benchmarking utilities
"""

import time
import functools
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import gc
import numpy as np
import torch as th


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    enable_memory_optimization: bool = True
    enable_cache_optimization: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_management: bool = True
    memory_threshold: float = 0.8  # 80% memory usage threshold
    cache_size_limit: int = 1000
    performance_monitoring_interval: float = 1.0
    resource_cleanup_interval: float = 60.0


class OptimizationUtils:
    """Utility functions for optimization."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimization utils.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._benchmarks: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function's performance.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profile results
        """
        # Get initial system state
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu = psutil.cpu_percent(interval=None)
        
        # Measure execution time
        start_time = time.time()
        start_perf_counter = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_perf_counter = time.perf_counter()
        
        # Get final system state
        final_memory = psutil.Process().memory_info().rss
        final_cpu = psutil.cpu_percent(interval=None)
        
        # Calculate metrics
        wall_time = end_time - start_time
        cpu_time = end_perf_counter - start_perf_counter
        memory_used = final_memory - initial_memory
        cpu_usage = (initial_cpu + final_cpu) / 2
        
        profile = {
            'function_name': func.__name__,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'memory_used': memory_used,
            'cpu_usage': cpu_usage,
            'success': success,
            'error': error,
            'result': result,
            'timestamp': start_time
        }
        
        # Store profile
        with self._lock:
            if func.__name__ not in self._profiles:
                self._profiles[func.__name__] = []
            self._profiles[func.__name__].append(profile)
        
        return profile
    
    def benchmark_function(self, func: Callable, iterations: int = 10, 
                          *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function over multiple iterations.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        times = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(iterations):
            # Get initial state
            initial_memory = psutil.Process().memory_info().rss
            initial_cpu = psutil.cpu_percent(interval=None)
            
            # Measure execution time
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                print(f"Error in iteration {i}: {e}")
            
            end_time = time.perf_counter()
            
            # Get final state
            final_memory = psutil.Process().memory_info().rss
            final_cpu = psutil.cpu_percent(interval=None)
            
            # Record metrics
            times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)
            cpu_usage.append((initial_cpu + final_cpu) / 2)
        
        # Calculate statistics
        times_array = np.array(times)
        memory_array = np.array(memory_usage)
        cpu_array = np.array(cpu_usage)
        
        benchmark = {
            'function_name': func.__name__,
            'iterations': iterations,
            'times': {
                'mean': np.mean(times_array),
                'std': np.std(times_array),
                'min': np.min(times_array),
                'max': np.max(times_array),
                'median': np.median(times_array),
                'p95': np.percentile(times_array, 95),
                'p99': np.percentile(times_array, 99)
            },
            'memory': {
                'mean': np.mean(memory_array),
                'std': np.std(memory_array),
                'min': np.min(memory_array),
                'max': np.max(memory_array),
                'median': np.median(memory_array)
            },
            'cpu': {
                'mean': np.mean(cpu_array),
                'std': np.std(cpu_array),
                'min': np.min(cpu_array),
                'max': np.max(cpu_array),
                'median': np.median(cpu_array)
            },
            'success_rate': sum(1 for t in times if t > 0) / len(times)
        }
        
        # Store benchmark
        with self._lock:
            if func.__name__ not in self._benchmarks:
                self._benchmarks[func.__name__] = []
            self._benchmarks[func.__name__].append(benchmark)
        
        return benchmark
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Returns:
            Optimization results
        """
        initial_memory = psutil.Process().memory_info().rss
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear unused variables
        self._clear_unused_variables()
        
        # Optimize PyTorch memory
        if th.cuda.is_available():
            th.cuda.empty_cache()
        
        final_memory = psutil.Process().memory_info().rss
        memory_freed = initial_memory - final_memory
        
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_freed': memory_freed,
            'garbage_collected': collected,
            'optimization_ratio': memory_freed / initial_memory if initial_memory > 0 else 0
        }
    
    def _clear_unused_variables(self):
        """Clear unused variables from memory."""
        # This is a simplified version - in practice, you'd need more sophisticated
        # memory management
        pass
    
    def optimize_cache_performance(self, cache_manager) -> Dict[str, Any]:
        """
        Optimize cache performance.
        
        Args:
            cache_manager: Cache manager instance
            
        Returns:
            Optimization results
        """
        before_stats = cache_manager.get_combined_stats()
        
        # Optimize cache
        optimization_results = cache_manager.optimize()
        
        after_stats = cache_manager.get_combined_stats()
        
        return {
            'before': before_stats,
            'after': after_stats,
            'optimization_results': optimization_results,
            'hit_ratio_improvement': after_stats.hit_ratio - before_stats.hit_ratio,
            'memory_usage_reduction': before_stats.total_size - after_stats.total_size
        }
    
    def optimize_resource_usage(self, resource_manager) -> Dict[str, Any]:
        """
        Optimize resource usage.
        
        Args:
            resource_manager: Resource manager instance
            
        Returns:
            Optimization results
        """
        return resource_manager.optimize_resources()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Performance summary
        """
        with self._lock:
            summary = {
                'profiled_functions': len(self._profiles),
                'benchmarked_functions': len(self._benchmarks),
                'total_profiles': sum(len(profiles) for profiles in self._profiles.values()),
                'total_benchmarks': sum(len(benchmarks) for benchmarks in self._benchmarks.values())
            }
            
            # Add function-specific summaries
            for func_name, profiles in self._profiles.items():
                if profiles:
                    times = [p['wall_time'] for p in profiles]
                    summary[f'{func_name}_avg_time'] = np.mean(times)
                    summary[f'{func_name}_max_time'] = np.max(times)
                    summary[f'{func_name}_min_time'] = np.min(times)
            
            return summary
    
    def clear_profiles(self):
        """Clear all profiles."""
        with self._lock:
            self._profiles.clear()
            self._benchmarks.clear()
    
    def export_profiles(self, filename: str):
        """
        Export profiles to file.
        
        Args:
            filename: Output filename
        """
        import json
        
        with self._lock:
            data = {
                'profiles': self._profiles,
                'benchmarks': self._benchmarks,
                'export_timestamp': time.time()
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_profiles(self, filename: str):
        """
        Load profiles from file.
        
        Args:
            filename: Input filename
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        with self._lock:
            self._profiles = data.get('profiles', {})
            self._benchmarks = data.get('benchmarks', {})


def profile_method(func: Callable) -> Callable:
    """
    Decorator to profile a method.
    
    Args:
        func: Method to profile
        
    Returns:
        Decorated method
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_optimization_utils'):
            return self._optimization_utils.profile_function(func, self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    
    return wrapper


def benchmark_method(iterations: int = 10) -> Callable:
    """
    Decorator to benchmark a method.
    
    Args:
        iterations: Number of benchmark iterations
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, '_optimization_utils'):
                return self._optimization_utils.benchmark_function(
                    func, iterations, self, *args, **kwargs
                )
            else:
                return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def memory_monitor(threshold: float = 0.8):
    """
    Context manager for memory monitoring.
    
    Args:
        threshold: Memory usage threshold
    """
    initial_memory = psutil.Process().memory_info().rss
    initial_system_memory = psutil.virtual_memory().percent / 100
    
    try:
        yield
    finally:
        final_memory = psutil.Process().memory_info().rss
        final_system_memory = psutil.virtual_memory().percent / 100
        
        memory_used = final_memory - initial_memory
        system_memory_usage = final_system_memory
        
        if system_memory_usage > threshold:
            print(f"Warning: High system memory usage: {system_memory_usage:.2%}")
        
        if memory_used > 0:
            print(f"Memory used in context: {memory_used / 1024 / 1024:.2f} MB")


@contextmanager
def performance_monitor(operation_name: str):
    """
    Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation
    """
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"Operation '{operation_name}': {execution_time:.4f}s, {memory_used / 1024 / 1024:.2f} MB")


def optimize_tensor_operations(tensor: th.Tensor) -> th.Tensor:
    """
    Optimize tensor operations.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Optimized tensor
    """
    # Use half precision if possible
    if tensor.dtype == th.float32 and tensor.numel() > 1000:
        return tensor.half()
    
    # Use contiguous memory layout
    if not tensor.is_contiguous():
        return tensor.contiguous()
    
    return tensor


def optimize_numpy_operations(array: np.ndarray) -> np.ndarray:
    """
    Optimize numpy operations.
    
    Args:
        array: Input array
        
    Returns:
        Optimized array
    """
    # Use appropriate dtype
    if array.dtype == np.float64 and array.size > 1000:
        return array.astype(np.float32)
    
    # Use C-contiguous layout
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    
    return array


