"""
Memory Optimization System for Text4GCN.

This module provides comprehensive memory optimization features including:
- Memory profiling and monitoring
- Garbage collection optimization
- Memory pooling
- Lazy loading strategies
- Memory-efficient data structures
"""

import gc
import psutil
import tracemalloc
import weakref
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import defaultdict
import numpy as np
import torch as th


class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    LAZY_LOADING = "lazy_loading"
    MEMORY_POOLING = "memory_pooling"
    GARBAGE_COLLECTION = "garbage_collection"
    WEAK_REFERENCES = "weak_references"
    COMPRESSION = "compression"


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_memory: float = 0.0
    used_memory: float = 0.0
    available_memory: float = 0.0
    memory_percentage: float = 0.0
    peak_memory: float = 0.0
    gc_count: int = 0
    gc_collected: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryAllocation:
    """Memory allocation tracking."""
    size: int
    object_type: str
    timestamp: float
    stack_trace: List[str] = field(default_factory=list)
    weak_ref: Optional[weakref.ref] = None


class MemoryProfiler:
    """Memory profiler for tracking memory usage."""
    
    def __init__(self, enable_tracing: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            enable_tracing: Whether to enable memory tracing
        """
        self.enable_tracing = enable_tracing
        self.allocations: Dict[int, MemoryAllocation] = {}
        self.memory_history: List[MemoryStats] = []
        self.peak_memory = 0.0
        self._lock = threading.Lock()
        
        if enable_tracing:
            tracemalloc.start()
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            Memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        
        # Get GC stats
        gc_stats = gc.get_stats()
        total_collected = sum(stat['collected'] for stat in gc_stats)
        
        stats = MemoryStats(
            total_memory=system_memory.total,
            used_memory=memory_info.rss,
            available_memory=system_memory.available,
            memory_percentage=system_memory.percent,
            peak_memory=self.peak_memory,
            gc_count=len(gc_stats),
            gc_collected=total_collected
        )
        
        # Update peak memory
        if memory_info.rss > self.peak_memory:
            self.peak_memory = memory_info.rss
        
        with self._lock:
            self.memory_history.append(stats)
            # Keep only last 1000 entries
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]
        
        return stats
    
    def track_allocation(self, obj: Any, size: int = None) -> int:
        """
        Track memory allocation.
        
        Args:
            obj: Object to track
            size: Size of allocation (if known)
            
        Returns:
            Allocation ID
        """
        if not self.enable_tracing:
            return 0
        
        allocation_id = id(obj)
        
        if size is None:
            try:
                size = obj.__sizeof__()
            except (AttributeError, TypeError):
                size = 0
        
        # Get stack trace
        stack_trace = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            for frame in snapshot.traceback_limit(10):
                stack_trace.append(f"{frame.filename}:{frame.lineno}")
        
        # Create weak reference
        weak_ref = weakref.ref(obj, self._cleanup_allocation)
        
        allocation = MemoryAllocation(
            size=size,
            object_type=type(obj).__name__,
            timestamp=time.time(),
            stack_trace=stack_trace,
            weak_ref=weak_ref
        )
        
        with self._lock:
            self.allocations[allocation_id] = allocation
        
        return allocation_id
    
    def _cleanup_allocation(self, weak_ref):
        """Cleanup allocation when object is garbage collected."""
        allocation_id = None
        for aid, allocation in self.allocations.items():
            if allocation.weak_ref is weak_ref:
                allocation_id = aid
                break
        
        if allocation_id:
            with self._lock:
                if allocation_id in self.allocations:
                    del self.allocations[allocation_id]
    
    def get_memory_usage_by_type(self) -> Dict[str, int]:
        """
        Get memory usage by object type.
        
        Returns:
            Dictionary mapping object types to memory usage
        """
        usage_by_type = defaultdict(int)
        
        with self._lock:
            for allocation in self.allocations.values():
                usage_by_type[allocation.object_type] += allocation.size
        
        return dict(usage_by_type)
    
    def get_memory_growth_rate(self, window_size: int = 10) -> float:
        """
        Get memory growth rate.
        
        Args:
            window_size: Number of recent measurements to consider
            
        Returns:
            Memory growth rate (bytes per second)
        """
        if len(self.memory_history) < 2:
            return 0.0
        
        recent_stats = self.memory_history[-window_size:]
        if len(recent_stats) < 2:
            return 0.0
        
        time_diff = recent_stats[-1].timestamp - recent_stats[0].timestamp
        memory_diff = recent_stats[-1].used_memory - recent_stats[0].used_memory
        
        if time_diff == 0:
            return 0.0
        
        return memory_diff / time_diff
    
    def cleanup(self):
        """Cleanup profiler resources."""
        if self.enable_tracing:
            tracemalloc.stop()
        
        with self._lock:
            self.allocations.clear()
            self.memory_history.clear()


class MemoryOptimizer:
    """Memory optimizer for Text4GCN."""
    
    def __init__(self, profiler: Optional[MemoryProfiler] = None):
        """
        Initialize memory optimizer.
        
        Args:
            profiler: Memory profiler instance
        """
        self.profiler = profiler or MemoryProfiler()
        self.strategies: Dict[MemoryStrategy, bool] = {
            strategy: False for strategy in MemoryStrategy
        }
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)
        self.lazy_loaders: Dict[str, Callable] = {}
        self.compression_enabled = False
        self._lock = threading.Lock()
    
    def enable_strategy(self, strategy: MemoryStrategy) -> 'MemoryOptimizer':
        """
        Enable a memory optimization strategy.
        
        Args:
            strategy: Strategy to enable
            
        Returns:
            Self for method chaining
        """
        self.strategies[strategy] = True
        
        if strategy == MemoryStrategy.GARBAGE_COLLECTION:
            self._optimize_garbage_collection()
        elif strategy == MemoryStrategy.MEMORY_POOLING:
            self._setup_memory_pools()
        elif strategy == MemoryStrategy.COMPRESSION:
            self.compression_enabled = True
        
        return self
    
    def disable_strategy(self, strategy: MemoryStrategy) -> 'MemoryOptimizer':
        """
        Disable a memory optimization strategy.
        
        Args:
            strategy: Strategy to disable
            
        Returns:
            Self for method chaining
        """
        self.strategies[strategy] = False
        
        if strategy == MemoryStrategy.MEMORY_POOLING:
            self._cleanup_memory_pools()
        elif strategy == MemoryStrategy.COMPRESSION:
            self.compression_enabled = False
        
        return self
    
    def _optimize_garbage_collection(self):
        """Optimize garbage collection settings."""
        # Set GC thresholds
        gc.set_threshold(700, 10, 10)
        
        # Enable debug flags
        gc.set_debug(gc.DEBUG_LEAK)
    
    def _setup_memory_pools(self):
        """Setup memory pools for common objects."""
        # Tensor pools
        self.memory_pools['tensors'] = []
        self.memory_pools['arrays'] = []
        self.memory_pools['lists'] = []
    
    def _cleanup_memory_pools(self):
        """Cleanup memory pools."""
        with self._lock:
            for pool in self.memory_pools.values():
                pool.clear()
    
    def get_pooled_object(self, pool_name: str, factory: Callable) -> Any:
        """
        Get object from pool or create new one.
        
        Args:
            pool_name: Name of the pool
            factory: Factory function to create new object
            
        Returns:
            Object from pool or newly created
        """
        if not self.strategies[MemoryStrategy.MEMORY_POOLING]:
            return factory()
        
        with self._lock:
            if self.memory_pools[pool_name]:
                return self.memory_pools[pool_name].pop()
            else:
                return factory()
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """
        Return object to pool.
        
        Args:
            pool_name: Name of the pool
            obj: Object to return
        """
        if not self.strategies[MemoryStrategy.MEMORY_POOLING]:
            return
        
        with self._lock:
            if len(self.memory_pools[pool_name]) < 100:  # Limit pool size
                self.memory_pools[pool_name].append(obj)
    
    def register_lazy_loader(self, name: str, loader: Callable):
        """
        Register a lazy loader.
        
        Args:
            name: Name of the loader
            loader: Loader function
        """
        self.lazy_loaders[name] = loader
    
    def lazy_load(self, name: str, *args, **kwargs) -> Any:
        """
        Lazy load an object.
        
        Args:
            name: Name of the loader
            *args: Arguments for loader
            **kwargs: Keyword arguments for loader
            
        Returns:
            Loaded object
        """
        if not self.strategies[MemoryStrategy.LAZY_LOADING]:
            return self.lazy_loaders[name](*args, **kwargs)
        
        # Create lazy proxy
        class LazyProxy:
            def __init__(self, loader, args, kwargs):
                self._loader = loader
                self._args = args
                self._kwargs = kwargs
                self._obj = None
                self._loaded = False
            
            def __getattr__(self, name):
                if not self._loaded:
                    self._obj = self._loader(*self._args, **self._kwargs)
                    self._loaded = True
                return getattr(self._obj, name)
            
            def __call__(self, *args, **kwargs):
                if not self._loaded:
                    self._obj = self._loader(*self._args, **self._kwargs)
                    self._loaded = True
                return self._obj(*args, **kwargs)
        
        return LazyProxy(self.lazy_loaders[name], args, kwargs)
    
    def compress_tensor(self, tensor: th.Tensor) -> th.Tensor:
        """
        Compress tensor if compression is enabled.
        
        Args:
            tensor: Tensor to compress
            
        Returns:
            Compressed tensor
        """
        if not self.compression_enabled:
            return tensor
        
        # Use half precision if possible
        if tensor.dtype == th.float32:
            return tensor.half()
        
        return tensor
    
    def decompress_tensor(self, tensor: th.Tensor, target_dtype: th.dtype = th.float32) -> th.Tensor:
        """
        Decompress tensor.
        
        Args:
            tensor: Tensor to decompress
            target_dtype: Target dtype
            
        Returns:
            Decompressed tensor
        """
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        
        return tensor
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Returns:
            Optimization results
        """
        results = {
            'before': self.profiler.get_memory_stats(),
            'strategies_applied': [],
            'memory_freed': 0,
            'objects_collected': 0
        }
        
        # Apply garbage collection
        if self.strategies[MemoryStrategy.GARBAGE_COLLECTION]:
            collected = gc.collect()
            results['objects_collected'] = collected
            results['strategies_applied'].append('garbage_collection')
        
        # Clear unused pools
        if self.strategies[MemoryStrategy.MEMORY_POOLING]:
            with self._lock:
                for pool in self.memory_pools.values():
                    pool.clear()
            results['strategies_applied'].append('memory_pooling')
        
        # Get after stats
        results['after'] = self.profiler.get_memory_stats()
        results['memory_freed'] = results['before'].used_memory - results['after'].used_memory
        
        return results
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory report.
        
        Returns:
            Memory report
        """
        stats = self.profiler.get_memory_stats()
        
        return {
            'current_stats': stats,
            'memory_by_type': self.profiler.get_memory_usage_by_type(),
            'growth_rate': self.profiler.get_memory_growth_rate(),
            'active_strategies': [strategy.value for strategy, enabled in self.strategies.items() if enabled],
            'pool_sizes': {name: len(pool) for name, pool in self.memory_pools.items()},
            'lazy_loaders': list(self.lazy_loaders.keys())
        }
    
    def cleanup(self):
        """Cleanup optimizer resources."""
        self._cleanup_memory_pools()
        self.lazy_loaders.clear()
        self.profiler.cleanup()


