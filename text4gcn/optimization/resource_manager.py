"""
Resource Management System for Text4GCN.

This module provides comprehensive resource management including:
- Resource pooling and allocation
- Resource lifecycle management
- Resource monitoring and cleanup
- Resource limits and quotas
- Resource optimization strategies
"""

import time
import threading
import weakref
import gc
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import torch as th
from contextlib import contextmanager


class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    THREAD = "thread"
    CONNECTION = "connection"
    CUSTOM = "custom"


class ResourceState(Enum):
    """Resource states."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    BUSY = "busy"
    ERROR = "error"
    CLEANUP = "cleanup"


@dataclass
class Resource:
    """Resource definition."""
    id: str
    resource_type: ResourceType
    capacity: float
    current_usage: float = 0.0
    state: ResourceState = ResourceState.AVAILABLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_callback: Optional[Callable] = None


@dataclass
class ResourceQuota:
    """Resource quota definition."""
    resource_type: ResourceType
    max_usage: float
    current_usage: float = 0.0
    soft_limit: float = 0.8  # 80% of max_usage
    hard_limit: float = 0.95  # 95% of max_usage
    exceeded: bool = False


@dataclass
class ResourcePool:
    """Resource pool definition."""
    name: str
    resource_type: ResourceType
    resources: List[Resource] = field(default_factory=list)
    max_size: int = 100
    min_size: int = 0
    auto_cleanup: bool = True
    cleanup_interval: float = 60.0  # seconds
    last_cleanup: float = field(default_factory=time.time)


class ResourceManager:
    """Resource manager for Text4GCN."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.pools: Dict[str, ResourcePool] = {}
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        self.allocated_resources: Dict[str, Resource] = {}
        self.cleanup_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._monitoring = False
    
    def create_pool(self, name: str, resource_type: ResourceType, 
                   max_size: int = 100, min_size: int = 0,
                   auto_cleanup: bool = True) -> ResourcePool:
        """
        Create a resource pool.
        
        Args:
            name: Pool name
            resource_type: Type of resources in pool
            max_size: Maximum pool size
            min_size: Minimum pool size
            auto_cleanup: Whether to enable auto cleanup
            
        Returns:
            Created resource pool
        """
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            max_size=max_size,
            min_size=min_size,
            auto_cleanup=auto_cleanup
        )
        
        with self._lock:
            self.pools[name] = pool
        
        return pool
    
    def add_resource(self, pool_name: str, resource_id: str, capacity: float,
                   metadata: Dict[str, Any] = None,
                   cleanup_callback: Optional[Callable] = None) -> bool:
        """
        Add resource to pool.
        
        Args:
            pool_name: Name of the pool
            resource_id: Unique resource ID
            capacity: Resource capacity
            metadata: Resource metadata
            cleanup_callback: Cleanup callback function
            
        Returns:
            True if successfully added
        """
        if pool_name not in self.pools:
            return False
        
        pool = self.pools[pool_name]
        
        if len(pool.resources) >= pool.max_size:
            return False
        
        resource = Resource(
            id=resource_id,
            resource_type=pool.resource_type,
            capacity=capacity,
            metadata=metadata or {},
            cleanup_callback=cleanup_callback
        )
        
        with self._lock:
            pool.resources.append(resource)
        
        return True
    
    def allocate_resource(self, pool_name: str, required_capacity: float = 1.0,
                         timeout: float = 30.0) -> Optional[Resource]:
        """
        Allocate resource from pool.
        
        Args:
            pool_name: Name of the pool
            required_capacity: Required resource capacity
            timeout: Allocation timeout in seconds
            
        Returns:
            Allocated resource or None
        """
        if pool_name not in self.pools:
            return None
        
        pool = self.pools[pool_name]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                # Find available resource with sufficient capacity
                for resource in pool.resources:
                    if (resource.state == ResourceState.AVAILABLE and 
                        resource.capacity >= required_capacity):
                        
                        # Allocate resource
                        resource.state = ResourceState.ALLOCATED
                        resource.current_usage = required_capacity
                        resource.last_used = time.time()
                        
                        self.allocated_resources[resource.id] = resource
                        
                        return resource
            
            # Wait before retrying
            time.sleep(0.1)
        
        return None
    
    def release_resource(self, resource_id: str) -> bool:
        """
        Release allocated resource.
        
        Args:
            resource_id: ID of the resource to release
            
        Returns:
            True if successfully released
        """
        with self._lock:
            if resource_id not in self.allocated_resources:
                return False
            
            resource = self.allocated_resources[resource_id]
            resource.state = ResourceState.AVAILABLE
            resource.current_usage = 0.0
            
            del self.allocated_resources[resource_id]
        
        return True
    
    def set_quota(self, resource_type: ResourceType, max_usage: float,
                 soft_limit: float = 0.8, hard_limit: float = 0.95):
        """
        Set resource quota.
        
        Args:
            resource_type: Type of resource
            max_usage: Maximum usage limit
            soft_limit: Soft limit percentage
            hard_limit: Hard limit percentage
        """
        quota = ResourceQuota(
            resource_type=resource_type,
            max_usage=max_usage,
            soft_limit=soft_limit,
            hard_limit=hard_limit
        )
        
        with self._lock:
            self.quotas[resource_type] = quota
    
    def check_quota(self, resource_type: ResourceType, requested_usage: float) -> bool:
        """
        Check if resource quota allows allocation.
        
        Args:
            resource_type: Type of resource
            requested_usage: Requested usage amount
            
        Returns:
            True if quota allows allocation
        """
        if resource_type not in self.quotas:
            return True
        
        quota = self.quotas[resource_type]
        new_usage = quota.current_usage + requested_usage
        
        return new_usage <= quota.max_usage * quota.hard_limit
    
    def update_quota_usage(self, resource_type: ResourceType, usage: float):
        """
        Update quota usage.
        
        Args:
            resource_type: Type of resource
            usage: Usage amount
        """
        if resource_type not in self.quotas:
            return
        
        with self._lock:
            quota = self.quotas[resource_type]
            quota.current_usage += usage
            
            # Check if quota is exceeded
            if quota.current_usage > quota.max_usage * quota.soft_limit:
                quota.exceeded = True
                self._trigger_quota_warning(resource_type, quota)
    
    def _trigger_quota_warning(self, resource_type: ResourceType, quota: ResourceQuota):
        """Trigger quota warning."""
        for callback in self.cleanup_callbacks.get(f"quota_{resource_type.value}", []):
            try:
                callback(resource_type, quota)
            except Exception as e:
                print(f"Error in quota callback: {e}")
    
    def add_quota_callback(self, resource_type: ResourceType, callback: Callable):
        """
        Add quota warning callback.
        
        Args:
            resource_type: Type of resource
            callback: Callback function
        """
        self.cleanup_callbacks[f"quota_{resource_type.value}"].append(callback)
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._cleanup_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._cleanup_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._cleanup_thread:
            self._cleanup_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                self._update_system_quotas()
                self._cleanup_unused_resources()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
    
    def _update_system_quotas(self):
        """Update system resource quotas."""
        # Update CPU quota
        if ResourceType.CPU in self.quotas:
            cpu_usage = psutil.cpu_percent(interval=None)
            self.quotas[ResourceType.CPU].current_usage = cpu_usage
        
        # Update memory quota
        if ResourceType.MEMORY in self.quotas:
            memory = psutil.virtual_memory()
            self.quotas[ResourceType.MEMORY].current_usage = memory.percent
        
        # Update GPU quota
        if ResourceType.GPU in self.quotas and th.cuda.is_available():
            gpu_memory = th.cuda.memory_allocated()
            gpu_memory_percent = (gpu_memory / th.cuda.get_device_properties(0).total_memory) * 100
            self.quotas[ResourceType.GPU].current_usage = gpu_memory_percent
    
    def _cleanup_unused_resources(self):
        """Cleanup unused resources."""
        current_time = time.time()
        
        with self._lock:
            for pool in self.pools.values():
                if not pool.auto_cleanup:
                    continue
                
                if current_time - pool.last_cleanup < pool.cleanup_interval:
                    continue
                
                # Cleanup resources that haven't been used for a long time
                resources_to_remove = []
                for resource in pool.resources:
                    if (resource.state == ResourceState.AVAILABLE and 
                        current_time - resource.last_used > pool.cleanup_interval):
                        
                        if resource.cleanup_callback:
                            try:
                                resource.cleanup_callback(resource)
                            except Exception as e:
                                print(f"Error in resource cleanup: {e}")
                        
                        resources_to_remove.append(resource)
                
                # Remove resources
                for resource in resources_to_remove:
                    pool.resources.remove(resource)
                
                pool.last_cleanup = current_time
    
    def get_pool_stats(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get pool statistics.
        
        Args:
            pool_name: Name of the pool
            
        Returns:
            Pool statistics or None
        """
        if pool_name not in self.pools:
            return None
        
        pool = self.pools[pool_name]
        
        with self._lock:
            total_resources = len(pool.resources)
            available_resources = len([r for r in pool.resources if r.state == ResourceState.AVAILABLE])
            allocated_resources = len([r for r in pool.resources if r.state == ResourceState.ALLOCATED])
            
            return {
                'pool_name': pool_name,
                'resource_type': pool.resource_type.value,
                'total_resources': total_resources,
                'available_resources': available_resources,
                'allocated_resources': allocated_resources,
                'utilization': (allocated_resources / total_resources * 100) if total_resources > 0 else 0,
                'max_size': pool.max_size,
                'min_size': pool.min_size
            }
    
    def get_quota_stats(self) -> Dict[str, Any]:
        """
        Get quota statistics.
        
        Returns:
            Quota statistics
        """
        with self._lock:
            return {
                resource_type.value: {
                    'max_usage': quota.max_usage,
                    'current_usage': quota.current_usage,
                    'utilization': (quota.current_usage / quota.max_usage * 100) if quota.max_usage > 0 else 0,
                    'soft_limit': quota.soft_limit,
                    'hard_limit': quota.hard_limit,
                    'exceeded': quota.exceeded
                }
                for resource_type, quota in self.quotas.items()
            }
    
    def optimize_resources(self) -> Dict[str, Any]:
        """
        Optimize resource allocation.
        
        Returns:
            Optimization results
        """
        results = {
            'before': {
                'pools': {name: self.get_pool_stats(name) for name in self.pools.keys()},
                'quotas': self.get_quota_stats()
            },
            'optimizations_applied': []
        }
        
        # Cleanup unused resources
        self._cleanup_unused_resources()
        results['optimizations_applied'].append('cleanup_unused_resources')
        
        # Force garbage collection
        collected = gc.collect()
        results['optimizations_applied'].append(f'garbage_collection_{collected}_objects')
        
        # Balance resource pools
        self._balance_resource_pools()
        results['optimizations_applied'].append('balance_resource_pools')
        
        results['after'] = {
            'pools': {name: self.get_pool_stats(name) for name in self.pools.keys()},
            'quotas': self.get_quota_stats()
        }
        
        return results
    
    def _balance_resource_pools(self):
        """Balance resource pools."""
        with self._lock:
            for pool in self.pools.values():
                # Ensure minimum pool size
                while len(pool.resources) < pool.min_size:
                    # Create new resource
                    resource_id = f"{pool.name}_resource_{len(pool.resources)}"
                    self.add_resource(pool.name, resource_id, 1.0)
                
                # Remove excess resources if over max size
                while len(pool.resources) > pool.max_size:
                    # Remove least used resource
                    if pool.resources:
                        oldest_resource = min(pool.resources, key=lambda r: r.last_used)
                        if oldest_resource.state == ResourceState.AVAILABLE:
                            pool.resources.remove(oldest_resource)
    
    @contextmanager
    def resource_context(self, pool_name: str, required_capacity: float = 1.0):
        """
        Context manager for resource allocation.
        
        Args:
            pool_name: Name of the pool
            required_capacity: Required capacity
            
        Yields:
            Allocated resource
        """
        resource = self.allocate_resource(pool_name, required_capacity)
        if resource is None:
            raise RuntimeError(f"Failed to allocate resource from pool {pool_name}")
        
        try:
            yield resource
        finally:
            self.release_resource(resource.id)
    
    def cleanup(self):
        """Cleanup all resources."""
        self.stop_monitoring()
        
        with self._lock:
            # Release all allocated resources
            for resource in list(self.allocated_resources.values()):
                if resource.cleanup_callback:
                    try:
                        resource.cleanup_callback(resource)
                    except Exception as e:
                        print(f"Error in resource cleanup: {e}")
            
            self.allocated_resources.clear()
            
            # Clear all pools
            for pool in self.pools.values():
                for resource in pool.resources:
                    if resource.cleanup_callback:
                        try:
                            resource.cleanup_callback(resource)
                        except Exception as e:
                            print(f"Error in resource cleanup: {e}")
                pool.resources.clear()
            
            self.pools.clear()
            self.quotas.clear()
            self.cleanup_callbacks.clear()


