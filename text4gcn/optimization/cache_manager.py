"""
Advanced Caching System for Text4GCN.

This module provides comprehensive caching capabilities including:
- Multi-level caching (L1, L2, L3)
- Cache strategies (LRU, LFU, FIFO, TTL)
- Memory-aware caching
- Distributed caching support
- Cache invalidation strategies
"""

import time
import threading
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import weakref
import json
import numpy as np
import torch as th


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"   # Time To Live
    SIZE_BASED = "size_based"  # Based on object size


class CacheLevel(Enum):
    """Cache levels."""
    L1 = "l1"  # Fastest, smallest
    L2 = "l2"  # Medium speed, medium size
    L3 = "l3"  # Slowest, largest


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    level: CacheLevel = CacheLevel.L1
    compressed: bool = False


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    hit_ratio: float = 0.0


class CacheLevelImpl:
    """Individual cache level implementation."""
    
    def __init__(self, level: CacheLevel, max_size: int, max_memory: int, 
                 strategy: CacheStrategy = CacheStrategy.LRU):
        """
        Initialize cache level.
        
        Args:
            level: Cache level
            max_size: Maximum number of entries
            max_memory: Maximum memory usage in bytes
            strategy: Eviction strategy
        """
        self.level = level
        self.max_size = max_size
        self.max_memory = max_memory
        self.strategy = strategy
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self.entries:
                self.stats.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.created_at > entry.ttl:
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.entries.move_to_end(key)
            
            self.stats.hits += 1
            self._update_hit_ratio()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            size: Optional[int] = None) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            size: Size of the value in bytes
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate size if not provided
            if size is None:
                size = self._calculate_size(value)
            
            # Check if we can fit this entry
            if not self._can_fit_entry(size):
                return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                ttl=ttl,
                level=self.level
            )
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self.stats.entry_count += 1
            self.stats.total_size += size
            
            # Evict if necessary
            self._evict_if_needed()
            
            return True
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if isinstance(value, (np.ndarray, th.Tensor)):
                return value.nbytes
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except (pickle.PicklingError, TypeError):
            return 1024  # Default size
    
    def _can_fit_entry(self, size: int) -> bool:
        """Check if entry can fit in cache."""
        return (self.stats.total_size + size <= self.max_memory and 
                len(self.entries) < self.max_size)
    
    def _evict_if_needed(self):
        """Evict entries if cache is full."""
        while (self.stats.total_size > self.max_memory or 
               len(self.entries) >= self.max_size):
            if not self.entries:
                break
            
            # Choose entry to evict based on strategy
            key_to_evict = self._choose_eviction_candidate()
            if key_to_evict:
                self._remove_entry(key_to_evict)
                self.stats.evictions += 1
    
    def _choose_eviction_candidate(self) -> Optional[str]:
        """Choose entry to evict based on strategy."""
        if not self.entries:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            return next(iter(self.entries))
        elif self.strategy == CacheStrategy.LFU:
            return min(self.entries.keys(), 
                      key=lambda k: self.entries[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            return next(iter(self.entries))
        elif self.strategy == CacheStrategy.SIZE_BASED:
            return max(self.entries.keys(), 
                      key=lambda k: self.entries[k].size)
        else:
            return next(iter(self.entries))
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.entries:
            entry = self.entries[key]
            self.stats.total_size -= entry.size
            self.stats.entry_count -= 1
            del self.entries[key]
    
    def _update_hit_ratio(self):
        """Update hit ratio."""
        total = self.stats.hits + self.stats.misses
        if total > 0:
            self.stats.hit_ratio = self.stats.hits / total
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self.entries.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size=self.stats.total_size,
                entry_count=self.stats.entry_count,
                hit_ratio=self.stats.hit_ratio
            )


class CacheManager:
    """Advanced cache manager with multi-level caching."""
    
    def __init__(self, l1_size: int = 100, l1_memory: int = 100 * 1024 * 1024,
                 l2_size: int = 1000, l2_memory: int = 500 * 1024 * 1024,
                 l3_size: int = 10000, l3_memory: int = 2 * 1024 * 1024 * 1024):
        """
        Initialize cache manager.
        
        Args:
            l1_size: L1 cache max entries
            l1_memory: L1 cache max memory
            l2_size: L2 cache max entries  
            l2_memory: L2 cache max memory
            l3_size: L3 cache max entries
            l3_memory: L3 cache max memory
        """
        self.l1 = CacheLevelImpl(CacheLevel.L1, l1_size, l1_memory, CacheStrategy.LRU)
        self.l2 = CacheLevelImpl(CacheLevel.L2, l2_size, l2_memory, CacheStrategy.LFU)
        self.l3 = CacheLevelImpl(CacheLevel.L3, l3_size, l3_memory, CacheStrategy.TTL)
        
        self.compression_enabled = False
        self.distributed_enabled = False
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (L1 -> L2 -> L3).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            return self._decompress_if_needed(value)
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1
            self.l1.put(key, value)
            return self._decompress_if_needed(value)
        
        # Try L3
        value = self.l3.get(key)
        if value is not None:
            # Promote to L2
            self.l2.put(key, value)
            return self._decompress_if_needed(value)
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            level: Optional[CacheLevel] = None) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            level: Specific cache level (if None, auto-select)
            
        Returns:
            True if successfully cached
        """
        # Compress if needed
        if self.compression_enabled:
            value = self._compress_if_needed(value)
        
        # Auto-select level based on size
        if level is None:
            size = self._calculate_size(value)
            if size < 1024:  # < 1KB
                level = CacheLevel.L1
            elif size < 1024 * 1024:  # < 1MB
                level = CacheLevel.L2
            else:  # >= 1MB
                level = CacheLevel.L3
        
        # Try to put in selected level
        if level == CacheLevel.L1:
            return self.l1.put(key, value, ttl)
        elif level == CacheLevel.L2:
            return self.l2.put(key, value, ttl)
        else:
            return self.l3.put(key, value, ttl)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value."""
        try:
            if isinstance(value, (np.ndarray, th.Tensor)):
                return value.nbytes
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except (pickle.PicklingError, TypeError):
            return 1024
    
    def _compress_if_needed(self, value: Any) -> Any:
        """Compress value if compression is enabled."""
        if not self.compression_enabled:
            return value
        
        # Simple compression for demonstration
        if isinstance(value, str):
            return value.encode('utf-8')
        elif isinstance(value, (list, tuple)):
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        return value
    
    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if needed."""
        if not self.compression_enabled:
            return value
        
        # Simple decompression for demonstration
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except UnicodeDecodeError:
                return pickle.loads(value)
        
        return value
    
    def invalidate(self, key: str):
        """Invalidate cache entry."""
        with self._lock:
            self.l1._remove_entry(key)
            self.l2._remove_entry(key)
            self.l3._remove_entry(key)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate entries matching pattern."""
        with self._lock:
            for level in [self.l1, self.l2, self.l3]:
                keys_to_remove = [key for key in level.entries.keys() 
                                if pattern in key]
                for key in keys_to_remove:
                    level._remove_entry(key)
    
    def clear(self):
        """Clear all cache levels."""
        with self._lock:
            self.l1.clear()
            self.l2.clear()
            self.l3.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        return {
            'l1': self.l1.get_stats(),
            'l2': self.l2.get_stats(),
            'l3': self.l3.get_stats()
        }
    
    def get_combined_stats(self) -> CacheStats:
        """Get combined statistics."""
        l1_stats = self.l1.get_stats()
        l2_stats = self.l2.get_stats()
        l3_stats = self.l3.get_stats()
        
        return CacheStats(
            hits=l1_stats.hits + l2_stats.hits + l3_stats.hits,
            misses=l1_stats.misses + l2_stats.misses + l3_stats.misses,
            evictions=l1_stats.evictions + l2_stats.evictions + l3_stats.evictions,
            total_size=l1_stats.total_size + l2_stats.total_size + l3_stats.total_size,
            entry_count=l1_stats.entry_count + l2_stats.entry_count + l3_stats.entry_count,
            hit_ratio=(l1_stats.hit_ratio + l2_stats.hit_ratio + l3_stats.hit_ratio) / 3
        )
    
    def enable_compression(self):
        """Enable compression."""
        self.compression_enabled = True
    
    def disable_compression(self):
        """Disable compression."""
        self.compression_enabled = False
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimize cache performance.
        
        Returns:
            Optimization results
        """
        results = {
            'before': self.get_combined_stats(),
            'optimizations_applied': []
        }
        
        # Clear empty levels
        for level_name, level in [('l1', self.l1), ('l2', self.l2), ('l3', self.l3)]:
            if level.stats.entry_count == 0:
                level.clear()
                results['optimizations_applied'].append(f'cleared_empty_{level_name}')
        
        # Balance cache levels
        self._balance_cache_levels()
        results['optimizations_applied'].append('balanced_cache_levels')
        
        results['after'] = self.get_combined_stats()
        
        return results
    
    def _balance_cache_levels(self):
        """Balance cache levels by moving entries."""
        # Move frequently accessed items to L1
        for key, entry in list(self.l2.entries.items()):
            if entry.access_count > 5:
                value = self.l2.get(key)
                if value and self.l1.put(key, value):
                    self.l2._remove_entry(key)
        
        # Move large items to L3
        for key, entry in list(self.l1.entries.items()):
            if entry.size > 1024 * 1024:  # > 1MB
                value = self.l1.get(key)
                if value and self.l3.put(key, value):
                    self.l1._remove_entry(key)


class DistributedCache:
    """Distributed cache implementation."""
    
    def __init__(self, nodes: List[str], cache_manager: CacheManager):
        """
        Initialize distributed cache.
        
        Args:
            nodes: List of cache nodes
            cache_manager: Local cache manager
        """
        self.nodes = nodes
        self.cache_manager = cache_manager
        self.node_hashes = {node: hashlib.md5(node.encode()).hexdigest() 
                           for node in nodes}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        value = self.cache_manager.get(key)
        if value is not None:
            return value
        
        # Try distributed nodes
        node = self._get_node_for_key(key)
        # In a real implementation, this would make a network call
        # For now, return None
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in distributed cache."""
        # Put in local cache
        success = self.cache_manager.put(key, value, ttl)
        
        # Replicate to other nodes
        node = self._get_node_for_key(key)
        # In a real implementation, this would replicate to other nodes
        
        return success
    
    def _get_node_for_key(self, key: str) -> str:
        """Get node for a given key using consistent hashing."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Find the first node with hash >= key_hash
        for node, node_hash in sorted(self.node_hashes.items(), 
                                     key=lambda x: x[1]):
            if node_hash >= key_hash:
                return node
        
        # If no node found, use the first node
        return self.nodes[0]
