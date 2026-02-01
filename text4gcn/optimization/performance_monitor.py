"""
Performance Monitoring System for Text4GCN.

This module provides comprehensive performance monitoring including:
- Real-time performance metrics
- Profiling and benchmarking
- Resource usage tracking
- Performance alerts and thresholds
- Historical performance data
"""

import time
import threading
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
import torch as th
from contextlib import contextmanager


class MetricType(Enum):
    """Types of performance metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert."""
    metric_name: str
    current_value: float
    threshold: float
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class PerformanceStats:
    """Performance statistics."""
    metric_name: str
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsCollector:
    """Collector for performance metrics."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical metrics to keep
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.stats: Dict[str, PerformanceStats] = {}
        self._lock = threading.RLock()
    
    def record_metric(self, metric: PerformanceMetric):
        """
        Record a performance metric.
        
        Args:
            metric: Performance metric to record
        """
        with self._lock:
            self.metrics[metric.name].append(metric)
            self._update_stats(metric)
    
    def _update_stats(self, metric: PerformanceMetric):
        """Update statistics for a metric."""
        if metric.name not in self.stats:
            self.stats[metric.name] = PerformanceStats(metric_name=metric.name)
        
        stats = self.stats[metric.name]
        stats.count += 1
        stats.sum += metric.value
        stats.min = min(stats.min, metric.value)
        stats.max = max(stats.max, metric.value)
        stats.mean = stats.sum / stats.count
        
        # Calculate standard deviation
        if stats.count > 1:
            values = [m.value for m in self.metrics[metric.name]]
            stats.std = np.std(values) if len(values) > 1 else 0.0
        
        # Calculate percentiles
        if stats.count >= 10:
            values = sorted([m.value for m in self.metrics[metric.name]])
            n = len(values)
            stats.p50 = values[int(n * 0.5)]
            stats.p95 = values[int(n * 0.95)]
            stats.p99 = values[int(n * 0.99)]
    
    def get_metric_history(self, metric_name: str, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> List[PerformanceMetric]:
        """
        Get metric history.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of metrics
        """
        with self._lock:
            if metric_name not in self.metrics:
                return []
            
            metrics = list(self.metrics[metric_name])
            
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return metrics
    
    def get_stats(self, metric_name: str) -> Optional[PerformanceStats]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Performance statistics or None
        """
        with self._lock:
            return self.stats.get(metric_name)
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Get all statistics."""
        with self._lock:
            return self.stats.copy()
    
    def clear_metric(self, metric_name: str):
        """Clear metrics for a specific metric."""
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
            if metric_name in self.stats:
                del self.stats[metric_name]
    
    def clear_all(self):
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()
            self.stats.clear()


class PerformanceMonitor:
    """Performance monitor for Text4GCN."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.
        
        Args:
            collector: Metrics collector instance
        """
        self.collector = collector or MetricsCollector()
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[AlertLevel, float]] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.record_metric("system.cpu_usage", cpu_percent, MetricType.CPU_USAGE, "%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric("system.memory_usage", memory.percent, MetricType.MEMORY_USAGE, "%")
        self.record_metric("system.memory_available", memory.available, MetricType.MEMORY_USAGE, "bytes")
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.record_metric("system.disk_read", disk_io.read_bytes, MetricType.DISK_IO, "bytes")
            self.record_metric("system.disk_write", disk_io.write_bytes, MetricType.DISK_IO, "bytes")
        
        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            self.record_metric("system.network_sent", net_io.bytes_sent, MetricType.NETWORK_IO, "bytes")
            self.record_metric("system.network_recv", net_io.bytes_recv, MetricType.NETWORK_IO, "bytes")
        
        # GPU usage (if available)
        if th.cuda.is_available():
            gpu_memory = th.cuda.memory_allocated()
            gpu_memory_max = th.cuda.max_memory_allocated()
            self.record_metric("system.gpu_memory", gpu_memory, MetricType.GPU_USAGE, "bytes")
            self.record_metric("system.gpu_memory_max", gpu_memory_max, MetricType.GPU_USAGE, "bytes")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     unit: str = "", tags: Dict[str, str] = None, 
                     metadata: Dict[str, Any] = None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            unit: Unit of measurement
            tags: Additional tags
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.collector.record_metric(metric)
        self._check_thresholds(metric)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds."""
        if metric.name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric.name]
        
        for level, threshold in thresholds.items():
            if metric.value > threshold:
                alert = PerformanceAlert(
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=threshold,
                    level=level,
                    message=f"{metric.name} exceeded {level.value} threshold: {metric.value} > {threshold}"
                )
                
                with self._lock:
                    self.alerts.append(alert)
                
                # Trigger callbacks
                for callback in self.callbacks.get(metric.name, []):
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Error in alert callback: {e}")
    
    def set_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """
        Set alert threshold for a metric.
        
        Args:
            metric_name: Name of the metric
            level: Alert level
            threshold: Threshold value
        """
        with self._lock:
            if metric_name not in self.thresholds:
                self.thresholds[metric_name] = {}
            self.thresholds[metric_name][level] = threshold
    
    def add_alert_callback(self, metric_name: str, callback: Callable[[PerformanceAlert], None]):
        """
        Add alert callback for a metric.
        
        Args:
            metric_name: Name of the metric
            callback: Callback function
        """
        self.callbacks[metric_name].append(callback)
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                   resolved: Optional[bool] = None) -> List[PerformanceAlert]:
        """
        Get performance alerts.
        
        Args:
            level: Filter by alert level
            resolved: Filter by resolved status
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self.alerts.copy()
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts
    
    def resolve_alert(self, alert_index: int):
        """Resolve an alert."""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
    
    def get_metric_stats(self, metric_name: str) -> Optional[PerformanceStats]:
        """Get statistics for a metric."""
        return self.collector.get_stats(metric_name)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance report
        """
        stats = self.collector.get_all_stats()
        alerts = self.get_alerts()
        
        return {
            'timestamp': time.time(),
            'metrics_count': len(stats),
            'alerts_count': len(alerts),
            'unresolved_alerts': len([a for a in alerts if not a.resolved]),
            'metrics': {name: {
                'count': stat.count,
                'mean': stat.mean,
                'min': stat.min,
                'max': stat.max,
                'std': stat.std,
                'p95': getattr(stat, 'p95', None)
            } for name, stat in stats.items()},
            'alerts': [{
                'metric_name': alert.metric_name,
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'resolved': alert.resolved
            } for alert in alerts]
        }
    
    @contextmanager
    def measure_execution_time(self, operation_name: str, tags: Dict[str, str] = None):
        """
        Context manager for measuring execution time.
        
        Args:
            operation_name: Name of the operation
            tags: Additional tags
        """
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.record_metric(
                f"execution_time.{operation_name}",
                execution_time,
                MetricType.EXECUTION_TIME,
                "seconds",
                tags or {}
            )
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        benchmark_result = {
            'function_name': func.__name__,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': success,
            'result': result if success else None,
            'error': error if not success else None,
            'timestamp': start_time
        }
        
        # Record metrics
        self.record_metric(f"benchmark.{func.__name__}.execution_time", 
                          execution_time, MetricType.EXECUTION_TIME, "seconds")
        self.record_metric(f"benchmark.{func.__name__}.memory_used", 
                          memory_used, MetricType.MEMORY_USAGE, "bytes")
        
        return benchmark_result
    
    def cleanup(self):
        """Cleanup monitor resources."""
        self.stop_monitoring()
        self.collector.clear_all()
        
        with self._lock:
            self.alerts.clear()
            self.thresholds.clear()
            self.callbacks.clear()
