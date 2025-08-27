"""
パフォーマンス最適化システム

包括的なパフォーマンス最適化機能：
- メモリ使用量最適化
- CPU使用量最適化
- ネットワーク最適化
- キャッシュ最適化
- 非同期処理最適化
"""

import asyncio
import gc
import psutil
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import json


# ================================
# パフォーマンス測定とメトリクス
# ================================

class MetricType(Enum):
    """メトリクスタイプ"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMetric:
    """パフォーマンスメトリクス"""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'tags': self.tags
        }


class PerformanceMonitor:
    """パフォーマンス監視"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: List[PerformanceMetric] = []
        self.max_history = max_history
        self._lock = threading.RLock()
        self._start_time = datetime.now()
        
        # システムメトリクス取得用
        self.system_monitor_enabled = True
        self.system_monitor_task: Optional[asyncio.Task] = None
        
    def record_metric(self, metric: PerformanceMetric):
        """メトリクス記録"""
        with self._lock:
            self.metrics.append(metric)
            
            # 履歴サイズ制限
            if len(self.metrics) > self.max_history:
                self.metrics.pop(0)
    
    def get_metrics(
        self, 
        metric_type: MetricType = None, 
        component: str = None, 
        since: datetime = None,
        limit: int = None
    ) -> List[PerformanceMetric]:
        """メトリクス取得"""
        with self._lock:
            filtered_metrics = list(self.metrics)
            
            # フィルタリング
            if metric_type:
                filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
            
            if component:
                filtered_metrics = [m for m in filtered_metrics if m.component == component]
            
            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]
            
            # 最新順でソート
            filtered_metrics.sort(key=lambda m: m.timestamp, reverse=True)
            
            if limit:
                filtered_metrics = filtered_metrics[:limit]
            
            return filtered_metrics
    
    def get_system_metrics(self) -> Dict[str, float]:
        """システムメトリクス取得"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # メモリ使用率
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # プロセス情報
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            # ディスク使用率
            disk_info = psutil.disk_usage('/')
            disk_usage = disk_info.percent
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'process_memory_mb': process_memory,
                'process_cpu': process_cpu,
                'disk_usage': disk_usage
            }
        except Exception as e:
            logging.error(f"システムメトリクス取得エラー: {e}")
            return {}
    
    async def start_system_monitoring(self, interval: float = 30.0):
        """システム監視開始"""
        if self.system_monitor_task is not None:
            return
        
        async def monitor_loop():
            while self.system_monitor_enabled:
                try:
                    system_metrics = self.get_system_metrics()
                    now = datetime.now()
                    
                    for metric_name, value in system_metrics.items():
                        if metric_name == 'cpu_usage':
                            metric_type = MetricType.CPU_USAGE
                        elif 'memory' in metric_name:
                            metric_type = MetricType.MEMORY_USAGE
                        else:
                            continue
                        
                        metric = PerformanceMetric(
                            metric_type=metric_type,
                            value=value,
                            timestamp=now,
                            component='system',
                            tags={'metric_name': metric_name}
                        )
                        self.record_metric(metric)
                    
                    await asyncio.sleep(interval)
                
                except Exception as e:
                    logging.error(f"システム監視エラー: {e}")
                    await asyncio.sleep(interval)
        
        self.system_monitor_task = asyncio.create_task(monitor_loop())
    
    def stop_system_monitoring(self):
        """システム監視停止"""
        self.system_monitor_enabled = False
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            self.system_monitor_task = None
    
    def get_performance_summary(self, duration: timedelta = None) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        if duration is None:
            duration = timedelta(hours=1)
        
        since = datetime.now() - duration
        recent_metrics = self.get_metrics(since=since)
        
        summary = {
            'total_metrics': len(recent_metrics),
            'duration_minutes': duration.total_seconds() / 60,
            'metrics_by_type': defaultdict(list),
            'average_values': {},
            'peak_values': {}
        }
        
        # タイプ別分類
        for metric in recent_metrics:
            summary['metrics_by_type'][metric.metric_type.value].append(metric.value)
        
        # 統計計算
        for metric_type, values in summary['metrics_by_type'].items():
            if values:
                summary['average_values'][metric_type] = sum(values) / len(values)
                summary['peak_values'][metric_type] = max(values)
        
        return summary


# ================================
# 最適化されたデータ構造
# ================================

class OptimizedBuffer:
    """最適化されたバッファ"""
    
    def __init__(self, max_size: int = 1000, overflow_strategy: str = "drop_oldest"):
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy  # drop_oldest, drop_newest, resize
        self._buffer = deque(maxlen=max_size if overflow_strategy == "drop_oldest" else None)
        self._lock = threading.RLock()
        self._total_added = 0
        self._total_dropped = 0
    
    def add(self, item: Any) -> bool:
        """アイテム追加"""
        with self._lock:
            self._total_added += 1
            
            if self.overflow_strategy == "drop_oldest":
                if len(self._buffer) >= self.max_size:
                    self._total_dropped += 1
                self._buffer.append(item)
                return True
            
            elif self.overflow_strategy == "drop_newest":
                if len(self._buffer) >= self.max_size:
                    self._total_dropped += 1
                    return False
                self._buffer.append(item)
                return True
            
            elif self.overflow_strategy == "resize":
                if len(self._buffer) >= self.max_size:
                    # バッファサイズを動的に調整
                    self.max_size = int(self.max_size * 1.5)
                    logging.info(f"バッファサイズを {self.max_size} に拡張")
                
                self._buffer.append(item)
                return True
            
            return False
    
    def get_all(self, clear: bool = False) -> List[Any]:
        """全アイテム取得"""
        with self._lock:
            items = list(self._buffer)
            if clear:
                self._buffer.clear()
            return items
    
    def get_stats(self) -> Dict[str, Any]:
        """バッファ統計"""
        with self._lock:
            return {
                'current_size': len(self._buffer),
                'max_size': self.max_size,
                'total_added': self._total_added,
                'total_dropped': self._total_dropped,
                'drop_rate': self._total_dropped / self._total_added if self._total_added > 0 else 0,
                'utilization': len(self._buffer) / self.max_size if self.max_size > 0 else 0
            }


class AdaptiveCache:
    """適応型キャッシュ"""
    
    def __init__(self, initial_size: int = 128, max_size: int = 10000):
        self.initial_size = initial_size
        self.max_size = max_size
        self.current_size = initial_size
        
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._lock = threading.RLock()
        
        # 統計
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: Any) -> Any:
        """値取得"""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._access_times[key] = datetime.now()
                self._access_counts[key] += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
    
    def put(self, key: Any, value: Any):
        """値設定"""
        with self._lock:
            # キャッシュサイズチェック
            if len(self._cache) >= self.current_size:
                self._evict_least_recently_used()
            
            self._cache[key] = value
            self._access_times[key] = datetime.now()
            self._access_counts[key] += 1
            
            # 適応的サイズ調整
            self._adjust_cache_size()
    
    def _evict_least_recently_used(self):
        """LRU エビクション"""
        if not self._cache:
            return
        
        # 最も古いアクセス時刻のキーを見つける
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
        del self._access_counts[oldest_key]
        
        self._evictions += 1
    
    def _adjust_cache_size(self):
        """適応的キャッシュサイズ調整"""
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        
        # ヒット率に基づく調整
        if hit_rate > 0.8 and self.current_size < self.max_size:
            # ヒット率が高い場合、キャッシュサイズを増やす
            self.current_size = min(int(self.current_size * 1.2), self.max_size)
        elif hit_rate < 0.5 and self.current_size > self.initial_size:
            # ヒット率が低い場合、キャッシュサイズを減らす
            self.current_size = max(int(self.current_size * 0.8), self.initial_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'current_size': self.current_size,
                'used_entries': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'utilization': len(self._cache) / self.current_size if self.current_size > 0 else 0
            }


# ================================
# パフォーマンス最適化デコレーター
# ================================

def performance_optimized(
    cache_enabled: bool = True,
    cache_size: int = 128,
    timeout: float = 30.0,
    memory_limit_mb: int = 100
):
    """パフォーマンス最適化デコレーター"""
    
    # 関数ごとのキャッシュとメトリクス
    caches = {}
    monitors = {}
    
    def decorator(func: Callable):
        func_name = func.__name__
        
        # キャッシュとモニター初期化
        if cache_enabled:
            caches[func_name] = AdaptiveCache(initial_size=cache_size)
        monitors[func_name] = PerformanceMonitor()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            cache_key = None
            
            try:
                # キャッシュキー生成
                if cache_enabled:
                    cache_key = f"{func_name}:{hash((args, tuple(sorted(kwargs.items()))))}"
                    cached_result = caches[func_name].get(cache_key)
                    if cached_result is not None:
                        execution_time = (time.time() - start_time) * 1000
                        
                        # キャッシュヒットメトリクス記録
                        monitors[func_name].record_metric(
                            PerformanceMetric(
                                metric_type=MetricType.RESPONSE_TIME,
                                value=execution_time,
                                component=func_name,
                                tags={'cache': 'hit'}
                            )
                        )
                        
                        return cached_result
                
                # タイムアウト付き実行
                try:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except asyncio.TimeoutError:
                    logging.warning(f"関数 {func_name} がタイムアウトしました ({timeout}秒)")
                    raise
                
                execution_time = (time.time() - start_time) * 1000
                
                # キャッシュに保存
                if cache_enabled and cache_key:
                    caches[func_name].put(cache_key, result)
                
                # メトリクス記録
                monitors[func_name].record_metric(
                    PerformanceMetric(
                        metric_type=MetricType.RESPONSE_TIME,
                        value=execution_time,
                        component=func_name,
                        tags={'cache': 'miss' if cache_enabled else 'disabled'}
                    )
                )
                
                return result
            
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # エラーメトリクス記録
                monitors[func_name].record_metric(
                    PerformanceMetric(
                        metric_type=MetricType.ERROR_RATE,
                        value=1.0,
                        component=func_name,
                        tags={'error_type': type(e).__name__}
                    )
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            cache_key = None
            
            try:
                # キャッシュキー生成
                if cache_enabled:
                    cache_key = f"{func_name}:{hash((args, tuple(sorted(kwargs.items()))))}"
                    cached_result = caches[func_name].get(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # キャッシュに保存
                if cache_enabled and cache_key:
                    caches[func_name].put(cache_key, result)
                
                # メトリクス記録
                monitors[func_name].record_metric(
                    PerformanceMetric(
                        metric_type=MetricType.RESPONSE_TIME,
                        value=execution_time,
                        component=func_name,
                        tags={'cache': 'miss' if cache_enabled else 'disabled'}
                    )
                )
                
                return result
            
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # エラーメトリクス記録
                monitors[func_name].record_metric(
                    PerformanceMetric(
                        metric_type=MetricType.ERROR_RATE,
                        value=1.0,
                        component=func_name,
                        tags={'error_type': type(e).__name__}
                    )
                )
                
                raise
        
        # パフォーマンス統計取得メソッドを関数に追加
        def get_performance_stats():
            stats = {
                'function_name': func_name,
                'monitor_stats': monitors[func_name].get_performance_summary()
            }
            
            if cache_enabled:
                stats['cache_stats'] = caches[func_name].get_stats()
            
            return stats
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.get_performance_stats = get_performance_stats
        
        return wrapper
    
    return decorator


# ================================
# メモリ最適化
# ================================

class MemoryOptimizer:
    """メモリ最適化"""
    
    def __init__(self):
        self._weak_references: Set[weakref.ref] = set()
        self._memory_threshold_mb = 500  # MB
        self._optimization_interval = 60  # 秒
        self._last_optimization = datetime.now()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """メモリ最適化実行"""
        start_time = time.time()
        
        # ガベージコレクション強制実行
        collected_objects = gc.collect()
        
        # 弱参照のクリーンアップ
        dead_refs = [ref for ref in self._weak_references if ref() is None]
        for ref in dead_refs:
            self._weak_references.discard(ref)
        
        optimization_time = (time.time() - start_time) * 1000
        
        result = {
            'collected_objects': collected_objects,
            'cleaned_weak_refs': len(dead_refs),
            'optimization_time_ms': optimization_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self._last_optimization = datetime.now()
        logging.info(f"メモリ最適化完了: {result}")
        
        return result
    
    def should_optimize(self) -> bool:
        """最適化が必要かチェック"""
        # 時間ベースチェック
        time_since_last = (datetime.now() - self._last_optimization).total_seconds()
        if time_since_last > self._optimization_interval:
            return True
        
        # メモリ使用量ベースチェック
        try:
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            return memory_usage_mb > self._memory_threshold_mb
        except Exception:
            return False
    
    def register_weak_reference(self, obj: Any) -> weakref.ref:
        """弱参照登録"""
        weak_ref = weakref.ref(obj)
        self._weak_references.add(weak_ref)
        return weak_ref


# ================================
# パフォーマンス最適化マネージャー
# ================================

class PerformanceOptimizer:
    """統合パフォーマンス最適化マネージャー"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.adaptive_caches: Dict[str, AdaptiveCache] = {}
        self.optimized_buffers: Dict[str, OptimizedBuffer] = {}
        
        # 最適化設定
        self.auto_optimization_enabled = True
        self.optimization_task: Optional[asyncio.Task] = None
        
    def create_optimized_buffer(self, name: str, max_size: int = 1000, overflow_strategy: str = "drop_oldest") -> OptimizedBuffer:
        """最適化バッファ作成"""
        buffer = OptimizedBuffer(max_size, overflow_strategy)
        self.optimized_buffers[name] = buffer
        return buffer
    
    def create_adaptive_cache(self, name: str, initial_size: int = 128, max_size: int = 10000) -> AdaptiveCache:
        """適応型キャッシュ作成"""
        cache = AdaptiveCache(initial_size, max_size)
        self.adaptive_caches[name] = cache
        return cache
    
    async def start_auto_optimization(self, interval: float = 300.0):
        """自動最適化開始"""
        if self.optimization_task is not None:
            return
        
        async def optimization_loop():
            while self.auto_optimization_enabled:
                try:
                    # メモリ最適化チェック
                    if self.memory_optimizer.should_optimize():
                        self.memory_optimizer.optimize_memory()
                    
                    # システム監視メトリクス記録
                    system_metrics = self.monitor.get_system_metrics()
                    for metric_name, value in system_metrics.items():
                        metric_type = MetricType.CPU_USAGE if 'cpu' in metric_name else MetricType.MEMORY_USAGE
                        
                        self.monitor.record_metric(
                            PerformanceMetric(
                                metric_type=metric_type,
                                value=value,
                                component='system',
                                tags={'metric': metric_name}
                            )
                        )
                    
                    await asyncio.sleep(interval)
                
                except Exception as e:
                    logging.error(f"自動最適化エラー: {e}")
                    await asyncio.sleep(interval)
        
        self.optimization_task = asyncio.create_task(optimization_loop())
    
    def stop_auto_optimization(self):
        """自動最適化停止"""
        self.auto_optimization_enabled = False
        if self.optimization_task:
            self.optimization_task.cancel()
            self.optimization_task = None
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報取得"""
        return {
            'system_metrics': self.monitor.get_system_metrics(),
            'performance_summary': self.monitor.get_performance_summary(),
            'buffer_stats': {name: buf.get_stats() for name, buf in self.optimized_buffers.items()},
            'cache_stats': {name: cache.get_stats() for name, cache in self.adaptive_caches.items()},
            'optimization_status': {
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'memory_optimization_needed': self.memory_optimizer.should_optimize(),
                'last_memory_optimization': self.memory_optimizer._last_optimization.isoformat()
            }
        }


# グローバル最適化インスタンス
_global_optimizer = PerformanceOptimizer()

def get_global_performance_optimizer() -> PerformanceOptimizer:
    """グローバルパフォーマンス最適化取得"""
    return _global_optimizer


# エクスポート
__all__ = [
    # 列挙型とデータクラス
    'MetricType', 'PerformanceMetric',
    
    # 監視クラス
    'PerformanceMonitor',
    
    # 最適化されたデータ構造
    'OptimizedBuffer', 'AdaptiveCache',
    
    # 最適化クラス
    'MemoryOptimizer', 'PerformanceOptimizer',
    
    # デコレーター
    'performance_optimized',
    
    # グローバル関数
    'get_global_performance_optimizer'
]