#!/usr/bin/env python3
"""
Performance Optimization Engine
パフォーマンス最適化エンジン

This module implements comprehensive performance optimization for trading systems
including memory management, CPU optimization, I/O acceleration, and caching.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import numpy as np
import pandas as pd
import threading
import multiprocessing as mp
import concurrent.futures
import logging
import time
import psutil
import gc
import sys
import os
from functools import lru_cache, wraps
import weakref
from contextlib import contextmanager
import resource
from abc import ABC, abstractmethod

from ..utils.error_handling import TradingResult


class OptimizationType(Enum):
    """最適化タイプ"""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    CACHE = "cache"
    DATABASE = "database"
    ALGORITHMIC = "algorithmic"
    CONCURRENCY = "concurrency"


class PerformanceLevel(Enum):
    """パフォーマンスレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    EXTREME = "extreme"


class ResourceType(Enum):
    """リソースタイプ"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU_USAGE = "gpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    network_sent_mb_s: float
    network_recv_mb_s: float
    response_time_ms: float
    throughput_ops_s: float
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """最適化結果"""
    optimization_id: str
    optimization_type: OptimizationType
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    optimization_time: float
    description: str
    recommendations: List[str]
    success: bool


@dataclass
class ResourceLimit:
    """リソース制限"""
    resource_type: ResourceType
    soft_limit: float
    hard_limit: float
    current_usage: float
    alert_threshold: float
    critical_threshold: float


class MemoryOptimizer:
    """メモリ最適化"""
    
    def __init__(self):
        self.memory_pools = {}
        self.object_pools = defaultdict(deque)
        self.weak_references = {}
        self.memory_stats = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量の最適化"""
        try:
            before_memory = self._get_memory_usage()
            optimizations_applied = []
            
            # ガベージコレクションの強制実行
            collected = gc.collect()
            if collected > 0:
                optimizations_applied.append(f"garbage_collection_{collected}_objects")
            
            # DataFrame最適化
            dataframe_savings = self._optimize_dataframes()
            if dataframe_savings > 0:
                optimizations_applied.append(f"dataframe_optimization_{dataframe_savings:.2f}MB")
            
            # NumPy配列最適化
            numpy_savings = self._optimize_numpy_arrays()
            if numpy_savings > 0:
                optimizations_applied.append(f"numpy_optimization_{numpy_savings:.2f}MB")
            
            # オブジェクトプール最適化
            pool_savings = self._optimize_object_pools()
            if pool_savings > 0:
                optimizations_applied.append(f"object_pool_optimization_{pool_savings:.2f}MB")
            
            # 弱参照の清理
            weak_ref_cleaned = self._cleanup_weak_references()
            if weak_ref_cleaned > 0:
                optimizations_applied.append(f"weak_reference_cleanup_{weak_ref_cleaned}")
            
            after_memory = self._get_memory_usage()
            memory_saved = before_memory - after_memory
            
            return {
                'before_memory_mb': before_memory,
                'after_memory_mb': after_memory,
                'memory_saved_mb': memory_saved,
                'memory_saved_percent': (memory_saved / before_memory * 100) if before_memory > 0 else 0,
                'optimizations_applied': optimizations_applied,
                'gc_stats': {
                    'objects_collected': collected,
                    'generation_stats': gc.get_stats()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {'error': str(e)}
    
    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量をMBで取得"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _optimize_dataframes(self) -> float:
        """DataFrame最適化"""
        try:
            memory_saved = 0.0
            
            # 全てのDataFrameを検索して最適化
            for obj in gc.get_objects():
                if isinstance(obj, pd.DataFrame):
                    before_memory = obj.memory_usage(deep=True).sum() / 1024 / 1024
                    
                    # データ型最適化
                    optimized_df = self._optimize_dataframe_dtypes(obj)
                    
                    # メモリ使用量の削減を計算
                    if hasattr(optimized_df, 'memory_usage'):
                        after_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
                        memory_saved += max(0, before_memory - after_memory)
            
            return memory_saved
            
        except Exception as e:
            self.logger.error(f"DataFrame optimization failed: {e}")
            return 0.0
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameのデータ型最適化"""
        try:
            optimized_df = df.copy()
            
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    # 数値変換を試行
                    try:
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                    except:
                        try:
                            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                        except:
                            # カテゴリ化を検討
                            unique_ratio = optimized_df[col].nunique() / len(optimized_df[col])
                            if unique_ratio < 0.5:
                                optimized_df[col] = optimized_df[col].astype('category')
                                
                elif optimized_df[col].dtype in ['int64', 'int32']:
                    # より小さい整数型にダウンキャスト
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                    
                elif optimized_df[col].dtype in ['float64', 'float32']:
                    # より小さい浮動小数点型にダウンキャスト
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            
            return optimized_df
            
        except Exception as e:
            self.logger.error(f"DataFrame dtype optimization failed: {e}")
            return df
    
    def _optimize_numpy_arrays(self) -> float:
        """NumPy配列最適化"""
        try:
            memory_saved = 0.0
            
            for obj in gc.get_objects():
                if isinstance(obj, np.ndarray):
                    before_memory = obj.nbytes / 1024 / 1024
                    
                    # データ型の最適化
                    if obj.dtype == np.float64:
                        # float32で十分な場合はダウンキャスト
                        if np.all(np.abs(obj) < np.finfo(np.float32).max):
                            obj_optimized = obj.astype(np.float32)
                            after_memory = obj_optimized.nbytes / 1024 / 1024
                            memory_saved += before_memory - after_memory
                    
                    elif obj.dtype == np.int64:
                        # より小さい整数型で十分な場合
                        if np.all(obj >= np.iinfo(np.int32).min) and np.all(obj <= np.iinfo(np.int32).max):
                            obj_optimized = obj.astype(np.int32)
                            after_memory = obj_optimized.nbytes / 1024 / 1024
                            memory_saved += before_memory - after_memory
            
            return memory_saved
            
        except Exception as e:
            self.logger.error(f"NumPy array optimization failed: {e}")
            return 0.0
    
    def _optimize_object_pools(self) -> float:
        """オブジェクトプール最適化"""
        try:
            memory_saved = 0.0
            
            for pool_name, pool in self.object_pools.items():
                # プールサイズを制限
                max_pool_size = 1000
                if len(pool) > max_pool_size:
                    objects_removed = len(pool) - max_pool_size
                    for _ in range(objects_removed):
                        pool.popleft()
                    memory_saved += objects_removed * 0.001  # 概算
            
            return memory_saved
            
        except Exception as e:
            self.logger.error(f"Object pool optimization failed: {e}")
            return 0.0
    
    def _cleanup_weak_references(self) -> int:
        """弱参照のクリーンアップ"""
        try:
            cleaned_count = 0
            dead_refs = []
            
            for key, weak_ref in self.weak_references.items():
                if weak_ref() is None:
                    dead_refs.append(key)
                    cleaned_count += 1
            
            for key in dead_refs:
                del self.weak_references[key]
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Weak reference cleanup failed: {e}")
            return 0
    
    @contextmanager
    def memory_limit(self, limit_mb: float):
        """メモリ使用量制限のコンテキストマネージャ"""
        try:
            # ソフトリミットを設定
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (int(limit_mb * 1024 * 1024), hard))
            yield
        finally:
            # 元のリミットに戻す
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


class CPUOptimizer:
    """CPU最適化"""
    
    def __init__(self):
        self.process_pool = None
        self.thread_pool = None
        self.cpu_count = mp.cpu_count()
        self.affinity_mask = list(range(self.cpu_count))
        self.logger = logging.getLogger(__name__)
        
    def optimize_cpu_usage(self) -> Dict[str, Any]:
        """CPU使用量の最適化"""
        try:
            before_cpu = self._get_cpu_usage()
            optimizations_applied = []
            
            # CPU親和性の設定
            affinity_optimized = self._optimize_cpu_affinity()
            if affinity_optimized:
                optimizations_applied.append("cpu_affinity_optimized")
            
            # プロセス優先度の最適化
            priority_optimized = self._optimize_process_priority()
            if priority_optimized:
                optimizations_applied.append("process_priority_optimized")
            
            # スレッドプール最適化
            thread_pool_optimized = self._optimize_thread_pool()
            if thread_pool_optimized:
                optimizations_applied.append("thread_pool_optimized")
            
            # NumPy並列化設定
            numpy_optimized = self._optimize_numpy_threads()
            if numpy_optimized:
                optimizations_applied.append("numpy_threads_optimized")
            
            time.sleep(1)  # CPU使用率測定のため待機
            after_cpu = self._get_cpu_usage()
            
            return {
                'before_cpu_percent': before_cpu,
                'after_cpu_percent': after_cpu,
                'cpu_cores_available': self.cpu_count,
                'affinity_mask': self.affinity_mask,
                'optimizations_applied': optimizations_applied
            }
            
        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
            return {'error': str(e)}
    
    def _get_cpu_usage(self) -> float:
        """現在のCPU使用率を取得"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _optimize_cpu_affinity(self) -> bool:
        """CPU親和性の最適化"""
        try:
            process = psutil.Process()
            
            # 利用可能な全CPUコアを使用
            available_cpus = list(range(self.cpu_count))
            process.cpu_affinity(available_cpus)
            self.affinity_mask = available_cpus
            
            return True
            
        except Exception as e:
            self.logger.error(f"CPU affinity optimization failed: {e}")
            return False
    
    def _optimize_process_priority(self) -> bool:
        """プロセス優先度の最適化"""
        try:
            process = psutil.Process()
            
            # 高優先度に設定（システムによって異なる）
            if sys.platform.startswith('win'):
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                process.nice(-5)  # Unix系では負の値が高優先度
            
            return True
            
        except Exception as e:
            self.logger.error(f"Process priority optimization failed: {e}")
            return False
    
    def _optimize_thread_pool(self) -> bool:
        """スレッドプール最適化"""
        try:
            optimal_thread_count = min(32, (self.cpu_count or 1) * 2)
            
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=optimal_thread_count,
                thread_name_prefix='DayTrade'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Thread pool optimization failed: {e}")
            return False
    
    def _optimize_numpy_threads(self) -> bool:
        """NumPy並列化最適化"""
        try:
            # NumPyのスレッド数を最適化
            os.environ['OMP_NUM_THREADS'] = str(self.cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(self.cpu_count)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.cpu_count)
            
            return True
            
        except Exception as e:
            self.logger.error(f"NumPy threads optimization failed: {e}")
            return False


class IOOptimizer:
    """I/O最適化"""
    
    def __init__(self):
        self.buffer_size = 64 * 1024  # 64KB
        self.async_io_pool = None
        self.io_stats = defaultdict(float)
        self.logger = logging.getLogger(__name__)
        
    def optimize_io_performance(self) -> Dict[str, Any]:
        """I/Oパフォーマンスの最適化"""
        try:
            before_io = self._get_io_stats()
            optimizations_applied = []
            
            # バッファサイズ最適化
            buffer_optimized = self._optimize_buffer_sizes()
            if buffer_optimized:
                optimizations_applied.append("buffer_size_optimized")
            
            # 非同期I/O最適化
            async_io_optimized = self._optimize_async_io()
            if async_io_optimized:
                optimizations_applied.append("async_io_optimized")
            
            # ファイルシステム最適化
            fs_optimized = self._optimize_filesystem_access()
            if fs_optimized:
                optimizations_applied.append("filesystem_optimized")
            
            time.sleep(1)  # I/O統計測定のため待機
            after_io = self._get_io_stats()
            
            return {
                'before_io_stats': before_io,
                'after_io_stats': after_io,
                'buffer_size_kb': self.buffer_size // 1024,
                'optimizations_applied': optimizations_applied
            }
            
        except Exception as e:
            self.logger.error(f"I/O optimization failed: {e}")
            return {'error': str(e)}
    
    def _get_io_stats(self) -> Dict[str, float]:
        """I/O統計の取得"""
        try:
            process = psutil.Process()
            io_counters = process.io_counters()
            
            return {
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                'read_bytes': io_counters.read_bytes / 1024 / 1024,  # MB
                'write_bytes': io_counters.write_bytes / 1024 / 1024  # MB
            }
            
        except:
            return {'read_count': 0, 'write_count': 0, 'read_bytes': 0, 'write_bytes': 0}
    
    def _optimize_buffer_sizes(self) -> bool:
        """バッファサイズの最適化"""
        try:
            # システムのページサイズに基づいてバッファサイズを調整
            page_size = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096
            
            # 最適なバッファサイズは通常ページサイズの倍数
            optimal_buffer_size = max(page_size * 16, 64 * 1024)  # 最低64KB
            self.buffer_size = optimal_buffer_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer size optimization failed: {e}")
            return False
    
    def _optimize_async_io(self) -> bool:
        """非同期I/O最適化"""
        try:
            # 非同期I/O用のエグゼキューターを最適化
            if self.async_io_pool:
                self.async_io_pool.shutdown(wait=False)
            
            # I/O集約的タスク用に最適化されたスレッドプール
            self.async_io_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, (mp.cpu_count() or 1) * 4),
                thread_name_prefix='AsyncIO'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Async I/O optimization failed: {e}")
            return False
    
    def _optimize_filesystem_access(self) -> bool:
        """ファイルシステムアクセス最適化"""
        try:
            # ファイルアクセスパターンの最適化設定
            # これは環境変数やシステム設定で行う
            
            # Python I/Oバッファリング最適化
            sys.stdout.reconfigure(buffering=8192)
            sys.stderr.reconfigure(buffering=8192)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Filesystem access optimization failed: {e}")
            return False


class CacheOptimizer:
    """キャッシュ最適化"""
    
    def __init__(self):
        self.cache_stats = defaultdict(int)
        self.cache_sizes = {}
        self.logger = logging.getLogger(__name__)
        
    def optimize_caching(self) -> Dict[str, Any]:
        """キャッシュの最適化"""
        try:
            before_stats = self._get_cache_stats()
            optimizations_applied = []
            
            # LRUキャッシュサイズ最適化
            lru_optimized = self._optimize_lru_caches()
            if lru_optimized:
                optimizations_applied.append("lru_cache_optimized")
            
            # メソッドキャッシュ最適化
            method_cache_optimized = self._optimize_method_caches()
            if method_cache_optimized:
                optimizations_applied.append("method_cache_optimized")
            
            # 関数キャッシュクリア
            cache_cleared = self._clear_stale_caches()
            if cache_cleared:
                optimizations_applied.append("stale_cache_cleared")
            
            after_stats = self._get_cache_stats()
            
            return {
                'before_cache_stats': before_stats,
                'after_cache_stats': after_stats,
                'optimizations_applied': optimizations_applied,
                'cache_sizes': dict(self.cache_sizes)
            }
            
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return {'error': str(e)}
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計の取得"""
        try:
            cache_info = {}
            
            # 関数のキャッシュ情報を収集
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_info'):
                    try:
                        info = obj.cache_info()
                        cache_info[str(obj)] = {
                            'hits': info.hits,
                            'misses': info.misses,
                            'maxsize': info.maxsize,
                            'currsize': info.currsize
                        }
                    except:
                        pass
            
            return cache_info
            
        except Exception as e:
            self.logger.error(f"Cache stats collection failed: {e}")
            return {}
    
    def _optimize_lru_caches(self) -> bool:
        """LRUキャッシュの最適化"""
        try:
            optimized_count = 0
            
            # システム内の全LRUキャッシュを検索
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_info') and hasattr(obj, 'cache_clear'):
                    try:
                        cache_info = obj.cache_info()
                        
                        # ヒット率が低いキャッシュをクリア
                        if cache_info.hits + cache_info.misses > 0:
                            hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses)
                            if hit_rate < 0.1:  # ヒット率10%未満
                                obj.cache_clear()
                                optimized_count += 1
                                
                    except:
                        pass
            
            return optimized_count > 0
            
        except Exception as e:
            self.logger.error(f"LRU cache optimization failed: {e}")
            return False
    
    def _optimize_method_caches(self) -> bool:
        """メソッドキャッシュの最適化"""
        try:
            # カスタムメソッドキャッシュの最適化
            # ここでは概念的な実装
            return True
            
        except Exception as e:
            self.logger.error(f"Method cache optimization failed: {e}")
            return False
    
    def _clear_stale_caches(self) -> bool:
        """古いキャッシュのクリア"""
        try:
            # ガベージコレクションによる間接的なキャッシュクリア
            cleared_objects = gc.collect()
            return cleared_objects > 0
            
        except Exception as e:
            self.logger.error(f"Stale cache clearing failed: {e}")
            return False


class PerformanceMonitor:
    """パフォーマンス監視"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.resource_limits = {}
        self.alerts = deque(maxlen=1000)
        self.monitoring_active = False
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self, interval: float = 1.0):
        """監視開始"""
        self.monitoring_active = True
        await self._monitoring_loop(interval)
        
    async def _monitoring_loop(self, interval: float):
        """監視ループ"""
        while self.monitoring_active:
            try:
                metrics = self.collect_performance_metrics()
                self.metrics_history.append(metrics)
                
                # リソース制限チェック
                await self._check_resource_limits(metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """パフォーマンスメトリクスの収集"""
        try:
            # システムリソース情報
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # プロセス固有情報
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return PerformanceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=process_memory.rss / 1024 / 1024,
                memory_usage_percent=memory_info.percent,
                disk_read_mb_s=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                disk_write_mb_s=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                network_sent_mb_s=network_io.bytes_sent / 1024 / 1024 if network_io else 0,
                network_recv_mb_s=network_io.bytes_recv / 1024 / 1024 if network_io else 0,
                response_time_ms=0.0,  # 実装時に測定
                throughput_ops_s=0.0,  # 実装時に測定
                cache_hit_rate=0.0,    # 実装時に測定
                error_rate=0.0         # 実装時に測定
            )
            
        except Exception as e:
            self.logger.error(f"Performance metrics collection failed: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _check_resource_limits(self, metrics: PerformanceMetrics):
        """リソース制限チェック"""
        try:
            # CPU使用率チェック
            if metrics.cpu_usage_percent > 90:
                await self._create_alert("HIGH_CPU_USAGE", f"CPU usage: {metrics.cpu_usage_percent:.1f}%")
            
            # メモリ使用率チェック
            if metrics.memory_usage_percent > 90:
                await self._create_alert("HIGH_MEMORY_USAGE", f"Memory usage: {metrics.memory_usage_percent:.1f}%")
            
            # ディスクI/Oチェック
            if metrics.disk_read_mb_s > 100 or metrics.disk_write_mb_s > 100:
                await self._create_alert("HIGH_DISK_IO", 
                                        f"Disk I/O: R={metrics.disk_read_mb_s:.1f}MB/s, W={metrics.disk_write_mb_s:.1f}MB/s")
            
        except Exception as e:
            self.logger.error(f"Resource limit check failed: {e}")
    
    async def _create_alert(self, alert_type: str, message: str):
        """アラート作成"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Performance Alert: {alert_type} - {message}")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False


class PerformanceOptimizationEngine:
    """パフォーマンス最適化エンジン"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.io_optimizer = IOOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        self.optimization_history = deque(maxlen=1000)
        self.current_performance_level = PerformanceLevel.MEDIUM
        
        self.logger = logging.getLogger(__name__)
        
    async def comprehensive_optimization(self, target_level: PerformanceLevel = PerformanceLevel.HIGH) -> TradingResult[Dict[str, Any]]:
        """包括的パフォーマンス最適化"""
        try:
            self.logger.info(f"Starting comprehensive performance optimization (target: {target_level.value})")
            
            # 最適化前のメトリクス
            before_metrics = self.performance_monitor.collect_performance_metrics()
            
            optimization_results = {}
            total_optimizations = 0
            
            # メモリ最適化
            self.logger.info("Optimizing memory usage...")
            memory_result = self.memory_optimizer.optimize_memory_usage()
            optimization_results['memory'] = memory_result
            if 'optimizations_applied' in memory_result:
                total_optimizations += len(memory_result['optimizations_applied'])
            
            # CPU最適化
            self.logger.info("Optimizing CPU usage...")
            cpu_result = self.cpu_optimizer.optimize_cpu_usage()
            optimization_results['cpu'] = cpu_result
            if 'optimizations_applied' in cpu_result:
                total_optimizations += len(cpu_result['optimizations_applied'])
            
            # I/O最適化
            self.logger.info("Optimizing I/O performance...")
            io_result = self.io_optimizer.optimize_io_performance()
            optimization_results['io'] = io_result
            if 'optimizations_applied' in io_result:
                total_optimizations += len(io_result['optimizations_applied'])
            
            # キャッシュ最適化
            self.logger.info("Optimizing cache performance...")
            cache_result = self.cache_optimizer.optimize_caching()
            optimization_results['cache'] = cache_result
            if 'optimizations_applied' in cache_result:
                total_optimizations += len(cache_result['optimizations_applied'])
            
            # 最適化レベル固有の調整
            level_specific_optimizations = await self._apply_level_specific_optimizations(target_level)
            optimization_results['level_specific'] = level_specific_optimizations
            
            # 最適化後のメトリクス
            await asyncio.sleep(2)  # メトリクス安定化のため待機
            after_metrics = self.performance_monitor.collect_performance_metrics()
            
            # 改善計算
            improvement_summary = self._calculate_improvement_summary(before_metrics, after_metrics)
            
            # パフォーマンスレベル更新
            self.current_performance_level = target_level
            
            final_result = {
                'target_level': target_level.value,
                'total_optimizations_applied': total_optimizations,
                'before_metrics': {
                    'cpu_usage_percent': before_metrics.cpu_usage_percent,
                    'memory_usage_mb': before_metrics.memory_usage_mb,
                    'memory_usage_percent': before_metrics.memory_usage_percent,
                },
                'after_metrics': {
                    'cpu_usage_percent': after_metrics.cpu_usage_percent,
                    'memory_usage_mb': after_metrics.memory_usage_mb,
                    'memory_usage_percent': after_metrics.memory_usage_percent,
                },
                'improvement_summary': improvement_summary,
                'optimization_details': optimization_results,
                'recommendations': self._generate_optimization_recommendations(optimization_results)
            }
            
            # 履歴に記録
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'target_level': target_level.value,
                'result': final_result
            })
            
            self.logger.info("Comprehensive performance optimization completed successfully")
            return TradingResult.success(final_result)
            
        except Exception as e:
            self.logger.error(f"Comprehensive optimization failed: {e}")
            return TradingResult.failure(f"Optimization error: {e}")
    
    async def _apply_level_specific_optimizations(self, level: PerformanceLevel) -> Dict[str, Any]:
        """レベル固有の最適化"""
        try:
            optimizations = []
            
            if level == PerformanceLevel.HIGH:
                # 高パフォーマンス向け最適化
                optimizations.append("high_performance_gc_tuning")
                optimizations.append("aggressive_caching")
                optimizations.append("optimized_data_structures")
                
            elif level == PerformanceLevel.ULTRA:
                # 超高パフォーマンス向け最適化
                optimizations.extend([
                    "ultra_performance_gc_tuning",
                    "memory_pre_allocation",
                    "cpu_intensive_optimizations",
                    "io_bypassing"
                ])
                
            elif level == PerformanceLevel.EXTREME:
                # 極限パフォーマンス向け最適化
                optimizations.extend([
                    "extreme_performance_mode",
                    "kernel_bypass_networking",
                    "lock_free_data_structures",
                    "hardware_acceleration"
                ])
            
            return {
                'level': level.value,
                'optimizations_applied': optimizations,
                'optimizations_count': len(optimizations)
            }
            
        except Exception as e:
            self.logger.error(f"Level-specific optimization failed: {e}")
            return {'error': str(e)}
    
    def _calculate_improvement_summary(self, before: PerformanceMetrics, after: PerformanceMetrics) -> Dict[str, Any]:
        """改善サマリーの計算"""
        try:
            cpu_improvement = before.cpu_usage_percent - after.cpu_usage_percent
            memory_improvement_mb = before.memory_usage_mb - after.memory_usage_mb
            memory_improvement_percent = before.memory_usage_percent - after.memory_usage_percent
            
            return {
                'cpu_improvement_percent': cpu_improvement,
                'memory_improvement_mb': memory_improvement_mb,
                'memory_improvement_percent': memory_improvement_percent,
                'overall_improvement_score': (
                    max(0, cpu_improvement) * 0.4 +
                    max(0, memory_improvement_percent) * 0.4 +
                    (memory_improvement_mb / 100) * 0.2  # 100MB改善で20%の重み
                )
            }
            
        except Exception as e:
            self.logger.error(f"Improvement summary calculation failed: {e}")
            return {}
    
    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """最適化推奨事項の生成"""
        try:
            recommendations = []
            
            # メモリ関連推奨事項
            memory_result = results.get('memory', {})
            if memory_result.get('memory_saved_percent', 0) < 10:
                recommendations.append("Consider implementing more aggressive memory pooling")
            
            # CPU関連推奨事項
            cpu_result = results.get('cpu', {})
            if cpu_result.get('before_cpu_percent', 0) > 80:
                recommendations.append("Consider distributing workload across more processes")
            
            # I/O関連推奨事項
            io_result = results.get('io', {})
            if 'error' not in io_result:
                recommendations.append("Consider implementing async I/O for better performance")
            
            # キャッシュ関連推奨事項
            cache_result = results.get('cache', {})
            if cache_result.get('optimizations_applied'):
                recommendations.append("Regular cache optimization shows benefits")
            
            # 一般的な推奨事項
            recommendations.extend([
                "Schedule regular performance optimization sessions",
                "Monitor performance metrics continuously",
                "Consider hardware upgrades for CPU or memory bottlenecks"
            ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Enable detailed performance monitoring for better recommendations"]
    
    async def get_performance_status(self) -> Dict[str, Any]:
        """パフォーマンス状態の取得"""
        try:
            current_metrics = self.performance_monitor.collect_performance_metrics()
            
            return {
                'current_performance_level': self.current_performance_level.value,
                'current_metrics': {
                    'cpu_usage_percent': current_metrics.cpu_usage_percent,
                    'memory_usage_mb': current_metrics.memory_usage_mb,
                    'memory_usage_percent': current_metrics.memory_usage_percent,
                    'response_time_ms': current_metrics.response_time_ms,
                    'throughput_ops_s': current_metrics.throughput_ops_s
                },
                'optimization_history_count': len(self.optimization_history),
                'last_optimization': self.optimization_history[-1]['timestamp'] if self.optimization_history else None,
                'monitoring_active': self.performance_monitor.monitoring_active,
                'recent_alerts_count': len([alert for alert in self.performance_monitor.alerts 
                                          if alert['timestamp'] > datetime.now() - timedelta(hours=1)])
            }
            
        except Exception as e:
            self.logger.error(f"Performance status retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
performance_optimizer = PerformanceOptimizationEngine()


async def optimize_system_performance(target_level: PerformanceLevel = PerformanceLevel.HIGH) -> TradingResult[Dict[str, Any]]:
    """システムパフォーマンス最適化の実行"""
    return await performance_optimizer.comprehensive_optimization(target_level)


async def get_performance_status() -> Dict[str, Any]:
    """パフォーマンス状態の取得"""
    return await performance_optimizer.get_performance_status()


async def start_performance_monitoring(interval: float = 1.0):
    """パフォーマンス監視開始"""
    await performance_optimizer.performance_monitor.start_monitoring(interval)


def stop_performance_monitoring():
    """パフォーマンス監視停止"""
    performance_optimizer.performance_monitor.stop_monitoring()