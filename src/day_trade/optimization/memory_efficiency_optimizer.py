#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Efficiency Optimization System
メモリ効率化最適化システム - 第4世代メモリ管理技術
"""

import gc
import sys
import tracemalloc
import psutil
import threading
import weakref
from typing import Dict, List, Any, Optional, Callable, Union, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import mmap
import contextlib
from functools import wraps, lru_cache
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """メモリメトリクス"""
    timestamp: datetime
    total_memory: float  # MB
    used_memory: float   # MB
    free_memory: float   # MB
    cached_memory: float # MB
    process_memory: float # MB
    gc_collections: int
    large_objects_count: int
    memory_leaks: List[str]


class ObjectPool:
    """オブジェクトプール"""
    
    def __init__(self, factory: Callable, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def acquire(self):
        """オブジェクト取得"""
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            return self.factory()
            
    def release(self, obj):
        """オブジェクト返却"""
        with self.lock:
            if len(self.pool) < self.max_size:
                # オブジェクトをクリア
                if hasattr(obj, 'clear'):
                    obj.clear()
                elif hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
                
    @contextlib.contextmanager
    def get_object(self):
        """コンテキストマネージャー"""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)


class MemoryMappedArray:
    """メモリマップ配列"""
    
    def __init__(self, filename: str, shape: tuple, dtype=np.float32, mode='w+'):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.array = None
        self.file = None
        self.mmap = None
        
    def __enter__(self):
        """コンテキスト開始"""
        # ファイルサイズ計算
        itemsize = np.dtype(self.dtype).itemsize
        size = int(np.prod(self.shape) * itemsize)
        
        # ファイル作成・オープン
        self.file = open(self.filename, 'r+b' if self.mode == 'r+' else 'w+b')
        if self.mode == 'w+':
            self.file.write(b'\x00' * size)
            self.file.flush()
            
        # メモリマップ作成
        self.mmap = mmap.mmap(self.file.fileno(), size)
        
        # NumPy配列作成
        self.array = np.frombuffer(self.mmap, dtype=self.dtype).reshape(self.shape)
        
        return self.array
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了"""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()


class SmartCache:
    """インテリジェントキャッシュ"""
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory = max_memory_mb * 1024 * 1024  # bytes
        self.cache = {}
        self.access_times = {}
        self.memory_usage = {}
        self.lock = threading.RLock()
        
    def get(self, key: str, default=None):
        """キャッシュ取得"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                return self.cache[key]
            return default
            
    def set(self, key: str, value: Any):
        """キャッシュ設定"""
        with self.lock:
            # メモリ使用量計算
            try:
                memory_size = sys.getsizeof(pickle.dumps(value))
            except:
                memory_size = sys.getsizeof(value)
                
            # メモリ制限チェック
            current_memory = sum(self.memory_usage.values())
            
            if current_memory + memory_size > self.max_memory:
                self._evict_lru(memory_size)
                
            # キャッシュ設定
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            self.memory_usage[key] = memory_size
            
    def _evict_lru(self, required_memory: int):
        """LRU退避"""
        # アクセス時間順にソート
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        freed_memory = 0
        for key in sorted_keys:
            if freed_memory >= required_memory:
                break
                
            freed_memory += self.memory_usage[key]
            del self.cache[key]
            del self.access_times[key]
            del self.memory_usage[key]
            
    def clear(self):
        """キャッシュクリア"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.memory_usage.clear()


class MemoryProfiler:
    """メモリプロファイラ"""
    
    def __init__(self):
        self.snapshots = []
        self.is_tracing = False
        
    def start_tracing(self):
        """トレース開始"""
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
            
    def stop_tracing(self):
        """トレース停止"""
        if self.is_tracing:
            tracemalloc.stop()
            self.is_tracing = False
            
    def take_snapshot(self, name: str = None):
        """スナップショット取得"""
        if self.is_tracing:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append({
                'name': name or f"snapshot_{len(self.snapshots)}",
                'timestamp': datetime.now(),
                'snapshot': snapshot
            })
            return snapshot
        return None
        
    def compare_snapshots(self, snapshot1_idx: int = -2, snapshot2_idx: int = -1):
        """スナップショット比較"""
        if len(self.snapshots) < 2:
            return None
            
        snapshot1 = self.snapshots[snapshot1_idx]['snapshot']
        snapshot2 = self.snapshots[snapshot2_idx]['snapshot']
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        memory_diff = []
        for stat in top_stats[:10]:  # 上位10件
            memory_diff.append({
                'filename': stat.traceback.format()[0],
                'size_diff': stat.size_diff,
                'count_diff': stat.count_diff,
                'size': stat.size,
                'count': stat.count
            })
            
        return memory_diff
        
    def get_top_memory_consumers(self, limit: int = 10):
        """メモリ使用量上位"""
        if not self.snapshots:
            return []
            
        snapshot = self.snapshots[-1]['snapshot']
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for stat in top_stats[:limit]:
            consumers.append({
                'filename': stat.traceback.format()[0],
                'size': stat.size,
                'count': stat.count
            })
            
        return consumers


class DataFrameOptimizer:
    """DataFrame最適化"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, downcast_integers: bool = True) -> pd.DataFrame:
        """データ型最適化"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # 数値型最適化
            if col_type in ['int64', 'int32']:
                if downcast_integers:
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            elif col_type in ['float64']:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                
            # カテゴリ型変換（重複の多い文字列）
            elif col_type == 'object':
                unique_ratio = len(optimized_df[col].unique()) / len(optimized_df[col])
                if unique_ratio < 0.5:  # 50%未満がユニークの場合
                    optimized_df[col] = optimized_df[col].astype('category')
                    
        return optimized_df
        
    @staticmethod
    def chunked_processing(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """チャンク処理"""
        for start in range(0, len(df), chunk_size):
            yield df[start:start + chunk_size]
            
    @staticmethod
    def memory_usage_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """メモリ使用量分析"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_memory_mb': total_memory / 1024 / 1024,
            'column_memory': {
                col: usage / 1024 / 1024 
                for col, usage in memory_usage.items()
            },
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict()
        }


class MemoryEfficiencyOptimizer:
    """メモリ効率化最適化システム"""
    
    def __init__(self, max_cache_memory_mb: int = 500):
        self.smart_cache = SmartCache(max_cache_memory_mb)
        self.object_pools = {}
        self.profiler = MemoryProfiler()
        self.metrics_history: List[MemoryMetrics] = []
        
        # ガベージコレクション設定
        gc.set_threshold(700, 10, 10)
        
        # プロファイリング開始
        self.profiler.start_tracing()
        
    def memory_monitor(self, func: Callable = None):
        """メモリ監視デコレータ"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # 実行前スナップショット
                self.profiler.take_snapshot(f"before_{f.__name__}")
                
                # 実行前メモリ状況
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    # 実行後スナップショット
                    self.profiler.take_snapshot(f"after_{f.__name__}")
                    
                    # メモリ使用量チェック
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_diff = memory_after - memory_before
                    
                    if memory_diff > 100:  # 100MB以上増加
                        logger.warning(f"Function {f.__name__} increased memory by {memory_diff:.2f}MB")
                        self.analyze_memory_usage()
                        
            return wrapper
            
        if func is None:
            return decorator
        return decorator(func)
        
    def get_memory_pool(self, pool_name: str, factory: Callable, max_size: int = 100):
        """オブジェクトプール取得"""
        if pool_name not in self.object_pools:
            self.object_pools[pool_name] = ObjectPool(factory, max_size)
        return self.object_pools[pool_name]
        
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame メモリ最適化"""
        return DataFrameOptimizer.optimize_dtypes(df)
        
    def create_memory_mapped_array(self, filename: str, shape: tuple, 
                                 dtype=np.float32, mode='w+'):
        """メモリマップ配列作成"""
        return MemoryMappedArray(filename, shape, dtype, mode)
        
    def smart_caching(self, key: str, generator: Callable, 
                     cache_ttl: timedelta = timedelta(hours=1)):
        """スマートキャッシング"""
        cached_value = self.smart_cache.get(key)
        
        if cached_value is not None:
            # TTL チェック
            cache_time = cached_value.get('timestamp')
            if cache_time and datetime.now() - cache_time < cache_ttl:
                return cached_value['data']
                
        # キャッシュミス - データ生成
        data = generator()
        self.smart_cache.set(key, {
            'data': data,
            'timestamp': datetime.now()
        })
        
        return data
        
    def force_garbage_collection(self):
        """強制ガベージコレクション"""
        # 全世代のガベージコレクション実行
        collected = [gc.collect(i) for i in range(3)]
        
        # 弱参照のクリーンアップ
        weakref_count_before = len(gc.get_referrers(weakref.WeakSet))
        gc.collect()
        weakref_count_after = len(gc.get_referrers(weakref.WeakSet))
        
        logger.info(f"GC collected objects: {sum(collected)}")
        logger.info(f"WeakRef cleaned: {weakref_count_before - weakref_count_after}")
        
        return sum(collected)
        
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量分析"""
        # システムメモリ情報
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # ガベージコレクション統計
        gc_stats = gc.get_stats()
        
        # プロファイラ統計
        top_consumers = self.profiler.get_top_memory_consumers()
        
        # 大きなオブジェクト検出
        large_objects = []
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # 1MB以上
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': size / 1024 / 1024,
                        'id': id(obj)
                    })
            except:
                continue
                
        large_objects.sort(key=lambda x: x['size_mb'], reverse=True)
        
        analysis = {
            'system_memory': {
                'total_mb': system_memory.total / 1024 / 1024,
                'available_mb': system_memory.available / 1024 / 1024,
                'used_percent': system_memory.percent,
                'cached_mb': getattr(system_memory, 'cached', 0) / 1024 / 1024
            },
            'process_memory': {
                'rss_mb': process_memory.rss / 1024 / 1024,
                'vms_mb': process_memory.vms / 1024 / 1024
            },
            'gc_stats': gc_stats,
            'top_memory_consumers': top_consumers[:5],
            'large_objects': large_objects[:10],
            'cache_stats': {
                'cache_items': len(self.smart_cache.cache),
                'cache_memory_mb': sum(self.smart_cache.memory_usage.values()) / 1024 / 1024
            }
        }
        
        # メトリクス保存
        metrics = MemoryMetrics(
            timestamp=datetime.now(),
            total_memory=system_memory.total / 1024 / 1024,
            used_memory=(system_memory.total - system_memory.available) / 1024 / 1024,
            free_memory=system_memory.available / 1024 / 1024,
            cached_memory=getattr(system_memory, 'cached', 0) / 1024 / 1024,
            process_memory=process_memory.rss / 1024 / 1024,
            gc_collections=sum(stat['collections'] for stat in gc_stats),
            large_objects_count=len(large_objects),
            memory_leaks=[]
        )
        
        self.metrics_history.append(metrics)
        
        return analysis
        
    def detect_memory_leaks(self) -> List[str]:
        """メモリリーク検出"""
        if len(self.metrics_history) < 2:
            return []
            
        recent_metrics = self.metrics_history[-10:]  # 直近10回
        
        # プロセスメモリの増加傾向チェック
        memory_trend = [m.process_memory for m in recent_metrics]
        if len(memory_trend) > 3:
            # 単調増加をチェック
            increasing_count = sum(
                1 for i in range(1, len(memory_trend))
                if memory_trend[i] > memory_trend[i-1]
            )
            
            if increasing_count / (len(memory_trend) - 1) > 0.8:  # 80%以上増加
                return ["Potential memory leak detected: Process memory continuously increasing"]
                
        # 大きなオブジェクトの増加チェック
        large_obj_trend = [m.large_objects_count for m in recent_metrics]
        if len(large_obj_trend) > 3:
            avg_increase = sum(
                large_obj_trend[i] - large_obj_trend[i-1]
                for i in range(1, len(large_obj_trend))
            ) / (len(large_obj_trend) - 1)
            
            if avg_increase > 5:  # 平均5個以上増加
                return ["Potential memory leak: Large objects count increasing"]
                
        return []
        
    def optimize_memory_usage(self):
        """メモリ使用量最適化"""
        # ガベージコレクション実行
        collected = self.force_garbage_collection()
        
        # キャッシュクリーンアップ
        self.smart_cache.clear()
        
        # オブジェクトプールクリーンアップ
        for pool in self.object_pools.values():
            pool.pool.clear()
            
        # メモリ分析実行
        analysis = self.analyze_memory_usage()
        
        return {
            'gc_collected': collected,
            'memory_analysis': analysis,
            'optimization_timestamp': datetime.now()
        }
        
    def get_memory_metrics(self) -> Dict[str, Any]:
        """メモリメトリクス取得"""
        if not self.metrics_history:
            return {"message": "No metrics available"}
            
        recent_metrics = self.metrics_history[-10:]
        
        avg_memory = sum(m.process_memory for m in recent_metrics) / len(recent_metrics)
        memory_trend = "increasing" if recent_metrics[-1].process_memory > recent_metrics[0].process_memory else "stable"
        
        return {
            'current_memory_mb': recent_metrics[-1].process_memory,
            'avg_memory_mb': avg_memory,
            'memory_trend': memory_trend,
            'gc_collections': recent_metrics[-1].gc_collections,
            'large_objects': recent_metrics[-1].large_objects_count,
            'cache_memory_mb': sum(self.smart_cache.memory_usage.values()) / 1024 / 1024,
            'potential_leaks': self.detect_memory_leaks()
        }


# グローバルメモリオプティマイザ
memory_optimizer = MemoryEfficiencyOptimizer()


def memory_efficient(func: Callable = None):
    """メモリ効率化デコレータ"""
    return memory_optimizer.memory_monitor(func)


def get_memory_pool(pool_name: str, factory: Callable, max_size: int = 100):
    """メモリプール取得ヘルパー"""
    return memory_optimizer.get_memory_pool(pool_name, factory, max_size)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameメモリ最適化ヘルパー"""
    return memory_optimizer.optimize_dataframe_memory(df)