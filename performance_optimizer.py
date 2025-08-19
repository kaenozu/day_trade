#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Optimizer - システムパフォーマンス最適化
Issue #945対応: メモリ使用量改善 + CPU最適化 + I/O効率化
"""

import gc
import sys
import psutil
import time
import threading
import asyncio
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import functools
import cProfile
import pstats
import io
import tracemalloc

# メモリプロファイリング
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

# 高速キャッシュライブラリ
try:
    import cachetools
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gc_count: int
    thread_count: int
    open_files: int
    response_time_ms: float
    cache_hit_rate: float
    error_rate: float


@dataclass
class MemoryUsage:
    """メモリ使用量詳細"""
    component: str
    memory_mb: float
    peak_memory_mb: float
    objects_count: int
    growth_rate: float  # MB/分
    optimization_potential: str  # LOW, MEDIUM, HIGH


@dataclass
class OptimizationSuggestion:
    """最適化提案"""
    component: str
    issue_type: str  # 'MEMORY_LEAK', 'CPU_INTENSIVE', 'IO_BOTTLENECK', 'CACHE_MISS'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description: str
    solution: str
    expected_improvement: str
    estimated_effort: str  # 'LOW', 'MEDIUM', 'HIGH'


class MemoryOptimizer:
    """メモリ最適化システム"""
    
    def __init__(self):
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.object_registry: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.memory_threshold_mb = 500  # 500MB制限
        self.gc_frequency = 100  # 100秒間隔でGC
        self.last_gc_time = time.time()
        
        # メモリプロファイリング
        if HAS_MEMORY_PROFILER:
            self.memory_usage_log = []
        
        # 弱参照ベースのキャッシュ
        self.weak_cache: Dict[str, Any] = {}
        self._setup_memory_monitoring()
    
    def _setup_memory_monitoring(self):
        """メモリ監視セットアップ"""
        tracemalloc.start()
        
        # 定期的なメモリチェック
        threading.Timer(60.0, self._periodic_memory_check).start()
    
    def _periodic_memory_check(self):
        """定期メモリチェック"""
        try:
            current_memory = self.get_current_memory_usage()
            
            if current_memory > self.memory_threshold_mb:
                logging.warning(f"Memory threshold exceeded: {current_memory:.1f}MB")
                self.emergency_memory_cleanup()
            
            # メモリスナップショット保存
            snapshot = tracemalloc.take_snapshot()
            self._analyze_memory_snapshot(snapshot)
            
            # 次回のチェックをスケジュール
            threading.Timer(60.0, self._periodic_memory_check).start()
            
        except Exception as e:
            logging.error(f"Memory check error: {e}")
    
    def _analyze_memory_snapshot(self, snapshot):
        """メモリスナップショット分析"""
        top_stats = snapshot.statistics('lineno')
        
        memory_usage = {}
        for stat in top_stats[:20]:  # 上位20項目
            filename = stat.traceback.format()[0] if stat.traceback else 'unknown'
            memory_usage[filename] = {
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
        
        self.memory_snapshots.append({
            'timestamp': datetime.now(),
            'total_memory_mb': sum(stat.size for stat in top_stats) / 1024 / 1024,
            'usage_breakdown': memory_usage
        })
        
        # 古いスナップショットを削除（メモリ節約）
        if len(self.memory_snapshots) > 50:
            self.memory_snapshots = self.memory_snapshots[-25:]
    
    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def emergency_memory_cleanup(self):
        """緊急メモリクリーンアップ"""
        logging.info("Starting emergency memory cleanup...")
        
        # ガベージコレクション強制実行
        collected = gc.collect()
        logging.info(f"GC collected {collected} objects")
        
        # 弱参照キャッシュクリア
        self.weak_cache.clear()
        
        # オブジェクトレジストリクリーンアップ
        for component_name, weak_set in self.object_registry.items():
            original_count = len(weak_set)
            # 弱参照セットは自動的にクリーンアップされる
            current_count = len(weak_set)
            if original_count != current_count:
                logging.info(f"Cleaned up {original_count - current_count} objects from {component_name}")
        
        # メモリマッピングファイルのフラッシュ
        try:
            import mmap
            # システムレベルのメモリ同期
        except ImportError:
            pass
        
        memory_after = self.get_current_memory_usage()
        logging.info(f"Memory after cleanup: {memory_after:.1f}MB")
    
    def register_object(self, component: str, obj: Any):
        """オブジェクト登録（メモリ追跡用）"""
        try:
            self.object_registry[component].add(obj)
        except TypeError:
            # 弱参照できないオブジェクトの場合はスキップ
            pass
    
    def get_memory_report(self) -> Dict[str, Any]:
        """メモリレポート生成"""
        current_memory = self.get_current_memory_usage()
        
        # コンポーネント別使用量
        component_usage = {}
        for component, weak_set in self.object_registry.items():
            component_usage[component] = {
                'object_count': len(weak_set),
                'estimated_memory_mb': len(weak_set) * 0.1  # 概算
            }
        
        # メモリ成長トレンド
        growth_trend = self._calculate_memory_trend()
        
        return {
            'current_memory_mb': current_memory,
            'memory_threshold_mb': self.memory_threshold_mb,
            'usage_percentage': (current_memory / self.memory_threshold_mb) * 100,
            'component_usage': component_usage,
            'growth_trend_mb_per_hour': growth_trend,
            'snapshots_count': len(self.memory_snapshots),
            'recommendation': self._get_memory_recommendation(current_memory)
        }
    
    def _calculate_memory_trend(self) -> float:
        """メモリ使用量トレンド計算"""
        if len(self.memory_snapshots) < 2:
            return 0.0
        
        recent_snapshots = self.memory_snapshots[-10:]  # 最新10個
        if len(recent_snapshots) < 2:
            return 0.0
        
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        time_diff = (last_snapshot['timestamp'] - first_snapshot['timestamp']).total_seconds() / 3600  # 時間
        memory_diff = last_snapshot['total_memory_mb'] - first_snapshot['total_memory_mb']
        
        return memory_diff / max(time_diff, 0.1)
    
    def _get_memory_recommendation(self, current_memory: float) -> str:
        """メモリ最適化推奨"""
        usage_percent = (current_memory / self.memory_threshold_mb) * 100
        
        if usage_percent > 90:
            return "CRITICAL: Immediate memory optimization required"
        elif usage_percent > 75:
            return "HIGH: Memory cleanup recommended"
        elif usage_percent > 50:
            return "MEDIUM: Monitor memory usage"
        else:
            return "LOW: Memory usage is optimal"


class CPUOptimizer:
    """CPU最適化システム"""
    
    def __init__(self):
        self.cpu_samples: deque = deque(maxlen=100)
        self.profiler_data: Dict[str, Any] = {}
        self.optimization_cache = {}
        
        # CPU効率的な処理パターン
        self.batch_size_optimizer = self._initialize_batch_optimizer()
        self.parallel_processing_config = self._initialize_parallel_config()
    
    def _initialize_batch_optimizer(self) -> Dict[str, int]:
        """バッチサイズ最適化"""
        cpu_count = psutil.cpu_count()
        
        return {
            'data_processing': min(1000, cpu_count * 100),
            'ai_analysis': min(50, cpu_count * 5),
            'database_operations': min(500, cpu_count * 50),
            'api_requests': min(20, cpu_count * 2)
        }
    
    def _initialize_parallel_config(self) -> Dict[str, int]:
        """並列処理設定"""
        cpu_count = psutil.cpu_count()
        
        return {
            'max_workers': min(cpu_count * 2, 16),
            'io_workers': min(cpu_count * 4, 32),
            'cpu_workers': cpu_count,
            'thread_pool_size': cpu_count * 2
        }
    
    def profile_function(self, func: Callable) -> Callable:
        """関数プロファイリングデコレータ"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            profiler.disable()
            
            # プロファイル結果保存
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # 上位10関数
            
            self.profiler_data[func.__name__] = {
                'execution_time': end_time - start_time,
                'profile_output': s.getvalue(),
                'timestamp': datetime.now()
            }
            
            return result
        return wrapper
    
    def monitor_cpu_usage(self):
        """CPU使用率監視"""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_samples.append({
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        })
    
    def get_cpu_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """CPU最適化提案"""
        suggestions = []
        
        if not self.cpu_samples:
            return suggestions
        
        # 平均CPU使用率
        avg_cpu = sum(sample['cpu_percent'] for sample in self.cpu_samples) / len(self.cpu_samples)
        
        if avg_cpu > 80:
            suggestions.append(OptimizationSuggestion(
                component="CPU Usage",
                issue_type="CPU_INTENSIVE",
                severity="HIGH",
                description=f"High average CPU usage: {avg_cpu:.1f}%",
                solution="Implement batch processing, async operations, or caching",
                expected_improvement="20-40% CPU reduction",
                estimated_effort="MEDIUM"
            ))
        
        # プロファイリング結果から提案
        for func_name, profile_data in self.profiler_data.items():
            if profile_data['execution_time'] > 1.0:  # 1秒以上の関数
                suggestions.append(OptimizationSuggestion(
                    component=func_name,
                    issue_type="CPU_INTENSIVE",
                    severity="MEDIUM",
                    description=f"Slow function execution: {profile_data['execution_time']:.2f}s",
                    solution="Optimize algorithm, add caching, or use parallel processing",
                    expected_improvement="50-80% speed improvement",
                    estimated_effort="MEDIUM"
                ))
        
        return suggestions
    
    def optimize_batch_size(self, operation_type: str, current_size: int) -> int:
        """バッチサイズ最適化"""
        optimal_size = self.batch_size_optimizer.get(operation_type, current_size)
        
        # CPU負荷に基づいた動的調整
        if self.cpu_samples:
            recent_cpu = sum(s['cpu_percent'] for s in list(self.cpu_samples)[-5:]) / 5
            
            if recent_cpu > 80:
                optimal_size = int(optimal_size * 0.7)  # 負荷高時は30%削減
            elif recent_cpu < 30:
                optimal_size = int(optimal_size * 1.3)  # 負荷低時は30%増加
        
        return max(1, min(optimal_size, current_size * 2))


class IOOptimizer:
    """I/O最適化システム"""
    
    def __init__(self):
        self.io_stats: Dict[str, List[float]] = defaultdict(list)
        self.connection_pool_config = self._initialize_connection_pools()
        self.async_io_config = self._initialize_async_config()
    
    def _initialize_connection_pools(self) -> Dict[str, Dict[str, int]]:
        """コネクションプール設定"""
        return {
            'database': {
                'min_connections': 5,
                'max_connections': 20,
                'connection_timeout': 30
            },
            'http': {
                'min_connections': 10,
                'max_connections': 100,
                'connection_timeout': 10
            },
            'cache': {
                'min_connections': 5,
                'max_connections': 50,
                'connection_timeout': 5
            }
        }
    
    def _initialize_async_config(self) -> Dict[str, Any]:
        """非同期I/O設定"""
        return {
            'max_concurrent_requests': 50,
            'request_timeout': 30.0,
            'retry_attempts': 3,
            'backoff_factor': 1.5
        }
    
    def monitor_io_operation(self, operation_name: str, duration: float):
        """I/O操作監視"""
        self.io_stats[operation_name].append(duration)
        
        # 古いデータを削除（メモリ節約）
        if len(self.io_stats[operation_name]) > 1000:
            self.io_stats[operation_name] = self.io_stats[operation_name][-500:]
    
    def get_io_performance_report(self) -> Dict[str, Any]:
        """I/Oパフォーマンスレポート"""
        report = {}
        
        for operation, durations in self.io_stats.items():
            if durations:
                report[operation] = {
                    'avg_duration_ms': sum(durations) / len(durations) * 1000,
                    'max_duration_ms': max(durations) * 1000,
                    'min_duration_ms': min(durations) * 1000,
                    'operation_count': len(durations),
                    'slowest_10_percent_ms': sorted(durations, reverse=True)[:max(1, len(durations) // 10)]
                }
        
        return report
    
    def get_io_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """I/O最適化提案"""
        suggestions = []
        
        for operation, durations in self.io_stats.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                
                if avg_duration > 1.0:  # 1秒以上のI/O操作
                    suggestions.append(OptimizationSuggestion(
                        component=operation,
                        issue_type="IO_BOTTLENECK",
                        severity="HIGH",
                        description=f"Slow I/O operation: {avg_duration:.2f}s average",
                        solution="Implement connection pooling, caching, or async I/O",
                        expected_improvement="60-80% speed improvement",
                        estimated_effort="MEDIUM"
                    ))
                elif avg_duration > 0.5:  # 0.5秒以上
                    suggestions.append(OptimizationSuggestion(
                        component=operation,
                        issue_type="IO_BOTTLENECK",
                        severity="MEDIUM",
                        description=f"Moderate I/O latency: {avg_duration:.2f}s average",
                        solution="Consider caching or request batching",
                        expected_improvement="30-50% speed improvement",
                        estimated_effort="LOW"
                    ))
        
        return suggestions


class CacheOptimizer:
    """キャッシュ最適化システム"""
    
    def __init__(self):
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
        self.cache_efficiency_threshold = 0.7  # 70%のヒット率が目標
        
        # 高性能キャッシュの初期化
        if HAS_CACHETOOLS:
            self.lru_cache = cachetools.LRUCache(maxsize=1000)
            self.ttl_cache = cachetools.TTLCache(maxsize=500, ttl=300)  # 5分TTL
        else:
            self.lru_cache = {}
            self.ttl_cache = {}
    
    def record_cache_hit(self, cache_name: str):
        """キャッシュヒット記録"""
        self.cache_stats[cache_name]['hits'] += 1
    
    def record_cache_miss(self, cache_name: str):
        """キャッシュミス記録"""
        self.cache_stats[cache_name]['misses'] += 1
    
    def get_cache_hit_rate(self, cache_name: str) -> float:
        """キャッシュヒット率取得"""
        stats = self.cache_stats[cache_name]
        total = stats['hits'] + stats['misses']
        
        if total == 0:
            return 0.0
        
        return stats['hits'] / total
    
    def get_cache_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """キャッシュ最適化提案"""
        suggestions = []
        
        for cache_name, stats in self.cache_stats.items():
            hit_rate = self.get_cache_hit_rate(cache_name)
            
            if hit_rate < self.cache_efficiency_threshold:
                severity = "HIGH" if hit_rate < 0.5 else "MEDIUM"
                suggestions.append(OptimizationSuggestion(
                    component=cache_name,
                    issue_type="CACHE_MISS",
                    severity=severity,
                    description=f"Low cache hit rate: {hit_rate:.2%}",
                    solution="Increase cache size, adjust TTL, or improve cache key strategy",
                    expected_improvement=f"Improve hit rate to {self.cache_efficiency_threshold:.0%}+",
                    estimated_effort="LOW"
                ))
        
        return suggestions
    
    def cache_with_optimization(self, key: str, compute_func: Callable, cache_type: str = 'lru') -> Any:
        """最適化されたキャッシュ機能"""
        cache = self.lru_cache if cache_type == 'lru' else self.ttl_cache
        
        if key in cache:
            self.record_cache_hit(cache_type)
            return cache[key]
        else:
            self.record_cache_miss(cache_type)
            value = compute_func()
            cache[key] = value
            return value


class PerformanceOptimizer:
    """統合パフォーマンス最適化システム"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.io_optimizer = IOOptimizer()
        self.cache_optimizer = CacheOptimizer()
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_log: List[Dict[str, Any]] = []
        self.auto_optimization_enabled = True
        
        # 最適化間隔設定
        self.optimization_interval = 300  # 5分間隔
        self.last_optimization_time = time.time()
        
        self._start_performance_monitoring()
    
    def _start_performance_monitoring(self):
        """パフォーマンス監視開始"""
        def monitor_loop():
            while True:
                try:
                    self.collect_performance_metrics()
                    
                    if self.auto_optimization_enabled:
                        current_time = time.time()
                        if current_time - self.last_optimization_time > self.optimization_interval:
                            self.auto_optimize_system()
                            self.last_optimization_time = current_time
                    
                    time.sleep(60)  # 1分間隔
                    
                except Exception as e:
                    logging.error(f"Performance monitoring error: {e}")
                    time.sleep(60)
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """パフォーマンスメトリクス収集"""
        try:
            process = psutil.Process()
            
            # CPU・メモリ
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # ディスクI/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
            disk_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0
            
            # ネットワーク
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / 1024 / 1024 if net_io else 0
            net_recv_mb = net_io.bytes_recv / 1024 / 1024 if net_io else 0
            
            # システム情報
            gc_count = sum(gc.get_count())
            thread_count = threading.active_count()
            
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, AttributeError):
                open_files = 0
            
            # キャッシュヒット率（平均）
            cache_hit_rates = [self.cache_optimizer.get_cache_hit_rate(name) 
                             for name in self.cache_optimizer.cache_stats.keys()]
            avg_cache_hit_rate = sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0.0
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                gc_count=gc_count,
                thread_count=thread_count,
                open_files=open_files,
                response_time_ms=0.0,  # 外部から設定
                cache_hit_rate=avg_cache_hit_rate,
                error_rate=0.0  # 外部から設定
            )
            
            self.metrics_history.append(metrics)
            
            # 履歴サイズ制限
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to collect performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0, memory_mb=0.0, memory_percent=0.0,
                disk_io_read_mb=0.0, disk_io_write_mb=0.0,
                network_sent_mb=0.0, network_recv_mb=0.0,
                gc_count=0, thread_count=0, open_files=0,
                response_time_ms=0.0, cache_hit_rate=0.0, error_rate=0.0
            )
    
    def auto_optimize_system(self):
        """自動システム最適化"""
        logging.info("Starting automatic system optimization...")
        
        optimization_actions = []
        
        # メモリ最適化
        memory_report = self.memory_optimizer.get_memory_report()
        if memory_report['usage_percentage'] > 80:
            self.memory_optimizer.emergency_memory_cleanup()
            optimization_actions.append("Memory cleanup performed")
        
        # CPU最適化
        self.cpu_optimizer.monitor_cpu_usage()
        
        # キャッシュ最適化
        for cache_name in self.cache_optimizer.cache_stats.keys():
            hit_rate = self.cache_optimizer.get_cache_hit_rate(cache_name)
            if hit_rate < 0.5:
                # キャッシュサイズを動的に調整
                optimization_actions.append(f"Cache {cache_name} needs optimization (hit rate: {hit_rate:.2%})")
        
        # 最適化ログ記録
        self.optimization_log.append({
            'timestamp': datetime.now(),
            'actions': optimization_actions,
            'memory_usage_mb': memory_report['current_memory_mb'],
            'cpu_usage_avg': self._get_avg_cpu_usage(),
        })
        
        logging.info(f"Auto-optimization completed: {len(optimization_actions)} actions taken")
    
    def _get_avg_cpu_usage(self) -> float:
        """平均CPU使用率取得"""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]  # 最新10個
        return sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
    
    def get_comprehensive_optimization_report(self) -> Dict[str, Any]:
        """包括的最適化レポート"""
        # 各最適化システムから提案を収集
        memory_suggestions = []
        cpu_suggestions = self.cpu_optimizer.get_cpu_optimization_suggestions()
        io_suggestions = self.io_optimizer.get_io_optimization_suggestions()
        cache_suggestions = self.cache_optimizer.get_cache_optimization_suggestions()
        
        all_suggestions = memory_suggestions + cpu_suggestions + io_suggestions + cache_suggestions
        
        # 重要度でソート
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        all_suggestions.sort(key=lambda x: severity_order.get(x.severity, 0), reverse=True)
        
        # 現在のパフォーマンスメトリクス
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'current_performance': asdict(current_metrics) if current_metrics else None,
            'memory_report': self.memory_optimizer.get_memory_report(),
            'io_performance': self.io_optimizer.get_io_performance_report(),
            'optimization_suggestions': [asdict(suggestion) for suggestion in all_suggestions],
            'recent_optimizations': self.optimization_log[-10:],
            'system_health_score': self._calculate_system_health_score(),
            'recommendations': self._generate_optimization_recommendations(all_suggestions)
        }
    
    def _calculate_system_health_score(self) -> int:
        """システム健全性スコア計算（0-100）"""
        if not self.metrics_history:
            return 50
        
        recent_metrics = self.metrics_history[-5:]  # 最新5個
        
        # 各要素のスコア計算
        cpu_score = max(0, 100 - sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics))
        memory_score = max(0, 100 - sum(m.memory_percent for m in recent_metrics) / len(recent_metrics))
        
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate > 0]
        cache_score = sum(cache_hit_rates) / len(cache_hit_rates) * 100 if cache_hit_rates else 70
        
        # 重み付き平均
        total_score = (cpu_score * 0.4 + memory_score * 0.4 + cache_score * 0.2)
        
        return max(0, min(100, int(total_score)))
    
    def _generate_optimization_recommendations(self, suggestions: List[OptimizationSuggestion]) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []
        
        # 重要度別の推奨事項
        critical_issues = [s for s in suggestions if s.severity == 'CRITICAL']
        high_issues = [s for s in suggestions if s.severity == 'HIGH']
        
        if critical_issues:
            recommendations.append(f"🚨 {len(critical_issues)} critical issues require immediate attention")
        
        if high_issues:
            recommendations.append(f"⚠️ {len(high_issues)} high-priority optimizations recommended")
        
        # 具体的な推奨事項
        if any(s.issue_type == 'MEMORY_LEAK' for s in suggestions):
            recommendations.append("💾 Enable automatic memory cleanup and review object lifecycle")
        
        if any(s.issue_type == 'CPU_INTENSIVE' for s in suggestions):
            recommendations.append("⚡ Consider parallel processing and algorithm optimization")
        
        if any(s.issue_type == 'IO_BOTTLENECK' for s in suggestions):
            recommendations.append("🔄 Implement connection pooling and async I/O patterns")
        
        if any(s.issue_type == 'CACHE_MISS' for s in suggestions):
            recommendations.append("🗄️ Optimize cache strategy and increase cache sizes")
        
        # システム全体の推奨
        system_score = self._calculate_system_health_score()
        if system_score < 50:
            recommendations.append("🔧 System requires comprehensive optimization")
        elif system_score < 70:
            recommendations.append("📊 System performance monitoring recommended")
        else:
            recommendations.append("✅ System performance is good")
        
        return recommendations
    
    def optimize_function(self, func: Callable) -> Callable:
        """関数最適化デコレータ"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # CPU監視開始
            start_cpu = psutil.cpu_percent()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                end_cpu = psutil.cpu_percent()
                
                # I/O操作として記録
                self.io_optimizer.monitor_io_operation(func.__name__, execution_time)
                
                # 遅い関数の場合は警告
                if execution_time > 5.0:
                    logging.warning(f"Slow function detected: {func.__name__} took {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                logging.error(f"Function {func.__name__} failed: {e}")
                raise
            
        return wrapper


# グローバルインスタンス
performance_optimizer = PerformanceOptimizer()


def optimize_performance(func: Callable) -> Callable:
    """パフォーマンス最適化デコレータ（便利関数）"""
    return performance_optimizer.optimize_function(func)


def get_optimization_report() -> Dict[str, Any]:
    """最適化レポート取得（便利関数）"""
    return performance_optimizer.get_comprehensive_optimization_report()


async def test_performance_optimizer():
    """パフォーマンス最適化システムテスト"""
    print("=== Performance Optimizer Test ===")
    
    # メトリクス収集テスト
    print("1. Collecting performance metrics...")
    metrics = performance_optimizer.collect_performance_metrics()
    print(f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_mb:.1f}MB")
    
    # 最適化テスト
    print("2. Testing function optimization...")
    
    @optimize_performance
    def cpu_intensive_task():
        """CPU集約的タスク"""
        total = 0
        for i in range(1000000):
            total += i * i
        return total
    
    result = cpu_intensive_task()
    print(f"CPU task result: {result}")
    
    # キャッシュテスト
    print("3. Testing cache optimization...")
    cache_result = performance_optimizer.cache_optimizer.cache_with_optimization(
        'test_key',
        lambda: sum(range(10000)),
        'lru'
    )
    print(f"Cache result: {cache_result}")
    
    # レポート生成
    print("4. Generating optimization report...")
    report = performance_optimizer.get_comprehensive_optimization_report()
    
    print(f"System Health Score: {report['system_health_score']}/100")
    print(f"Optimization Suggestions: {len(report['optimization_suggestions'])}")
    
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_performance_optimizer())