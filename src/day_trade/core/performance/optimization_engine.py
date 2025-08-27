"""
統一パフォーマンス最適化エンジン

システム全体のパフォーマンス監視と最適化を提供
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time
import psutil
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import cProfile
import pstats
import io

T = TypeVar("T")


class PerformanceMetricType(Enum):
    """パフォーマンスメトリクスタイプ"""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"


class OptimizationStrategy(Enum):
    """最適化戦略"""

    CACHING = "caching"
    PARALLEL_PROCESSING = "parallel_processing"
    LAZY_LOADING = "lazy_loading"
    MEMORY_POOLING = "memory_pooling"
    BATCH_PROCESSING = "batch_processing"
    ASYNC_PROCESSING = "async_processing"


@dataclass
class PerformanceMetric:
    """パフォーマンスメトリクス"""

    name: str
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """パフォーマンスベンチマーク"""

    operation_name: str
    target_response_time_ms: float
    target_throughput_ops: float
    max_memory_mb: float
    max_cpu_percent: float


@dataclass
class OptimizationResult:
    """最適化結果"""

    strategy: OptimizationStrategy
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    recommendation: str


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self._profiles: Dict[str, cProfile.Profile] = {}
        self._metrics: List[PerformanceMetric] = []
        self._benchmarks: Dict[str, PerformanceBenchmark] = {}

    def start_profiling(self, operation_name: str) -> None:
        """プロファイリング開始"""
        profile = cProfile.Profile()
        profile.enable()
        self._profiles[operation_name] = profile

    def stop_profiling(self, operation_name: str) -> Dict[str, Any]:
        """プロファイリング終了"""
        if operation_name not in self._profiles:
            return {}

        profile = self._profiles[operation_name]
        profile.disable()

        # 統計生成
        stats = pstats.Stats(profile)
        stats.sort_stats("cumulative")

        # 文字列バッファに出力
        s = io.StringIO()
        stats.print_stats(20)  # 上位20件

        del self._profiles[operation_name]

        return {
            "operation": operation_name,
            "profile_data": s.getvalue(),
            "total_calls": stats.total_calls,
            "total_time": stats.total_tt,
        }

    def add_metric(self, metric: PerformanceMetric) -> None:
        """メトリクス追加"""
        self._metrics.append(metric)

        # 古いメトリクスを削除（24時間以上古い）
        cutoff = datetime.now() - timedelta(hours=24)
        self._metrics = [m for m in self._metrics if m.timestamp > cutoff]

    def get_metrics(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        metric_type: Optional[PerformanceMetricType] = None,
        hours: int = 1,
    ) -> List[PerformanceMetric]:
        """メトリクス取得"""
        cutoff = datetime.now() - timedelta(hours=hours)

        filtered = [m for m in self._metrics if m.timestamp > cutoff]

        if component:
            filtered = [m for m in filtered if m.component == component]
        if operation:
            filtered = [m for m in filtered if m.operation == operation]
        if metric_type:
            filtered = [m for m in filtered if m.metric_type == metric_type]

        return filtered

    def set_benchmark(self, benchmark: PerformanceBenchmark) -> None:
        """ベンチマーク設定"""
        self._benchmarks[benchmark.operation_name] = benchmark

    def check_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """ベンチマークチェック"""
        results = {}

        for operation_name, benchmark in self._benchmarks.items():
            metrics = self.get_metrics(operation=operation_name, hours=1)

            if not metrics:
                continue

            # 平均値計算
            response_times = [
                m.value
                for m in metrics
                if m.metric_type == PerformanceMetricType.EXECUTION_TIME
            ]
            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )

            # ベンチマークとの比較
            results[operation_name] = {
                "benchmark_response_time_ms": benchmark.target_response_time_ms,
                "actual_response_time_ms": avg_response_time,
                "meets_benchmark": avg_response_time
                <= benchmark.target_response_time_ms,
                "variance_percent": (
                    (
                        (avg_response_time - benchmark.target_response_time_ms)
                        / benchmark.target_response_time_ms
                    )
                    * 100
                    if benchmark.target_response_time_ms > 0
                    else 0
                ),
            }

        return results


class ResourceMonitor:
    """リソースモニター"""

    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics: List[PerformanceMetric] = []

    def start_monitoring(self) -> None:
        """監視開始"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """監視停止"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_loop(self) -> None:
        """監視ループ"""
        while self._monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                self._metrics.append(
                    PerformanceMetric(
                        name="system_cpu_usage",
                        metric_type=PerformanceMetricType.CPU_USAGE,
                        value=cpu_percent,
                        component="system",
                    )
                )

                # メモリ使用率
                memory = psutil.virtual_memory()
                self._metrics.append(
                    PerformanceMetric(
                        name="system_memory_usage",
                        metric_type=PerformanceMetricType.MEMORY_USAGE,
                        value=memory.percent,
                        component="system",
                    )
                )

                # プロセス固有メトリクス
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                self._metrics.append(
                    PerformanceMetric(
                        name="process_memory_usage",
                        metric_type=PerformanceMetricType.MEMORY_USAGE,
                        value=process_memory,
                        component="process",
                    )
                )

                time.sleep(self.interval_seconds)

            except Exception as e:
                print(f"Resource monitoring error: {e}")

    def get_current_usage(self) -> Dict[str, float]:
        """現在の使用率取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024 / 1024 / 1024,
                "process_memory_mb": process_memory,
            }
        except Exception:
            return {}


class CacheOptimizer:
    """キャッシュ最適化"""

    def __init__(self):
        self._cache_stats: Dict[str, Dict[str, int]] = {}

    def record_cache_access(self, cache_name: str, hit: bool) -> None:
        """キャッシュアクセス記録"""
        if cache_name not in self._cache_stats:
            self._cache_stats[cache_name] = {"hits": 0, "misses": 0}

        if hit:
            self._cache_stats[cache_name]["hits"] += 1
        else:
            self._cache_stats[cache_name]["misses"] += 1

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """キャッシュ統計取得"""
        stats = {}
        for cache_name, data in self._cache_stats.items():
            total = data["hits"] + data["misses"]
            hit_rate = data["hits"] / total if total > 0 else 0

            stats[cache_name] = {
                "hits": data["hits"],
                "misses": data["misses"],
                "total_requests": total,
                "hit_rate": hit_rate,
                "efficiency": (
                    "high" if hit_rate > 0.8 else "medium" if hit_rate > 0.5 else "low"
                ),
            }

        return stats

    def optimize_cache_size(self, cache_name: str, current_size: int) -> int:
        """キャッシュサイズ最適化"""
        if cache_name not in self._cache_stats:
            return current_size

        stats = self._cache_stats[cache_name]
        total = stats["hits"] + stats["misses"]
        hit_rate = stats["hits"] / total if total > 0 else 0

        # ヒット率に基づくサイズ調整
        if hit_rate < 0.5:
            return int(current_size * 1.5)  # サイズ増加
        elif hit_rate > 0.9:
            return int(current_size * 0.8)  # サイズ減少
        else:
            return current_size  # 現状維持


class ParallelProcessingOptimizer:
    """並列処理最適化"""

    def __init__(self):
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def optimize_for_io_bound(
        self, tasks: List[Callable], max_workers: Optional[int] = None
    ) -> List[Any]:
        """IO集約的タスクの最適化"""
        if max_workers is None:
            max_workers = min(32, (psutil.cpu_count() or 1) + 4)

        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        futures = [self._thread_pool.submit(task) for task in tasks]
        return [future.result() for future in futures]

    def optimize_for_cpu_bound(
        self, tasks: List[Callable], max_workers: Optional[int] = None
    ) -> List[Any]:
        """CPU集約的タスクの最適化"""
        if max_workers is None:
            max_workers = psutil.cpu_count() or 1

        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)

        futures = [self._process_pool.submit(task) for task in tasks]
        return [future.result() for future in futures]

    def cleanup(self) -> None:
        """リソースクリーンアップ"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None


class MemoryOptimizer:
    """メモリ最適化"""

    @staticmethod
    def optimize_garbage_collection() -> Dict[str, Any]:
        """ガベージコレクション最適化"""
        before_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # ガベージコレクション実行
        collected = gc.collect()

        after_memory = psutil.Process().memory_info().rss / 1024 / 1024
        freed_mb = before_memory - after_memory

        return {
            "objects_collected": collected,
            "memory_freed_mb": freed_mb,
            "before_memory_mb": before_memory,
            "after_memory_mb": after_memory,
        }

    @staticmethod
    def get_memory_profile() -> Dict[str, Any]:
        """メモリプロファイル取得"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "gc_stats": {
                    "generation_0": gc.get_count()[0],
                    "generation_1": gc.get_count()[1],
                    "generation_2": gc.get_count()[2],
                },
                "gc_thresholds": gc.get_threshold(),
            }
        except Exception:
            return {}


class OptimizationEngine:
    """最適化エンジン"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.resource_monitor = ResourceMonitor()
        self.cache_optimizer = CacheOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self._optimization_results: List[OptimizationResult] = []

    def start_monitoring(self) -> None:
        """監視開始"""
        self.resource_monitor.start_monitoring()

    def stop_monitoring(self) -> None:
        """監視停止"""
        self.resource_monitor.stop_monitoring()

    def analyze_performance(self, component: str = "") -> Dict[str, Any]:
        """パフォーマンス分析"""
        # メトリクス取得
        execution_metrics = self.profiler.get_metrics(
            component=component, metric_type=PerformanceMetricType.EXECUTION_TIME
        )

        memory_metrics = self.profiler.get_metrics(
            component=component, metric_type=PerformanceMetricType.MEMORY_USAGE
        )

        # 統計計算
        avg_execution_time = (
            sum(m.value for m in execution_metrics) / len(execution_metrics)
            if execution_metrics
            else 0
        )
        max_memory_usage = max((m.value for m in memory_metrics), default=0)

        # ベンチマークチェック
        benchmark_results = self.profiler.check_benchmarks()

        return {
            "component": component,
            "average_execution_time_ms": avg_execution_time,
            "max_memory_usage_mb": max_memory_usage,
            "benchmark_results": benchmark_results,
            "cache_stats": self.cache_optimizer.get_cache_stats(),
            "memory_profile": self.memory_optimizer.get_memory_profile(),
            "current_resource_usage": self.resource_monitor.get_current_usage(),
        }

    def suggest_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """最適化提案"""
        suggestions = []

        # 実行時間チェック
        if analysis["average_execution_time_ms"] > 1000:
            suggestions.append(
                "実行時間が長いため、並列処理またはキャッシュの導入を検討してください"
            )

        # メモリ使用量チェック
        if analysis["max_memory_usage_mb"] > 1000:
            suggestions.append(
                "メモリ使用量が高いため、メモリプール化またはデータ分割処理を検討してください"
            )

        # キャッシュ効率チェック
        cache_stats = analysis.get("cache_stats", {})
        for cache_name, stats in cache_stats.items():
            if stats["hit_rate"] < 0.5:
                suggestions.append(
                    f"キャッシュ '{cache_name}' のヒット率が低いため、キャッシュ戦略の見直しを推奨します"
                )

        # リソース使用率チェック
        resource_usage = analysis.get("current_resource_usage", {})
        if resource_usage.get("cpu_percent", 0) > 80:
            suggestions.append("CPU使用率が高いため、処理の分散または最適化が必要です")

        if resource_usage.get("memory_percent", 0) > 80:
            suggestions.append(
                "メモリ使用率が高いため、メモリクリーンアップまたは最適化が必要です"
            )

        return suggestions

    def auto_optimize(self, component: str = "") -> Dict[str, Any]:
        """自動最適化"""
        # 分析実行
        analysis = self.analyze_performance(component)

        # 最適化実行
        optimizations_applied = []

        # ガベージコレクション最適化
        gc_result = self.memory_optimizer.optimize_garbage_collection()
        if gc_result["memory_freed_mb"] > 10:
            optimizations_applied.append("memory_cleanup")

        # キャッシュサイズ最適化
        cache_stats = analysis.get("cache_stats", {})
        for cache_name, stats in cache_stats.items():
            if stats["hit_rate"] < 0.5:
                # キャッシュサイズ調整の提案（実際の実装は各キャッシュシステムに依存）
                optimizations_applied.append(f"cache_optimization_{cache_name}")

        return {
            "analysis": analysis,
            "optimizations_applied": optimizations_applied,
            "suggestions": self.suggest_optimizations(analysis),
            "gc_result": gc_result,
        }


# グローバル最適化エンジン
global_optimization_engine = OptimizationEngine()


def performance_monitor(
    component: str = "", operation: str = "", enable_profiling: bool = False
):
    """パフォーマンス監視デコレーター"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation_name = operation or func.__name__

            if enable_profiling:
                global_optimization_engine.profiler.start_profiling(
                    f"{component}:{operation_name}"
                )

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                result = await func(*args, **kwargs)
                error_occurred = False
                return result
            except Exception as e:
                error_occurred = True
                raise e
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                execution_time_ms = (end_time - start_time) * 1000
                memory_delta_mb = end_memory - start_memory

                # メトリクス記録
                global_optimization_engine.profiler.add_metric(
                    PerformanceMetric(
                        name=f"{component}:{operation_name}:execution_time",
                        metric_type=PerformanceMetricType.EXECUTION_TIME,
                        value=execution_time_ms,
                        component=component,
                        operation=operation_name,
                    )
                )

                if memory_delta_mb > 0:
                    global_optimization_engine.profiler.add_metric(
                        PerformanceMetric(
                            name=f"{component}:{operation_name}:memory_usage",
                            metric_type=PerformanceMetricType.MEMORY_USAGE,
                            value=memory_delta_mb,
                            component=component,
                            operation=operation_name,
                        )
                    )

                if enable_profiling:
                    profile_result = global_optimization_engine.profiler.stop_profiling(
                        f"{component}:{operation_name}"
                    )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            operation_name = operation or func.__name__

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                execution_time_ms = (end_time - start_time) * 1000
                memory_delta_mb = end_memory - start_memory

                # メトリクス記録
                global_optimization_engine.profiler.add_metric(
                    PerformanceMetric(
                        name=f"{component}:{operation_name}:execution_time",
                        metric_type=PerformanceMetricType.EXECUTION_TIME,
                        value=execution_time_ms,
                        component=component,
                        operation=operation_name,
                    )
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
