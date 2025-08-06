"""
アプリケーションパフォーマンス分析ツール

アプリケーション全体の処理速度を分析し、ボトルネックを特定して
最適化案を提供する包括的なパフォーマンス分析システム。
"""

import cProfile
import functools
import io
import pstats
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Conditional import for psutil
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from ..utils.logging_config import get_context_logger, log_performance_metric

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    function_calls: int
    cache_hits: int = 0
    cache_misses: int = 0
    io_operations: int = 0
    network_calls: int = 0
    database_queries: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """キャッシュヒット率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class BottleneckResult:
    """ボトルネック分析結果"""

    component: str
    severity: str  # "critical", "high", "medium", "low"
    issue_type: str  # "cpu", "memory", "io", "network", "algorithm"
    current_performance: PerformanceMetrics
    optimization_suggestions: List[str] = field(default_factory=list)
    estimated_improvement: float = 0.0  # percentage


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self.metrics_history = {}
        self.active_profiles = {}
        self.cache_stats = {}

    def profile_function(self, func: Callable):
        """関数のパフォーマンスプロファイリングデコレータ"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Initialize monitoring variables with fallback
            if HAS_PSUTIL:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                process.cpu_percent()
            else:
                process = None
                initial_memory = 0.0

            # プロファイラ開始
            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)

                # プロファイラ停止
                profiler.disable()

                # メトリクス計算
                end_time = time.time()
                execution_time = end_time - start_time

                if HAS_PSUTIL and process:
                    final_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage = final_memory - initial_memory
                    cpu_usage = process.cpu_percent()
                else:
                    memory_usage = 0.0
                    cpu_usage = 0.0

                # プロファイリング結果解析
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats("cumulative")
                function_calls = ps.total_calls

                # メトリクス記録
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    function_calls=function_calls,
                )

                func_name = f"{func.__module__}.{func.__name__}"
                self.metrics_history[func_name] = metrics

                # ログ記録
                log_performance_metric(
                    metric_name=f"function_performance_{func_name}",
                    value=execution_time,
                    unit="seconds",
                    additional_data={
                        "memory_usage_mb": memory_usage,
                        "cpu_usage_percent": cpu_usage,
                        "function_calls": function_calls,
                    },
                )

                logger.debug(
                    f"関数パフォーマンス: {func_name}",
                    section="performance_profiling",
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    function_calls=function_calls,
                )

                return result

            except Exception as e:
                profiler.disable()
                logger.error(
                    f"プロファイリング中にエラー: {func.__name__}", error=str(e)
                )
                raise

        return wrapper

    def get_metrics_summary(self) -> Dict[str, Any]:
        """メトリクスサマリー取得"""
        if not self.metrics_history:
            return {}

        total_execution_time = sum(
            m.execution_time for m in self.metrics_history.values()
        )
        avg_memory_usage = np.mean(
            [m.memory_usage_mb for m in self.metrics_history.values()]
        )
        avg_cpu_usage = np.mean(
            [m.cpu_usage_percent for m in self.metrics_history.values()]
        )
        total_function_calls = sum(
            m.function_calls for m in self.metrics_history.values()
        )

        return {
            "total_functions_profiled": len(self.metrics_history),
            "total_execution_time": total_execution_time,
            "average_memory_usage_mb": avg_memory_usage,
            "average_cpu_usage_percent": avg_cpu_usage,
            "total_function_calls": total_function_calls,
            "slowest_functions": self._get_slowest_functions(5),
            "memory_intensive_functions": self._get_memory_intensive_functions(5),
        }

    def _get_slowest_functions(self, count: int) -> List[Tuple[str, float]]:
        """最も遅い関数を取得"""
        sorted_funcs = sorted(
            self.metrics_history.items(),
            key=lambda x: x[1].execution_time,
            reverse=True,
        )
        return [
            (name, metrics.execution_time) for name, metrics in sorted_funcs[:count]
        ]

    def _get_memory_intensive_functions(self, count: int) -> List[Tuple[str, float]]:
        """メモリ使用量の多い関数を取得"""
        sorted_funcs = sorted(
            self.metrics_history.items(),
            key=lambda x: x[1].memory_usage_mb,
            reverse=True,
        )
        return [
            (name, metrics.memory_usage_mb) for name, metrics in sorted_funcs[:count]
        ]


class SystemPerformanceMonitor:
    """システム全体のパフォーマンス監視"""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.system_metrics = []

    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitoring_thread.start()

        logger.info("システムパフォーマンス監視開始", section="system_monitoring")

    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("システムパフォーマンス監視停止", section="system_monitoring")

    def _monitor_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # システムメトリクス収集
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)

                # 古いメトリクスの削除（最新1時間分のみ保持）
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.system_metrics = [
                    m for m in self.system_metrics if m["timestamp"] > cutoff_time
                ]

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error("システム監視エラー", error=str(e))
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """システムメトリクス収集"""
        if not HAS_PSUTIL:
            return {
                "timestamp": datetime.now(),
                "cpu_percent": 0.0,
                "process_cpu_percent": 0.0,
                "memory_total": 0,
                "memory_available": 0,
                "memory_percent": 0.0,
                "process_memory_rss": 0,
                "process_memory_vms": 0,
                "disk_read_bytes": 0,
                "disk_write_bytes": 0,
                "network_bytes_sent": 0,
                "network_bytes_recv": 0,
            }

        process = psutil.Process()

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        process_cpu = process.cpu_percent()

        # メモリ使用量
        memory = psutil.virtual_memory()
        process_memory = process.memory_info()

        # ディスクI/O
        disk_io = psutil.disk_io_counters()

        # ネットワークI/O
        network_io = psutil.net_io_counters()

        return {
            "timestamp": datetime.now(),
            "cpu_percent_system": cpu_percent,
            "cpu_percent_process": process_cpu,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "process_memory_rss_mb": process_memory.rss / (1024**2),
            "process_memory_vms_mb": process_memory.vms / (1024**2),
            "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
            "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
            "network_bytes_sent": network_io.bytes_sent if network_io else 0,
            "network_bytes_recv": network_io.bytes_recv if network_io else 0,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        if not self.system_metrics:
            return {}

        # 統計計算
        cpu_system = [m["cpu_percent_system"] for m in self.system_metrics]
        cpu_process = [m["cpu_percent_process"] for m in self.system_metrics]
        memory_percent = [m["memory_percent"] for m in self.system_metrics]
        process_memory = [m["process_memory_rss_mb"] for m in self.system_metrics]

        return {
            "monitoring_duration_minutes": len(self.system_metrics)
            * self.monitoring_interval
            / 60,
            "avg_cpu_system": np.mean(cpu_system),
            "max_cpu_system": np.max(cpu_system),
            "avg_cpu_process": np.mean(cpu_process),
            "max_cpu_process": np.max(cpu_process),
            "avg_memory_percent": np.mean(memory_percent),
            "max_memory_percent": np.max(memory_percent),
            "avg_process_memory_mb": np.mean(process_memory),
            "max_process_memory_mb": np.max(process_memory),
            "total_samples": len(self.system_metrics),
        }


class BottleneckAnalyzer:
    """ボトルネック分析器"""

    def __init__(self):
        self.performance_thresholds = {
            "cpu_critical": 80.0,
            "cpu_high": 60.0,
            "memory_critical": 85.0,
            "memory_high": 70.0,
            "execution_time_critical": 5.0,
            "execution_time_high": 2.0,
            "function_calls_critical": 10000,
            "function_calls_high": 5000,
        }

    def analyze_bottlenecks(
        self, profiler: PerformanceProfiler, monitor: SystemPerformanceMonitor
    ) -> List[BottleneckResult]:
        """ボトルネック分析実行"""
        bottlenecks = []

        # 関数レベルのボトルネック分析
        function_bottlenecks = self._analyze_function_bottlenecks(profiler)
        bottlenecks.extend(function_bottlenecks)

        # システムレベルのボトルネック分析
        system_bottlenecks = self._analyze_system_bottlenecks(monitor)
        bottlenecks.extend(system_bottlenecks)

        # 重要度でソート
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        bottlenecks.sort(key=lambda x: severity_order.get(x.severity, 3))

        logger.info(
            "ボトルネック分析完了",
            section="bottleneck_analysis",
            total_bottlenecks=len(bottlenecks),
            critical_issues=len([b for b in bottlenecks if b.severity == "critical"]),
            high_issues=len([b for b in bottlenecks if b.severity == "high"]),
        )

        return bottlenecks

    def _analyze_function_bottlenecks(
        self, profiler: PerformanceProfiler
    ) -> List[BottleneckResult]:
        """関数レベルのボトルネック分析"""
        bottlenecks = []

        for func_name, metrics in profiler.metrics_history.items():
            severity = "low"
            suggestions = []
            issue_type = "algorithm"
            estimated_improvement = 0.0

            # 実行時間分析
            if (
                metrics.execution_time
                > self.performance_thresholds["execution_time_critical"]
            ):
                severity = "critical"
                estimated_improvement = 60.0
                suggestions.extend(
                    [
                        "アルゴリズムの最適化を検討",
                        "並列処理の導入",
                        "データ構造の見直し",
                        "キャッシュの導入",
                    ]
                )
            elif (
                metrics.execution_time
                > self.performance_thresholds["execution_time_high"]
            ):
                severity = "high"
                estimated_improvement = 30.0
                suggestions.extend(["処理ロジックの最適化", "不要な計算の除去"])

            # メモリ使用量分析
            if metrics.memory_usage_mb > 100:  # 100MB以上
                if severity in ["low", "medium"]:
                    severity = "high" if metrics.memory_usage_mb > 500 else "medium"
                issue_type = "memory"
                estimated_improvement = max(estimated_improvement, 40.0)
                suggestions.extend(
                    [
                        "メモリ使用量の最適化",
                        "データの分割処理",
                        "不要なオブジェクトの削除",
                    ]
                )

            # 関数呼び出し数分析
            if (
                metrics.function_calls
                > self.performance_thresholds["function_calls_critical"]
            ):
                if severity in ["low", "medium"]:
                    severity = "high"
                suggestions.append("関数呼び出し回数の削減")
                estimated_improvement = max(estimated_improvement, 25.0)

            if severity != "low" or suggestions:
                bottlenecks.append(
                    BottleneckResult(
                        component=func_name,
                        severity=severity,
                        issue_type=issue_type,
                        current_performance=metrics,
                        optimization_suggestions=suggestions,
                        estimated_improvement=estimated_improvement,
                    )
                )

        return bottlenecks

    def _analyze_system_bottlenecks(
        self, monitor: SystemPerformanceMonitor
    ) -> List[BottleneckResult]:
        """システムレベルのボトルネック分析"""
        bottlenecks = []
        summary = monitor.get_performance_summary()

        if not summary:
            return bottlenecks

        # CPU使用率分析
        if summary["max_cpu_system"] > self.performance_thresholds["cpu_critical"]:
            bottlenecks.append(
                BottleneckResult(
                    component="system_cpu",
                    severity="critical",
                    issue_type="cpu",
                    current_performance=PerformanceMetrics(
                        execution_time=0,
                        memory_usage_mb=0,
                        cpu_usage_percent=summary["max_cpu_system"],
                        function_calls=0,
                    ),
                    optimization_suggestions=[
                        "CPU集約的な処理の最適化",
                        "並列処理の導入",
                        "処理の分散化",
                    ],
                    estimated_improvement=50.0,
                )
            )

        # メモリ使用率分析
        if (
            summary["max_memory_percent"]
            > self.performance_thresholds["memory_critical"]
        ):
            bottlenecks.append(
                BottleneckResult(
                    component="system_memory",
                    severity="critical",
                    issue_type="memory",
                    current_performance=PerformanceMetrics(
                        execution_time=0,
                        memory_usage_mb=summary["max_process_memory_mb"],
                        cpu_usage_percent=0,
                        function_calls=0,
                    ),
                    optimization_suggestions=[
                        "メモリリークの調査",
                        "データ処理の最適化",
                        "ガベージコレクションの調整",
                    ],
                    estimated_improvement=60.0,
                )
            )

        return bottlenecks


class PerformanceOptimizer:
    """パフォーマンス最適化実行器"""

    def __init__(self):
        self.optimization_history = []

    def apply_optimizations(
        self, bottlenecks: List[BottleneckResult]
    ) -> Dict[str, Any]:
        """最適化の適用"""
        optimization_results = {
            "applied_optimizations": [],
            "total_estimated_improvement": 0.0,
            "optimization_errors": [],
        }

        for bottleneck in bottlenecks:
            if bottleneck.severity in ["critical", "high"]:
                try:
                    result = self._apply_optimization(bottleneck)
                    optimization_results["applied_optimizations"].append(result)
                    optimization_results[
                        "total_estimated_improvement"
                    ] += bottleneck.estimated_improvement

                except Exception as e:
                    error_msg = f"最適化適用エラー ({bottleneck.component}): {e}"
                    logger.error(error_msg)
                    optimization_results["optimization_errors"].append(error_msg)

        logger.info(
            "パフォーマンス最適化完了",
            section="performance_optimization",
            applied_count=len(optimization_results["applied_optimizations"]),
            estimated_improvement=optimization_results["total_estimated_improvement"],
        )

        return optimization_results

    def _apply_optimization(self, bottleneck: BottleneckResult) -> Dict[str, Any]:
        """個別最適化の適用"""
        # 実際の最適化実装はボトルネックの種類に応じて異なる
        # ここでは最適化の提案と記録のみ実装

        optimization_plan = {
            "component": bottleneck.component,
            "issue_type": bottleneck.issue_type,
            "severity": bottleneck.severity,
            "suggestions": bottleneck.optimization_suggestions,
            "estimated_improvement": bottleneck.estimated_improvement,
            "status": "planned",  # 実際の適用は別途実装が必要
        }

        self.optimization_history.append(optimization_plan)

        logger.info(
            f"最適化プラン作成: {bottleneck.component}",
            section="performance_optimization",
            issue_type=bottleneck.issue_type,
            severity=bottleneck.severity,
            estimated_improvement=bottleneck.estimated_improvement,
        )

        return optimization_plan


# グローバルプロファイラインスタンス
global_profiler = PerformanceProfiler()


def profile_performance(func: Callable):
    """パフォーマンスプロファイリングデコレータ（グローバル）"""
    return global_profiler.profile_function(func)


# 使用例とデモ
if __name__ == "__main__":
    # デモ用の重い処理関数
    @profile_performance
    def heavy_computation(n: int = 100000):
        """CPU集約的な処理のデモ"""
        result = sum(i**2 for i in range(n))
        return result

    @profile_performance
    def memory_intensive_operation():
        """メモリ集約的な処理のデモ"""
        large_list = [i for i in range(1000000)]
        df = pd.DataFrame({"col1": large_list, "col2": [x * 2 for x in large_list]})
        return df.sum().sum()

    # パフォーマンス分析デモ
    logger.info("パフォーマンス分析デモ開始", section="demo")

    # システム監視開始
    monitor = SystemPerformanceMonitor(monitoring_interval=0.5)
    monitor.start_monitoring()

    try:
        # 重い処理実行
        result1 = heavy_computation(50000)
        result2 = memory_intensive_operation()

        # 少し待機
        time.sleep(2)

        # 監視停止
        monitor.stop_monitoring()

        # 分析実行
        analyzer = BottleneckAnalyzer()
        bottlenecks = analyzer.analyze_bottlenecks(global_profiler, monitor)

        # 最適化プラン作成
        optimizer = PerformanceOptimizer()
        optimization_results = optimizer.apply_optimizations(bottlenecks)

        # 結果表示
        profiler_summary = global_profiler.get_metrics_summary()
        system_summary = monitor.get_performance_summary()

        logger.info(
            "パフォーマンス分析デモ完了",
            section="demo",
            profiler_summary=profiler_summary,
            system_summary=system_summary,
            bottlenecks_found=len(bottlenecks),
            optimization_plans=len(optimization_results["applied_optimizations"]),
        )

    finally:
        monitor.stop_monitoring()
