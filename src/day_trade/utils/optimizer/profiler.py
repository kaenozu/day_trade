#!/usr/bin/env python3
"""
パフォーマンスプロファイラー

関数のパフォーマンス測定と詳細プロファイリングを提供します。
"""

import cProfile
import io
import pstats
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import psutil

from .metrics import PerformanceMetrics


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed = enable_detailed_profiling
        self.metrics: List[PerformanceMetrics] = []
        self.profiler: Optional[cProfile.Profile] = None

    def profile_function(self, func: Callable) -> Callable:
        """関数のパフォーマンスをプロファイルするデコレータ"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._measure_performance(func, *args, **kwargs)

        return wrapper

    def _measure_performance(self, func: Callable, *args, **kwargs) -> Any:
        """パフォーマンス測定の実行"""
        # システムリソース測定開始
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        start_time = time.perf_counter()

        # 詳細プロファイリング開始
        if self.enable_detailed and self.profiler is None:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        try:
            # 関数実行
            result = func(*args, **kwargs)

            # 測定終了
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = (
                process.memory_info().peak_wss / 1024 / 1024
                if hasattr(process.memory_info(), "peak_wss")
                else end_memory
            )
            end_cpu = psutil.cpu_percent()

            # メトリクス記録
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                peak_memory_mb=peak_memory,
                data_size=self._estimate_data_size(result),
            )

            self.metrics.append(metrics)
            return result

        finally:
            if self.enable_detailed and self.profiler:
                self.profiler.disable()

    def _estimate_data_size(self, data: Any) -> Optional[int]:
        """データサイズの推定"""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, (list, tuple, dict)):
            return len(data)
        return None

    def get_profile_stats(self) -> str:
        """詳細プロファイル統計を取得"""
        if not self.profiler:
            return "詳細プロファイリングが有効化されていません"

        stats_buffer = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # 上位20関数
        return stats_buffer.getvalue()

    def get_summary_report(self) -> Dict[str, Any]:
        """サマリーレポートを取得"""
        if not self.metrics:
            return {"message": "測定データがありません"}

        total_time = sum(m.execution_time for m in self.metrics)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics) / len(
            self.metrics
        )
        max_memory = max(m.peak_memory_mb for m in self.metrics)

        return {
            "total_functions_profiled": len(self.metrics),
            "total_execution_time": total_time,
            "average_memory_usage_mb": avg_memory,
            "peak_memory_usage_mb": max_memory,
            "slowest_functions": sorted(
                self.metrics, key=lambda x: x.execution_time, reverse=True
            )[:5],
            "memory_intensive_functions": sorted(
                self.metrics, key=lambda x: x.memory_usage_mb, reverse=True
            )[:5],
        }
