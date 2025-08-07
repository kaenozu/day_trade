#!/usr/bin/env python3
"""
パフォーマンス監視システム

実行時間、メモリ使用量、キャッシュ効率の監視
"""

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil

from .logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    operation_name: str
    execution_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0


class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    """

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()

    @contextmanager
    def measure_operation(self, operation_name: str):
        """
        操作のパフォーマンス測定コンテキストマネージャ

        Usage:
            with monitor.measure_operation("data_fetch"):
                # 測定対象の処理
                result = fetch_data()
        """
        # 開始時計測
        start_time = time.time()
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = self.process.cpu_percent()

        try:
            yield
        finally:
            # 終了時計測
            end_time = time.time()
            memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = self.process.cpu_percent()

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=end_time - start_time,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - memory_before,
                cpu_percent=(cpu_before + cpu_after) / 2,
            )

            self.metrics_history.append(metrics)

            logger.info(f"パフォーマンス測定完了: {operation_name}")
            logger.info(f"  実行時間: {metrics.execution_time:.3f}秒")
            logger.info(f"  メモリ変化: {metrics.memory_delta_mb:+.1f}MB")
            logger.info(f"  CPU使用率: {metrics.cpu_percent:.1f}%")

    def measure_cache_performance(self, data_manager) -> Dict[str, float]:
        """
        キャッシュパフォーマンス測定

        Args:
            data_manager: RealMarketDataManager インスタンス

        Returns:
            Dict: キャッシュ効率指標
        """
        try:
            memory_cache_size = len(data_manager.memory_cache)
            expiry_cache_size = len(data_manager.cache_expiry)

            # SQLiteキャッシュサイズ
            sqlite_size = 0
            try:
                import sqlite3

                with sqlite3.connect(data_manager.cache_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM price_cache")
                    sqlite_size = cursor.fetchone()[0]
            except Exception:
                sqlite_size = 0

            # キャッシュ効率計算
            total_cache_entries = memory_cache_size + sqlite_size
            memory_cache_ratio = memory_cache_size / max(1, total_cache_entries)

            metrics = {
                "memory_cache_entries": memory_cache_size,
                "sqlite_cache_entries": sqlite_size,
                "total_cache_entries": total_cache_entries,
                "memory_cache_ratio": memory_cache_ratio,
                "expiry_entries": expiry_cache_size,
            }

            logger.info(f"キャッシュパフォーマンス: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"キャッシュパフォーマンス測定エラー: {e}")
            return {}

    def get_system_stats(self) -> Dict[str, float]:
        """システム統計情報取得"""
        try:
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "memory_percent": self.process.memory_percent(),
                "thread_count": self.process.num_threads(),
                "open_files": len(self.process.open_files())
                if hasattr(self.process, "open_files")
                else 0,
            }
        except Exception as e:
            logger.error(f"システム統計取得エラー: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, float]:
        """パフォーマンス要約統計"""
        if not self.metrics_history:
            return {}

        execution_times = [m.execution_time for m in self.metrics_history]
        memory_deltas = [m.memory_delta_mb for m in self.metrics_history]
        cpu_percentages = [m.cpu_percent for m in self.metrics_history]

        return {
            "total_operations": len(self.metrics_history),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
            "avg_cpu_percent": sum(cpu_percentages) / len(cpu_percentages),
        }

    def optimize_memory(self) -> Dict[str, int]:
        """メモリ最適化実行"""
        logger.info("メモリ最適化開始")

        # ガベージコレクション強制実行
        collected_counts = gc.collect()

        # メモリ使用量確認
        memory_after = self.process.memory_info().rss / 1024 / 1024

        stats = {
            "garbage_collected": collected_counts,
            "memory_after_mb": int(memory_after),
            "gc_generation_counts": [len(gc.get_objects(i)) for i in range(3)],
        }

        logger.info(f"メモリ最適化完了: {stats}")
        return stats

    def clear_metrics_history(self):
        """測定履歴クリア"""
        cleared_count = len(self.metrics_history)
        self.metrics_history.clear()
        logger.info(f"パフォーマンス履歴クリア: {cleared_count}件")

    def export_performance_report(self, filepath: Optional[str] = None) -> str:
        """パフォーマンスレポートエクスポート"""
        if not self.metrics_history:
            return "パフォーマンス履歴が空です"

        lines = ["# パフォーマンスレポート\n"]

        # 要約統計
        summary = self.get_performance_summary()
        lines.append("## 要約統計")
        for key, value in summary.items():
            lines.append(f"- {key}: {value:.3f}")

        lines.append("\n## 詳細履歴")
        lines.append("| 操作名 | 実行時間(秒) | メモリ変化(MB) | CPU使用率(%) |")
        lines.append("|--------|-------------|--------------|-------------|")

        for metric in self.metrics_history:
            lines.append(
                f"| {metric.operation_name} | {metric.execution_time:.3f} | "
                f"{metric.memory_delta_mb:+.1f} | {metric.cpu_percent:.1f} |"
            )

        report = "\n".join(lines)

        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"パフォーマンスレポート出力: {filepath}")
            except Exception as e:
                logger.error(f"レポート出力エラー: {e}")

        return report


# グローバル監視インスタンス
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """グローバルパフォーマンス監視インスタンス取得"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def measure_performance(operation_name: str):
    """パフォーマンス測定デコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.measure_operation(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # テスト実行
    monitor = PerformanceMonitor()

    print("=== パフォーマンス監視テスト ===")

    # 測定テスト
    with monitor.measure_operation("test_operation"):
        # 何らかの処理をシミュレート
        time.sleep(0.1)
        data = [i**2 for i in range(10000)]

    # システム統計
    system_stats = monitor.get_system_stats()
    print(f"システム統計: {system_stats}")

    # パフォーマンス要約
    summary = monitor.get_performance_summary()
    print(f"パフォーマンス要約: {summary}")

    # メモリ最適化
    memory_stats = monitor.optimize_memory()
    print(f"メモリ最適化結果: {memory_stats}")

    # レポート出力
    report = monitor.export_performance_report()
    print("レポート生成完了")
    print(report[:500] + "...")  # 先頭500文字表示
