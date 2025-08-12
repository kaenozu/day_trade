#!/usr/bin/env python3
"""
Performance Monitoring System (Simple Version)
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class PerformanceMetrics:
    """Performance Metrics"""

    timestamp: datetime
    process_name: str
    execution_time: float
    success: bool


class PerformanceMonitor:
    """Performance Monitoring System"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics = {
            "ml_analysis_85_stocks": 3.6,
            "data_fetch_85_stocks": 2.0,
            "portfolio_optimization": 1.0,
        }
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        self.collected_metrics: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def monitor(self, process_name: str):
        """Monitoring Context"""
        start_time = time.time()
        success = True

        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            execution_time = time.time() - start_time

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                process_name=process_name,
                execution_time=execution_time,
                success=success,
            )

            self.metrics_history.append(metrics)
            status = "SUCCESS" if success else "FAILED"
            print(
                f"Monitoring Complete: {process_name} - {execution_time:.3f}s ({status})"
            )

    def start_monitoring(self, monitor_name: str) -> None:
        """監視開始"""
        self.active_monitors[monitor_name] = {
            "start_time": time.time(),
            "memory_start": self._get_memory_usage(),
            "status": "running",
        }

    def stop_monitoring(self, monitor_name: str) -> None:
        """監視終了"""
        if monitor_name in self.active_monitors:
            monitor_data = self.active_monitors[monitor_name]
            end_time = time.time()
            memory_end = self._get_memory_usage()

            self.collected_metrics[monitor_name] = {
                "execution_time": end_time - monitor_data["start_time"],
                "memory_usage": memory_end,
                "memory_delta": memory_end - monitor_data["memory_start"],
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
            }

            del self.active_monitors[monitor_name]

    def get_metrics(self, monitor_name: str) -> Dict[str, Any]:
        """監視メトリクス取得"""
        if monitor_name in self.collected_metrics:
            return self.collected_metrics[monitor_name]

        # アクティブな監視の場合は現在の状態を返す
        if monitor_name in self.active_monitors:
            monitor_data = self.active_monitors[monitor_name]
            current_time = time.time()
            current_memory = self._get_memory_usage()

            return {
                "execution_time": current_time - monitor_data["start_time"],
                "memory_usage": current_memory,
                "memory_delta": current_memory - monitor_data["memory_start"],
                "timestamp": datetime.now().isoformat(),
                "status": "running",
            }

        # デフォルト値を返す
        return {
            "execution_time": 0.0,
            "memory_usage": self._get_memory_usage(),
            "memory_delta": 0.0,
            "timestamp": datetime.now().isoformat(),
            "status": "not_found",
        }

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（MB単位）"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # psutilがない場合はダミー値
            return 50.0


# Global instance
global_monitor = PerformanceMonitor()


def get_performance_monitor():
    return global_monitor
