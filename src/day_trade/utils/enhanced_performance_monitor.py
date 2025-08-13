"""
強化されたパフォーマンスモニタリングシステム

システム全体のパフォーマンス監視と分析を提供します。
- リアルタイム性能監視
- メモリ使用量監視
- 異常検知とアラート
- 履歴分析とレポート生成
"""

import gc
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

from .logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    timestamp: datetime
    process_name: str
    execution_time: float
    success: bool
    memory_usage: float  # MB単位
    cpu_usage: float  # パーセンテージ
    thread_count: int
    gc_collections: int
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """システム全体のメトリクス"""

    timestamp: datetime
    total_memory: float  # MB
    available_memory: float  # MB
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    active_threads: int
    process_count: int


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""

    timestamp: datetime
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    process_name: str
    message: str
    metrics: Dict[str, Any]
    threshold: float
    actual_value: float


class PerformanceThresholds:
    """パフォーマンス閾値定義"""

    def __init__(self):
        self.execution_time = {
            "ml_analysis": 5.0,  # 5秒以内
            "data_fetch": 3.0,  # 3秒以内
            "analysis_cycle": 2.0,  # 2秒以内
            "database_operation": 1.0,  # 1秒以内
            "api_call": 10.0,  # 10秒以内
        }

        self.memory_usage = {
            "total_mb": 1024,  # 1GB以内
            "process_mb": 512,  # 512MB以内
            "growth_rate": 0.1,  # 10%成長率以内
        }

        self.cpu_usage = {
            "average": 70.0,  # 70%以内
            "peak": 90.0,  # 90%以内
        }

        self.system = {
            "memory_percent": 80.0,  # 80%以内
            "disk_percent": 85.0,  # 85%以内
        }


class EnhancedPerformanceMonitor:
    """強化されたパフォーマンスモニタリングシステム"""

    def __init__(self, history_limit: int = 10000):
        self.history_limit = history_limit
        self.metrics_history = deque(maxlen=history_limit)
        self.system_metrics_history = deque(maxlen=1000)
        self.alerts_history = deque(maxlen=1000)

        # 基準値とベースライン
        self.baseline_metrics = {
            "ml_analysis_85_stocks": 3.6,
            "data_fetch_85_stocks": 2.0,
            "portfolio_optimization": 1.0,
            "analysis_cycle": 1.5,
        }

        # 閾値設定
        self.thresholds = PerformanceThresholds()

        # 統計情報
        self.process_stats = defaultdict(
            lambda: {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float("inf"),
            }
        )

        # システム監視用
        self._monitoring_active = False
        self._monitoring_thread = None
        self._system_monitor_interval = 30.0  # 30秒間隔

        # アラート設定
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        logger.info("強化されたパフォーマンスモニターを初期化しました")

    def start_system_monitoring(self, interval: float = 30.0):
        """システム監視を開始"""
        if self._monitoring_active:
            logger.warning("システム監視は既に開始されています")
            return

        self._system_monitor_interval = interval
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"システム監視を開始しました (間隔: {interval}秒)")

    def stop_system_monitoring(self):
        """システム監視を停止"""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logger.info("システム監視を停止しました")

    def _system_monitoring_loop(self):
        """システム監視ループ"""
        while self._monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)

                # システムレベルのアラートチェック
                self._check_system_alerts(metrics)

            except Exception as e:
                logger.error(f"システム監視エラー: {e}")

            time.sleep(self._system_monitor_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        process = psutil.Process()

        # メモリ情報
        memory_info = psutil.virtual_memory()

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # ディスク使用率
        disk_usage = psutil.disk_usage("/")

        return SystemMetrics(
            timestamp=datetime.now(),
            total_memory=memory_info.total / 1024 / 1024,  # MB
            available_memory=memory_info.available / 1024 / 1024,  # MB
            memory_usage_percent=memory_info.percent,
            cpu_usage_percent=cpu_percent,
            disk_usage_percent=disk_usage.percent,
            active_threads=threading.active_count(),
            process_count=len(psutil.pids()),
        )

    @contextmanager
    def monitor(self, process_name: str, category: Optional[str] = None):
        """
        パフォーマンス監視コンテキスト

        Args:
            process_name: プロセス名
            category: カテゴリ（オプション）
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        start_gc = sum(
            gc.get_stats()[i]["collections"] for i in range(len(gc.get_stats()))
        )

        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # メトリクス収集
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            end_gc = sum(
                gc.get_stats()[i]["collections"] for i in range(len(gc.get_stats()))
            )

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                process_name=process_name,
                execution_time=execution_time,
                success=success,
                memory_usage=end_memory - start_memory,
                cpu_usage=end_cpu - start_cpu,
                thread_count=threading.active_count(),
                gc_collections=end_gc - start_gc,
                error_message=error_message,
            )

            # 履歴に追加
            self.metrics_history.append(metrics)

            # 統計情報更新
            self._update_process_stats(process_name, execution_time, success)

            # アラートチェック
            self._check_performance_alerts(metrics, category)

            # ログ出力
            self._log_performance_metrics(metrics, category)

    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB単位）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _update_process_stats(
        self, process_name: str, execution_time: float, success: bool
    ):
        """プロセス統計情報を更新"""
        stats = self.process_stats[process_name]
        stats["total_calls"] += 1

        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1

        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)

    def _check_performance_alerts(
        self, metrics: PerformanceMetrics, category: Optional[str]
    ):
        """パフォーマンスアラートをチェック"""
        alerts = []

        # 実行時間チェック
        threshold_key = category or "default"
        execution_threshold = self.thresholds.execution_time.get(threshold_key, 5.0)

        if metrics.execution_time > execution_threshold:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="EXECUTION_TIME",
                    severity=(
                        "HIGH"
                        if metrics.execution_time > execution_threshold * 2
                        else "MEDIUM"
                    ),
                    process_name=metrics.process_name,
                    message=f"実行時間が閾値を超過: {metrics.execution_time:.2f}秒",
                    metrics=asdict(metrics),
                    threshold=execution_threshold,
                    actual_value=metrics.execution_time,
                )
            )

        # メモリ使用量チェック
        memory_threshold = self.thresholds.memory_usage["process_mb"]
        current_memory = self._get_memory_usage()

        if current_memory > memory_threshold:
            alerts.append(
                PerformanceAlert(
                    timestamp=metrics.timestamp,
                    alert_type="MEMORY_USAGE",
                    severity=(
                        "HIGH" if current_memory > memory_threshold * 2 else "MEDIUM"
                    ),
                    process_name=metrics.process_name,
                    message=f"メモリ使用量が閾値を超過: {current_memory:.2f}MB",
                    metrics=asdict(metrics),
                    threshold=memory_threshold,
                    actual_value=current_memory,
                )
            )

        # アラートを履歴に追加し、コールバックを実行
        for alert in alerts:
            self.alerts_history.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"アラートコールバックエラー: {e}")

    def _check_system_alerts(self, system_metrics: SystemMetrics):
        """システムレベルのアラートをチェック"""
        alerts = []

        # メモリ使用率チェック
        if (
            system_metrics.memory_usage_percent
            > self.thresholds.system["memory_percent"]
        ):
            alerts.append(
                PerformanceAlert(
                    timestamp=system_metrics.timestamp,
                    alert_type="SYSTEM_MEMORY",
                    severity="HIGH",
                    process_name="SYSTEM",
                    message=f"システムメモリ使用率が高い: {system_metrics.memory_usage_percent:.1f}%",
                    metrics=asdict(system_metrics),
                    threshold=self.thresholds.system["memory_percent"],
                    actual_value=system_metrics.memory_usage_percent,
                )
            )

        # ディスク使用率チェック
        if system_metrics.disk_usage_percent > self.thresholds.system["disk_percent"]:
            alerts.append(
                PerformanceAlert(
                    timestamp=system_metrics.timestamp,
                    alert_type="SYSTEM_DISK",
                    severity="MEDIUM",
                    process_name="SYSTEM",
                    message=f"ディスク使用率が高い: {system_metrics.disk_usage_percent:.1f}%",
                    metrics=asdict(system_metrics),
                    threshold=self.thresholds.system["disk_percent"],
                    actual_value=system_metrics.disk_usage_percent,
                )
            )

        # アラートを処理
        for alert in alerts:
            self.alerts_history.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"システムアラートコールバックエラー: {e}")

    def _log_performance_metrics(
        self, metrics: PerformanceMetrics, category: Optional[str]
    ):
        """パフォーマンスメトリクスをログ出力"""
        log_data = {
            "process_name": metrics.process_name,
            "execution_time": f"{metrics.execution_time:.3f}s",
            "memory_delta": f"{metrics.memory_usage:.2f}MB",
            "success": metrics.success,
            "category": category,
        }

        if metrics.success:
            if metrics.execution_time > 1.0:  # 1秒以上の場合は警告
                logger.warning("長時間実行プロセス検出", extra=log_data)
            else:
                logger.debug("パフォーマンスメトリクス", extra=log_data)
        else:
            logger.error(
                "プロセス実行失敗", extra={**log_data, "error": metrics.error_message}
            )

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """アラートコールバックを追加"""
        self.alert_callbacks.append(callback)
        logger.info(f"アラートコールバックを追加しました: {callback.__name__}")

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 指定時間内のメトリクスを抽出
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"message": "指定期間内のデータがありません"}

        # 統計計算
        total_processes = len(recent_metrics)
        successful_processes = len([m for m in recent_metrics if m.success])
        avg_execution_time = (
            sum(m.execution_time for m in recent_metrics) / total_processes
        )
        max_execution_time = max(m.execution_time for m in recent_metrics)
        total_memory_usage = sum(m.memory_usage for m in recent_metrics)

        # プロセス別統計
        process_summary = {}
        process_groups = defaultdict(list)
        for metric in recent_metrics:
            process_groups[metric.process_name].append(metric)

        for process_name, metrics in process_groups.items():
            process_summary[process_name] = {
                "count": len(metrics),
                "success_rate": len([m for m in metrics if m.success]) / len(metrics),
                "avg_time": sum(m.execution_time for m in metrics) / len(metrics),
                "max_time": max(m.execution_time for m in metrics),
            }

        # 最新のシステムメトリクス
        latest_system = (
            self.system_metrics_history[-1] if self.system_metrics_history else None
        )

        return {
            "period": f"{hours}時間",
            "total_processes": total_processes,
            "success_rate": successful_processes / total_processes,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "total_memory_delta": total_memory_usage,
            "process_summary": process_summary,
            "recent_alerts": len(
                [a for a in self.alerts_history if a.timestamp >= cutoff_time]
            ),
            "system_status": asdict(latest_system) if latest_system else None,
        }

    def get_bottleneck_analysis(self, limit: int = 10) -> Dict[str, Any]:
        """ボトルネック分析"""
        process_performance = []

        for process_name, stats in self.process_stats.items():
            if stats["total_calls"] > 0:
                process_performance.append(
                    {
                        "process_name": process_name,
                        "avg_time": stats["avg_time"],
                        "max_time": stats["max_time"],
                        "total_time": stats["total_time"],
                        "call_count": stats["total_calls"],
                        "failure_rate": stats["failed_calls"] / stats["total_calls"],
                    }
                )

        # 平均実行時間でソート
        slowest_by_avg = sorted(
            process_performance, key=lambda x: x["avg_time"], reverse=True
        )[:limit]

        # 最大実行時間でソート
        slowest_by_max = sorted(
            process_performance, key=lambda x: x["max_time"], reverse=True
        )[:limit]

        # 総実行時間でソート
        heaviest_by_total = sorted(
            process_performance, key=lambda x: x["total_time"], reverse=True
        )[:limit]

        return {
            "slowest_by_average": slowest_by_avg,
            "slowest_by_maximum": slowest_by_max,
            "heaviest_by_total_time": heaviest_by_total,
            "analysis_timestamp": datetime.now(),
        }


# グローバルインスタンス
global_performance_monitor = EnhancedPerformanceMonitor()


def get_performance_monitor() -> EnhancedPerformanceMonitor:
    """グローバルパフォーマンスモニターを取得"""
    return global_performance_monitor


# アラート処理のデフォルト実装
def default_alert_handler(alert: PerformanceAlert):
    """デフォルトアラートハンドラ"""
    severity_prefix = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "🔴"}.get(
        alert.severity, "⚠️"
    )

    logger.warning(
        f"{severity_prefix} パフォーマンスアラート: {alert.message}",
        extra={
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "process_name": alert.process_name,
            "threshold": alert.threshold,
            "actual_value": alert.actual_value,
        },
    )


# デフォルトアラートハンドラーを登録
global_performance_monitor.add_alert_callback(default_alert_handler)
