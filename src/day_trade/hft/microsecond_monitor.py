#!/usr/bin/env python3
"""
マイクロ秒精度パフォーマンス監視システム
Issue #366: 高頻度取引最適化エンジン - 超高精度監視

ナノ秒レベルレイテンシー計測、リアルタイム統計、
アラート機能を備えた次世代HFT監視システム
"""

import asyncio
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

# プロジェクトモジュール
try:
    from ..cache.advanced_cache_system import AdvancedCacheSystem
    from ..distributed.distributed_computing_manager import DistributedComputingManager
    from ..utils.logging_config import get_context_logger, log_performance_metric
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # モッククラス
    class DistributedComputingManager:
        def __init__(self):
            pass

        async def execute_distributed_task(self, task):
            return type("MockResult", (), {"success": True, "result": None})()

    class AdvancedCacheSystem:
        def __init__(self):
            pass


logger = get_context_logger(__name__)


class MetricType(IntEnum):
    """メトリクス タイプ"""

    LATENCY = 1
    THROUGHPUT = 2
    ERROR_RATE = 3
    QUEUE_DEPTH = 4
    CPU_USAGE = 5
    MEMORY_USAGE = 6
    NETWORK_IO = 7


class AlertSeverity(IntEnum):
    """アラート重要度"""

    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class LatencyMetrics:
    """レイテンシー測定値"""

    # 基本統計
    count: int = 0
    sum_us: float = 0.0
    min_us: float = float("inf")
    max_us: float = 0.0

    # 高精度統計 (P50, P95, P99, P99.9)
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    p999_us: float = 0.0

    # 時系列データ（最新N個）
    recent_values: deque = field(default_factory=lambda: deque(maxlen=1000))

    # レイテンシー分布（ヒストグラム）
    histogram_buckets: Dict[int, int] = field(default_factory=dict)

    # タイムスタンプ
    last_update_ns: int = field(default_factory=time.perf_counter_ns)

    def update(self, latency_us: float):
        """レイテンシー更新"""
        self.count += 1
        self.sum_us += latency_us
        self.min_us = min(self.min_us, latency_us)
        self.max_us = max(self.max_us, latency_us)

        self.recent_values.append(latency_us)
        self.last_update_ns = time.perf_counter_ns()

        # ヒストグラム更新
        bucket = int(latency_us // 10) * 10  # 10μsバケット
        self.histogram_buckets[bucket] = self.histogram_buckets.get(bucket, 0) + 1

    def calculate_percentiles(self):
        """百分位数計算"""
        if not self.recent_values:
            return

        values = sorted(self.recent_values)
        n = len(values)

        if n > 0:
            self.p50_us = values[int(n * 0.5)]
            self.p95_us = values[int(n * 0.95)]
            self.p99_us = values[int(n * 0.99)]
            self.p999_us = values[int(n * 0.999)] if n >= 1000 else values[-1]

    def get_average_us(self) -> float:
        """平均レイテンシー取得"""
        return self.sum_us / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で取得"""
        return {
            "count": self.count,
            "avg_us": self.get_average_us(),
            "min_us": self.min_us if self.min_us != float("inf") else 0.0,
            "max_us": self.max_us,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "p999_us": self.p999_us,
            "last_update_ns": self.last_update_ns,
        }


@dataclass
class ThroughputMetrics:
    """スループット測定値"""

    total_operations: int = 0
    operations_per_second: float = 0.0
    peak_ops_per_second: float = 0.0

    # 時系列データ
    recent_ops: deque = field(default_factory=lambda: deque(maxlen=60))  # 60秒分
    recent_timestamps: deque = field(default_factory=lambda: deque(maxlen=60))

    last_update_ns: int = field(default_factory=time.perf_counter_ns)

    def update(self, operations: int = 1):
        """スループット更新"""
        now_ns = time.perf_counter_ns()

        self.total_operations += operations
        self.recent_ops.append(operations)
        self.recent_timestamps.append(now_ns)
        self.last_update_ns = now_ns

        # 1秒間のスループット計算
        self._calculate_throughput()

    def _calculate_throughput(self):
        """スループット計算"""
        if len(self.recent_timestamps) < 2:
            return

        # 直近1秒のデータでスループット計算
        now_ns = self.recent_timestamps[-1]
        one_second_ago_ns = now_ns - 1_000_000_000

        ops_in_last_second = 0
        for i in range(len(self.recent_timestamps) - 1, -1, -1):
            if self.recent_timestamps[i] >= one_second_ago_ns:
                ops_in_last_second += self.recent_ops[i]
            else:
                break

        self.operations_per_second = ops_in_last_second
        self.peak_ops_per_second = max(self.peak_ops_per_second, ops_in_last_second)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で取得"""
        return {
            "total_operations": self.total_operations,
            "operations_per_second": self.operations_per_second,
            "peak_ops_per_second": self.peak_ops_per_second,
            "last_update_ns": self.last_update_ns,
        }


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0

    # ネットワークI/O
    network_bytes_sent: int = 0
    network_bytes_received: int = 0

    # ディスクI/O
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0

    # プロセス固有
    process_threads: int = 0
    process_handles: int = 0

    last_update_ns: int = field(default_factory=time.perf_counter_ns)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で取得"""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "memory_usage_percent": self.memory_usage_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_received": self.network_bytes_received,
            "process_threads": self.process_threads,
            "process_handles": self.process_handles,
            "last_update_ns": self.last_update_ns,
        }


@dataclass
class AlertCondition:
    """アラート条件"""

    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True

    # ヒステリシス（チャタリング防止）
    hysteresis_percent: float = 10.0
    consecutive_violations: int = 0
    required_violations: int = 3

    # 通知制限
    last_alert_time: float = 0.0
    alert_cooldown_seconds: float = 60.0


@dataclass
class AlertEvent:
    """アラートイベント"""

    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で取得"""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp_ns": self.timestamp_ns,
        }


class HighResolutionTimer:
    """高解像度タイマー（ナノ秒精度）"""

    def __init__(self):
        self.start_times = {}
        self.cpu_frequency = self._estimate_cpu_frequency()

    def _estimate_cpu_frequency(self) -> float:
        """CPU周波数推定"""
        try:
            if os.name == "posix":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "cpu MHz" in line:
                            return float(line.split(":")[1].strip()) * 1e6
        except:
            pass
        return 3.0e9  # 3GHz fallback

    def start_timing(self, operation_id: str) -> int:
        """タイミング開始"""
        start_time_ns = time.perf_counter_ns()
        self.start_times[operation_id] = start_time_ns
        return start_time_ns

    def end_timing(self, operation_id: str) -> Optional[float]:
        """タイミング終了（マイクロ秒で返す）"""
        end_time_ns = time.perf_counter_ns()
        start_time_ns = self.start_times.pop(operation_id, None)

        if start_time_ns is None:
            return None

        return (end_time_ns - start_time_ns) / 1000.0

    def time_operation(self, operation_id: str):
        """コンテキストマネージャー"""

        class TimingContext:
            def __init__(self, timer, op_id):
                self.timer = timer
                self.operation_id = op_id
                self.duration_us = None

            def __enter__(self):
                self.timer.start_timing(self.operation_id)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.duration_us = self.timer.end_timing(self.operation_id)

        return TimingContext(self, operation_id)


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""

    report_id: str
    generation_time_ns: int = field(default_factory=time.perf_counter_ns)

    # メトリクス集約
    latency_metrics: Dict[str, LatencyMetrics] = field(default_factory=dict)
    throughput_metrics: Dict[str, ThroughputMetrics] = field(default_factory=dict)
    system_metrics: SystemMetrics = field(default_factory=SystemMetrics)

    # アラート状況
    active_alerts: List[AlertEvent] = field(default_factory=list)
    resolved_alerts: List[AlertEvent] = field(default_factory=list)

    # パフォーマンス要約
    overall_health_score: float = 100.0  # 0-100
    performance_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で取得"""
        return {
            "report_id": self.report_id,
            "generation_time_ns": self.generation_time_ns,
            "latency_metrics": {
                name: metrics.to_dict() for name, metrics in self.latency_metrics.items()
            },
            "throughput_metrics": {
                name: metrics.to_dict() for name, metrics in self.throughput_metrics.items()
            },
            "system_metrics": self.system_metrics.to_dict(),
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "resolved_alerts": [alert.to_dict() for alert in self.resolved_alerts],
            "overall_health_score": self.overall_health_score,
            "performance_summary": self.performance_summary,
        }


class MicrosecondMonitor:
    """
    マイクロ秒精度パフォーマンス監視システム

    ナノ秒レベル計測、リアルタイム統計分析、
    インテリジェントアラート機能を提供
    """

    def __init__(
        self,
        distributed_manager: Optional[DistributedComputingManager] = None,
        cache_system: Optional[AdvancedCacheSystem] = None,
        monitoring_interval_ms: int = 100,
        history_retention_hours: int = 24,
    ):
        """
        初期化

        Args:
            distributed_manager: 分散処理マネージャー
            cache_system: キャッシュシステム
            monitoring_interval_ms: 監視間隔（ミリ秒）
            history_retention_hours: 履歴保持時間
        """
        self.distributed_manager = distributed_manager or DistributedComputingManager()
        self.cache_system = cache_system or AdvancedCacheSystem()
        self.monitoring_interval_ms = monitoring_interval_ms
        self.history_retention_hours = history_retention_hours

        # 高精度タイマー
        self.timer = HighResolutionTimer()

        # メトリクス集約
        self.latency_metrics: Dict[str, LatencyMetrics] = {}
        self.throughput_metrics: Dict[str, ThroughputMetrics] = {}
        self.system_metrics = SystemMetrics()

        # アラート管理
        self.alert_conditions: Dict[str, AlertCondition] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[AlertEvent], None]] = []

        # 監視スレッド
        self.monitoring_thread = None
        self.running = False

        # パフォーマンスレポート履歴
        self.report_history: deque = deque(maxlen=100)

        # 統計キャッシュ
        self.stats_cache = {}
        self.last_stats_update = 0

        # デフォルトアラート条件設定
        self._setup_default_alerts()

        logger.info("MicrosecondMonitor初期化完了")

    def _setup_default_alerts(self):
        """デフォルトアラート条件設定"""
        # 実行レイテンシーアラート
        self.add_alert_condition(
            AlertCondition(
                metric_name="execution_latency_avg",
                threshold_value=100.0,  # 100μs
                comparison="gt",
                severity=AlertSeverity.WARNING,
                required_violations=5,
            )
        )

        self.add_alert_condition(
            AlertCondition(
                metric_name="execution_latency_p99",
                threshold_value=500.0,  # 500μs
                comparison="gt",
                severity=AlertSeverity.ERROR,
                required_violations=3,
            )
        )

        # スループットアラート
        self.add_alert_condition(
            AlertCondition(
                metric_name="execution_throughput",
                threshold_value=1000.0,  # 1000 ops/sec
                comparison="lt",
                severity=AlertSeverity.WARNING,
                required_violations=10,
            )
        )

        # システムリソースアラート
        self.add_alert_condition(
            AlertCondition(
                metric_name="cpu_usage",
                threshold_value=90.0,  # 90%
                comparison="gt",
                severity=AlertSeverity.WARNING,
                required_violations=5,
            )
        )

        self.add_alert_condition(
            AlertCondition(
                metric_name="memory_usage",
                threshold_value=85.0,  # 85%
                comparison="gt",
                severity=AlertSeverity.ERROR,
                required_violations=3,
            )
        )

    def record_latency(self, metric_name: str, latency_us: float):
        """レイテンシー記録"""
        if metric_name not in self.latency_metrics:
            self.latency_metrics[metric_name] = LatencyMetrics()

        self.latency_metrics[metric_name].update(latency_us)

        # 百分位数更新（適度な頻度で）
        if self.latency_metrics[metric_name].count % 100 == 0:
            self.latency_metrics[metric_name].calculate_percentiles()

        # アラート条件チェック
        self._check_metric_alerts(
            f"{metric_name}_avg", self.latency_metrics[metric_name].get_average_us()
        )
        self._check_metric_alerts(f"{metric_name}_max", latency_us)

    def record_throughput(self, metric_name: str, operations: int = 1):
        """スループット記録"""
        if metric_name not in self.throughput_metrics:
            self.throughput_metrics[metric_name] = ThroughputMetrics()

        self.throughput_metrics[metric_name].update(operations)

        # アラート条件チェック
        self._check_metric_alerts(
            f"{metric_name}_throughput",
            self.throughput_metrics[metric_name].operations_per_second,
        )

    def time_operation(self, operation_name: str):
        """操作時間測定（コンテキストマネージャー）"""

        class MonitoringContext:
            def __init__(self, monitor, op_name):
                self.monitor = monitor
                self.operation_name = op_name
                self.start_time_ns = None

            def __enter__(self):
                self.start_time_ns = time.perf_counter_ns()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time_ns = time.perf_counter_ns()
                duration_us = (end_time_ns - self.start_time_ns) / 1000.0

                self.monitor.record_latency(self.operation_name, duration_us)
                self.monitor.record_throughput(self.operation_name, 1)

        return MonitoringContext(self, operation_name)

    def add_alert_condition(self, condition: AlertCondition):
        """アラート条件追加"""
        alert_id = f"{condition.metric_name}_{condition.comparison}_{condition.threshold_value}"
        self.alert_conditions[alert_id] = condition
        logger.debug(f"アラート条件追加: {alert_id}")

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    def _check_metric_alerts(self, metric_name: str, current_value: float):
        """メトリクスアラート条件チェック"""
        for alert_id, condition in self.alert_conditions.items():
            if not condition.enabled or condition.metric_name != metric_name:
                continue

            # 閾値チェック
            violation = False
            if condition.comparison == "gt" and current_value > condition.threshold_value:
                violation = True
            elif condition.comparison == "lt" and current_value < condition.threshold_value:
                violation = True
            elif (
                condition.comparison == "eq"
                and abs(current_value - condition.threshold_value) < 0.001
            ):
                violation = True

            if violation:
                condition.consecutive_violations += 1

                # 連続違反回数チェック
                if condition.consecutive_violations >= condition.required_violations:
                    self._trigger_alert(alert_id, condition, current_value)
            else:
                # 違反解決
                if condition.consecutive_violations > 0:
                    condition.consecutive_violations = 0
                    self._resolve_alert(alert_id)

    def _trigger_alert(self, alert_id: str, condition: AlertCondition, current_value: float):
        """アラート発火"""
        now = time.time()

        # クールダウンチェック
        if now - condition.last_alert_time < condition.alert_cooldown_seconds:
            return

        # アラートイベント作成
        alert_event = AlertEvent(
            alert_id=alert_id,
            metric_name=condition.metric_name,
            current_value=current_value,
            threshold_value=condition.threshold_value,
            severity=condition.severity,
            message=f"{condition.metric_name} {condition.comparison} {condition.threshold_value}: current={current_value:.3f}",
        )

        # アクティブアラートに追加
        self.active_alerts[alert_id] = alert_event
        self.alert_history.append(alert_event)

        condition.last_alert_time = now

        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                callback(alert_event)
            except Exception as e:
                logger.error(f"アラートコールバックエラー: {e}")

        logger.warning(f"アラート発火: {alert_event.message}")

    def _resolve_alert(self, alert_id: str):
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            logger.info(f"アラート解決: {alert.metric_name}")

    def _update_system_metrics(self):
        """システムメトリクス更新"""
        try:
            # CPU使用率（簡易実装）
            import psutil

            self.system_metrics.cpu_usage_percent = psutil.cpu_percent()

            # メモリ使用率
            memory = psutil.virtual_memory()
            self.system_metrics.memory_usage_percent = memory.percent
            self.system_metrics.memory_usage_mb = memory.used / (1024 * 1024)

            # プロセス情報
            process = psutil.Process()
            self.system_metrics.process_threads = process.num_threads()

        except ImportError:
            # psutilが利用できない場合の簡易実装
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                self.system_metrics.memory_usage_mb = usage.ru_maxrss / 1024  # KB to MB
            except:
                pass
        except Exception as e:
            logger.debug(f"システムメトリクス更新エラー: {e}")

        self.system_metrics.last_update_ns = time.perf_counter_ns()

        # アラート条件チェック
        self._check_metric_alerts("cpu_usage", self.system_metrics.cpu_usage_percent)
        self._check_metric_alerts("memory_usage", self.system_metrics.memory_usage_percent)

    def start_monitoring(self):
        """監視開始"""
        if self.running:
            return

        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="MicrosecondMonitor", daemon=True
        )
        self.monitoring_thread.start()
        logger.info("マイクロ秒監視開始")

    def stop_monitoring(self):
        """監視停止"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("マイクロ秒監視停止")

    def _monitoring_loop(self):
        """監視メインループ"""
        logger.info("マイクロ秒監視ループ開始")

        while self.running:
            try:
                # システムメトリクス更新
                self._update_system_metrics()

                # 百分位数定期計算
                for metrics in self.latency_metrics.values():
                    if metrics.count > 0 and len(metrics.recent_values) >= 10:
                        metrics.calculate_percentiles()

                # 統計キャッシュ更新
                self._update_stats_cache()

                # パフォーマンスレポート生成（定期的）
                if (
                    len(self.report_history) == 0
                    or time.perf_counter_ns() - self.report_history[-1].generation_time_ns
                    > 60_000_000_000
                ):  # 60秒
                    self._generate_performance_report()

                # 監視間隔待機
                time.sleep(self.monitoring_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")

    def _update_stats_cache(self):
        """統計キャッシュ更新"""
        now_ns = time.perf_counter_ns()
        if now_ns - self.last_stats_update < 1_000_000_000:  # 1秒間隔
            return

        self.stats_cache = {
            "latency_stats": {
                name: metrics.to_dict() for name, metrics in self.latency_metrics.items()
            },
            "throughput_stats": {
                name: metrics.to_dict() for name, metrics in self.throughput_metrics.items()
            },
            "system_stats": self.system_metrics.to_dict(),
            "active_alerts_count": len(self.active_alerts),
            "cache_update_time_ns": now_ns,
        }

        self.last_stats_update = now_ns

    def _generate_performance_report(self):
        """パフォーマンスレポート生成"""
        report_id = f"report_{int(time.time())}"

        report = PerformanceReport(
            report_id=report_id,
            latency_metrics=self.latency_metrics.copy(),
            throughput_metrics=self.throughput_metrics.copy(),
            system_metrics=self.system_metrics,
            active_alerts=list(self.active_alerts.values()),
            resolved_alerts=[],  # 最近解決されたアラート
        )

        # 健全性スコア計算
        report.overall_health_score = self._calculate_health_score()

        # パフォーマンス要約
        report.performance_summary = self._create_performance_summary()

        self.report_history.append(report)

        logger.info(
            f"パフォーマンスレポート生成: {report_id}, ヘルススコア: {report.overall_health_score:.1f}"
        )

    def _calculate_health_score(self) -> float:
        """健全性スコア計算（0-100）"""
        score = 100.0

        # アクティブアラート数による減点
        score -= len(self.active_alerts) * 5.0

        # レイテンシー目標達成度
        for name, metrics in self.latency_metrics.items():
            if metrics.count > 0:
                avg_latency = metrics.get_average_us()
                if avg_latency > 100:  # 100μs目標
                    score -= min(20.0, (avg_latency - 100) / 10)

        # システムリソース使用率
        if self.system_metrics.cpu_usage_percent > 80:
            score -= (self.system_metrics.cpu_usage_percent - 80) * 0.5

        if self.system_metrics.memory_usage_percent > 80:
            score -= (self.system_metrics.memory_usage_percent - 80) * 0.5

        return max(0.0, min(100.0, score))

    def _create_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約作成"""
        summary = {}

        # レイテンシー要約
        if self.latency_metrics:
            total_ops = sum(m.count for m in self.latency_metrics.values())
            avg_latency = (
                sum(m.sum_us for m in self.latency_metrics.values()) / total_ops
                if total_ops > 0
                else 0
            )

            summary["latency_summary"] = {
                "total_operations": total_ops,
                "average_latency_us": avg_latency,
                "sub50us_operations": sum(
                    sum(1 for v in m.recent_values if v < 50) for m in self.latency_metrics.values()
                ),
            }

        # スループット要約
        if self.throughput_metrics:
            total_throughput = sum(
                m.operations_per_second for m in self.throughput_metrics.values()
            )
            peak_throughput = max(m.peak_ops_per_second for m in self.throughput_metrics.values())

            summary["throughput_summary"] = {
                "current_total_ops_per_sec": total_throughput,
                "peak_total_ops_per_sec": peak_throughput,
            }

        return summary

    def get_current_stats(self) -> Dict[str, Any]:
        """現在の統計情報取得"""
        return self.stats_cache.copy() if self.stats_cache else {}

    def get_latest_report(self) -> Optional[PerformanceReport]:
        """最新パフォーマンスレポート取得"""
        return self.report_history[-1] if self.report_history else None

    def get_alert_summary(self) -> Dict[str, Any]:
        """アラート要約取得"""
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.name] += 1

        return {
            "active_alerts_count": len(self.active_alerts),
            "severity_breakdown": dict(severity_counts),
            "recent_alerts": [alert.to_dict() for alert in list(self.alert_history)[-10:]],
        }

    def reset_metrics(self):
        """メトリクスリセット"""
        self.latency_metrics.clear()
        self.throughput_metrics.clear()
        self.active_alerts.clear()
        self.alert_history.clear()
        logger.info("メトリクスリセット完了")

    async def cleanup(self):
        """クリーンアップ"""
        logger.info("MicrosecondMonitor クリーンアップ開始")

        self.stop_monitoring()

        # アラートコールバッククリア
        self.alert_callbacks.clear()

        # メトリクスクリア
        self.reset_metrics()

        logger.info("クリーンアップ完了")


# Factory function
def create_microsecond_monitor(
    distributed_manager: Optional[DistributedComputingManager] = None,
    cache_system: Optional[AdvancedCacheSystem] = None,
    **config,
) -> MicrosecondMonitor:
    """MicrosecondMonitorファクトリ関数"""
    return MicrosecondMonitor(
        distributed_manager=distributed_manager, cache_system=cache_system, **config
    )


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #366 マイクロ秒精度パフォーマンス監視テスト ===")

        monitor = None
        try:
            # 監視システム初期化
            monitor = create_microsecond_monitor(monitoring_interval_ms=50)  # 50ms間隔

            # アラートコールバック設定
            def alert_handler(alert: AlertEvent):
                print(f"🚨 アラート: {alert.message} (重要度: {alert.severity.name})")

            monitor.add_alert_callback(alert_handler)

            # 監視開始
            monitor.start_monitoring()

            print("\n1. レイテンシー測定テスト")
            # 各種操作のレイテンシー測定
            test_operations = [
                ("order_validation", lambda: time.sleep(0.000005)),  # 5μs
                ("order_execution", lambda: time.sleep(0.000050)),  # 50μs
                ("market_data_processing", lambda: time.sleep(0.000010)),  # 10μs
            ]

            for op_name, operation in test_operations:
                for i in range(100):
                    with monitor.time_operation(op_name):
                        operation()

                    # 一部で意図的に高レイテンシー
                    if i % 20 == 19:
                        with monitor.time_operation(op_name):
                            time.sleep(0.000200)  # 200μs (高レイテンシー)

            # 統計確認のため少し待機
            await asyncio.sleep(2)

            print("\n2. 現在の統計情報")
            stats = monitor.get_current_stats()
            if stats.get("latency_stats"):
                for metric_name, data in stats["latency_stats"].items():
                    print(f"  {metric_name}:")
                    print(f"    平均: {data['avg_us']:.1f}μs")
                    print(f"    P95: {data['p95_us']:.1f}μs")
                    print(f"    P99: {data['p99_us']:.1f}μs")
                    print(f"    処理回数: {data['count']}")

            print("\n3. スループット統計")
            if stats.get("throughput_stats"):
                for metric_name, data in stats["throughput_stats"].items():
                    print(f"  {metric_name}: {data['operations_per_second']:.1f} ops/sec")
                    print(f"    ピーク: {data['peak_ops_per_second']:.1f} ops/sec")

            print("\n4. システムメトリクス")
            sys_stats = stats.get("system_stats", {})
            print(f"  CPU使用率: {sys_stats.get('cpu_usage_percent', 0):.1f}%")
            print(f"  メモリ使用率: {sys_stats.get('memory_usage_percent', 0):.1f}%")
            print(f"  メモリ使用量: {sys_stats.get('memory_usage_mb', 0):.1f}MB")

            print("\n5. アラート状況")
            alert_summary = monitor.get_alert_summary()
            print(f"  アクティブアラート: {alert_summary['active_alerts_count']}")
            if alert_summary["severity_breakdown"]:
                print("  重要度別:")
                for severity, count in alert_summary["severity_breakdown"].items():
                    print(f"    {severity}: {count}")

            print("\n6. 最新パフォーマンスレポート")
            latest_report = monitor.get_latest_report()
            if latest_report:
                print(f"  レポートID: {latest_report.report_id}")
                print(f"  健全性スコア: {latest_report.overall_health_score:.1f}/100")

                if latest_report.performance_summary:
                    summary = latest_report.performance_summary
                    if "latency_summary" in summary:
                        lat_sum = summary["latency_summary"]
                        print(f"  総処理数: {lat_sum['total_operations']}")
                        print(f"  平均レイテンシー: {lat_sum['average_latency_us']:.1f}μs")
                        print(f"  <50μs処理: {lat_sum['sub50us_operations']}")

            # 高レイテンシーでアラート発生テスト
            print("\n7. アラート発生テスト（高レイテンシー）")
            for i in range(10):
                with monitor.time_operation("test_high_latency"):
                    time.sleep(0.000150)  # 150μs（閾値超過）

            await asyncio.sleep(1)  # アラート処理待機

            print("\n8. 最終統計確認")
            final_stats = monitor.get_current_stats()
            final_alert_summary = monitor.get_alert_summary()
            print(f"  最終アラート数: {final_alert_summary['active_alerts_count']}")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if monitor:
                await monitor.cleanup()

        print("\n=== マイクロ秒精度パフォーマンス監視テスト完了 ===")

    asyncio.run(main())
