#!/usr/bin/env python3
"""
本番運用監視システム
Issue #436: 本番運用監視システム完成 - 24/7安定運用のためのアラート精度向上とダッシュボード完成

APM・オブザーバビリティ統合基盤による包括的監視システム
分散トレーシング、ログ集約、メトリクス統合、SLO/SLI監視
"""

import asyncio
import json
import logging
import os
import platform
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AlertSeverity(Enum):
    """アラート重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringScope(Enum):
    """監視スコープ"""

    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"


class HealthStatus(Enum):
    """ヘルス状態"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SLOStatus(Enum):
    """SLO状態"""

    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"


@dataclass
class MetricPoint:
    """メトリクスデータポイント"""

    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metric_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
            "metric_name": self.metric_name,
        }


@dataclass
class TraceSpan:
    """分散トレースのスパン"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    tags: Dict[str, str]
    logs: List[Dict[str, Any]]
    status: str  # ok, error, timeout

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogEntry:
    """構造化ログエントリ"""

    timestamp: datetime
    level: str
    message: str
    component: str
    trace_id: Optional[str]
    span_id: Optional[str]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Alert:
    """監視アラート"""

    alert_id: str
    severity: AlertSeverity
    scope: MonitoringScope
    title: str
    description: str
    timestamp: datetime
    source_component: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SLOConfig:
    """SLO設定"""

    name: str
    description: str
    target_percentage: float  # 99.9% = 99.9
    time_window_hours: int  # 24, 168(1week), 720(30days)
    metric_query: str
    error_budget_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SLOStatus:
    """SLO状態"""

    config: SLOConfig
    current_percentage: float
    error_budget_consumed: float
    status: str  # meeting, at_risk, breached
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "current_percentage": self.current_percentage,
            "error_budget_consumed": self.error_budget_consumed,
            "status": self.status,
            "last_updated": self.last_updated.isoformat(),
        }


class DistributedTracer:
    """分散トレーシングシステム"""

    def __init__(self):
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        self.completed_traces: Dict[str, List[TraceSpan]] = {}
        self.trace_lock = Lock()

    def start_trace(self, operation_name: str, parent_span: TraceSpan = None) -> TraceSpan:
        """トレース開始"""
        trace_id = str(uuid.uuid4()) if not parent_span else parent_span.trace_id
        span_id = str(uuid.uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
            tags={},
            logs=[],
            status="active",
        )

        with self.trace_lock:
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = []
            self.active_traces[trace_id].append(span)

        return span

    def finish_span(self, span: TraceSpan, status: str = "ok", error: str = None):
        """スパン終了"""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status

        if error:
            span.logs.append(
                {"timestamp": datetime.now().isoformat(), "level": "error", "message": error}
            )

        # トレースが完了したかチェック
        with self.trace_lock:
            if span.trace_id in self.active_traces:
                active_spans = self.active_traces[span.trace_id]
                all_finished = all(s.end_time is not None for s in active_spans)

                if all_finished:
                    self.completed_traces[span.trace_id] = active_spans
                    del self.active_traces[span.trace_id]

    def add_span_tag(self, span: TraceSpan, key: str, value: str):
        """スパンタグ追加"""
        span.tags[key] = value

    def add_span_log(self, span: TraceSpan, message: str, level: str = "info"):
        """スパンログ追加"""
        span.logs.append(
            {"timestamp": datetime.now().isoformat(), "level": level, "message": message}
        )

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """トレース取得"""
        with self.trace_lock:
            if trace_id in self.active_traces:
                return self.active_traces[trace_id][:]
            elif trace_id in self.completed_traces:
                return self.completed_traces[trace_id][:]
            return []

    @asynccontextmanager
    async def trace_operation(self, operation_name: str, parent_span: TraceSpan = None):
        """トレーシングコンテキストマネージャー"""
        span = self.start_trace(operation_name, parent_span)
        try:
            yield span
            self.finish_span(span, "ok")
        except Exception as e:
            self.finish_span(span, "error", str(e))
            raise


class AnomalyDetector:
    """機械学習による異常検知"""

    def __init__(self):
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.training_threshold = 50  # 50データポイントで学習開始

    def add_metric_point(self, metric_name: str, value: float):
        """メトリクス追加"""
        if not ML_AVAILABLE:
            return

        self.metric_history[metric_name].append({"timestamp": time.time(), "value": value})

        # 十分なデータが蓄積されたら学習
        if len(self.metric_history[metric_name]) >= self.training_threshold:
            self._train_model(metric_name)

    def _train_model(self, metric_name: str):
        """異常検知モデル学習"""
        if not ML_AVAILABLE:
            return

        try:
            values = [point["value"] for point in self.metric_history[metric_name]]
            X = np.array(values).reshape(-1, 1)

            # データ正規化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 異常検知モデル学習
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_scaled)

            self.models[metric_name] = model
            self.scalers[metric_name] = scaler

        except Exception as e:
            logger.error(f"異常検知モデル学習エラー ({metric_name}): {e}")

    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """異常検知"""
        if not ML_AVAILABLE or metric_name not in self.models:
            return False, 0.0

        try:
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]

            X = np.array([[value]])
            X_scaled = scaler.transform(X)

            # 異常スコア計算
            anomaly_score = model.decision_function(X_scaled)[0]
            is_anomaly = model.predict(X_scaled)[0] == -1

            return is_anomaly, abs(anomaly_score)

        except Exception as e:
            logger.error(f"異常検知エラー ({metric_name}): {e}")
            return False, 0.0


class SLOManager:
    """SLO（Service Level Objective）管理"""

    def __init__(self):
        self.slo_configs: Dict[str, SLOConfig] = {}
        self.slo_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.slo_lock = Lock()

        # デフォルトSLO設定
        self._setup_default_slos()

    def _setup_default_slos(self):
        """デフォルトSLO設定"""
        default_slos = [
            SLOConfig(
                name="api_latency",
                description="API応答時間 99.9% < 50ms",
                target_percentage=99.9,
                time_window_hours=24,
                metric_query="api_response_time_p99",
                error_budget_percentage=0.1,
            ),
            SLOConfig(
                name="system_availability",
                description="システム稼働率 99.99%",
                target_percentage=99.99,
                time_window_hours=720,  # 30日
                metric_query="system_uptime",
                error_budget_percentage=0.01,
            ),
            SLOConfig(
                name="error_rate",
                description="エラー率 < 0.1%",
                target_percentage=99.9,
                time_window_hours=168,  # 7日
                metric_query="error_rate",
                error_budget_percentage=0.1,
            ),
            SLOConfig(
                name="data_accuracy",
                description="データ精度 99.95%",
                target_percentage=99.95,
                time_window_hours=24,
                metric_query="data_accuracy",
                error_budget_percentage=0.05,
            ),
        ]

        for slo in default_slos:
            self.slo_configs[slo.name] = slo

    def add_slo_metric(self, slo_name: str, value: float, success: bool):
        """SLOメトリクス追加"""
        if slo_name not in self.slo_configs:
            return

        with self.slo_lock:
            self.slo_history[slo_name].append(
                {"timestamp": datetime.now(), "value": value, "success": success}
            )

    def get_slo_status(self, slo_name: str) -> Optional[SLOStatus]:
        """SLO状態取得"""
        if slo_name not in self.slo_configs:
            return None

        config = self.slo_configs[slo_name]

        with self.slo_lock:
            history = list(self.slo_history[slo_name])

        if not history:
            return SLOStatus(
                config=config,
                current_percentage=0.0,
                error_budget_consumed=0.0,
                status="unknown",
                last_updated=datetime.now(),
            )

        # 時間窓での成功率計算
        cutoff_time = datetime.now() - timedelta(hours=config.time_window_hours)
        relevant_history = [h for h in history if h["timestamp"] >= cutoff_time]

        if not relevant_history:
            return SLOStatus(
                config=config,
                current_percentage=0.0,
                error_budget_consumed=0.0,
                status="unknown",
                last_updated=datetime.now(),
            )

        success_count = sum(1 for h in relevant_history if h["success"])
        total_count = len(relevant_history)
        current_percentage = (success_count / total_count) * 100

        # エラーバジェット消費率計算
        error_budget_consumed = max(
            0, (config.target_percentage - current_percentage) / config.error_budget_percentage
        )

        # 状態判定
        if current_percentage >= config.target_percentage:
            status = "meeting"
        elif error_budget_consumed >= 0.8:
            status = "at_risk"
        else:
            status = "breached"

        return SLOStatus(
            config=config,
            current_percentage=current_percentage,
            error_budget_consumed=error_budget_consumed,
            status=status,
            last_updated=datetime.now(),
        )

    def get_all_slo_status(self) -> Dict[str, SLOStatus]:
        """全SLO状態取得"""
        return {name: self.get_slo_status(name) for name in self.slo_configs.keys()}


class ProductionMonitoringSystem:
    """本番運用監視システム"""

    def __init__(self):
        self.tracer = DistributedTracer()
        self.anomaly_detector = AnomalyDetector()
        self.slo_manager = SLOManager()

        # メトリクス収集
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metrics_lock = Lock()

        # ログ収集
        self.logs: deque = deque(maxlen=50000)
        self.logs_lock = Lock()

        # アラート管理
        self.alerts: List[Alert] = []
        self.alerts_lock = Lock()

        # システム状態
        self.system_health = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now()
        self.monitoring_active = False

        # 統計情報
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "alerts_triggered": 0,
            "anomalies_detected": 0,
            "slo_breaches": 0,
        }

        logger.info("本番運用監視システム初期化完了")

    def start_monitoring(self):
        """監視開始"""
        self.monitoring_active = True

        # 監視スレッド開始
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # ヘルスチェックスレッド開始
        self.health_thread = Thread(target=self._health_check_loop, daemon=True)
        self.health_thread.start()

        logger.info("本番運用監視システム開始")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        logger.info("本番運用監視システム停止")

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """メトリクス記録"""
        if labels is None:
            labels = {}

        point = MetricPoint(timestamp=datetime.now(), value=value, labels=labels, metric_name=name)

        with self.metrics_lock:
            self.metrics[name].append(point)

        # 異常検知
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(name, value)
        if is_anomaly:
            self.stats["anomalies_detected"] += 1
            self._create_alert(
                severity=AlertSeverity.HIGH,
                scope=MonitoringScope.SYSTEM,
                title=f"異常値検知: {name}",
                description=f"メトリクス {name} で異常値を検知しました。値: {value}, 異常スコア: {anomaly_score:.3f}",
                metrics={"metric_name": name, "value": value, "anomaly_score": anomaly_score},
            )

        # ML学習用にデータ追加
        self.anomaly_detector.add_metric_point(name, value)

    def log_structured(
        self,
        level: str,
        message: str,
        component: str,
        trace_id: str = None,
        span_id: str = None,
        **context,
    ):
        """構造化ログ記録"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            context=context,
        )

        with self.logs_lock:
            self.logs.append(entry)

        # 重要レベルのログはアラート化
        if level in ["ERROR", "CRITICAL"]:
            self._create_alert(
                severity=AlertSeverity.HIGH if level == "ERROR" else AlertSeverity.CRITICAL,
                scope=MonitoringScope.APPLICATION,
                title=f"{level}ログ検出: {component}",
                description=message,
                metrics={"log_level": level, "component": component, **context},
            )

    def record_api_call(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        success: bool = True,
    ):
        """API呼び出し記録"""
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1

        # 平均応答時間更新
        self.stats["avg_response_time"] = (
            self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time
        ) / self.stats["total_requests"]

        # メトリクス記録
        self.record_metric(
            "api_response_time",
            response_time,
            {"endpoint": endpoint, "method": method, "status_code": str(status_code)},
        )

        # SLO記録
        self.slo_manager.add_slo_metric("api_latency", response_time, response_time < 50)
        self.slo_manager.add_slo_metric("error_rate", 1.0, success)

    def _create_alert(
        self,
        severity: AlertSeverity,
        scope: MonitoringScope,
        title: str,
        description: str,
        metrics: Dict[str, Any] = None,
    ):
        """アラート作成"""
        if metrics is None:
            metrics = {}

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            scope=scope,
            title=title,
            description=description,
            timestamp=datetime.now(),
            source_component="ProductionMonitoringSystem",
            metrics=metrics,
        )

        with self.alerts_lock:
            self.alerts.append(alert)

        self.stats["alerts_triggered"] += 1
        logger.warning(f"アラート発生: {title}", extra={"alert_id": alert.alert_id})

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                # システムメトリクス収集
                if PSUTIL_AVAILABLE:
                    self._collect_system_metrics()

                # SLO状態チェック
                self._check_slo_status()

                # 30秒間隔
                time.sleep(30)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(60)

    def _health_check_loop(self):
        """ヘルスチェックループ"""
        while self.monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(60)  # 1分間隔
            except Exception as e:
                logger.error(f"ヘルスチェックエラー: {e}")
                time.sleep(120)

    def _collect_system_metrics(self):
        """システムメトリクス収集"""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage_percent", cpu_percent)

            # メモリ使用率
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage_percent", memory.percent)
            self.record_metric("system.memory_available_mb", memory.available / (1024**2))

            # ディスク使用率
            disk_path = "C:\\" if platform.system() == "Windows" else "/"
            disk = psutil.disk_usage(disk_path)
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system.disk_usage_percent", disk_percent)

            # ネットワークI/O
            network = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", network.bytes_sent)
            self.record_metric("system.network_bytes_recv", network.bytes_recv)

            # プロセス数
            process_count = len(psutil.pids())
            self.record_metric("system.process_count", process_count)

        except Exception as e:
            logger.error(f"システムメトリクス収集エラー: {e}")

    def _check_slo_status(self):
        """SLO状態チェック"""
        all_slo_status = self.slo_manager.get_all_slo_status()

        for name, status in all_slo_status.items():
            if not status:
                continue

            if status.status == "breached":
                self.stats["slo_breaches"] += 1
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    scope=MonitoringScope.BUSINESS,
                    title=f"SLO違反: {status.config.name}",
                    description=f"SLO '{status.config.description}' が違反されました。現在値: {status.current_percentage:.2f}%",
                    metrics={
                        "slo_name": name,
                        "target": status.config.target_percentage,
                        "current": status.current_percentage,
                        "error_budget_consumed": status.error_budget_consumed,
                    },
                )
            elif status.status == "at_risk":
                self._create_alert(
                    severity=AlertSeverity.HIGH,
                    scope=MonitoringScope.BUSINESS,
                    title=f"SLOリスク: {status.config.name}",
                    description=f"SLO '{status.config.description}' がリスク状態です。エラーバジェット消費率: {status.error_budget_consumed:.1f}%",
                    metrics={
                        "slo_name": name,
                        "error_budget_consumed": status.error_budget_consumed,
                    },
                )

    def _perform_health_check(self):
        """ヘルスチェック実行"""
        try:
            # 基本的なヘルスチェック
            health_issues = []

            # メトリクス収集状況チェック
            if not self.metrics:
                health_issues.append("メトリクス収集が停止している")

            # アラート過多チェック
            recent_alerts = [
                a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 300
            ]
            if len(recent_alerts) > 10:
                health_issues.append(f"5分間で{len(recent_alerts)}件のアラートが発生")

            # システムリソースチェック
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                if cpu_percent > 90:
                    health_issues.append(f"CPU使用率が高い: {cpu_percent:.1f}%")

                if memory.percent > 90:
                    health_issues.append(f"メモリ使用率が高い: {memory.percent:.1f}%")

            # ヘルス状態決定
            if not health_issues:
                self.system_health = HealthStatus.HEALTHY
            elif len(health_issues) <= 2:
                self.system_health = HealthStatus.DEGRADED
            else:
                self.system_health = HealthStatus.UNHEALTHY

            self.last_health_check = datetime.now()

            # 重要な問題がある場合はアラート
            if health_issues and self.system_health == HealthStatus.UNHEALTHY:
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    scope=MonitoringScope.SYSTEM,
                    title="システムヘルス悪化",
                    description=f"システムヘルスが悪化しています: {', '.join(health_issues)}",
                    metrics={"health_issues": health_issues},
                )

        except Exception as e:
            logger.error(f"ヘルスチェック実行エラー: {e}")
            self.system_health = HealthStatus.UNKNOWN

    def get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ取得"""
        # アクティブアラート統計
        active_alerts = [a for a in self.alerts if not a.resolved]
        alert_stats = {
            "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "high": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            "medium": len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
            "low": len([a for a in active_alerts if a.severity == AlertSeverity.LOW]),
            "total": len(active_alerts),
        }

        # システムメトリクス
        system_metrics = {}
        for metric_name in [
            "system.cpu_usage_percent",
            "system.memory_usage_percent",
            "system.disk_usage_percent",
        ]:
            with self.metrics_lock:
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_point = self.metrics[metric_name][-1]
                    system_metrics[metric_name.replace("system.", "")] = latest_point.value

        # SLO状態
        slo_status = {}
        all_slo_status = self.slo_manager.get_all_slo_status()
        for name, status in all_slo_status.items():
            if status:
                slo_status[name] = {
                    "current_percentage": status.current_percentage,
                    "target_percentage": status.config.target_percentage,
                    "error_budget_consumed": status.error_budget_consumed,
                    "status": status.status,
                }

        return {
            "system_health": {
                "status": self.system_health.value,
                "last_check": self.last_health_check.isoformat(),
                "monitoring_active": self.monitoring_active,
            },
            "alerts": alert_stats,
            "system_metrics": system_metrics,
            "slo_status": slo_status,
            "statistics": self.stats.copy(),
            "traces": {
                "active_traces": len(self.tracer.active_traces),
                "completed_traces": len(self.tracer.completed_traces),
            },
        }

    def get_metrics_summary(self, metric_name: str, duration_minutes: int = 30) -> Dict[str, Any]:
        """メトリクスサマリー取得"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        with self.metrics_lock:
            if metric_name not in self.metrics:
                return {}

            recent_points = [p for p in self.metrics[metric_name] if p.timestamp >= cutoff_time]

        if not recent_points:
            return {}

        values = [p.value for p in recent_points]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "trend": "rising" if len(values) > 1 and values[-1] > values[0] else "falling",
        }

    async def trace_operation(self, operation_name: str, parent_span: TraceSpan = None):
        """トレーシング付きオペレーション"""
        return self.tracer.trace_operation(operation_name, parent_span)

    def get_recent_logs(
        self, level: str = None, component: str = None, limit: int = 100
    ) -> List[LogEntry]:
        """最近のログ取得"""
        with self.logs_lock:
            logs = list(self.logs)

        # フィルタリング
        if level:
            logs = [log for log in logs if log.level == level]

        if component:
            logs = [log for log in logs if log.component == component]

        # 最新順でソートして制限
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]


# 使用例とテスト関数
async def demo_production_monitoring():
    """本番監視システムデモ"""
    print("=== 本番運用監視システムデモ ===")

    # システム初期化
    monitoring = ProductionMonitoringSystem()
    monitoring.start_monitoring()

    print("監視システム開始しました")

    # サンプルデータ記録
    await asyncio.sleep(2)

    # API呼び出し記録
    monitoring.record_api_call("/api/trades", "GET", 25.5, 200, True)
    monitoring.record_api_call("/api/trades", "POST", 45.2, 201, True)
    monitoring.record_api_call("/api/trades", "GET", 150.0, 500, False)  # 異常値

    # 構造化ログ記録
    monitoring.log_structured(
        "INFO", "取引処理開始", "TradingEngine", user_id="12345", symbol="USDJPY"
    )
    monitoring.log_structured(
        "ERROR", "データベース接続エラー", "DatabaseManager", error_code="DB001"
    )

    # カスタムメトリクス記録
    monitoring.record_metric("custom.trading_volume", 1000000.0, {"symbol": "USDJPY"})
    monitoring.record_metric("custom.profit_loss", 5000.0, {"strategy": "scalping"})

    print("\nサンプルデータ記録完了")

    # 少し待機してからダッシュボード表示
    await asyncio.sleep(3)

    dashboard = monitoring.get_dashboard_data()
    print("\n=== ダッシュボード ===")
    print(f"システムヘルス: {dashboard['system_health']['status']}")
    print(f"アクティブアラート: {dashboard['alerts']['total']}件")
    print(f"- Critical: {dashboard['alerts']['critical']}")
    print(f"- High: {dashboard['alerts']['high']}")
    print(f"総リクエスト: {dashboard['statistics']['total_requests']}")
    print(f"平均応答時間: {dashboard['statistics']['avg_response_time']:.2f}ms")
    print(f"異常検知: {dashboard['statistics']['anomalies_detected']}件")

    # SLO状態表示
    if dashboard["slo_status"]:
        print("\n=== SLO状態 ===")
        for name, status in dashboard["slo_status"].items():
            print(
                f"{name}: {status['current_percentage']:.2f}% (目標: {status['target_percentage']:.2f}%) - {status['status']}"
            )

    # トレーシングデモ
    print("\n=== 分散トレーシングデモ ===")
    async with monitoring.trace_operation("sample_trade_execution") as span:
        monitoring.tracer.add_span_tag(span, "symbol", "USDJPY")
        monitoring.tracer.add_span_tag(span, "quantity", "10000")

        # 子スパン
        async with monitoring.trace_operation("market_data_fetch", span) as child_span:
            monitoring.tracer.add_span_log(child_span, "市場データ取得開始")
            await asyncio.sleep(0.1)  # 模擬処理時間
            monitoring.tracer.add_span_log(child_span, "市場データ取得完了")

        # 別の子スパン
        async with monitoring.trace_operation("order_submission", span) as child_span:
            monitoring.tracer.add_span_log(child_span, "注文送信開始")
            await asyncio.sleep(0.05)
            monitoring.tracer.add_span_log(child_span, "注文送信完了")

    print(f"アクティブトレース: {dashboard['traces']['active_traces']}")
    print(f"完了トレース: {dashboard['traces']['completed_traces']}")

    # 最近のログ表示
    recent_logs = monitoring.get_recent_logs(limit=5)
    print(f"\n=== 最近のログ ({len(recent_logs)}件) ===")
    for log in recent_logs[:3]:
        print(
            f"[{log.timestamp.strftime('%H:%M:%S')}] {log.level} - {log.component}: {log.message}"
        )

    monitoring.stop_monitoring()
    print("\n監視システムデモ完了")


if __name__ == "__main__":
    # デモ実行
    asyncio.run(demo_production_monitoring())
