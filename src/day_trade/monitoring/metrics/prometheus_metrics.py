#!/usr/bin/env python3
"""
Prometheus リアルタイムメトリクス収集システム
Day Trade システム専用メトリクス定義・収集・アラート機能

Features:
- リアルタイムメトリクス収集
- インテリジェントアラート
- 異常検知統合
- パフォーマンス最適化
- SLA監視
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricConfig:
    """メトリクス設定"""

    name: str
    help: str
    labels: List[str] = None
    buckets: List[float] = None


@dataclass
class AlertRule:
    """アラートルール定義"""

    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    duration: int  # seconds
    description: str
    labels: Dict[str, str] = None


@dataclass
class MetricSnapshot:
    """メトリクススナップショット"""

    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AnomalyDetectionResult:
    """異常検知結果"""

    is_anomaly: bool
    confidence: float
    explanation: str
    metric_name: str
    timestamp: datetime
    value: float
    expected_range: tuple = None


class PrometheusMetricsCollector:
    """メインメトリクス収集器"""

    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        self._last_collection_time = time.time()

    def _initialize_metrics(self):
        """基本メトリクス初期化"""

        # システム情報
        self.system_info = Info(
            "day_trade_system_info", "Day Trade システム情報", registry=self.registry
        )

        # 収集統計
        self.metrics_collection_total = Counter(
            "day_trade_metrics_collection_total",
            "メトリクス収集回数",
            ["collector_type"],
            registry=self.registry,
        )

        self.metrics_collection_duration = Histogram(
            "day_trade_metrics_collection_duration_seconds",
            "メトリクス収集時間",
            ["collector_type"],
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0],
            registry=self.registry,
        )

        # リアルタイム監視メトリクス
        self.realtime_data_latency = Histogram(
            "day_trade_realtime_data_latency_seconds",
            "リアルタイムデータ遅延",
            ["data_source", "symbol"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        self.active_websocket_connections = Gauge(
            "day_trade_active_websocket_connections",
            "アクティブWebSocket接続数",
            ["endpoint"],
            registry=self.registry,
        )

        self.market_data_updates_total = Counter(
            "day_trade_market_data_updates_total",
            "マーケットデータ更新回数",
            ["symbol", "data_type"],
            registry=self.registry,
        )

        self.prediction_confidence_score = Gauge(
            "day_trade_prediction_confidence_score",
            "予測信頼度スコア",
            ["model_type", "symbol"],
            registry=self.registry,
        )

        # SLA監視メトリクス
        self.sla_availability = Gauge(
            "day_trade_sla_availability_percent",
            "SLA可用性",
            ["service"],
            registry=self.registry,
        )

        self.response_time_percentile = Gauge(
            "day_trade_response_time_percentile_seconds",
            "レスポンス時間パーセンタイル",
            ["service", "percentile"],
            registry=self.registry,
        )

        # システム初期化
        self.system_info.info(
            {
                "version": "2.1.0",
                "environment": "production",
                "start_time": datetime.now().isoformat(),
                "features": "realtime,alerts,anomaly_detection",
            }
        )

        # リアルタイム監視データ構造
        self._realtime_snapshots = deque(maxlen=1000)
        self._metric_buffers = defaultdict(lambda: deque(maxlen=100))
        self._alert_history = deque(maxlen=500)
        self._last_alert_times = {}
        self._anomaly_detection_enabled = True

    def collect_all_metrics(self) -> Dict[str, Any]:
        """全メトリクス収集"""

        start_time = time.time()

        try:
            # 基本統計更新
            self.metrics_collection_total.labels(collector_type="all").inc()

            collection_time = time.time() - start_time
            self.metrics_collection_duration.labels(collector_type="all").observe(collection_time)

            # リアルタイムスナップショット作成
            snapshot = self._create_realtime_snapshot()
            self._store_metric_snapshot(snapshot)

            # 異常検知実行
            if self._anomaly_detection_enabled:
                anomalies = self._detect_anomalies()
                if anomalies:
                    self._process_anomalies(anomalies)

            return {
                "status": "success",
                "collection_time": collection_time,
                "timestamp": datetime.now().isoformat(),
                "snapshots_count": len(self._realtime_snapshots),
                "anomalies_detected": len(getattr(self, "_last_anomalies", [])),
            }

        except Exception as e:
            logger.error(f"メトリクス収集エラー: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _create_realtime_snapshot(self) -> Dict[str, Any]:
        """リアルタイムスナップショット作成"""

        # SystemHealthMetricsから最新のシステムメトリクスを取得
        from .prometheus_metrics import get_health_metrics  # 自己参照インポート

        health_metrics = get_health_metrics()

        timestamp = datetime.now()
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "system_metrics": {
                "cpu_usage": health_metrics.cpu_usage.get(),
                "memory_usage": (
                    health_metrics.memory_usage.labels(type="used").get()
                    / health_metrics.memory_usage.labels(type="total").get()
                    * 100
                    if health_metrics.memory_usage.labels(type="total").get()
                    else 0
                ),  # Convert bytes to percentage
                "disk_usage": (
                    health_metrics.disk_usage.labels(mount_point="/", type="used").get()
                    / health_metrics.disk_usage.labels(mount_point="/", type="total").get()
                    * 100
                    if health_metrics.disk_usage.labels(mount_point="/", type="total").get()
                    else 0
                ),
            },
            "collection_count": self.metrics_collection_total._value._value,
        }

        return snapshot

    def _store_metric_snapshot(self, snapshot: Dict[str, Any]):
        """メトリクススナップショット保存"""

        self._realtime_snapshots.append(snapshot)

        # バッファにも保存（異常検知用）
        timestamp = datetime.fromisoformat(snapshot["timestamp"].replace("Z", "+00:00"))
        for metric_name, value in snapshot.get("system_metrics", {}).items():
            self._metric_buffers[metric_name].append({"timestamp": timestamp, "value": value})

    def _detect_anomalies(self) -> List[AnomalyDetectionResult]:
        """異常検知実行"""

        anomalies = []

        try:
            for metric_name, buffer in self._metric_buffers.items():
                if len(buffer) < 10:  # 最小サンプル数
                    continue

                values = [item["value"] for item in buffer]
                timestamps = [item["timestamp"] for item in buffer]

                # 簡単な統計的異常検知
                mean_val = np.mean(values)
                std_val = np.std(values)
                current_val = values[-1]
                current_timestamp = timestamps[-1]

                # Z-score based anomaly detection
                if std_val > 0:
                    z_score = abs(current_val - mean_val) / std_val

                    if z_score > 3.0:  # 3σを超える場合
                        anomalies.append(
                            AnomalyDetectionResult(
                                is_anomaly=True,
                                confidence=min(z_score / 3.0, 1.0),
                                explanation=f"{metric_name}の値が異常: {current_val:.2f} (平均: {mean_val:.2f}, Z-score: {z_score:.2f})",
                                metric_name=metric_name,
                                timestamp=current_timestamp,
                                value=current_val,
                                expected_range=(
                                    mean_val - 2 * std_val,
                                    mean_val + 2 * std_val,
                                ),
                            )
                        )

        except Exception as e:
            logger.error(f"異常検知エラー: {e}")

        return anomalies

    def _process_anomalies(self, anomalies: List[AnomalyDetectionResult]):
        """異常処理"""

        self._last_anomalies = anomalies

        for anomaly in anomalies:
            # アラート履歴に追加
            alert_data = {
                "timestamp": anomaly.timestamp.isoformat(),
                "type": "anomaly",
                "severity": "warning" if anomaly.confidence < 0.8 else "critical",
                "metric_name": anomaly.metric_name,
                "value": anomaly.value,
                "confidence": anomaly.confidence,
                "explanation": anomaly.explanation,
            }

            self._alert_history.append(alert_data)

            logger.warning(f"異常検知: {anomaly.explanation}")

    def record_realtime_data_latency(self, data_source: str, symbol: str, latency: float):
        """リアルタイムデータ遅延記録"""

        self.realtime_data_latency.labels(data_source=data_source, symbol=symbol).observe(latency)

    def update_websocket_connections(self, endpoint: str, count: int):
        """WebSocket接続数更新"""

        self.active_websocket_connections.labels(endpoint=endpoint).set(count)

    def record_market_data_update(self, symbol: str, data_type: str):
        """マーケットデータ更新記録"""

        self.market_data_updates_total.labels(symbol=symbol, data_type=data_type).inc()

    def update_prediction_confidence(self, model_type: str, symbol: str, confidence: float):
        """予測信頼度更新"""

        self.prediction_confidence_score.labels(model_type=model_type, symbol=symbol).set(
            confidence
        )

    def update_sla_availability(self, service: str, availability_percent: float):
        """SLA可用性更新"""

        self.sla_availability.labels(service=service).set(availability_percent)

    def update_response_time_percentile(self, service: str, percentile: str, time_seconds: float):
        """レスポンス時間パーセンタイル更新"""

        self.response_time_percentile.labels(service=service, percentile=percentile).set(
            time_seconds
        )

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """アラート履歴取得"""

        return list(self._alert_history)[-limit:]

    def get_realtime_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """リアルタイムスナップショット取得"""

        return list(self._realtime_snapshots)[-limit:]

    def enable_anomaly_detection(self, enabled: bool = True):
        """異常検知有効化/無効化"""

        self._anomaly_detection_enabled = enabled
        logger.info(f"異常検知: {'有効' if enabled else '無効'}")


class RiskManagementMetrics:
    """リスク管理システムメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_risk_metrics()

    def _initialize_risk_metrics(self):
        """リスク管理メトリクス初期化"""

        # リスク分析実行回数
        self.risk_analysis_total = Counter(
            "day_trade_risk_analysis_total",
            "リスク分析実行回数",
            ["analysis_type", "risk_level"],
            registry=self.registry,
        )

        # リスク分析処理時間
        self.risk_analysis_duration = Histogram(
            "day_trade_risk_analysis_duration_seconds",
            "リスク分析処理時間",
            ["analysis_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # 現在のリスクスコア
        self.current_risk_score = Gauge(
            "day_trade_current_risk_score",
            "現在のリスクスコア",
            ["component", "symbol"],
            registry=self.registry,
        )

        # リスクアラート発生回数
        self.risk_alerts_total = Counter(
            "day_trade_risk_alerts_total",
            "リスクアラート発生回数",
            ["alert_level", "component"],
            registry=self.registry,
        )

        # 不正検知結果
        self.fraud_detection_total = Counter(
            "day_trade_fraud_detection_total",
            "不正検知実行回数",
            ["result", "confidence_level"],
            registry=self.registry,
        )

        # 不正検知精度
        self.fraud_detection_accuracy = Gauge(
            "day_trade_fraud_detection_accuracy",
            "不正検知精度",
            ["model_type"],
            registry=self.registry,
        )

        # リスク分析コンポーネント状態
        self.risk_component_status = Gauge(
            "day_trade_risk_component_status",
            "リスクコンポーネント状態 (1=正常, 0=異常)",
            ["component"],
            registry=self.registry,
        )

    def record_risk_analysis(self, analysis_type: str, risk_level: str, duration: float):
        """リスク分析記録"""

        self.risk_analysis_total.labels(analysis_type=analysis_type, risk_level=risk_level).inc()

        self.risk_analysis_duration.labels(analysis_type=analysis_type).observe(duration)

    def update_risk_score(self, component: str, symbol: str, score: float):
        """リスクスコア更新"""

        self.current_risk_score.labels(component=component, symbol=symbol).set(score)

    def record_fraud_detection(
        self,
        result: str,
        confidence: str,
        accuracy: float = None,
        model_type: str = "ensemble",
    ):
        """不正検知結果記録"""

        self.fraud_detection_total.labels(result=result, confidence_level=confidence).inc()

        if accuracy is not None:
            self.fraud_detection_accuracy.labels(model_type=model_type).set(accuracy)


class TradingPerformanceMetrics:
    """取引パフォーマンスメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_trading_metrics()

    def _initialize_trading_metrics(self):
        """取引メトリクス初期化"""

        # 取引実行回数
        self.trades_total = Counter(
            "day_trade_trades_total",
            "取引実行回数",
            ["trade_type", "symbol", "result"],
            registry=self.registry,
        )

        # 取引執行時間
        self.trade_execution_duration = Histogram(
            "day_trade_execution_duration_seconds",
            "取引執行時間",
            ["trade_type"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        # ポートフォリオ価値
        self.portfolio_value = Gauge(
            "day_trade_portfolio_value",
            "ポートフォリオ価値",
            ["currency"],
            registry=self.registry,
        )

        # 未実現損益
        self.unrealized_pnl = Gauge(
            "day_trade_unrealized_pnl",
            "未実現損益",
            ["symbol", "currency"],
            registry=self.registry,
        )

        # 実現損益
        self.realized_pnl = Counter(
            "day_trade_realized_pnl_total",
            "実現損益累計",
            ["symbol", "currency"],
            registry=self.registry,
        )

        # 勝率
        self.win_rate = Gauge(
            "day_trade_win_rate",
            "勝率",
            ["strategy", "timeframe"],
            registry=self.registry,
        )

        # 最大ドローダウン
        self.max_drawdown = Gauge(
            "day_trade_max_drawdown",
            "最大ドローダウン",
            ["strategy"],
            registry=self.registry,
        )


class AIEngineMetrics:
    """AI エンジンメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_ai_metrics()

    def _initialize_ai_metrics(self):
        """AI エンジンメトリクス初期化"""

        # AI予測実行回数
        self.ai_predictions_total = Counter(
            "day_trade_ai_predictions_total",
            "AI予測実行回数",
            ["model_type", "symbol"],
            registry=self.registry,
        )

        # AI予測処理時間
        self.ai_prediction_duration = Histogram(
            "day_trade_ai_prediction_duration_seconds",
            "AI予測処理時間",
            ["model_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # 予測精度
        self.prediction_accuracy = Gauge(
            "day_trade_prediction_accuracy",
            "予測精度",
            ["model_type", "timeframe"],
            registry=self.registry,
        )

        # 生成AI API呼び出し
        self.generative_ai_calls_total = Counter(
            "day_trade_generative_ai_calls_total",
            "生成AI API呼び出し回数",
            ["provider", "model", "result"],
            registry=self.registry,
        )

        # GPU使用率
        self.gpu_utilization = Gauge(
            "day_trade_gpu_utilization_percent",
            "GPU使用率",
            ["gpu_id"],
            registry=self.registry,
        )

        # モデル訓練状況
        self.model_training_status = Gauge(
            "day_trade_model_training_status",
            "モデル訓練状況 (1=訓練中, 0=停止)",
            ["model_type"],
            registry=self.registry,
        )


class SystemHealthMetrics:
    """システムヘルスメトリクス"""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self._initialize_health_metrics()
        self._start_system_monitoring()

    def _initialize_health_metrics(self):
        """ヘルスメトリクス初期化"""

        # CPU使用率
        self.cpu_usage = Gauge("day_trade_cpu_usage_percent", "CPU使用率", registry=self.registry)

        # メモリ使用量
        self.memory_usage = Gauge(
            "day_trade_memory_usage_bytes",
            "メモリ使用量",
            ["type"],  # used, available, total
            registry=self.registry,
        )

        # ディスク使用量
        self.disk_usage = Gauge(
            "day_trade_disk_usage_bytes",
            "ディスク使用量",
            ["mount_point", "type"],
            registry=self.registry,
        )

        # アクティブ接続数
        self.active_connections = Gauge(
            "day_trade_active_connections",
            "アクティブ接続数",
            ["connection_type"],
            registry=self.registry,
        )

        # データベース接続プール
        self.db_connections = Gauge(
            "day_trade_db_connections",
            "データベース接続数",
            ["status"],  # active, idle, total
            registry=self.registry,
        )

        # アプリケーションエラー
        self.application_errors = Counter(
            "day_trade_application_errors_total",
            "アプリケーションエラー回数",
            ["error_type", "component"],
            registry=self.registry,
        )

    def _start_system_monitoring(self):
        """システム監視開始"""

        def update_system_metrics():
            while True:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)

                    # メモリ使用量
                    memory = psutil.virtual_memory()
                    self.memory_usage.labels(type="used").set(memory.used)
                    self.memory_usage.labels(type="available").set(memory.available)
                    self.memory_usage.labels(type="total").set(memory.total)

                    # ディスク使用量
                    disk = psutil.disk_usage("/")
                    self.disk_usage.labels(mount_point="/", type="used").set(disk.used)
                    self.disk_usage.labels(mount_point="/", type="total").set(disk.total)

                except Exception as e:
                    logger.error(f"システムメトリクス更新エラー: {e}")

                time.sleep(10)  # 10秒間隔で更新

        # バックグラウンドで実行
        monitoring_thread = threading.Thread(target=update_system_metrics, daemon=True)
        monitoring_thread.start()

    def record_error(self, error_type: str, component: str):
        """エラー記録"""

        self.application_errors.labels(error_type=error_type, component=component).inc()


# グローバル インスタンス
_default_registry = CollectorRegistry()
_metrics_collector = PrometheusMetricsCollector(_default_registry)
_risk_metrics = RiskManagementMetrics(_default_registry)
_trading_metrics = TradingPerformanceMetrics(_default_registry)
_ai_metrics = AIEngineMetrics(_default_registry)
_health_metrics = SystemHealthMetrics(_default_registry)


def get_metrics_collector() -> PrometheusMetricsCollector:
    """デフォルトメトリクス収集器取得"""
    return _metrics_collector


def get_risk_metrics() -> RiskManagementMetrics:
    """リスク管理メトリクス取得"""
    return _risk_metrics


def get_trading_metrics() -> TradingPerformanceMetrics:
    """取引メトリクス取得"""
    return _trading_metrics


def get_ai_metrics() -> AIEngineMetrics:
    """AI メトリクス取得"""
    return _ai_metrics


def get_health_metrics() -> SystemHealthMetrics:
    """ヘルスメトリクス取得"""
    return _health_metrics


def get_registry() -> CollectorRegistry:
    """デフォルトレジストリ取得"""
    return _default_registry
