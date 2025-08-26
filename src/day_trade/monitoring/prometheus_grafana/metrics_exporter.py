"""
Prometheusメトリクス出力システム

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

Prometheusメトリクスの登録・更新・出力を管理するシステム。
"""

import logging
from typing import Any, Dict, Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False

from .configs import MetricDefinition, PrometheusMetricType


class PrometheusMetricsExporter:
    """Prometheusメトリクス出力システム"""

    def __init__(self, registry: Optional[CollectorRegistry] = None, port: int = 8000):
        self.registry = registry or CollectorRegistry()
        self.port = port
        self.metrics: Dict[str, Any] = {}
        self.http_server_started = False
        self.logger = logging.getLogger(__name__)

    def register_metric(self, definition: MetricDefinition) -> bool:
        """メトリクス登録"""
        try:
            if not PROMETHEUS_CLIENT_AVAILABLE:
                self.logger.error("prometheus_clientライブラリが利用できません")
                return False

            if definition.name in self.metrics:
                self.logger.warning(
                    f"メトリクス '{definition.name}' は既に登録済みです"
                )
                return True

            # メトリクスタイプに応じてPrometheusメトリクスを作成
            if definition.metric_type == PrometheusMetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry,
                )
            elif definition.metric_type == PrometheusMetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry,
                )
            elif definition.metric_type == PrometheusMetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self.registry,
                )
            elif definition.metric_type == PrometheusMetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry,
                )
            elif definition.metric_type == PrometheusMetricType.INFO:
                metric = Info(
                    definition.name, definition.description, registry=self.registry
                )
            else:
                self.logger.error(
                    f"サポートされていないメトリクスタイプ: {definition.metric_type}"
                )
                return False

            self.metrics[definition.name] = metric
            self.logger.info(f"メトリクス登録完了: {definition.name}")
            return True

        except Exception as e:
            self.logger.error(f"メトリクス登録エラー: {e}")
            return False

    def update_metric(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """メトリクス更新"""
        try:
            if name not in self.metrics:
                self.logger.warning(f"未登録のメトリクス: {name}")
                return

            metric = self.metrics[name]

            if isinstance(metric, Counter):
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif isinstance(metric, Gauge):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif isinstance(metric, (Histogram, Summary)):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)

        except Exception as e:
            self.logger.error(f"メトリクス更新エラー: {e}")

    def start_http_server(self) -> bool:
        """HTTPメトリクスサーバー開始"""
        try:
            if not PROMETHEUS_CLIENT_AVAILABLE:
                self.logger.error("prometheus_clientライブラリが利用できません")
                return False

            if self.http_server_started:
                self.logger.warning("HTTPサーバーは既に開始されています")
                return True

            start_http_server(self.port, registry=self.registry)
            self.http_server_started = True
            self.logger.info(
                f"Prometheusメトリクスサーバー開始: http://localhost:{self.port}/metrics"
            )
            return True

        except Exception as e:
            self.logger.error(f"HTTPサーバー開始エラー: {e}")
            return False

    def get_metrics_output(self) -> str:
        """メトリクス出力（Prometheus形式）"""
        try:
            if not PROMETHEUS_CLIENT_AVAILABLE:
                return ""

            return generate_latest(self.registry).decode("utf-8")

        except Exception as e:
            self.logger.error(f"メトリクス出力エラー: {e}")
            return ""