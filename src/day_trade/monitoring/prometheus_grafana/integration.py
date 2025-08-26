"""
Prometheus + Grafana統合システム

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

メインの統合システム。各コンポーネントを統合して監視システムを提供する。
"""

import asyncio
import logging
from typing import Any, Dict

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .alert_manager import AlertManager
from .configs import (
    GrafanaConfig,
    MetricDefinition,
    PrometheusConfig,
    PrometheusMetricType,
)
from .dashboard_factory import DashboardFactory
from .grafana_client import GrafanaClient
from .metrics_exporter import PrometheusMetricsExporter
from .metrics_updater import SystemMetricsUpdater
from .prometheus_client import PrometheusClient


class EnhancedPrometheusGrafanaIntegration:
    """強化Prometheus + Grafana統合システム"""

    def __init__(
        self,
        prometheus_config: PrometheusConfig,
        grafana_config: GrafanaConfig,
        metrics_port: int = 8000,
    ):
        self.prometheus_config = prometheus_config
        self.grafana_config = grafana_config

        self.metrics_exporter = PrometheusMetricsExporter(port=metrics_port)
        self.prometheus_client = PrometheusClient(prometheus_config)
        self.grafana_client = GrafanaClient(grafana_config)
        self.metrics_updater = SystemMetricsUpdater(self.metrics_exporter)

        self.logger = logging.getLogger(__name__)

        # デフォルトメトリクス定義
        self._setup_default_metrics()

    def _setup_default_metrics(self):
        """デフォルトメトリクス設定"""
        default_metrics = [
            MetricDefinition(
                name="day_trade_system_uptime_seconds",
                metric_type=PrometheusMetricType.COUNTER,
                description="システム稼働時間（秒）",
            ),
            MetricDefinition(
                name="day_trade_api_requests_total",
                metric_type=PrometheusMetricType.COUNTER,
                description="API リクエスト総数",
                labels=["method", "endpoint", "status"],
            ),
            MetricDefinition(
                name="day_trade_api_request_duration_seconds",
                metric_type=PrometheusMetricType.HISTOGRAM,
                description="API リクエスト処理時間",
                labels=["method", "endpoint"],
                buckets=[
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                ],
            ),
            MetricDefinition(
                name="day_trade_active_trades",
                metric_type=PrometheusMetricType.GAUGE,
                description="アクティブな取引数",
            ),
            MetricDefinition(
                name="day_trade_portfolio_value_jpy",
                metric_type=PrometheusMetricType.GAUGE,
                description="ポートフォリオ価値（日本円）",
            ),
            MetricDefinition(
                name="day_trade_ml_predictions_total",
                metric_type=PrometheusMetricType.COUNTER,
                description="ML予測実行回数",
                labels=["model", "symbol", "prediction_type"],
            ),
            MetricDefinition(
                name="day_trade_ml_prediction_accuracy",
                metric_type=PrometheusMetricType.GAUGE,
                description="ML予測精度",
                labels=["model", "timeframe"],
            ),
            MetricDefinition(
                name="day_trade_security_events_total",
                metric_type=PrometheusMetricType.COUNTER,
                description="セキュリティイベント数",
                labels=["severity", "component"],
            ),
            MetricDefinition(
                name="day_trade_log_entries_total",
                metric_type=PrometheusMetricType.COUNTER,
                description="ログエントリ総数",
                labels=["level", "source", "component"],
            ),
            MetricDefinition(
                name="day_trade_cache_hit_ratio",
                metric_type=PrometheusMetricType.GAUGE,
                description="キャッシュヒット率",
                labels=["cache_type", "cache_level"],
            ),
        ]

        for metric_def in default_metrics:
            self.metrics_exporter.register_metric(metric_def)

    async def initialize_monitoring_stack(self) -> bool:
        """監視スタック初期化"""
        try:
            self.logger.info("Prometheus + Grafana監視スタック初期化開始")

            # Prometheusメトリクスサーバー開始
            if not self.metrics_exporter.start_http_server():
                self.logger.error("Prometheusメトリクスサーバー開始失敗")
                return False

            # Grafanaデータソース作成
            if not await self.grafana_client.create_datasource(
                "Prometheus", self.prometheus_config
            ):
                self.logger.warning("Grafanaデータソース作成失敗")

            # デフォルトダッシュボード作成
            await self._create_default_dashboards()

            # デフォルトアラートルール作成
            await self._create_default_alert_rules()

            self.logger.info("Prometheus + Grafana監視スタック初期化完了")
            return True

        except Exception as e:
            self.logger.error(f"監視スタック初期化エラー: {e}")
            return False

    async def _create_default_dashboards(self):
        """デフォルトダッシュボード作成"""
        try:
            # システム概要ダッシュボード
            system_dashboard = DashboardFactory.create_system_overview_dashboard()
            await self.grafana_client.create_dashboard(system_dashboard)

            # パフォーマンス詳細ダッシュボード
            performance_dashboard = DashboardFactory.create_performance_dashboard()
            await self.grafana_client.create_dashboard(performance_dashboard)

        except Exception as e:
            self.logger.error(f"デフォルトダッシュボード作成エラー: {e}")

    async def _create_default_alert_rules(self):
        """デフォルトアラートルール作成"""
        try:
            alert_rules = AlertManager.get_default_alert_rules()

            for rule in alert_rules:
                await self.grafana_client.create_alert_rule(rule)

        except Exception as e:
            self.logger.error(f"デフォルトアラートルール作成エラー: {e}")

    def update_system_metrics(self, metrics_data: Dict[str, Any]):
        """システムメトリクス更新"""
        self.metrics_updater.update_system_metrics(metrics_data)

    async def get_monitoring_health(self) -> Dict[str, Any]:
        """監視システム健全性取得"""
        try:
            health_status = {
                "prometheus": {"status": "unknown", "details": {}},
                "grafana": {"status": "unknown", "details": {}},
                "metrics_exporter": {"status": "unknown", "details": {}},
                "overall_status": "unknown",
            }

            # Prometheusターゲット確認
            try:
                targets = await self.prometheus_client.get_targets()
                healthy_targets = len([t for t in targets if t.get("health") == "up"])
                health_status["prometheus"]["status"] = (
                    "healthy" if healthy_targets > 0 else "unhealthy"
                )
                health_status["prometheus"]["details"] = {
                    "total_targets": len(targets),
                    "healthy_targets": healthy_targets,
                }
            except Exception as e:
                health_status["prometheus"]["status"] = "error"
                health_status["prometheus"]["error"] = str(e)

            # Grafana健全性確認
            try:
                if REQUESTS_AVAILABLE:
                    headers = self.grafana_client._get_auth_headers()
                    response = requests.get(
                        f"{self.grafana_client.base_url}/api/health",
                        headers=headers,
                        timeout=10,
                    )
                    if response.status_code == 200:
                        health_status["grafana"]["status"] = "healthy"
                        health_status["grafana"]["details"] = response.json()
                    else:
                        health_status["grafana"]["status"] = "unhealthy"
                else:
                    health_status["grafana"]["status"] = "no_requests_library"
            except Exception as e:
                health_status["grafana"]["status"] = "error"
                health_status["grafana"]["error"] = str(e)

            # メトリクス出力確認
            metrics_output = self.metrics_exporter.get_metrics_output()
            if metrics_output:
                health_status["metrics_exporter"]["status"] = "healthy"
                health_status["metrics_exporter"]["details"] = {
                    "metrics_count": len(self.metrics_exporter.metrics),
                    "server_started": self.metrics_exporter.http_server_started,
                }
            else:
                health_status["metrics_exporter"]["status"] = "unhealthy"

            # 総合ステータス判定
            statuses = [
                health_status["prometheus"]["status"],
                health_status["grafana"]["status"],
                health_status["metrics_exporter"]["status"],
            ]

            if all(status == "healthy" for status in statuses):
                health_status["overall_status"] = "healthy"
            elif any(status == "healthy" for status in statuses):
                health_status["overall_status"] = "partial"
            else:
                health_status["overall_status"] = "unhealthy"

            return health_status

        except Exception as e:
            self.logger.error(f"監視健全性チェックエラー: {e}")
            return {"overall_status": "error", "error": str(e)}


# Factory function
def create_enhanced_prometheus_grafana_integration(
    prometheus_host: str = "localhost",
    prometheus_port: int = 9090,
    grafana_host: str = "localhost",
    grafana_port: int = 3000,
    grafana_username: str = "admin",
    grafana_password: str = "admin",
    metrics_port: int = 8000,
) -> EnhancedPrometheusGrafanaIntegration:
    """強化Prometheus + Grafana統合システム作成"""

    prometheus_config = PrometheusConfig(host=prometheus_host, port=prometheus_port)

    grafana_config = GrafanaConfig(
        host=grafana_host,
        port=grafana_port,
        username=grafana_username,
        password=grafana_password,
    )

    return EnhancedPrometheusGrafanaIntegration(
        prometheus_config, grafana_config, metrics_port
    )