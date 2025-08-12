"""
強化Prometheus + Grafana統合システム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

エンタープライズレベルのPrometheus監視とGrafana可視化の
統合リアルタイムダッシュボードシステム。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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

try:
    from .metrics_collection_system import MetricsCollectionSystem
    from .performance_dashboard import ChartType, TimeRange

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class PrometheusMetricType(Enum):
    """Prometheusメトリクスタイプ"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class GrafanaPanelType(Enum):
    """Grafanaパネルタイプ"""

    GRAPH = "graph"
    SINGLESTAT = "singlestat"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    STAT = "stat"
    TEXT = "text"
    PIE_CHART = "piechart"
    WORLD_MAP = "worldmap"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PrometheusConfig:
    """Prometheus設定"""

    host: str = "localhost"
    port: int = 9090
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30


@dataclass
class GrafanaConfig:
    """Grafana設定"""

    host: str = "localhost"
    port: int = 3000
    scheme: str = "http"
    username: str = "admin"
    password: str = "admin"
    api_key: Optional[str] = None
    timeout: int = 30


@dataclass
class MetricDefinition:
    """メトリクス定義"""

    name: str
    metric_type: PrometheusMetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Histogram用
    objectives: Optional[Dict[float, float]] = None  # Summary用


@dataclass
class AlertRule:
    """アラートルール"""

    name: str
    query: str
    condition: str  # e.g., "> 0.8", "== 0"
    duration: str = "5m"
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class GrafanaDashboard:
    """Grafanaダッシュボード定義"""

    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    time_range: str = "1h"
    refresh_interval: str = "30s"
    panels: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)


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
                self.logger.warning(f"メトリクス '{definition.name}' は既に登録済みです")
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
                metric = Info(definition.name, definition.description, registry=self.registry)
            else:
                self.logger.error(f"サポートされていないメトリクスタイプ: {definition.metric_type}")
                return False

            self.metrics[definition.name] = metric
            self.logger.info(f"メトリクス登録完了: {definition.name}")
            return True

        except Exception as e:
            self.logger.error(f"メトリクス登録エラー: {e}")
            return False

    def update_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
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

    def start_http_server(self):
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


class PrometheusClient:
    """Prometheusクライアント"""

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.base_url = f"{config.scheme}://{config.host}:{config.port}"
        self.logger = logging.getLogger(__name__)

    async def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """Prometheusクエリ実行"""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.error("requestsライブラリが利用できません")
                return {"status": "error", "data": {"result": []}}

            params = {"query": query}
            if time:
                params["time"] = time.timestamp()

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            response = requests.get(
                f"{self.base_url}/api/v1/query",
                params=params,
                auth=auth,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Prometheusクエリエラー: {response.status_code}")
                return {"status": "error", "data": {"result": []}}

        except Exception as e:
            self.logger.error(f"Prometheusクエリ例外: {e}")
            return {"status": "error", "data": {"result": []}}

    async def query_range(
        self, query: str, start: datetime, end: datetime, step: str = "15s"
    ) -> Dict[str, Any]:
        """Prometheus範囲クエリ実行"""
        try:
            if not REQUESTS_AVAILABLE:
                return {"status": "error", "data": {"result": []}}

            params = {
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step,
            }

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            response = requests.get(
                f"{self.base_url}/api/v1/query_range",
                params=params,
                auth=auth,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "data": {"result": []}}

        except Exception as e:
            self.logger.error(f"Prometheus範囲クエリ例外: {e}")
            return {"status": "error", "data": {"result": []}}

    async def get_targets(self) -> List[Dict[str, Any]]:
        """Prometheusターゲット一覧取得"""
        try:
            if not REQUESTS_AVAILABLE:
                return []

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            response = requests.get(
                f"{self.base_url}/api/v1/targets",
                auth=auth,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("activeTargets", [])
            else:
                return []

        except Exception as e:
            self.logger.error(f"ターゲット取得エラー: {e}")
            return []


class GrafanaClient:
    """Grafanaクライアント"""

    def __init__(self, config: GrafanaConfig):
        self.config = config
        self.base_url = f"{config.scheme}://{config.host}:{config.port}"
        self.logger = logging.getLogger(__name__)

    async def create_datasource(self, name: str, prometheus_config: PrometheusConfig) -> bool:
        """データソース作成"""
        try:
            if not REQUESTS_AVAILABLE:
                return False

            headers = self._get_auth_headers()

            datasource_config = {
                "name": name,
                "type": "prometheus",
                "access": "proxy",
                "url": f"{prometheus_config.scheme}://{prometheus_config.host}:{prometheus_config.port}",
                "isDefault": True,
                "jsonData": {"httpMethod": "POST", "keepCookies": []},
            }

            if prometheus_config.username and prometheus_config.password:
                datasource_config["basicAuth"] = True
                datasource_config["basicAuthUser"] = prometheus_config.username
                datasource_config["secureJsonData"] = {
                    "basicAuthPassword": prometheus_config.password
                }

            response = requests.post(
                f"{self.base_url}/api/datasources",
                json=datasource_config,
                headers=headers,
                timeout=self.config.timeout,
            )

            if response.status_code in [200, 201, 409]:  # 409 = already exists
                self.logger.info(f"Grafanaデータソース作成/確認完了: {name}")
                return True
            else:
                self.logger.error(f"データソース作成失敗: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"データソース作成エラー: {e}")
            return False

    async def create_dashboard(self, dashboard: GrafanaDashboard) -> bool:
        """ダッシュボード作成"""
        try:
            if not REQUESTS_AVAILABLE:
                return False

            headers = self._get_auth_headers()

            dashboard_json = {
                "dashboard": {
                    "id": None,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "tags": dashboard.tags,
                    "timezone": "browser",
                    "panels": dashboard.panels,
                    "templating": {"list": dashboard.variables},
                    "time": {"from": f"now-{dashboard.time_range}", "to": "now"},
                    "refresh": dashboard.refresh_interval,
                    "version": 1,
                    "editable": True,
                    "gnetId": None,
                    "graphTooltip": 1,
                    "hideControls": False,
                    "links": [],
                    "rows": [],
                    "schemaVersion": 16,
                    "style": "dark",
                    "uid": None,
                },
                "folderId": 0,
                "overwrite": True,
            }

            response = requests.post(
                f"{self.base_url}/api/dashboards/db",
                json=dashboard_json,
                headers=headers,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Grafanaダッシュボード作成完了: {dashboard.title}")
                return True
            else:
                self.logger.error(
                    f"ダッシュボード作成失敗: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            self.logger.error(f"ダッシュボード作成エラー: {e}")
            return False

    async def create_alert_rule(self, rule: AlertRule, datasource_name: str = "Prometheus") -> bool:
        """アラートルール作成"""
        try:
            if not REQUESTS_AVAILABLE:
                return False

            headers = self._get_auth_headers()

            alert_rule = {
                "alert": {
                    "name": rule.name,
                    "message": rule.description,
                    "frequency": "10s",
                    "conditions": [
                        {
                            "query": {
                                "queryType": "range",
                                "refId": "A",
                                "model": {
                                    "expr": rule.query,
                                    "format": "time_series",
                                    "intervalMs": 1000,
                                    "maxDataPoints": 43200,
                                    "refId": "A",
                                },
                                "datasource": {
                                    "type": "prometheus",
                                    "name": datasource_name,
                                },
                            },
                            "reducer": {"type": "last", "params": []},
                            "evaluator": {
                                "params": self._parse_condition(rule.condition),
                                "type": self._get_condition_type(rule.condition),
                            },
                        }
                    ],
                    "executionErrorState": "alerting",
                    "noDataState": "no_data",
                    "for": rule.duration,
                }
            }

            # アラートルール作成API（Grafana 8.0以降の場合）
            response = requests.post(
                f"{self.base_url}/api/v1/provisioning/alert-rules",
                json=alert_rule,
                headers=headers,
                timeout=self.config.timeout,
            )

            if response.status_code in [200, 201, 202]:
                self.logger.info(f"アラートルール作成完了: {rule.name}")
                return True
            else:
                self.logger.error(f"アラートルール作成失敗: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"アラートルール作成エラー: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """認証ヘッダー取得"""
        headers = {"Content-Type": "application/json"}

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        else:
            # Basic認証用のエンコーディング
            import base64

            credentials = f"{self.config.username}:{self.config.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

        return headers

    def _parse_condition(self, condition: str) -> List[float]:
        """条件解析"""
        # 簡単な条件解析（例: "> 0.8", "== 0", "< 100"）
        import re

        match = re.match(r"([><]=?|==|!=)\s*([\d.]+)", condition)
        if match:
            return [float(match.group(2))]
        return [0.0]

    def _get_condition_type(self, condition: str) -> str:
        """条件タイプ取得"""
        if condition.startswith(">"):
            return "gt"
        elif condition.startswith("<"):
            return "lt"
        elif condition.startswith("=="):
            return "eq"
        elif condition.startswith("!="):
            return "ne"
        else:
            return "gt"


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
            system_dashboard = GrafanaDashboard(
                title="Day Trade System Overview",
                description="システム全体の概要監視",
                tags=["day-trade", "overview", "system"],
                time_range="1h",
                refresh_interval="30s",
                panels=[
                    self._create_api_requests_panel(),
                    self._create_response_time_panel(),
                    self._create_active_trades_panel(),
                    self._create_portfolio_value_panel(),
                    self._create_ml_accuracy_panel(),
                    self._create_security_events_panel(),
                    self._create_log_levels_panel(),
                    self._create_cache_performance_panel(),
                ],
            )

            await self.grafana_client.create_dashboard(system_dashboard)

            # パフォーマンス詳細ダッシュボード
            performance_dashboard = GrafanaDashboard(
                title="Day Trade Performance Details",
                description="パフォーマンス詳細監視",
                tags=["day-trade", "performance", "details"],
                time_range="6h",
                refresh_interval="15s",
                panels=[
                    self._create_request_rate_panel(),
                    self._create_error_rate_panel(),
                    self._create_latency_percentiles_panel(),
                    self._create_throughput_panel(),
                ],
            )

            await self.grafana_client.create_dashboard(performance_dashboard)

        except Exception as e:
            self.logger.error(f"デフォルトダッシュボード作成エラー: {e}")

    async def _create_default_alert_rules(self):
        """デフォルトアラートルール作成"""
        try:
            alert_rules = [
                AlertRule(
                    name="High API Error Rate",
                    query="rate(day_trade_api_requests_total{status=~'5..'}[5m]) > 0.05",
                    condition="> 0.05",
                    duration="2m",
                    severity=AlertSeverity.WARNING,
                    description="APIエラー率が高くなっています",
                ),
                AlertRule(
                    name="High API Response Time",
                    query="histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m])) > 2",
                    condition="> 2",
                    duration="3m",
                    severity=AlertSeverity.WARNING,
                    description="API応答時間が長くなっています",
                ),
                AlertRule(
                    name="Low ML Prediction Accuracy",
                    query="day_trade_ml_prediction_accuracy < 0.7",
                    condition="< 0.7",
                    duration="5m",
                    severity=AlertSeverity.CRITICAL,
                    description="ML予測精度が低下しています",
                ),
                AlertRule(
                    name="Security Events Spike",
                    query="rate(day_trade_security_events_total{severity='critical'}[5m]) > 0.01",
                    condition="> 0.01",
                    duration="1m",
                    severity=AlertSeverity.CRITICAL,
                    description="セキュリティイベントが急増しています",
                ),
                AlertRule(
                    name="Low Cache Hit Ratio",
                    query="day_trade_cache_hit_ratio < 0.8",
                    condition="< 0.8",
                    duration="10m",
                    severity=AlertSeverity.INFO,
                    description="キャッシュヒット率が低下しています",
                ),
            ]

            for rule in alert_rules:
                await self.grafana_client.create_alert_rule(rule)

        except Exception as e:
            self.logger.error(f"デフォルトアラートルール作成エラー: {e}")

    def _create_api_requests_panel(self) -> Dict[str, Any]:
        """APIリクエストパネル作成"""
        return {
            "id": 1,
            "title": "API Requests Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(day_trade_api_requests_total[5m])",
                    "legendFormat": "{{method}} {{endpoint}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Requests/sec", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        }

    def _create_response_time_panel(self) -> Dict[str, Any]:
        """応答時間パネル作成"""
        return {
            "id": 2,
            "title": "API Response Time (95th percentile)",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Seconds", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        }

    def _create_active_trades_panel(self) -> Dict[str, Any]:
        """アクティブ取引パネル作成"""
        return {
            "id": 3,
            "title": "Active Trades",
            "type": "singlestat",
            "targets": [{"expr": "day_trade_active_trades", "refId": "A"}],
            "valueName": "current",
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        }

    def _create_portfolio_value_panel(self) -> Dict[str, Any]:
        """ポートフォリオ価値パネル作成"""
        return {
            "id": 4,
            "title": "Portfolio Value (JPY)",
            "type": "singlestat",
            "targets": [{"expr": "day_trade_portfolio_value_jpy", "refId": "A"}],
            "valueName": "current",
            "format": "currencyJPY",
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
        }

    def _create_ml_accuracy_panel(self) -> Dict[str, Any]:
        """ML予測精度パネル作成"""
        return {
            "id": 5,
            "title": "ML Prediction Accuracy",
            "type": "gauge",
            "targets": [{"expr": "avg(day_trade_ml_prediction_accuracy)", "refId": "A"}],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 0.7},
                            {"color": "green", "value": 0.8},
                        ]
                    },
                }
            },
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        }

    def _create_security_events_panel(self) -> Dict[str, Any]:
        """セキュリティイベントパネル作成"""
        return {
            "id": 6,
            "title": "Security Events Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(day_trade_security_events_total[5m])",
                    "legendFormat": "{{severity}} {{component}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Events/sec", "min": 0}],
            "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8},
        }

    def _create_log_levels_panel(self) -> Dict[str, Any]:
        """ログレベルパネル作成"""
        return {
            "id": 7,
            "title": "Log Levels Distribution",
            "type": "piechart",
            "targets": [
                {
                    "expr": "sum by (level) (rate(day_trade_log_entries_total[5m]))",
                    "legendFormat": "{{level}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
        }

    def _create_cache_performance_panel(self) -> Dict[str, Any]:
        """キャッシュパフォーマンスパネル作成"""
        return {
            "id": 8,
            "title": "Cache Hit Ratio",
            "type": "graph",
            "targets": [
                {
                    "expr": "day_trade_cache_hit_ratio",
                    "legendFormat": "{{cache_type}} {{cache_level}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Hit Ratio", "min": 0, "max": 1}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        }

    def _create_request_rate_panel(self) -> Dict[str, Any]:
        """リクエスト率パネル作成"""
        return {
            "id": 9,
            "title": "Request Rate by Endpoint",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum by (endpoint) (rate(day_trade_api_requests_total[5m]))",
                    "legendFormat": "{{endpoint}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        }

    def _create_error_rate_panel(self) -> Dict[str, Any]:
        """エラー率パネル作成"""
        return {
            "id": 10,
            "title": "Error Rate by Status Code",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum by (status) (rate(day_trade_api_requests_total{status=~'[4-5]..'}[5m]))",
                    "legendFormat": "{{status}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        }

    def _create_latency_percentiles_panel(self) -> Dict[str, Any]:
        """レイテンシパーセンタイルパネル作成"""
        return {
            "id": 11,
            "title": "Response Time Percentiles",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile",
                    "refId": "A",
                },
                {
                    "expr": "histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "refId": "B",
                },
                {
                    "expr": "histogram_quantile(0.99, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "99th percentile",
                    "refId": "C",
                },
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        }

    def _create_throughput_panel(self) -> Dict[str, Any]:
        """スループットパネル作成"""
        return {
            "id": 12,
            "title": "System Throughput",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum(rate(day_trade_api_requests_total[5m]))",
                    "legendFormat": "Total Throughput",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Requests/sec", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        }

    def update_system_metrics(self, metrics_data: Dict[str, Any]):
        """システムメトリクス更新"""
        try:
            # API関連メトリクス
            if "api_requests" in metrics_data:
                for request_data in metrics_data["api_requests"]:
                    labels = {
                        "method": request_data.get("method", "GET"),
                        "endpoint": request_data.get("endpoint", "unknown"),
                        "status": str(request_data.get("status", 200)),
                    }
                    self.metrics_exporter.update_metric("day_trade_api_requests_total", 1, labels)

            # 応答時間メトリクス
            if "response_time" in metrics_data:
                response_time = metrics_data["response_time"]
                labels = {
                    "method": response_time.get("method", "GET"),
                    "endpoint": response_time.get("endpoint", "unknown"),
                }
                self.metrics_exporter.update_metric(
                    "day_trade_api_request_duration_seconds",
                    response_time.get("duration", 0),
                    labels,
                )

            # 取引関連メトリクス
            if "active_trades" in metrics_data:
                self.metrics_exporter.update_metric(
                    "day_trade_active_trades", metrics_data["active_trades"]
                )

            if "portfolio_value" in metrics_data:
                self.metrics_exporter.update_metric(
                    "day_trade_portfolio_value_jpy", metrics_data["portfolio_value"]
                )

            # ML関連メトリクス
            if "ml_predictions" in metrics_data:
                for prediction_data in metrics_data["ml_predictions"]:
                    labels = {
                        "model": prediction_data.get("model", "unknown"),
                        "symbol": prediction_data.get("symbol", "unknown"),
                        "prediction_type": prediction_data.get("type", "unknown"),
                    }
                    self.metrics_exporter.update_metric("day_trade_ml_predictions_total", 1, labels)

            if "ml_accuracy" in metrics_data:
                for accuracy_data in metrics_data["ml_accuracy"]:
                    labels = {
                        "model": accuracy_data.get("model", "unknown"),
                        "timeframe": accuracy_data.get("timeframe", "unknown"),
                    }
                    self.metrics_exporter.update_metric(
                        "day_trade_ml_prediction_accuracy",
                        accuracy_data.get("accuracy", 0),
                        labels,
                    )

            # セキュリティイベント
            if "security_events" in metrics_data:
                for event_data in metrics_data["security_events"]:
                    labels = {
                        "severity": event_data.get("severity", "info"),
                        "component": event_data.get("component", "unknown"),
                    }
                    self.metrics_exporter.update_metric(
                        "day_trade_security_events_total", 1, labels
                    )

            # ログエントリ
            if "log_entries" in metrics_data:
                for log_data in metrics_data["log_entries"]:
                    labels = {
                        "level": log_data.get("level", "info"),
                        "source": log_data.get("source", "unknown"),
                        "component": log_data.get("component", "unknown"),
                    }
                    self.metrics_exporter.update_metric("day_trade_log_entries_total", 1, labels)

            # キャッシュメトリクス
            if "cache_performance" in metrics_data:
                for cache_data in metrics_data["cache_performance"]:
                    labels = {
                        "cache_type": cache_data.get("type", "unknown"),
                        "cache_level": cache_data.get("level", "unknown"),
                    }
                    self.metrics_exporter.update_metric(
                        "day_trade_cache_hit_ratio",
                        cache_data.get("hit_ratio", 0),
                        labels,
                    )

        except Exception as e:
            self.logger.error(f"システムメトリクス更新エラー: {e}")

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

    return EnhancedPrometheusGrafanaIntegration(prometheus_config, grafana_config, metrics_port)


if __name__ == "__main__":
    # テスト実行
    async def test_prometheus_grafana_integration():
        print("=== 強化Prometheus + Grafana統合システムテスト ===")

        try:
            # 統合システム初期化
            monitoring_system = create_enhanced_prometheus_grafana_integration()

            print("\n1. 強化Prometheus + Grafana統合システム初期化完了")
            print(f"   登録メトリクス数: {len(monitoring_system.metrics_exporter.metrics)}")

            # 監視スタック初期化テスト
            print("\n2. 監視スタック初期化テスト...")
            init_success = await monitoring_system.initialize_monitoring_stack()
            print(f"   初期化: {'成功' if init_success else '失敗'}")

            # システムメトリクス更新テスト
            print("\n3. システムメトリクス更新テスト...")
            test_metrics = {
                "api_requests": [
                    {"method": "GET", "endpoint": "/api/trades", "status": 200},
                    {"method": "POST", "endpoint": "/api/orders", "status": 201},
                ],
                "response_time": {
                    "method": "GET",
                    "endpoint": "/api/trades",
                    "duration": 0.15,
                },
                "active_trades": 5,
                "portfolio_value": 1250000,
                "ml_predictions": [{"model": "lstm", "symbol": "7203", "type": "price_forecast"}],
                "ml_accuracy": [{"model": "lstm", "timeframe": "1h", "accuracy": 0.85}],
                "security_events": [{"severity": "warning", "component": "auth"}],
                "log_entries": [
                    {"level": "info", "source": "application", "component": "trading"},
                    {"level": "error", "source": "database", "component": "connection"},
                ],
                "cache_performance": [{"type": "redis", "level": "l1", "hit_ratio": 0.92}],
            }

            monitoring_system.update_system_metrics(test_metrics)
            print("   メトリクス更新完了")

            # メトリクス出力確認
            print("\n4. Prometheusメトリクス出力確認...")
            metrics_output = monitoring_system.metrics_exporter.get_metrics_output()
            metrics_lines = len(metrics_output.split("\n")) if metrics_output else 0
            print(f"   メトリクス出力: {metrics_lines}行")

            if metrics_output:
                # サンプルメトリクス表示
                sample_lines = metrics_output.split("\n")[:10]
                print("   サンプル出力:")
                for line in sample_lines:
                    if line.strip() and not line.startswith("#"):
                        print(f"     {line}")

            # 監視システム健全性チェック
            print("\n5. 監視システム健全性チェック...")
            health_status = await monitoring_system.get_monitoring_health()
            print(f"   総合ステータス: {health_status.get('overall_status')}")
            print(f"   Prometheus: {health_status.get('prometheus', {}).get('status')}")
            print(f"   Grafana: {health_status.get('grafana', {}).get('status')}")
            print(f"   メトリクス出力: {health_status.get('metrics_exporter', {}).get('status')}")

            # Prometheusクエリテスト（モック）
            print("\n6. Prometheusクエリテスト...")
            query_result = await monitoring_system.prometheus_client.query("up")
            result_count = len(query_result.get("data", {}).get("result", []))
            print(f"   クエリ結果: {result_count}件")

            print("\n✅ 強化Prometheus + Grafana統合システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_prometheus_grafana_integration())
