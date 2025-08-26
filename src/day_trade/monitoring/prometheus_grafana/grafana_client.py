"""
Grafanaクライアント

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

GrafanaサーバーとのAPI通信を管理するクライアント。
"""

import base64
import logging
import re
from typing import Any, Dict, List

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .configs import AlertRule, GrafanaConfig, GrafanaDashboard, PrometheusConfig


class GrafanaClient:
    """Grafanaクライアント"""

    def __init__(self, config: GrafanaConfig):
        self.config = config
        self.base_url = f"{config.scheme}://{config.host}:{config.port}"
        self.logger = logging.getLogger(__name__)

    async def create_datasource(
        self, name: str, prometheus_config: PrometheusConfig
    ) -> bool:
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
                self.logger.error(
                    f"データソース作成失敗: {response.status_code} - {response.text}"
                )
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

    async def create_alert_rule(
        self, rule: AlertRule, datasource_name: str = "Prometheus"
    ) -> bool:
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
            credentials = f"{self.config.username}:{self.config.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_credentials}"

        return headers

    def _parse_condition(self, condition: str) -> List[float]:
        """条件解析"""
        # 簡単な条件解析（例: "> 0.8", "== 0", "< 100"）
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