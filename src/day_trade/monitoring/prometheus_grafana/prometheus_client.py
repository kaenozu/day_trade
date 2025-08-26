"""
Prometheusクライアント

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

PrometheusサーバーとのAPI通信を管理するクライアント。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .configs import PrometheusConfig


class PrometheusClient:
    """Prometheusクライアント"""

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.base_url = f"{config.scheme}://{config.host}:{config.port}"
        self.logger = logging.getLogger(__name__)

    async def query(
        self, query: str, time: Optional[datetime] = None
    ) -> Dict[str, Any]:
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