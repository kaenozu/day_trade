#!/usr/bin/env python3
"""
メトリクス エクスポーター
Prometheus向けHTTPサーバー・メトリクス公開機能
"""

import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from .prometheus_metrics import (
    get_registry,
    get_metrics_collector,
    get_risk_metrics,
    get_trading_metrics,
    get_ai_metrics,
    get_health_metrics
)
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

class MetricsExporter:
    """メトリクス エクスポーター"""

    def __init__(self, port: int = 8000):
        self.port = port
        self.app = FastAPI(
            title="Day Trade Metrics Exporter",
            description="Prometheus メトリクス エクスポーター",
            version="2.0.0"
        )
        self._setup_routes()

    def _setup_routes(self):
        """API ルート設定"""

        @self.app.get("/metrics")
        async def get_all_metrics():
            """全メトリクス取得（Prometheus形式）"""

            try:
                # メトリクス収集実行
                get_metrics_collector().collect_all_metrics()

                # Prometheus形式で出力
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"メトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/risk_metrics")
        async def get_risk_metrics_endpoint():
            """リスク管理メトリクス取得"""

            try:
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"リスクメトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/trading_metrics")
        async def get_trading_metrics_endpoint():
            """取引メトリクス取得"""

            try:
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"取引メトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/ai_metrics")
        async def get_ai_metrics_endpoint():
            """AIメトリクス取得"""

            try:
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"AIメトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/fraud_metrics")
        async def get_fraud_metrics_endpoint():
            """不正検知メトリクス取得"""

            try:
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"不正検知メトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/db_metrics")
        async def get_db_metrics_endpoint():
            """データベースメトリクス取得"""

            try:
                registry = get_registry()
                metrics_data = generate_latest(registry)

                return Response(
                    content=metrics_data,
                    media_type=CONTENT_TYPE_LATEST
                )

            except Exception as e:
                logger.error(f"DBメトリクス取得エラー: {e}")
                return Response(
                    content=f"# ERROR: {e}\n",
                    media_type=CONTENT_TYPE_LATEST,
                    status_code=500
                )

        @self.app.get("/health")
        async def health_check():
            """ヘルスチェック"""

            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "metrics_endpoints": [
                    "/metrics",
                    "/risk_metrics",
                    "/trading_metrics",
                    "/ai_metrics",
                    "/fraud_metrics",
                    "/db_metrics"
                ]
            }

        @self.app.get("/status")
        async def get_status():
            """システムステータス"""

            try:
                collector = get_metrics_collector()
                status = collector.collect_all_metrics()

                return {
                    "exporter_status": "running",
                    "port": self.port,
                    "metrics_collection": status,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"ステータス取得エラー: {e}")
                return {
                    "exporter_status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

    def start_server(self):
        """サーバー開始"""

        try:
            logger.info(f"メトリクスエクスポーター開始: port {self.port}")

            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )

        except Exception as e:
            logger.error(f"サーバー開始エラー: {e}")
            raise

def start_metrics_server(port: int = 8000, background: bool = True) -> Optional[MetricsExporter]:
    """メトリクスサーバー開始"""

    exporter = MetricsExporter(port)

    if background:
        # バックグラウンドで実行
        def run_server():
            exporter.start_server()

        server_thread = threading.Thread(
            target=run_server,
            daemon=True
        )
        server_thread.start()

        logger.info(f"メトリクスサーバーをバックグラウンドで開始: port {port}")
        return exporter
    else:
        # フォアグラウンドで実行
        exporter.start_server()
        return None

# グローバル エクスポーター インスタンス
_metrics_exporter: Optional[MetricsExporter] = None

def get_metrics_exporter() -> Optional[MetricsExporter]:
    """グローバル エクスポーター取得"""
    return _metrics_exporter

def initialize_metrics_server(port: int = 8000) -> MetricsExporter:
    """メトリクスサーバー初期化"""

    global _metrics_exporter

    if _metrics_exporter is None:
        _metrics_exporter = start_metrics_server(port, background=True)
        logger.info(f"グローバル メトリクスサーバー初期化完了: port {port}")

    return _metrics_exporter
