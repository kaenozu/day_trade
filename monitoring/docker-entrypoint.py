#!/usr/bin/env python3
"""
Day Trade メトリクスエクスポーター エントリーポイント
Docker コンテナ用スタートアップスクリプト
"""

import os
import sys

# Pythonパス設定
sys.path.insert(0, "/app/src")

from src.day_trade.monitoring.metrics import (
    get_health_metrics,
    get_metrics_collector,
    get_risk_metrics,
    initialize_metrics_server,
)
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def main():
    """メイン実行関数"""

    try:
        # 環境変数取得
        port = int(os.getenv("METRICS_PORT", "8000"))
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/trading.db")

        logger.info("Day Trade メトリクスエクスポーター開始")
        logger.info(f"設定: Port={port}, Redis={redis_url}, DB={db_url}")

        # メトリクス初期化
        logger.info("メトリクス収集システム初期化中...")

        # メトリクス収集器初期化
        collector = get_metrics_collector()
        risk_metrics = get_risk_metrics()
        health_metrics = get_health_metrics()

        # 初期メトリクス収集
        logger.info("初期メトリクス収集実行...")
        initial_metrics = collector.collect_all_metrics()
        logger.info(f"初期メトリクス収集結果: {initial_metrics}")

        # HTTPサーバー開始
        logger.info(f"メトリクスサーバー開始: port {port}")
        exporter = initialize_metrics_server(port)

        # メトリクス更新ループ開始
        logger.info("メトリクス更新ループ開始")

        # フォアグラウンドでサーバー実行
        from src.day_trade.monitoring.metrics.metrics_exporter import MetricsExporter

        exporter_server = MetricsExporter(port)
        exporter_server.start_server()

    except KeyboardInterrupt:
        logger.info("キーボード割り込み - サーバー停止")
    except Exception as e:
        logger.error(f"エクスポーター起動エラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
