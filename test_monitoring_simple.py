#!/usr/bin/env python3
"""
リアルタイムメトリクス・アラートシステム 簡易テスト
Simple Monitoring System Test
"""

import asyncio
import time
from datetime import datetime

print("リアルタイムメトリクス・アラートシステム - 簡易テスト")
print("="*60)
print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

async def test_basic_imports():
    """基本インポートテスト"""

    print("テスト1: 基本インポート")
    print("-" * 30)

    try:
        from src.day_trade.monitoring.metrics.prometheus_metrics import (
            get_metrics_collector,
            get_risk_metrics
        )
        print("  OK メトリクス収集器インポート成功")

        collector = get_metrics_collector()
        print("  OK メトリクス収集器初期化成功")

        risk_metrics = get_risk_metrics()
        print("  OK リスクメトリクス初期化成功")

        return True

    except Exception as e:
        print(f"  NG インポートエラー: {e}")
        return False

async def test_metrics_collection():
    """メトリクス収集テスト"""

    print("\nテスト2: メトリクス収集")
    print("-" * 30)

    try:
        from src.day_trade.monitoring.metrics.prometheus_metrics import (
            get_metrics_collector,
            get_risk_metrics
        )

        collector = get_metrics_collector()
        risk_metrics = get_risk_metrics()

        # メトリクス収集実行
        start_time = time.time()
        result = collector.collect_all_metrics()
        processing_time = time.time() - start_time

        print(f"  OK メトリクス収集完了: {processing_time:.3f}秒")
        print(f"  結果ステータス: {result.get('status', 'unknown')}")

        # リスクスコア更新テスト
        risk_metrics.update_risk_score('test_component', 'TEST_SYMBOL', 0.75)
        print("  OK リスクスコア更新テスト完了")

        return True

    except Exception as e:
        print(f"  NG メトリクス収集エラー: {e}")
        return False

async def test_metrics_server():
    """メトリクスサーバーテスト"""

    print("\nテスト3: メトリクスサーバー")
    print("-" * 30)

    try:
        from src.day_trade.monitoring.metrics.metrics_exporter import start_metrics_server

        # バックグラウンドでサーバー開始
        print("  メトリクスサーバー開始中...")

        exporter = start_metrics_server(port=8002, background=True)

        # 短時間待機
        await asyncio.sleep(1)

        print("  OK メトリクスサーバー開始完了")
        print("  エンドポイント: http://localhost:8002/metrics")
        print("  ヘルスチェック: http://localhost:8002/health")

        return True

    except Exception as e:
        print(f"  NG メトリクスサーバーエラー: {e}")
        return False

async def main():
    """メイン実行"""

    results = []

    try:
        # 基本テスト実行
        result1 = await test_basic_imports()
        results.append(result1)

        result2 = await test_metrics_collection()
        results.append(result2)

        result3 = await test_metrics_server()
        results.append(result3)

        # 結果表示
        print("\n" + "="*60)
        print("テスト結果サマリー")
        print("="*60)

        success_count = sum(results)
        total_count = len(results)
        success_rate = (success_count / total_count) * 100

        print(f"成功率: {success_count}/{total_count} ({success_rate:.1f}%)")

        if success_rate == 100:
            print("全テスト成功! 監視システムは正常に動作しています。")
            print()
            print("次のステップ:")
            print("1. docker-compose -f docker-compose.monitoring.yml up")
            print("2. Grafana: http://localhost:3000")
            print("3. Prometheus: http://localhost:9090")
        else:
            print("いくつかのテストが失敗しました。設定を確認してください。")

        print("="*60)

    except Exception as e:
        print(f"テスト実行エラー: {e}")

if __name__ == "__main__":
    asyncio.run(main())
