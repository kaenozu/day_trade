"""
Prometheus + Grafana統合システム テスト

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

分割後のモジュール動作確認テスト
"""

import asyncio


async def test_prometheus_grafana_integration():
    """統合システムテスト"""
    print("=== 強化Prometheus + Grafana統合システム（分割後）テスト ===")

    try:
        # 統合システム作成
        from . import create_enhanced_prometheus_grafana_integration

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
            "ml_predictions": [
                {"model": "lstm", "symbol": "7203", "type": "price_forecast"}
            ],
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

        # 監視システム健全性チェック
        print("\n5. 監視システム健全性チェック...")
        health_status = await monitoring_system.get_monitoring_health()
        print(f"   総合ステータス: {health_status.get('overall_status')}")
        print(f"   Prometheus: {health_status.get('prometheus', {}).get('status')}")
        print(f"   Grafana: {health_status.get('grafana', {}).get('status')}")
        print(
            f"   メトリクス出力: {health_status.get('metrics_exporter', {}).get('status')}"
        )

        print("\n✅ 強化Prometheus + Grafana統合システム（分割後）テスト完了")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_prometheus_grafana_integration())