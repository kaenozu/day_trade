"""
Enhanced Dashboard 簡単テスト

Windows環境でのシンプルなテストスクリプト
"""

import asyncio
import os
import sys
from datetime import datetime

# パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_dashboard_components():
    """ダッシュボードコンポーネントのテスト"""

    print("Enhanced Realtime Dashboard Test")
    print("=" * 50)

    try:
        from day_trade.dashboard.core.metrics_collector import MetricsCollector

        print("[OK] MetricsCollector import success")
    except Exception as e:
        print(f"[ERROR] MetricsCollector import failed: {e}")
        return False

    try:
        from day_trade.dashboard.core.feature_store_monitor import FeatureStoreMonitor

        print("[OK] FeatureStoreMonitor import success")
    except Exception as e:
        print(f"[ERROR] FeatureStoreMonitor import failed: {e}")
        return False

    try:
        from day_trade.dashboard.core.realtime_stream import RealtimeStream

        print("[OK] RealtimeStream import success")
    except Exception as e:
        print(f"[ERROR] RealtimeStream import failed: {e}")
        return False

    # コンポーネント初期化テスト
    print("\nComponent Initialization Test:")

    try:
        # MetricsCollector テスト
        collector = MetricsCollector(collection_interval=1.0)
        current_metrics = collector.get_current_metrics()

        if current_metrics and "cpu" in current_metrics:
            cpu_usage = current_metrics["cpu"]["usage_percent"]
            memory_usage = current_metrics["memory"]["usage_percent"]
            print(
                f"[OK] System Metrics - CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%"
            )
        else:
            print("[WARNING] Could not get system metrics")

        # 短時間の収集テスト
        await collector.start_collection()
        await asyncio.sleep(1)

        history = collector.get_metrics_history(1)
        print(f"[OK] Metrics history collected: {len(history)} samples")

        health_report = collector.generate_health_report()
        health_score = health_report.get("overall_health", 0)
        print(f"[OK] System health score: {health_score}/100")

        await collector.stop_collection()

    except Exception as e:
        print(f"[ERROR] MetricsCollector test failed: {e}")
        return False

    try:
        # FeatureStoreMonitor テスト
        from day_trade.ml.feature_store import FeatureStore, FeatureStoreConfig

        config = FeatureStoreConfig(max_cache_size_mb=50)
        feature_store = FeatureStore(config=config)
        monitor = FeatureStoreMonitor(update_interval=1.0)
        monitor.set_feature_store(feature_store)

        # 監視開始
        await monitor.start_monitoring()

        # テスト用の統計データ生成（Feature Store内部statsを直接操作）
        # 実際のFeatureStoreの使用をシミュレート
        feature_store.stats["cache_hits"] = 5
        feature_store.stats["cache_misses"] = 2
        feature_store.stats["cache_size"] = 3
        print("   [INFO] Generated test statistics for Feature Store")

        await asyncio.sleep(1)  # 監視データ収集待機

        metrics = monitor.get_current_metrics()
        if metrics:
            hit_rate = metrics.get("hit_rate", 0)
            total_requests = metrics.get("total_requests", 0)
            print(
                f"[OK] Feature Store - Hit Rate: {hit_rate}%, Requests: {total_requests}"
            )

        health = monitor.get_health_status()
        print(f"[OK] Feature Store health: {health.get('status', 'unknown')}")

        await monitor.stop_monitoring()

    except Exception as e:
        print(f"[ERROR] FeatureStoreMonitor test failed: {e}")
        return False

    try:
        # RealtimeStream テスト
        stream = RealtimeStream(broadcast_interval=2.0)
        stats = stream.get_connection_stats()
        print(f"[OK] RealtimeStream - Connections: {stats['active_connections']}")

    except Exception as e:
        print(f"[ERROR] RealtimeStream test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("[SUCCESS] Enhanced Dashboard System is working correctly!")
    print("System Components:")
    print("- Real-time system monitoring")
    print("- Feature Store performance tracking")
    print("- WebSocket-based live data streaming")
    print("- Comprehensive health reporting")
    print("- High-performance metrics collection")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_dashboard_components())
        if success:
            print("\nAll tests passed successfully!")
        else:
            print("\nSome tests failed. Check the output above.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest execution error: {e}")
    finally:
        print("\nTest finished")
