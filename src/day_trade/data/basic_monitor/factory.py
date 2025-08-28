#!/usr/bin/env python3
"""
Basic Monitor Factory
基本監視システムのファクトリー関数とテスト機能

監視システムの作成とテスト実行
"""

import asyncio
from datetime import datetime, timedelta

from .models import AlertSeverity, DataSourceHealth, MonitorAlert, MonitorRule
from .monitor_core import DataFreshnessMonitor

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


def create_data_freshness_monitor(
    storage_path: str = "data/monitoring",
    enable_cache: bool = True,
    alert_retention_days: int = 30,
    check_interval_seconds: int = 300,
) -> DataFreshnessMonitor:
    """データ鮮度・整合性監視システム作成"""
    return DataFreshnessMonitor(
        storage_path=storage_path,
        enable_cache=enable_cache,
        alert_retention_days=alert_retention_days,
        check_interval_seconds=check_interval_seconds,
    )


async def test_data_freshness_monitor():
    """テスト実行"""
    print("=== Issue #420 データ鮮度・整合性監視システムテスト ===")

    try:
        # 監視システム初期化
        monitor = create_data_freshness_monitor(
            storage_path="test_monitoring",
            enable_cache=True,
            alert_retention_days=7,
            check_interval_seconds=60,
        )

        print("\n1. データ鮮度・整合性監視システム初期化完了")
        print(f"   ストレージパス: {monitor.storage_path}")
        print(f"   チェック間隔: {monitor.check_interval_seconds}秒")
        print(f"   監視ルール数: {len(monitor.monitor_rules)}")

        # アラートコールバック登録
        def alert_handler(alert: MonitorAlert):
            print(f"   📢 アラートコールバック: {alert.title}")

        monitor.add_alert_callback(alert_handler)

        # データソースヘルス初期化（模擬）
        print("\n2. データソースヘルス初期化...")
        monitor.data_source_health["price_data"] = DataSourceHealth(
            source_id="price_data",
            source_type="api",
            last_update=datetime.utcnow() - timedelta(minutes=45),
            data_age_minutes=45.0,
            quality_score=0.95,
            availability=0.99,
            error_rate=0.01,
            response_time_ms=120,
            health_status="healthy",
        )
        print("   価格データソース登録完了")

        # 監視開始
        print("\n3. 監視開始...")
        await monitor.start_monitoring()
        print(f"   監視状態: {monitor.monitor_status.value}")

        # 手動チェックテスト
        print("\n4. 手動チェック実行...")

        # 鮮度チェックテスト
        import pandas as pd

        freshness_check = monitor.checks["freshness"]
        test_data = pd.DataFrame(
            {
                "Open": [2500],
                "High": [2550],
                "Low": [2480],
                "Close": [2530],
                "Volume": [1000000],
            },
            index=[datetime.utcnow() - timedelta(hours=2)],
        )  # 2時間前のデータ

        test_context = {"data_timestamp": datetime.utcnow() - timedelta(hours=2)}

        check_passed, alert = await freshness_check.execute_check(
            "test_source", test_data, test_context
        )
        print(f"   鮮度チェック結果: {'合格' if check_passed else '失敗'}")
        if alert:
            print(f"   生成アラート: {alert.title}")

        # 整合性チェックテスト
        consistency_check = monitor.checks["consistency"]

        # 不正な価格データ
        invalid_data = pd.DataFrame(
            {
                "Open": [2500],
                "High": [2400],
                "Low": [2600],
                "Close": [2530],
                "Volume": [-1000],  # High < Low, 負のVolume
            }
        )

        check_passed, alert = await consistency_check.execute_check(
            "test_source", invalid_data, {}
        )
        print(f"   整合性チェック結果: {'合格' if check_passed else '失敗'}")
        if alert:
            print(f"   生成アラート: {alert.title}")

        # ダッシュボード情報取得
        print("\n5. システムダッシュボード...")
        dashboard = monitor.get_system_dashboard()

        print(f"   監視状態: {dashboard['monitor_status']}")
        print(
            f"   アクティブアラート: {dashboard['alert_statistics']['total_active']}件"
        )
        print(f"   データソース数: {dashboard['health_statistics']['total_sources']}")
        print(
            f"   平均品質スコア: {dashboard['health_statistics']['avg_quality_score']:.3f}"
        )
        print(
            f"   平均可用性: {dashboard['health_statistics']['avg_availability']:.3f}"
        )

        # SLA状況
        print("\n   SLA状況:")
        for sla_id, sla_info in dashboard["sla_metrics"].items():
            print(
                f"     {sla_info['name']}: 可用性 {sla_info['availability']:.3f} "
                f"(目標: {sla_info['target_availability']:.3f})"
            )

        # アラート管理テスト
        print("\n6. アラート管理テスト...")
        if monitor.active_alerts:
            first_alert_id = list(monitor.active_alerts.keys())[0]
            print(f"   アクティブアラート確認: {first_alert_id}")
            await monitor.acknowledge_alert(first_alert_id, "test_user")
            await monitor.resolve_alert(first_alert_id, "test_user")
            print("   アラート解決完了")

        # しばらく監視継続
        print("\n7. 監視継続テスト（10秒間）...")
        await asyncio.sleep(10)

        # 最終ダッシュボード確認
        final_dashboard = monitor.get_system_dashboard()
        print(
            f"   最終アクティブアラート数: {final_dashboard['alert_statistics']['total_active']}"
        )

        # 監視停止
        print("\n8. 監視停止...")
        await monitor.stop_monitoring()
        print(f"   監視状態: {monitor.monitor_status.value}")

        # クリーンアップ
        await monitor.cleanup()

        print("\n✅ Issue #420 データ鮮度・整合性監視システムテスト完了")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_data_freshness_monitor())