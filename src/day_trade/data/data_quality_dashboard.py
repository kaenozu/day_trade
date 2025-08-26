#!/usr/bin/env python3
"""
統合データ品質ダッシュボードシステム
Issue #420: データ管理とデータ品質保証メカニズムの強化

DEPRECATION NOTICE:
このファイルは、モジュラー構造への移行のため非推奨となりました。
新しいパッケージ構造をご利用ください:
- src/day_trade/data_quality/dashboard/

バックワード互換性のため、すべてのインポートは引き続き機能します。
"""

import warnings

# Backward compatibility imports
from .dashboard import (
    ChartType,
    DashboardComponentType,
    DashboardLayout,
    DashboardWidget,
    DataQualityDashboard,
    QualityKPI,
    create_data_quality_dashboard,
)

# Deprecation warning
warnings.warn(
    "data_quality_dashboard.py is deprecated. Use 'from day_trade.data_quality.dashboard import DataQualityDashboard' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Preserve all original exports for backward compatibility
__all__ = [
    # Main classes
    "DataQualityDashboard",
    
    # Model classes
    "DashboardWidget",
    "DashboardLayout", 
    "QualityKPI",
    
    # Enums
    "DashboardComponentType",
    "ChartType",
    
    # Factory function
    "create_data_quality_dashboard",
]


# Original test code maintained for compatibility
if __name__ == "__main__":
    import asyncio
    import json
    import time
    from pathlib import Path

    # テスト実行
    async def test_data_quality_dashboard():
        print("=== Issue #420 統合データ品質ダッシュボードシステムテスト ===")
        print("注意: このテストは非推奨のエントリーポイントを使用しています。")
        print("新しいパッケージ構造 'day_trade.data_quality.dashboard' の使用を推奨します。")

        try:
            # ダッシュボードシステム初期化
            dashboard = create_data_quality_dashboard(
                storage_path="test_dashboard",
                enable_cache=True,
                refresh_interval_seconds=60,
                retention_days=30,
            )

            print("\n1. 統合データ品質ダッシュボード初期化完了")
            print(f"   ストレージパス: {dashboard.storage_path}")

            # メインダッシュボードデータ取得
            print("\n2. メインダッシュボードデータ取得...")
            main_data = await dashboard.get_dashboard_data("main_dashboard")

            if "error" not in main_data:
                print(f"   レイアウト: {main_data['layout']['name']}")
                print(f"   最終リフレッシュ: {main_data['metadata']['last_refresh']}")
                print(f"   ウィジェット数: {len(main_data['widgets'])}")

                # 主要メトリクス表示
                if "global_metrics" in main_data:
                    global_metrics = main_data["global_metrics"]
                    print(
                        f"   総データポイント: {global_metrics.get('total_data_points', 0):,}"
                    )
                    print(
                        f"   今日の品質チェック: {global_metrics.get('quality_checks_today', 0):,}"
                    )

                # システムヘルス表示
                if "system_health" in main_data:
                    health = main_data["system_health"]
                    print(f"   システム状態: {health.get('overall_status', 'unknown')}")
                    print(f"   CPU使用率: {health.get('cpu_usage', 0):.1%}")
                    print(f"   メモリ使用率: {health.get('memory_usage', 0):.1%}")
            else:
                print(f"   エラー: {main_data['error']}")

            # 品質レポート生成テスト
            print("\n3. データ品質レポート生成テスト...")
            try:
                report_file = await dashboard.generate_quality_report(
                    report_type="daily", include_charts=True, export_format="json"
                )
                print(f"   レポート生成成功: {report_file}")

                # レポート内容確認
                if Path(report_file).exists():
                    with open(report_file, encoding="utf-8") as f:
                        report_data = json.load(f)

                    print(f"   レポートID: {report_data['report_id']}")
                    print(f"   レポート期間: {report_data['period']['duration_hours']}時間")

                    if "executive_summary" in report_data:
                        summary = report_data["executive_summary"]
                        print(
                            f"   総合品質スコア: {summary.get('overall_quality_score', 0):.3f}"
                        )

            except Exception as e:
                print(f"   レポート生成エラー: {e}")

            # ダッシュボードエクスポートテスト
            print("\n4. ダッシュボードエクスポートテスト...")
            try:
                export_file = await dashboard.export_dashboard_data(
                    layout_id="main_dashboard", format="json"
                )
                print(f"   エクスポート成功: {export_file}")

            except Exception as e:
                print(f"   エクスポートエラー: {e}")

            # パフォーマンステスト
            print("\n5. パフォーマンステスト...")
            start_time = time.time()

            # 複数回のデータ取得（キャッシュ効果確認）
            for i in range(3):
                await dashboard.get_dashboard_data("main_dashboard")

            end_time = time.time()
            avg_response_time = (end_time - start_time) / 3 * 1000
            print(f"   平均応答時間: {avg_response_time:.1f}ms (3回平均)")

            # クリーンアップ
            await dashboard.cleanup()

            print("\n✅ Issue #420 統合データ品質ダッシュボードシステムテスト完了")
            print("\n推奨: 今後は新しいパッケージ構造をご利用ください:")
            print("from day_trade.data_quality.dashboard import DataQualityDashboard")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_data_quality_dashboard())