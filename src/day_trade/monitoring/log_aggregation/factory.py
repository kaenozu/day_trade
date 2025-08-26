#!/usr/bin/env python3
"""
統合ログ集約システムのファクトリ関数とユーティリティ
"""

import asyncio
import json
import time
from datetime import datetime

from .core_system import LogAggregationSystem


def create_log_aggregation_system(
    storage_path: str = "data/logs",
    enable_cache: bool = True,
    max_memory_logs: int = 10000,
    retention_days: int = 30,
) -> LogAggregationSystem:
    """統合ログ集約システム作成"""
    return LogAggregationSystem(
        storage_path=storage_path,
        enable_cache=enable_cache,
        max_memory_logs=max_memory_logs,
        retention_days=retention_days,
    )


# 後方互換性のためのエイリアス
def get_log_aggregation_system(*args, **kwargs) -> LogAggregationSystem:
    """後方互換性のためのエイリアス関数"""
    return create_log_aggregation_system(*args, **kwargs)


async def test_log_aggregation_system():
    """テスト実行関数"""
    print("=== Issue #417 統合ログ集約・分析システムテスト ===")

    try:
        # ログ集約システム初期化
        log_system = create_log_aggregation_system(
            storage_path="test_logs",
            enable_cache=True,
            max_memory_logs=1000,
            retention_days=7,
        )

        print("\n1. 統合ログ集約システム初期化完了")
        print(f"   ストレージパス: {log_system.storage_path}")
        print(f"   ログパターン数: {len(log_system.log_patterns)}")
        print(f"   パーサー数: {len(log_system.parsers)}")

        # ログ処理開始
        print("\n2. ログ処理開始...")
        await log_system.start_processing()

        # テストログの取り込み
        print("\n3. テストログ取り込みテスト...")

        # JSON構造化ログ
        json_log = json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "logger_name": "test.component",
                "message": "Application started successfully",
                "user_id": "test_user",
                "trace_id": "trace_123",
            }
        )

        success = await log_system.ingest_log(json_log)
        print(f"   JSON構造化ログ取り込み: {'成功' if success else '失敗'}")

        # エラーログ（アラート発生テスト用）
        for i in range(7):  # 閾値5を超える
            error_log = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "ERROR",
                    "logger_name": "test.error",
                    "message": f"Database connection failed: attempt {i + 1}",
                    "error": "ConnectionError",
                }
            )
            await log_system.ingest_log(error_log)

        print("   エラーログ取り込み: 7件")

        # 標準形式ログ
        standard_log = "2024-01-15 12:00:00 - api.handler - INFO - Request processed successfully"
        success = await log_system.ingest_log(standard_log)
        print(f"   標準形式ログ取り込み: {'成功' if success else '失敗'}")

        # パフォーマンスログ
        perf_log = json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "WARNING",
                "logger_name": "performance.monitor",
                "message": "Slow query detected: SELECT * FROM large_table took 5.2s",
                "duration": 5.2,
                "query": "SELECT * FROM large_table",
            }
        )
        await log_system.ingest_log(perf_log)
        print("   パフォーマンスログ取り込み: 成功")

        # 処理時間を与える
        await asyncio.sleep(2)

        # ログ検索テスト
        print("\n4. ログ検索テスト...")
        from .models import LogSearchQuery
        from datetime import timedelta

        search_query = LogSearchQuery(
            query_text="connection",
            start_time=datetime.utcnow() - timedelta(minutes=5),
            end_time=datetime.utcnow(),
            limit=50,
        )

        search_results = await log_system.search_logs(search_query)
        print(f"   検索結果: {len(search_results)}件")

        if search_results:
            print(
                f"   最新ログ: {search_results[0].level.value} - {search_results[0].message[:50]}"
            )

        # ログ分析テスト
        print("\n5. ログ分析テスト...")
        analytics = await log_system.get_log_analytics(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
        )

        if analytics:
            stats = analytics.get("basic_statistics", {})
            print(f"   総ログ数: {stats.get('total_logs', 0)}")
            print(
                f"   ユニークコンポーネント数: {stats.get('unique_components', 0)}"
            )
            print(f"   ユニークソース数: {stats.get('unique_sources', 0)}")

            level_dist = analytics.get("level_distribution", {})
            print(f"   レベル分布: {dict(level_dist)}")

            if analytics.get("top_errors"):
                print("   トップエラー:")
                for error in analytics["top_errors"][:3]:
                    print(f"     - {error['message'][:40]}... ({error['count']}回)")

        # アクティブアラート確認
        print("\n6. アクティブアラート確認...")
        print(f"   アクティブアラート数: {len(log_system.active_alerts)}")

        for alert_id, alert in list(log_system.active_alerts.items())[:3]:
            print(f"   アラート: {alert.pattern_name} - {alert.message}")
            print(
                f"     重要度: {alert.severity.value}, 発生回数: {alert.occurrence_count}"
            )

        # ログパターン確認
        print("\n7. ログパターン確認...")
        for pattern_id, pattern in log_system.log_patterns.items():
            enabled_status = "有効" if pattern.enabled else "無効"
            print(
                f"   {pattern.name}: {enabled_status} (閾値: {pattern.alert_threshold})"
            )

        # ログエクスポートテスト
        print("\n8. ログエクスポートテスト...")
        try:
            export_query = LogSearchQuery(limit=100)
            export_file = await log_system.export_logs(
                export_query, export_format="json"
            )
            print(f"   エクスポート成功: {export_file}")

            # エクスポートファイル確認
            with open(export_file, encoding="utf-8") as f:
                exported_data = json.load(f)
            print(f"   エクスポートデータ: {len(exported_data)}件")

        except Exception as e:
            print(f"   エクスポートエラー: {e}")

        # パフォーマンステスト
        print("\n9. パフォーマンステスト...")
        start_time = time.time()

        # 大量ログ取り込み
        for i in range(100):
            test_log = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "INFO",
                    "logger_name": f"test.batch.{i % 5}",
                    "message": f"Batch test message {i}",
                    "batch_id": i,
                }
            )
            await log_system.ingest_log(test_log)

        processing_time = time.time() - start_time
        print(f"   100件ログ取り込み時間: {processing_time:.2f}秒")
        print(f"   スループット: {100 / processing_time:.1f}件/秒")

        # しばらく処理継続
        await asyncio.sleep(3)

        # システム停止
        await log_system.stop_processing()

        # クリーンアップ
        await log_system.cleanup()

        print("\n✅ Issue #417 統合ログ集約・分析システムテスト完了")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_log_aggregation_system())