#!/usr/bin/env python3
"""
統一バッチ処理エンジン
Issue #376: バッチ処理の強化

データ取得・処理・保存の全体をバッチ最適化する統合エンジン
後方互換性を保つため、新しいモジュールからインポート
"""

# 新しいモジュールからすべてをインポート（後方互換性）
from .batch_processing_engine import *  # noqa: F401,F403

# テスト用関数は残す
import asyncio


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #376 統一バッチ処理エンジンテスト ===")

        engine = BatchProcessingEngine(max_concurrent_jobs=3)

        try:
            # 株価データパイプラインテスト
            print("\n1. 株価データパイプライン実行")
            test_symbols = ["AAPL", "GOOGL", "MSFT"]

            result = await execute_stock_batch_pipeline(
                symbols=test_symbols,
                include_historical=False,
                store_data=False,  # テストではデータベース保存をスキップ
                engine=engine,
            )

            print(f"ジョブ結果: success={result.success}")
            print(f"処理時間: {result.total_processing_time_ms:.2f}ms")
            print(f"ステージ数: {len(result.stage_results)}")

            for stage_result in result.stage_results:
                print(
                    f"  - {stage_result.stage.value}: "
                    f"success={stage_result.success}, "
                    f"time={stage_result.processing_time_ms:.2f}ms"
                )

            # 統計情報
            print("\n2. 統計情報")
            stats = engine.get_stats()
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

            # ヘルスステータス
            print("\n3. ヘルスステータス")
            health = engine.get_health_status()
            for key, value in health.items():
                print(f"  {key}: {value}")

        finally:
            await engine.cleanup()

    asyncio.run(main())
    print("\n=== 統一バッチ処理エンジンテスト完了 ===")