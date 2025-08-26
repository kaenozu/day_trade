#!/usr/bin/env python3
"""
多角的データ収集システム - テスト実行

Issue #322: ML Data Shortage Problem Resolution
システムのテスト実行スクリプト
"""

import asyncio

from .manager import MultiSourceDataManager
from .feature_engineer import ComprehensiveFeatureEngineer

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


async def test_multi_source_data_manager():
    """多角的データ収集システムテスト"""
    print("=== 多角的データ収集システムテスト ===")

    # データ管理システム初期化
    data_manager = MultiSourceDataManager(
        enable_cache=True, cache_ttl_minutes=5, max_concurrent=4
    )

    try:
        # 包括データ収集テスト
        print("\n1. 包括データ収集テスト...")
        test_symbols = ["7203", "8306"]

        for symbol in test_symbols:
            print(f"\n  {symbol}のデータ収集...")
            comprehensive_data = await data_manager.collect_comprehensive_data(symbol)

            print(f"    収集データ種類: {len(comprehensive_data)}")
            for data_type, data in comprehensive_data.items():
                print(f"    - {data_type}: 品質スコア {data.quality_score:.2f}")

        # 特徴量生成テスト
        print("\n2. 特徴量生成テスト...")
        if comprehensive_data:
            feature_engineer = ComprehensiveFeatureEngineer(data_manager)
            features = feature_engineer.generate_comprehensive_features(
                comprehensive_data
            )
            print(f"    生成特徴量数: {len(features)}")

            # 特徴量サンプル表示
            sample_features = dict(list(features.items())[:5])
            for feature, value in sample_features.items():
                print(f"    - {feature}: {value:.3f}")

        # システム統計
        print("\n3. システム統計...")
        stats = data_manager.get_comprehensive_stats()
        collection_stats = stats["collection_stats"]
        print(f"    総リクエスト: {collection_stats['total_requests']}")
        print(
            f"    成功率: {collection_stats['successful_requests'] / max(collection_stats['total_requests'], 1):.1%}"
        )
        print(f"    キャッシュヒット: {collection_stats['cache_hits']}")

        print("\n✅ 多角的データ収集システムテスト完了")

    finally:
        await data_manager.shutdown()


def main():
    """メイン実行関数"""
    try:
        asyncio.run(test_multi_source_data_manager())
    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()