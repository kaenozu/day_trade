#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider Utilities

リアルデータプロバイダーユーティリティ関数
"""

import asyncio
import logging
import os
import sys


# Windows環境での文字化け対策
def configure_windows_encoding():
    """Windows環境での文字化け対策を設定"""
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


async def test_improved_data_provider():
    """改善版データプロバイダーのテスト"""
    from .multi_source_provider import ImprovedMultiSourceDataProvider
    
    print("=== Improved Multi-Source Data Provider Test ===")

    try:
        # プロバイダー初期化
        provider = ImprovedMultiSourceDataProvider()
        print(f"✓ Provider initialized with {len(provider.providers)} sources")

        # 有効なソース確認
        enabled_sources = provider.config_manager.get_enabled_sources()
        print(f"✓ Enabled sources: {', '.join(enabled_sources)}")

        # テスト銘柄でデータ取得
        test_symbols = ["7203", "4751"]

        for symbol in test_symbols:
            print(f"\n--- Testing symbol: {symbol} ---")

            # データ取得
            result = await provider.get_stock_data(symbol, "1mo")

            if result.data is not None:
                print("✓ Data fetched successfully")
                print(f"  - Source: {result.source.value}")
                print(f"  - Quality: {result.quality_level.value} ({result.quality_score:.1f})")
                print(f"  - Data points: {len(result.data)}")
                print(f"  - Fetch time: {result.fetch_time:.2f}s")
                print(f"  - Cached: {result.cached}")

                # データ内容確認
                if not result.data.empty:
                    latest = result.data.iloc[-1]
                    print(f"  - Latest close: {latest['Close']:.2f}")
            else:
                print(f"❌ Data fetch failed: {result.error_message}")

        # 統計情報表示
        print("\n--- Provider Statistics ---")
        stats = provider.get_statistics()
        for source, data in stats.items():
            print(f"{source}:")
            print(f"  - Success rate: {data['success_rate']:.1f}%")
            print(f"  - Avg response time: {data['avg_response_time']:.2f}s")
            print(f"  - Avg quality: {data['avg_quality_score']:.1f}")

        # ソース状態表示
        print("\n--- Source Status ---")
        status = provider.get_source_status()
        for source, data in status.items():
            print(f"{source}:")
            print(f"  - Enabled: {data['enabled']}")
            print(f"  - Priority: {data['priority']}")
            print(f"  - Daily requests: {data['daily_requests']}/{data['daily_limit']}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    # Windows環境設定
    configure_windows_encoding()
    
    # ログ設定
    setup_logging()

    # テスト実行
    asyncio.run(test_improved_data_provider())