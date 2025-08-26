#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Package Initialization
リアルデータプロバイダー V2 - パッケージ初期化

分割されたモジュールを統合し、後方互換性を保つための初期化ファイル
"""

# Windows環境での文字化け対策
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 基本モデルとデータクラス
from .models import (
    DataSource,
    DataQualityLevel, 
    DataSourceConfig,
    DataFetchResult,
    ProviderStatistics,
    SourceStatus,
    DEFAULT_PERIOD_DAYS,
    REQUIRED_COLUMNS,
    COLUMN_MAPPING,
    YAHOO_SUFFIXES,
    STOOQ_SUFFIXES
)

# 設定管理
from .config_manager import DataSourceConfigManager

# キャッシュ管理
from .cache_manager import ImprovedCacheManager

# プロバイダー基底クラス
from .base_provider import BaseDataProvider

# 個別プロバイダー
from .yahoo_provider import ImprovedYahooFinanceProvider
from .stooq_provider import ImprovedStooqProvider
from .mock_provider import MockDataProvider

# 統合プロバイダー（メインクラス）
from .multi_source_provider import ImprovedMultiSourceDataProvider

# グローバルインスタンス（後方互換性のため）
improved_data_provider = ImprovedMultiSourceDataProvider()

# 後方互換性のためのエイリアス（元のファイルからインポートしているコードの対応）
# 元のファイル名: real_data_provider_v2_improved.py での使用を想定

# クラスエイリアス
DataSourceConfigManager = DataSourceConfigManager
ImprovedCacheManager = ImprovedCacheManager  
BaseDataProvider = BaseDataProvider
ImprovedYahooFinanceProvider = ImprovedYahooFinanceProvider
ImprovedStooqProvider = ImprovedStooqProvider
MockDataProvider = MockDataProvider
ImprovedMultiSourceDataProvider = ImprovedMultiSourceDataProvider

# グローバルインスタンスエイリアス
improved_data_provider = improved_data_provider

# テスト関数（元ファイルにあった）
async def test_improved_data_provider():
    """改善版データプロバイダーのテスト"""
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
                print(f"✓ Data fetched successfully")
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

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "Improved Real Data Provider V2 - Modularized"

# 公開インターフェース
__all__ = [
    # データクラス・列挙型
    'DataSource',
    'DataQualityLevel',
    'DataSourceConfig', 
    'DataFetchResult',
    'ProviderStatistics',
    'SourceStatus',
    
    # 管理クラス
    'DataSourceConfigManager',
    'ImprovedCacheManager',
    
    # プロバイダークラス
    'BaseDataProvider',
    'ImprovedYahooFinanceProvider',
    'ImprovedStooqProvider', 
    'MockDataProvider',
    'ImprovedMultiSourceDataProvider',
    
    # グローバルインスタンス
    'improved_data_provider',
    
    # テスト関数
    'test_improved_data_provider',
    
    # 定数
    'DEFAULT_PERIOD_DAYS',
    'REQUIRED_COLUMNS',
    'COLUMN_MAPPING',
    'YAHOO_SUFFIXES',
    'STOOQ_SUFFIXES'
]