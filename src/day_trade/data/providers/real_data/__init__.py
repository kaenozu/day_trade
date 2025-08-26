#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider Package

後方互換性を保つための再エクスポートモジュール
"""

# 列挙型とデータクラスのエクスポート
from .enums import (
    DataSource,
    DataQualityLevel,
    DataSourceConfig,
    DataFetchResult
)

# 設定管理
from .config_manager import DataSourceConfigManager

# キャッシュ管理
from .cache_manager import ImprovedCacheManager

# 基底プロバイダー
from .base_provider import BaseDataProvider

# 個別プロバイダー
from .yahoo_provider import ImprovedYahooFinanceProvider
from .stooq_provider import ImprovedStooqProvider
from .mock_provider import MockDataProvider

# メインプロバイダー
from .multi_source_provider import ImprovedMultiSourceDataProvider

# ユーティリティ
from .utils import (
    configure_windows_encoding,
    test_improved_data_provider,
    setup_logging
)

# 後方互換性のためのエイリアス
RealDataProvider = ImprovedMultiSourceDataProvider

# グローバルインスタンス
improved_data_provider = ImprovedMultiSourceDataProvider()
real_data_provider = improved_data_provider

# パッケージ公開インターフェース
__all__ = [
    # 列挙型とデータクラス
    'DataSource',
    'DataQualityLevel',  
    'DataSourceConfig',
    'DataFetchResult',
    
    # 管理クラス
    'DataSourceConfigManager',
    'ImprovedCacheManager',
    
    # プロバイダークラス
    'BaseDataProvider',
    'ImprovedYahooFinanceProvider',
    'ImprovedStooqProvider', 
    'MockDataProvider',
    'ImprovedMultiSourceDataProvider',
    
    # 後方互換性エイリアス
    'RealDataProvider',
    
    # グローバルインスタンス
    'improved_data_provider',
    'real_data_provider',
    
    # ユーティリティ
    'configure_windows_encoding',
    'test_improved_data_provider',
    'setup_logging',
]