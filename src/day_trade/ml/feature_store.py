#!/usr/bin/env python3
"""
特徴量ストアメインモジュール（互換性レイヤー）

元の feature_store.py との完全な後方互換性を提供します。
実際の実装は feature_store パッケージに移動されました。

Usage:
    # 従来通りの使用方法
    from day_trade.ml.feature_store import FeatureStore, FeatureStoreConfig
    
    # 新しいパッケージベースの使用方法
    from day_trade.ml.feature_store import create_feature_store
"""

# フルな後方互換性のためのインポート
from .feature_store import *  # noqa: F401,F403

# 元のモジュールで使用されていた全てのクラスと関数を再エクスポート
__all__ = [
    # メインクラス
    "FeatureStore",
    "create_feature_store",
    
    # データクラス・設定
    "FeatureStoreConfig", 
    "FeatureMetadata",
    
    # 詳細制御用（上級者向け）
    "MetadataManager",
    "StorageManager",
    "FeatureOperations",
    "CacheManager",
    "CacheAnalyzer", 
    "BatchProcessor",
    "ChunkedBatchProcessor",
    "StatisticsTracker",
]