#!/usr/bin/env python3
"""
多角的データ収集システム - パッケージ初期化

Issue #322: ML Data Shortage Problem Resolution
後方互換性を保つためのクラス・関数の再エクスポート
"""

# 基底クラスとモデル
from .base import DataCollector
from .models import (
    DataSource,
    CollectedData, 
    DataQualityReport,
)

# データ収集器
from .news_collector import NewsDataCollector
from .sentiment_analyzer import SentimentAnalyzer
from .macro_collector import MacroEconomicCollector

# メインマネージャー
from .manager import MultiSourceDataManager

# 特徴量エンジニアリング
from .feature_engineer import ComprehensiveFeatureEngineer

# 後方互換性のため、元のファイルからインポートしていたクラスをすべてエクスポート
__all__ = [
    # モデル
    "DataSource",
    "CollectedData",
    "DataQualityReport",
    
    # 基底クラス
    "DataCollector",
    
    # 収集器
    "NewsDataCollector", 
    "SentimentAnalyzer",
    "MacroEconomicCollector",
    
    # メインシステム
    "MultiSourceDataManager",
    
    # 特徴量エンジニアリング
    "ComprehensiveFeatureEngineer",
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade Sub System"
__description__ = "多角的データ収集システム - モジュール化版"