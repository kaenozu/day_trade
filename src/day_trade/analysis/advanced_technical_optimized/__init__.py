#!/usr/bin/env python3
"""
高度テクニカル指標システム（統合最適化版）
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化基盤フル活用版:
- Issue #324: 98%メモリ削減キャッシュ活用
- Issue #323: 100倍並列処理活用
- Issue #325: 97%ML高速化活用
- Issue #322: 89%精度データ拡張活用

モジュール構成:
- main: メインクラスとAPIエントリーポイント
- core_system: コアシステム機能
- data_structures: データ構造定義
- bollinger_bands_analyzer: Bollinger Bands分析
- ichimoku_analyzer: 一目均衡表分析
- batch_processor: バッチ処理機能
- performance_utils: パフォーマンス計算ユーティリティ
- testing: テスト関連機能
"""

# メインクラスのエクスポート（後方互換性のため）
from .main import AdvancedTechnicalIndicatorsOptimized

# データ構造のエクスポート
from .data_structures import (
    BollingerBandsAnalysis,
    ComplexMAAnalysis,
    FibonacciAnalysis,
    IchimokuAnalysis,
)

# 個別アナライザーのエクスポート
from .bollinger_bands_analyzer import BollingerBandsAnalyzer
from .ichimoku_analyzer import IchimokuAnalyzer

# バッチプロセッサーのエクスポート
from .batch_processor import BatchProcessor

# ユーティリティのエクスポート
from .performance_utils import PerformanceUtils

# コアシステムのエクスポート
from .core_system import CoreAdvancedTechnicalSystem

# テスト機能のエクスポート
from .testing import test_optimized_system, generate_test_data, run_comprehensive_tests

# バージョン情報
__version__ = "1.0.0"
__author__ = "Advanced Technical Analysis Team"
__license__ = "MIT"

# 公開API
__all__ = [
    # メインクラス
    "AdvancedTechnicalIndicatorsOptimized",
    
    # データ構造
    "BollingerBandsAnalysis",
    "IchimokuAnalysis", 
    "ComplexMAAnalysis",
    "FibonacciAnalysis",
    
    # アナライザー
    "BollingerBandsAnalyzer",
    "IchimokuAnalyzer",
    "BatchProcessor",
    
    # ユーティリティ
    "PerformanceUtils",
    "CoreAdvancedTechnicalSystem",
    
    # テスト機能
    "test_optimized_system",
    "generate_test_data",
    "run_comprehensive_tests",
]

# モジュール情報
MODULE_INFO = {
    "name": "advanced_technical_optimized",
    "version": __version__,
    "description": "高度テクニカル指標システム（統合最適化版）",
    "features": [
        "Bollinger Bands変動率分析",
        "一目均衡表総合判定", 
        "複合移動平均分析",
        "フィボナッチ retracement自動検出",
        "バッチ並列処理",
        "統合最適化基盤活用",
    ],
    "optimizations": {
        "cache_efficiency": "98%メモリ削減",
        "parallel_speedup": "100倍高速化",
        "ml_acceleration": "97%処理高速化", 
        "accuracy_improvement": "89%精度向上",
    },
    "modules": {
        "main": "メインクラスとAPIエントリーポイント",
        "core_system": "コアシステム機能",
        "data_structures": "データ構造定義",
        "bollinger_bands_analyzer": "Bollinger Bands分析",
        "ichimoku_analyzer": "一目均衡表分析",
        "batch_processor": "バッチ処理機能",
        "performance_utils": "パフォーマンス計算ユーティリティ",
        "testing": "テスト関連機能",
    }
}


# 使用方法のサンプル
def get_usage_example() -> str:
    """使用方法のサンプル取得"""
    return """
# 基本的な使用方法
from day_trade.analysis.advanced_technical_optimized import AdvancedTechnicalIndicatorsOptimized

# システム初期化
analyzer = AdvancedTechnicalIndicatorsOptimized(
    enable_cache=True,
    enable_parallel=True,
    enable_ml_optimization=True,
    max_concurrent=20,
)

# 単一銘柄分析
bb_result = await analyzer.analyze_bollinger_bands_optimized(data, "7203")
ichimoku_result = await analyzer.analyze_ichimoku_cloud_optimized(data, "7203")

# バッチ分析
batch_data = {"7203": data1, "6758": data2, "4755": data3}
batch_results = await analyzer.batch_analyze_symbols(batch_data, ["bb", "ichimoku"])

# 包括的分析
comprehensive_results = await analyzer.comprehensive_analysis(data, "7203")

# パフォーマンス統計取得
stats = analyzer.get_detailed_performance_stats()
"""


# 初期化時メッセージ
try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
    logger.info(f"高度テクニカル指標システム（統合最適化版）v{__version__} ロード完了")
except ImportError:
    pass