"""
銘柄スクリーニング機能 (リファクタリング後)

3つの重複実装を統合パッケージに分割し、後方互換性を維持しながら
保守性とパフォーマンスを向上させたスクリーニングモジュール

⚠️ 重要: このファイルは新しいパッケージ構造への移行ブリッジです
実際の実装は screening/ パッケージ内にあります
"""

# 新しいパッケージ構造からクラスをインポート
from .screening import (
    EnhancedStockScreener,
    ScreenerCondition,
    ScreenerCriteria,
    ScreenerResult,
    ScreeningReport,
    StockScreener,
    UnifiedStockScreener,
    create_screening_report,
)

# 後方互換性のため、すべてのクラスを再エクスポート
__all__ = [
    # 推奨: 新しい統合実装
    "UnifiedStockScreener",
    "ScreeningReport",
    # 後方互換性: 既存API
    "StockScreener",
    "EnhancedStockScreener",
    "ScreenerCondition",
    "ScreenerCriteria",
    "ScreenerResult",
    "create_screening_report",
]
