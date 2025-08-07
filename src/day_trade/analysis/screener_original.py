"""
銘柄スクリーニング機能（オリジナル版） - 後方互換性ブリッジ

⚠️ 非推奨: 新しいコードでは analysis.screening パッケージを使用してください
"""

import warnings

# 新しい統合実装からインポート
from .screening import (
    OriginalStockScreener,
    ScreenerCondition,
    ScreenerCriteria,
    ScreenerResult,
    StockScreener,
    UnifiedStockScreener,
)

# Deprecation警告を表示
warnings.warn(
    "screener_original.py は非推奨です。analysis.screening パッケージを使用してください。",
    DeprecationWarning,
    stacklevel=2,
)

# オリジナルの完全互換性のため
# StockScreener は既に上でインポートされているため、エイリアスを作成
StockScreenerOriginal = OriginalStockScreener

# すべてのクラスを再エクスポート（後方互換性のため）
__all__ = [
    "UnifiedStockScreener",
    "StockScreener",
    "OriginalStockScreener",
    "ScreenerCondition",
    "ScreenerCriteria",
    "ScreenerResult",
]
