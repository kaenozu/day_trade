"""
銘柄スクリーニング機能（拡張版） - 後方互換性ブリッジ

⚠️ 非推奨: 新しいコードでは analysis.screening パッケージを使用してください
"""

import warnings

# 新しい統合実装からインポート
from .screening import (
    EnhancedStockScreener,
    ScreenerCondition,
    ScreenerCriteria,
    ScreenerResult,
    ScreeningReport,
    UnifiedStockScreener,
)

# Deprecation警告を表示
warnings.warn(
    "screener_enhanced.py は非推奨です。analysis.screening パッケージを使用してください。",
    DeprecationWarning,
    stacklevel=2,
)

# すべてのクラスを再エクスポート（後方互換性のため）
__all__ = [
    "UnifiedStockScreener",
    "EnhancedStockScreener",
    "ScreenerCondition",
    "ScreenerCriteria",
    "ScreenerResult",
    "ScreeningReport",
]
