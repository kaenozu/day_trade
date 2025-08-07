"""
統合スクリーニングパッケージ

3つのScreener重複実装を統合し、最高の機能を提供
- screener.py, screener_enhanced.py, screener_original.py の統合
- パフォーマンス最適化とキャッシュ機能
- 後方互換性完全サポート
"""

# 新しい統合実装
# 後方互換性サポート
from .legacy import EnhancedStockScreener, OriginalStockScreener, StockScreener
from .types import ScreenerCondition, ScreenerCriteria, ScreenerResult, ScreeningReport
from .unified_screener import UnifiedStockScreener, create_screening_report

# 主要クラスのエクスポート
__all__ = [
    # 新しい統合実装
    "UnifiedStockScreener",
    "ScreenerCondition",
    "ScreenerCriteria",
    "ScreenerResult",
    "ScreeningReport",
    "create_screening_report",
    # 後方互換性
    "StockScreener",
    "EnhancedStockScreener",
    "OriginalStockScreener",
]
