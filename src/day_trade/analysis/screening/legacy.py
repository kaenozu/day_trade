"""
Screenerクラスの後方互換性サポート

既存の3つのScreenerファイルとの後方互換性を維持
- screener.py
- screener_enhanced.py
- screener_original.py
"""

import warnings
from typing import Any, Dict, List, Optional

from src.day_trade.analysis.screening_config import ScreeningConfig
from src.day_trade.data.stock_fetcher import StockFetcher

from .types import ScreenerCriteria, ScreenerResult
from .unified_screener import UnifiedStockScreener


class StockScreener:
    """
    後方互換性のためのStockScreener（screener.py互換）

    内部では統合版UnifiedStockScreenerを使用
    """

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
        """
        warnings.warn(
            "StockScreener は非推奨です。UnifiedStockScreener を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )

        self._unified = UnifiedStockScreener(stock_fetcher=stock_fetcher)

        # 既存のAPIとの互換性のため
        self.stock_fetcher = stock_fetcher or StockFetcher()

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        min_score: float = 0.6,
        max_results: Optional[int] = None,
    ) -> List[ScreenerResult]:
        """
        スクリーニング実行（互換性API）

        Returns:
            結果リスト（レポートではなく結果のみ）
        """
        report = self._unified.screen_stocks(symbols, criteria, min_score, max_results)
        return report.results

    def get_default_criteria(self) -> List[ScreenerCriteria]:
        """デフォルト基準取得"""
        return self._unified.get_default_criteria()


class EnhancedStockScreener:
    """
    後方互換性のためのEnhancedStockScreener（screener_enhanced.py互換）

    内部では統合版UnifiedStockScreenerを使用
    """

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        config: Optional[ScreeningConfig] = None,
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            config: スクリーニング設定
        """
        warnings.warn(
            "EnhancedStockScreener は非推奨です。UnifiedStockScreener を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )

        self._unified = UnifiedStockScreener(
            stock_fetcher=stock_fetcher,
            config=config,
            enable_caching=True,
            parallel_processing=True,
        )

        # 既存のAPIとの互換性のため
        self.stock_fetcher = stock_fetcher
        self.config = config

    def screen_stocks_enhanced(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        enable_parallel: bool = True,
        min_score: float = 0.6,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        拡張スクリーニング実行（互換性API）

        Returns:
            辞書形式の詳細結果
        """
        report = self._unified.screen_stocks(symbols, criteria, min_score, max_results)

        return {
            "results": report.results,
            "summary": report.summary,
            "screening_time": report.screening_time,
            "total_screened": report.total_screened,
            "passed_criteria": report.passed_criteria,
        }

    def get_default_criteria(self) -> List[ScreenerCriteria]:
        """デフォルト基準取得"""
        return self._unified.get_default_criteria()


# 完全な後方互換性のため、元のクラス名をそのまま提供
OriginalStockScreener = StockScreener  # screener_original.py互換
