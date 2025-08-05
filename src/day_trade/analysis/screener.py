"""
銘柄スクリーニング機能
テクニカル指標に基づいて銘柄をフィルタリングし、投資候補を抽出する

リファクタリング版：ストラテジーパターンと設定外部化による改善
- 巨大な条件評価メソッドを戦略パターンで分離
- 設定値の外部化
- データ取得の効率化
- 未実装条件の実装
- formattersを活用したレポート生成
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger
from ..utils.formatters import format_currency, format_percentage, format_volume
from .indicators import TechnicalIndicators
from .signals import TradingSignalGenerator
from .screening_strategies import ScreeningStrategyFactory
from .screening_config import get_screening_config, ScreeningConfig
from .screener_enhanced import EnhancedStockScreener

logger = get_context_logger(__name__, component="stock_screener")


# 後方互換性のためのエクスポート
class ScreenerCondition(Enum):
    """スクリーニング条件の種類"""

    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    GOLDEN_CROSS = "golden_cross"
    DEAD_CROSS = "dead_cross"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    VOLUME_SPIKE = "volume_spike"
    PRICE_NEAR_SUPPORT = "price_near_support"
    PRICE_NEAR_RESISTANCE = "price_near_resistance"
    STRONG_MOMENTUM = "strong_momentum"
    REVERSAL_PATTERN = "reversal_pattern"


@dataclass
class ScreenerCriteria:
    """スクリーニング基準"""

    condition: ScreenerCondition
    threshold: Optional[float] = None
    lookback_days: int = 20
    weight: float = 1.0
    description: str = ""


@dataclass
class ScreenerResult:
    """スクリーニング結果"""

    symbol: str
    score: float
    matched_conditions: List[ScreenerCondition]
    technical_data: Dict[str, Any]
    signal_data: Optional[Dict[str, Any]] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None


class StockScreener:
    """
    銘柄スクリーニングクラス（リファクタリング版）

    内部実装を改善しつつ、後方互換性を維持
    """

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
        """
        logger.info("スクリーナーを初期化（リファクタリング版）")

        # 拡張版スクリーナーを内部で使用
        self._enhanced_screener = EnhancedStockScreener(stock_fetcher)

        # 後方互換性のため既存のプロパティも維持
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = TradingSignalGenerator()
        self.config = get_screening_config()

        # デフォルトスクリーニング条件（設定ファイルベース）
        self.default_criteria = self._enhanced_screener.get_default_criteria()

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        min_score: float = 0.0,
        max_results: int = 50,
        period: str = "3mo",
    ) -> List[ScreenerResult]:
        """
        銘柄スクリーニングを実行（リファクタリング版）

        Args:
            symbols: 対象銘柄コードリスト
            criteria: スクリーニング基準
            min_score: 最小スコア閾値
            max_results: 最大結果数
            period: データ取得期間

        Returns:
            スクリーニング結果リスト（スコア順）
        """
        # 拡張版スクリーナーに処理を委譲
        return self._enhanced_screener.screen_stocks(
            symbols=symbols,
            criteria=criteria,
            min_score=min_score,
            max_results=max_results,
            period=period,
            use_cache=True
        )

    def _evaluate_symbol(
        self, symbol: str, criteria: List[ScreenerCriteria], period: str
    ) -> Optional[ScreenerResult]:
        """
        個別銘柄の評価（後方互換性のため残す）

        実際の処理は拡張版スクリーナーに委譲
        """
        return self._enhanced_screener._evaluate_symbol_enhanced(
            symbol, criteria, period, use_cache=True
        )

    def _evaluate_condition(
        self, df: pd.DataFrame, indicators: pd.DataFrame, criterion: ScreenerCriteria
    ) -> tuple[bool, float]:
        """
        スクリーニング条件の評価（後方互換性のため残す）

        実際の処理はストラテジーパターンで実行
        """
        strategy_factory = ScreeningStrategyFactory()
        strategy = strategy_factory.get_strategy(criterion.condition.value)

        if not strategy:
            logger.warning(f"未実装の条件: {criterion.condition.value}")
            return False, 0.0

        return strategy.evaluate(
            df=df,
            indicators=indicators,
            threshold=criterion.threshold,
            lookback_days=criterion.lookback_days
        )

    def _summarize_technical_data(
        self, df: pd.DataFrame, indicators: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        テクニカルデータの要約（後方互換性のため残す）

        実際の処理は拡張版スクリーナーに委譲
        """
        return self._enhanced_screener._summarize_technical_data_enhanced(df, indicators)

    def create_custom_screener(
        self, name: str, criteria: List[ScreenerCriteria], description: str = ""
    ) -> Callable:
        """
        カスタムスクリーナーを作成
        """
        return self._enhanced_screener.create_custom_screener(name, criteria, description)

    def get_predefined_screeners(self) -> Dict[str, Callable]:
        """
        事前定義されたスクリーナーを取得（設定ファイルベース）
        """
        return self._enhanced_screener.get_predefined_screeners()

    def clear_cache(self):
        """キャッシュをクリア"""
        self._enhanced_screener.clear_cache()

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        return self._enhanced_screener.get_cache_info()


def create_screening_report(results: List[ScreenerResult]) -> str:
    """
    スクリーニング結果のレポート生成（改良版）
    formattersを活用して一貫性のある出力を提供

    Args:
        results: スクリーニング結果

    Returns:
        レポート文字列
    """
    if not results:
        return "スクリーニング条件を満たす銘柄が見つかりませんでした。"

    config = get_screening_config()
    use_formatters = config.should_use_formatters()
    currency_precision = config.get_currency_precision()
    percentage_precision = config.get_percentage_precision()
    volume_compact = config.should_use_compact_volume()

    report_lines = [f"=== 銘柄スクリーニング結果 ({len(results)}銘柄) ===", ""]

    for i, result in enumerate(results, 1):
        # 基本情報の表示（formatters活用）
        if use_formatters and result.last_price:
            price_str = format_currency(result.last_price, decimal_places=currency_precision)
        else:
            price_str = f"¥{result.last_price:,.0f}" if result.last_price else "N/A"

        if use_formatters and volume_compact and result.volume:
            volume_str = format_volume(result.volume)
        else:
            volume_str = f"{result.volume:,}" if result.volume else "N/A"

        report_lines.extend(
            [
                f"{i}. 銘柄コード: {result.symbol}",
                f"   スコア: {result.score:.2f}",
                f"   現在価格: {price_str}",
                f"   出来高: {volume_str}",
                f"   マッチした条件: {', '.join([c.value for c in result.matched_conditions])}",
            ]
        )

        # テクニカルデータの一部を表示（formatters活用）
        if result.technical_data:
            if "rsi" in result.technical_data:
                report_lines.append(f"   RSI: {result.technical_data['rsi']:.1f}")

            if "price_change_1d" in result.technical_data:
                change = result.technical_data["price_change_1d"]
                if use_formatters:
                    change_str = format_percentage(change, decimal_places=percentage_precision)
                else:
                    change_str = f"{change:+.2f}%"
                report_lines.append(f"   1日変化率: {change_str}")

            # 52週レンジ位置（新機能）
            if "price_position" in result.technical_data:
                position = result.technical_data["price_position"]
                report_lines.append(f"   52週レンジ位置: {position:.1f}%")

        report_lines.append("")

    return "\n".join(report_lines)


# 使用例とテスト用のコード（改良版）
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # 拡張版スクリーナーの作成
    screener = StockScreener()

    # サンプル銘柄
    symbols = ["7203", "9984", "8306", "4063", "6758"]

    try:
        logger.info("拡張版スクリーニング実行開始")

        # 基本スクリーニング
        results = screener.screen_stocks(symbols, min_score=0.1, max_results=10)

        if results:
            report = create_screening_report(results)
            print(report)

            logger.info(
                "スクリーニング結果レポート生成完了",
                matching_stocks=len(results),
                report_generated=True,
                screened_symbols=symbols,
            )

            # キャッシュ情報の表示
            cache_info = screener.get_cache_info()
            logger.info(f"キャッシュ情報: {cache_info}")

        else:
            logger.info(
                "スクリーニング結果",
                result="no_matching_stocks",
                screened_symbols=symbols,
                symbols_count=len(symbols),
            )

        # 事前定義スクリーナーのテスト
        predefined_screeners = screener.get_predefined_screeners()
        logger.info(f"事前定義スクリーナー: {list(predefined_screeners.keys())}")

        # 成長株スクリーナーのテスト（設定ファイルで定義されている場合）
        if "growth" in predefined_screeners:
            growth_results = predefined_screeners["growth"](symbols, min_score=0.5, max_results=5)
            logger.info(f"成長株スクリーナー結果: {len(growth_results)}銘柄")

    except Exception as e:
        logger.error(f"スクリーニング実行エラー: {e}", exc_info=True)
