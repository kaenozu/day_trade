"""
統合銘柄スクリーニング機能

3つの重複実装を統合し、最高の機能を提供
- screener.py, screener_enhanced.py, screener_original.py の統合
- パフォーマンス最適化とキャッシュ機能
- 設定外部化とストラテジーパターン
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.day_trade.analysis.indicators import TechnicalIndicators
from src.day_trade.analysis.screening_config import (
    ScreeningConfig,
    get_screening_config,
)
from src.day_trade.analysis.screening_strategies import ScreeningStrategyFactory
from src.day_trade.analysis.signals import TradingSignalGenerator
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.utils.formatters import (
    format_currency,
    format_volume,
)
from src.day_trade.utils.logging_config import get_context_logger

from .types import ScreenerCondition, ScreenerCriteria, ScreenerResult, ScreeningReport

logger = get_context_logger(__name__, component="unified_screener")


class UnifiedStockScreener:
    """統合版銘柄スクリーニングクラス"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        config: Optional[ScreeningConfig] = None,
        enable_caching: bool = True,
        parallel_processing: bool = True,
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            config: スクリーニング設定
            enable_caching: キャッシュ機能有効化
            parallel_processing: 並列処理有効化
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = TradingSignalGenerator()
        self.config = config or get_screening_config()
        self.strategy_factory = ScreeningStrategyFactory()
        self.technical_indicators = TechnicalIndicators()

        # パフォーマンス設定
        self.enable_caching = enable_caching
        self.parallel_processing = parallel_processing
        self.max_workers = 4

        # データキャッシュ
        if enable_caching:
            self._data_cache = {}
            self._cache_timestamps = {}
            self._cache_ttl = timedelta(minutes=15)

        logger.info(
            "統合スクリーナーを初期化",
            extra={
                "caching_enabled": enable_caching,
                "parallel_processing": parallel_processing,
            },
        )

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        min_score: float = 0.6,
        max_results: Optional[int] = None,
    ) -> ScreeningReport:
        """
        銘柄スクリーニングを実行

        Args:
            symbols: 対象銘柄リスト
            criteria: スクリーニング基準
            min_score: 最小スコア閾値
            max_results: 最大結果数

        Returns:
            スクリーニングレポート
        """
        start_time = datetime.now()
        logger.info(f"スクリーニング開始: {len(symbols)}銘柄, 基準数: {len(criteria or [])}")

        # デフォルト基準を使用
        if not criteria:
            criteria = self.get_default_criteria()

        # スクリーニング実行
        if self.parallel_processing and len(symbols) > 10:
            results = self._screen_stocks_parallel(symbols, criteria, min_score)
        else:
            results = self._screen_stocks_sequential(symbols, criteria, min_score)

        # 結果のソートとフィルタリング
        results.sort(key=lambda x: x.score, reverse=True)
        if max_results:
            results = results[:max_results]

        # レポート作成
        screening_time = (datetime.now() - start_time).total_seconds()
        report = ScreeningReport(
            total_screened=len(symbols),
            passed_criteria=len(results),
            results=results,
            screening_time=screening_time,
            criteria_used=criteria,
            summary=self._create_summary(results, criteria),
        )

        logger.info(f"スクリーニング完了: {len(results)}銘柄が基準を満たしました")
        return report

    def get_default_criteria(self) -> List[ScreenerCriteria]:
        """デフォルトのスクリーニング基準を取得"""
        return [
            ScreenerCriteria(
                condition=ScreenerCondition.RSI_OVERSOLD,
                threshold=30.0,
                weight=1.2,
                description="RSI売られすぎ",
            ),
            ScreenerCriteria(
                condition=ScreenerCondition.GOLDEN_CROSS,
                lookback_days=10,
                weight=1.5,
                description="ゴールデンクロス",
            ),
            ScreenerCriteria(
                condition=ScreenerCondition.VOLUME_SPIKE,
                threshold=2.0,
                weight=1.0,
                description="出来高急増",
            ),
        ]

    def _screen_stocks_sequential(
        self, symbols: List[str], criteria: List[ScreenerCriteria], min_score: float
    ) -> List[ScreenerResult]:
        """逐次スクリーニング処理"""
        results = []

        for symbol in symbols:
            try:
                result = self._evaluate_symbol(symbol, criteria)
                if result and result.score >= min_score:
                    results.append(result)
            except Exception as e:
                logger.warning(f"銘柄 {symbol} の評価でエラー: {e}")

        return results

    def _screen_stocks_parallel(
        self, symbols: List[str], criteria: List[ScreenerCriteria], min_score: float
    ) -> List[ScreenerResult]:
        """並列スクリーニング処理"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 並列タスクを送信
            future_to_symbol = {
                executor.submit(self._evaluate_symbol, symbol, criteria): symbol
                for symbol in symbols
            }

            # 結果を収集
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result.score >= min_score:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"銘柄 {symbol} の並列評価でエラー: {e}")

        return results

    def _evaluate_symbol(
        self, symbol: str, criteria: List[ScreenerCriteria]
    ) -> Optional[ScreenerResult]:
        """個別銘柄の評価"""
        try:
            # 履歴データの取得（キャッシュ考慮）
            data = self._get_cached_data(symbol)
            if data is None or data.empty:
                return None

            # テクニカル指標の計算
            technical_data = self._calculate_technical_indicators(data)

            # 各基準の評価
            matched_conditions = []
            total_score = 0.0
            total_weight = 0.0

            for criterion in criteria:
                score = self._evaluate_condition(data, technical_data, criterion)
                if score > 0:
                    matched_conditions.append(criterion.condition)
                    total_score += score * criterion.weight
                    total_weight += criterion.weight

            # 最終スコア計算
            final_score = total_score / total_weight if total_weight > 0 else 0.0

            # 現在価格と出来高の取得
            last_price = float(data["Close"].iloc[-1]) if len(data) > 0 else None
            volume = int(data["Volume"].iloc[-1]) if len(data) > 0 else None

            return ScreenerResult(
                symbol=symbol,
                score=final_score,
                matched_conditions=matched_conditions,
                technical_data=technical_data,
                last_price=last_price,
                volume=volume,
            )

        except Exception as e:
            logger.error(f"銘柄 {symbol} の評価中にエラー: {e}")
            return None

    def _get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """キャッシュを考慮したデータ取得"""
        if not self.enable_caching:
            return self._fetch_data(symbol)

        # キャッシュチェック
        if symbol in self._cache_timestamps:
            cache_time = self._cache_timestamps[symbol]
            if datetime.now() - cache_time < self._cache_ttl:
                return self._data_cache.get(symbol)

        # データ取得とキャッシュ
        data = self._fetch_data(symbol)
        if data is not None:
            self._data_cache[symbol] = data
            self._cache_timestamps[symbol] = datetime.now()

        return data

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """データ取得"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # 100日分
            return self.stock_fetcher.get_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.warning(f"銘柄 {symbol} のデータ取得エラー: {e}")
            return None

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """テクニカル指標の計算"""
        indicators = {}

        try:
            # RSI
            indicators["rsi"] = self.technical_indicators.rsi(data, period=14).iloc[-1]

            # 移動平均
            indicators["sma_20"] = self.technical_indicators.sma(data, period=20).iloc[-1]
            indicators["sma_50"] = self.technical_indicators.sma(data, period=50).iloc[-1]

            # MACD
            macd_data = self.technical_indicators.macd(data)
            indicators["macd"] = macd_data["MACD"].iloc[-1]
            indicators["macd_signal"] = macd_data["Signal"].iloc[-1]

            # 出来高移動平均
            indicators["volume_avg"] = data["Volume"].rolling(window=20).mean().iloc[-1]

        except Exception as e:
            logger.warning(f"テクニカル指標計算エラー: {e}")

        return indicators

    def _evaluate_condition(
        self,
        data: pd.DataFrame,
        technical_data: Dict[str, Any],
        criterion: ScreenerCriteria,
    ) -> float:
        """条件評価"""
        try:
            condition = criterion.condition
            threshold = criterion.threshold or 0.0

            if condition == ScreenerCondition.RSI_OVERSOLD:
                rsi = technical_data.get("rsi", 50)
                return 1.0 if rsi < threshold else 0.0

            elif condition == ScreenerCondition.RSI_OVERBOUGHT:
                rsi = technical_data.get("rsi", 50)
                return 1.0 if rsi > threshold else 0.0

            elif condition == ScreenerCondition.GOLDEN_CROSS:
                sma_20 = technical_data.get("sma_20", 0)
                sma_50 = technical_data.get("sma_50", 0)
                return 1.0 if sma_20 > sma_50 else 0.0

            elif condition == ScreenerCondition.DEAD_CROSS:
                sma_20 = technical_data.get("sma_20", 0)
                sma_50 = technical_data.get("sma_50", 0)
                return 1.0 if sma_20 < sma_50 else 0.0

            elif condition == ScreenerCondition.VOLUME_SPIKE:
                current_volume = data["Volume"].iloc[-1]
                avg_volume = technical_data.get("volume_avg", 1)
                ratio = current_volume / avg_volume if avg_volume > 0 else 0
                return 1.0 if ratio > threshold else 0.0

            elif condition == ScreenerCondition.MACD_BULLISH:
                macd = technical_data.get("macd", 0)
                signal = technical_data.get("macd_signal", 0)
                return 1.0 if macd > signal else 0.0

            elif condition == ScreenerCondition.MACD_BEARISH:
                macd = technical_data.get("macd", 0)
                signal = technical_data.get("macd_signal", 0)
                return 1.0 if macd < signal else 0.0

        except Exception as e:
            logger.warning(f"条件評価エラー {criterion.condition}: {e}")

        return 0.0

    def _create_summary(
        self, results: List[ScreenerResult], criteria: List[ScreenerCriteria]
    ) -> Dict[str, Any]:
        """サマリー作成"""
        if not results:
            return {"message": "条件に一致する銘柄がありませんでした"}

        avg_score = sum(r.score for r in results) / len(results)
        max_score = max(r.score for r in results)

        # 最も多くマッチした条件
        condition_counts = {}
        for result in results:
            for condition in result.matched_conditions:
                condition_counts[condition.value] = condition_counts.get(condition.value, 0) + 1

        return {
            "average_score": avg_score,
            "max_score": max_score,
            "top_conditions": sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[
                :3
            ],
            "price_range": (
                {
                    "min": min(r.last_price for r in results if r.last_price),
                    "max": max(r.last_price for r in results if r.last_price),
                }
                if any(r.last_price for r in results)
                else None
            ),
        }

    def generate_report(self, report: ScreeningReport) -> str:
        """レポート生成"""
        lines = []
        lines.append("=" * 60)
        lines.append("        銘柄スクリーニングレポート")
        lines.append("=" * 60)
        lines.append(f"対象銘柄数: {report.total_screened}")
        lines.append(f"条件通過銘柄: {report.passed_criteria}")
        lines.append(f"実行時間: {report.screening_time:.2f}秒")
        lines.append("")

        if report.results:
            lines.append("【上位銘柄】")
            lines.append("-" * 40)
            for i, result in enumerate(report.results[:10], 1):
                price_str = format_currency(result.last_price) if result.last_price else "N/A"
                volume_str = format_volume(result.volume) if result.volume else "N/A"
                lines.append(
                    f"{i:2d}. {result.symbol:>6} | "
                    f"スコア: {result.score:.2f} | "
                    f"価格: {price_str} | "
                    f"出来高: {volume_str}"
                )

        if report.summary.get("top_conditions"):
            lines.append("")
            lines.append("【人気の条件】")
            lines.append("-" * 40)
            for condition, count in report.summary["top_conditions"]:
                lines.append(f"- {condition}: {count}銘柄")

        return "\n".join(lines)


def create_screening_report(results: List[ScreenerResult], criteria: List[ScreenerCriteria]) -> str:
    """
    スクリーニングレポートを作成（後方互換性関数）

    Args:
        results: スクリーニング結果リスト
        criteria: 使用した基準リスト

    Returns:
        レポート文字列
    """
    # 簡易レポート作成
    lines = []
    lines.append("=" * 50)
    lines.append("    スクリーニング結果レポート")
    lines.append("=" * 50)
    lines.append(f"条件に一致した銘柄数: {len(results)}")
    lines.append(f"使用した基準数: {len(criteria)}")
    lines.append("")

    if results:
        lines.append("【上位銘柄】")
        lines.append("-" * 30)
        for i, result in enumerate(results[:5], 1):
            lines.append(f"{i}. {result.symbol} (スコア: {result.score:.2f})")

    return "\n".join(lines)
