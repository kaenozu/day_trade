"""
銘柄スクリーニング機能
テクニカル指標に基づいて銘柄をフィルタリングし、投資候補を抽出する
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..data.stock_fetcher import StockFetcher
from .indicators import TechnicalIndicators
from .signals import TradingSignalGenerator
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="stock_screener")


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
    """銘柄スクリーニングクラス"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = TradingSignalGenerator()

        # デフォルトスクリーニング条件
        self.default_criteria = [
            ScreenerCriteria(
                ScreenerCondition.RSI_OVERSOLD,
                threshold=30.0,
                weight=1.0,
                description="RSI過売り（30以下）",
            ),
            ScreenerCriteria(
                ScreenerCondition.GOLDEN_CROSS,
                weight=2.0,
                description="ゴールデンクロス発生",
            ),
            ScreenerCriteria(
                ScreenerCondition.VOLUME_SPIKE,
                threshold=2.0,
                weight=1.5,
                description="出来高急増（平均の2倍以上）",
            ),
            ScreenerCriteria(
                ScreenerCondition.STRONG_MOMENTUM,
                threshold=0.05,
                weight=1.2,
                description="強い上昇モメンタム（5%以上）",
            ),
        ]

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        min_score: float = 0.0,
        max_results: int = 50,
        period: str = "3mo",
    ) -> List[ScreenerResult]:
        """
        銘柄スクリーニングを実行

        Args:
            symbols: 対象銘柄コードリスト
            criteria: スクリーニング基準
            min_score: 最小スコア閾値
            max_results: 最大結果数
            period: データ取得期間

        Returns:
            スクリーニング結果リスト（スコア順）
        """
        if criteria is None:
            criteria = self.default_criteria

        logger.info(f"銘柄スクリーニング開始: {len(symbols)}銘柄, {len(criteria)}条件")

        results = []

        # 並列処理で各銘柄を評価
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self._evaluate_symbol, symbol, criteria, period): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result.score >= min_score:
                        results.append(result)
                        logger.debug(f"銘柄 {symbol}: スコア {result.score:.2f}")
                except Exception as e:
                    logger.error(f"銘柄 {symbol} の評価でエラー: {e}")

        # スコア順でソート
        results.sort(key=lambda x: x.score, reverse=True)

        # 結果数を制限
        if max_results > 0:
            results = results[:max_results]

        logger.info(f"スクリーニング完了: {len(results)}銘柄が条件を満たしました")
        return results

    def _evaluate_symbol(
        self, symbol: str, criteria: List[ScreenerCriteria], period: str
    ) -> Optional[ScreenerResult]:
        """
        個別銘柄の評価

        Args:
            symbol: 銘柄コード
            criteria: スクリーニング基準
            period: データ取得期間

        Returns:
            スクリーニング結果
        """
        try:
            # ヒストリカルデータ取得
            df = self.stock_fetcher.get_historical_data(
                symbol, period=period, interval="1d"
            )

            if df is None or df.empty or len(df) < 30:
                logger.debug(f"銘柄 {symbol}: データ不足")
                return None

            # テクニカル指標計算
            indicators = TechnicalIndicators.calculate_all(df)

            # 各条件を評価
            matched_conditions = []
            total_score = 0.0
            total_weight = 0.0

            for criterion in criteria:
                meets_condition, condition_score = self._evaluate_condition(
                    df, indicators, criterion
                )

                if meets_condition:
                    matched_conditions.append(criterion.condition)
                    total_score += condition_score * criterion.weight

                total_weight += criterion.weight

            # 正規化されたスコア
            final_score = total_score / total_weight if total_weight > 0 else 0.0

            # 最低1つの条件を満たしている場合のみ結果を返す
            if not matched_conditions:
                return None

            # 基本情報を収集
            last_price = float(df["Close"].iloc[-1])
            volume = int(df["Volume"].iloc[-1])

            # テクニカルデータの要約
            technical_data = self._summarize_technical_data(df, indicators)

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

    def _evaluate_condition(
        self, df: pd.DataFrame, indicators: pd.DataFrame, criterion: ScreenerCriteria
    ) -> tuple[bool, float]:
        """
        スクリーニング条件の評価

        Args:
            df: 価格データ
            indicators: テクニカル指標
            criterion: 評価基準

        Returns:
            条件満足フラグ, スコア
        """
        condition = criterion.condition
        threshold = criterion.threshold

        try:
            if condition == ScreenerCondition.RSI_OVERSOLD:
                if "RSI" in indicators.columns:
                    rsi = indicators["RSI"].iloc[-1]
                    if pd.notna(rsi) and rsi <= (threshold or 30):
                        score = (30 - rsi) / 30 * 100  # RSIが低いほど高スコア
                        return True, min(score, 100)

            elif condition == ScreenerCondition.RSI_OVERBOUGHT:
                if "RSI" in indicators.columns:
                    rsi = indicators["RSI"].iloc[-1]
                    if pd.notna(rsi) and rsi >= (threshold or 70):
                        score = (rsi - 70) / 30 * 100  # RSIが高いほど高スコア
                        return True, min(score, 100)

            elif condition == ScreenerCondition.MACD_BULLISH:
                if "MACD" in indicators.columns and "MACD_Signal" in indicators.columns:
                    macd = indicators["MACD"].iloc[-2:]
                    signal = indicators["MACD_Signal"].iloc[-2:]

                    if (
                        len(macd) >= 2
                        and not macd.isna().any()
                        and not signal.isna().any()
                    ):
                        # MACDがシグナルを上抜け
                        if (
                            macd.iloc[-2] <= signal.iloc[-2]
                            and macd.iloc[-1] > signal.iloc[-1]
                        ):
                            crossover_strength = abs(macd.iloc[-1] - signal.iloc[-1])
                            score = min(crossover_strength * 1000, 100)
                            return True, score

            elif condition == ScreenerCondition.GOLDEN_CROSS:
                if "SMA_20" in indicators.columns and "SMA_50" in indicators.columns:
                    sma20 = indicators["SMA_20"].iloc[-2:]
                    sma50 = indicators["SMA_50"].iloc[-2:]

                    if (
                        len(sma20) >= 2
                        and not sma20.isna().any()
                        and not sma50.isna().any()
                    ):
                        # 20日線が50日線を上抜け
                        if (
                            sma20.iloc[-2] <= sma50.iloc[-2]
                            and sma20.iloc[-1] > sma50.iloc[-1]
                        ):
                            cross_strength = (
                                (sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1] * 100
                            )
                            score = min(cross_strength * 50, 100)
                            return True, score

            elif condition == ScreenerCondition.VOLUME_SPIKE:
                volume_threshold = threshold or 2.0
                recent_volume = df["Volume"].iloc[-1]
                avg_volume = df["Volume"].iloc[-20:-1].mean()

                if recent_volume > avg_volume * volume_threshold:
                    volume_ratio = recent_volume / avg_volume
                    score = min((volume_ratio - volume_threshold) * 20 + 50, 100)
                    return True, score

            elif condition == ScreenerCondition.STRONG_MOMENTUM:
                momentum_threshold = threshold or 0.05
                lookback = criterion.lookback_days

                if len(df) >= lookback:
                    current_price = df["Close"].iloc[-1]
                    past_price = df["Close"].iloc[-lookback]
                    momentum = (current_price - past_price) / past_price

                    if momentum >= momentum_threshold:
                        score = min(momentum * 100, 100)
                        return True, score

            elif condition == ScreenerCondition.BOLLINGER_BREAKOUT:
                if (
                    "BB_Upper" in indicators.columns
                    and "BB_Lower" in indicators.columns
                ):
                    current_price = df["Close"].iloc[-1]
                    bb_upper = indicators["BB_Upper"].iloc[-1]
                    bb_lower = indicators["BB_Lower"].iloc[-1]

                    if pd.notna(bb_upper) and pd.notna(bb_lower):
                        # 上限突破
                        if current_price > bb_upper:
                            breakout_strength = (
                                (current_price - bb_upper) / bb_upper * 100
                            )
                            score = min(breakout_strength * 50, 100)
                            return True, score
                        # 下限突破（買いシグナルとして）
                        elif current_price < bb_lower:
                            breakout_strength = (
                                (bb_lower - current_price) / bb_lower * 100
                            )
                            score = min(breakout_strength * 50, 100)
                            return True, score

        except Exception as e:
            logger.debug(f"条件評価エラー ({condition.value}): {e}")

        return False, 0.0

    def _summarize_technical_data(
        self, df: pd.DataFrame, indicators: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        テクニカルデータの要約

        Args:
            df: 価格データ
            indicators: テクニカル指標

        Returns:
            要約データ
        """
        summary = {}

        try:
            # 基本統計
            current_price = float(df["Close"].iloc[-1])
            high_52w = float(df["High"].max())
            low_52w = float(df["Low"].min())

            summary.update(
                {
                    "current_price": current_price,
                    "high_52w": high_52w,
                    "low_52w": low_52w,
                    "price_position": (current_price - low_52w)
                    / (high_52w - low_52w)
                    * 100,
                    "volume_avg_20d": int(df["Volume"].iloc[-20:].mean()),
                    "price_change_1d": (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                    / df["Close"].iloc[-2]
                    * 100,
                    "price_change_5d": (
                        (df["Close"].iloc[-1] - df["Close"].iloc[-6])
                        / df["Close"].iloc[-6]
                        * 100
                        if len(df) >= 6
                        else 0
                    ),
                    "price_change_20d": (
                        (df["Close"].iloc[-1] - df["Close"].iloc[-21])
                        / df["Close"].iloc[-21]
                        * 100
                        if len(df) >= 21
                        else 0
                    ),
                }
            )

            # テクニカル指標
            if "RSI" in indicators.columns:
                rsi = indicators["RSI"].iloc[-1]
                if pd.notna(rsi):
                    summary["rsi"] = float(rsi)

            if "MACD" in indicators.columns:
                macd = indicators["MACD"].iloc[-1]
                if pd.notna(macd):
                    summary["macd"] = float(macd)

            if "SMA_20" in indicators.columns:
                sma20 = indicators["SMA_20"].iloc[-1]
                if pd.notna(sma20):
                    summary["sma_20"] = float(sma20)
                    summary["price_vs_sma20"] = (current_price - sma20) / sma20 * 100

            if "SMA_50" in indicators.columns:
                sma50 = indicators["SMA_50"].iloc[-1]
                if pd.notna(sma50):
                    summary["sma_50"] = float(sma50)
                    summary["price_vs_sma50"] = (current_price - sma50) / sma50 * 100

        except Exception as e:
            logger.debug(f"テクニカルデータ要約エラー: {e}")

        return summary

    def create_custom_screener(
        self, name: str, criteria: List[ScreenerCriteria], description: str = ""
    ) -> Callable:
        """
        カスタムスクリーナーを作成

        Args:
            name: スクリーナー名
            criteria: スクリーニング基準
            description: 説明

        Returns:
            カスタムスクリーナー関数
        """

        def custom_screener(symbols: List[str], **kwargs) -> List[ScreenerResult]:
            return self.screen_stocks(symbols, criteria, **kwargs)

        custom_screener.__name__ = name
        custom_screener.__doc__ = description or f"カスタムスクリーナー: {name}"

        return custom_screener

    def get_predefined_screeners(self) -> Dict[str, Callable]:
        """
        事前定義されたスクリーナーを取得

        Returns:
            スクリーナー辞書
        """
        screeners = {}

        # 成長株スクリーナー
        growth_criteria = [
            ScreenerCriteria(
                ScreenerCondition.STRONG_MOMENTUM,
                threshold=0.1,  # 10%以上の上昇
                weight=2.0,
                description="強い上昇トレンド",
            ),
            ScreenerCriteria(
                ScreenerCondition.VOLUME_SPIKE,
                threshold=1.5,
                weight=1.5,
                description="出来高増加",
            ),
            ScreenerCriteria(
                ScreenerCondition.GOLDEN_CROSS,
                weight=1.8,
                description="ゴールデンクロス",
            ),
        ]
        screeners["growth"] = self.create_custom_screener(
            "growth_screener", growth_criteria, "成長株スクリーナー"
        )

        # バリュー株スクリーナー
        value_criteria = [
            ScreenerCriteria(
                ScreenerCondition.RSI_OVERSOLD,
                threshold=35,
                weight=1.5,
                description="過売り状態",
            ),
            ScreenerCriteria(
                ScreenerCondition.PRICE_NEAR_SUPPORT,
                weight=1.2,
                description="サポートライン付近",
            ),
        ]
        screeners["value"] = self.create_custom_screener(
            "value_screener", value_criteria, "バリュー株スクリーナー"
        )

        # モメンタム株スクリーナー
        momentum_criteria = [
            ScreenerCriteria(
                ScreenerCondition.STRONG_MOMENTUM,
                threshold=0.05,
                weight=2.0,
                description="モメンタム",
            ),
            ScreenerCriteria(
                ScreenerCondition.VOLUME_SPIKE,
                threshold=2.0,
                weight=1.8,
                description="出来高急増",
            ),
            ScreenerCriteria(
                ScreenerCondition.BOLLINGER_BREAKOUT,
                weight=1.5,
                description="ボリンジャーバンド突破",
            ),
        ]
        screeners["momentum"] = self.create_custom_screener(
            "momentum_screener", momentum_criteria, "モメンタム株スクリーナー"
        )

        return screeners


def create_screening_report(results: List[ScreenerResult]) -> str:
    """
    スクリーニング結果のレポート生成

    Args:
        results: スクリーニング結果

    Returns:
        レポート文字列
    """
    if not results:
        return "スクリーニング条件を満たす銘柄が見つかりませんでした。"

    report_lines = [f"=== 銘柄スクリーニング結果 ({len(results)}銘柄) ===", ""]

    for i, result in enumerate(results, 1):
        report_lines.extend(
            [
                f"{i}. 銘柄コード: {result.symbol}",
                f"   スコア: {result.score:.2f}",
                (
                    f"   現在価格: ¥{result.last_price:,.0f}"
                    if result.last_price
                    else "   現在価格: N/A"
                ),
                f"   出来高: {result.volume:,}" if result.volume else "   出来高: N/A",
                f"   マッチした条件: {', '.join([c.value for c in result.matched_conditions])}",
            ]
        )

        # テクニカルデータの一部を表示
        if result.technical_data:
            if "rsi" in result.technical_data:
                report_lines.append(f"   RSI: {result.technical_data['rsi']:.1f}")
            if "price_change_1d" in result.technical_data:
                change = result.technical_data["price_change_1d"]
                report_lines.append(f"   1日変化率: {change:+.2f}%")

        report_lines.append("")

    return "\n".join(report_lines)


if __name__ == "__main__":
    # サンプル実行
    logging.basicConfig(level=logging.INFO)

    screener = StockScreener()

    # サンプル銘柄
    symbols = ["7203", "9984", "8306", "4063", "6758"]

    try:
        results = screener.screen_stocks(symbols, min_score=0.1, max_results=10)

        if results:
            report = create_screening_report(results)
            logger.info("スクリーニング結果レポート",
                       matching_stocks=len(results),
                       report_generated=True,
                       screened_symbols=symbols)
        else:
            logger.info("スクリーニング結果",
                       result="no_matching_stocks",
                       screened_symbols=symbols,
                       symbols_count=len(symbols))

    except Exception as e:
        logger.error(f"スクリーニング実行エラー: {e}")
