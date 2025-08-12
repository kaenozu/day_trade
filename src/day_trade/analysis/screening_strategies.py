"""
銘柄スクリーニング戦略パターン
各スクリーニング条件に対応した評価ロジックを分離
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ScreeningStrategy(ABC):
    """スクリーニング戦略の基底クラス"""

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        """
        スクリーニング条件を評価

        Args:
            df: 価格データ
            indicators: テクニカル指標
            threshold: 閾値
            lookback_days: 参照期間
            **kwargs: その他のパラメータ

        Returns:
            (条件満足フラグ, スコア) のタプル
        """
        pass

    @property
    @abstractmethod
    def condition_name(self) -> str:
        """条件名を返す"""
        pass


class RSIOversoldStrategy(ScreeningStrategy):
    """RSI過売り戦略"""

    @property
    def condition_name(self) -> str:
        return "RSI_OVERSOLD"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 30.0,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if "RSI" not in indicators.columns:
                return False, 0.0

            rsi = indicators["RSI"].iloc[-1]
            if pd.notna(rsi) and rsi <= threshold:
                score = (threshold - rsi) / threshold * 100  # RSIが低いほど高スコア
                return True, min(score, 100)

        except Exception as e:
            logger.debug(f"RSI過売り評価エラー: {e}")

        return False, 0.0


class RSIOverboughtStrategy(ScreeningStrategy):
    """RSI買われすぎ戦略"""

    @property
    def condition_name(self) -> str:
        return "RSI_OVERBOUGHT"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 70.0,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if "RSI" not in indicators.columns:
                return False, 0.0

            rsi = indicators["RSI"].iloc[-1]
            if pd.notna(rsi) and rsi >= threshold:
                score = (
                    (rsi - threshold) / (100 - threshold) * 100
                )  # RSIが高いほど高スコア
                return True, min(score, 100)

        except Exception as e:
            logger.debug(f"RSI買われすぎ評価エラー: {e}")

        return False, 0.0


class MACDBullishStrategy(ScreeningStrategy):
    """MACD強気戦略"""

    @property
    def condition_name(self) -> str:
        return "MACD_BULLISH"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if (
                "MACD" not in indicators.columns
                or "MACD_Signal" not in indicators.columns
            ):
                return False, 0.0

            macd = indicators["MACD"].iloc[-2:]
            signal = indicators["MACD_Signal"].iloc[-2:]

            if (
                len(macd) >= 2
                and not macd.isna().any()
                and not signal.isna().any()
                and macd.iloc[-2] <= signal.iloc[-2]
                and macd.iloc[-1] > signal.iloc[-1]
            ):
                # MACDがシグナルを上抜け
                crossover_strength = abs(macd.iloc[-1] - signal.iloc[-1])
                score = min(crossover_strength * 1000, 100)
                return True, score

        except Exception as e:
            logger.debug(f"MACD強気評価エラー: {e}")

        return False, 0.0


class MACDBearishStrategy(ScreeningStrategy):
    """MACD弱気戦略"""

    @property
    def condition_name(self) -> str:
        return "MACD_BEARISH"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if (
                "MACD" not in indicators.columns
                or "MACD_Signal" not in indicators.columns
            ):
                return False, 0.0

            macd = indicators["MACD"].iloc[-2:]
            signal = indicators["MACD_Signal"].iloc[-2:]

            if (
                len(macd) >= 2
                and not macd.isna().any()
                and not signal.isna().any()
                and macd.iloc[-2] >= signal.iloc[-2]
                and macd.iloc[-1] < signal.iloc[-1]
            ):
                # MACDがシグナルを下抜け
                crossover_strength = abs(macd.iloc[-1] - signal.iloc[-1])
                score = min(crossover_strength * 1000, 100)
                return True, score

        except Exception as e:
            logger.debug(f"MACD弱気評価エラー: {e}")

        return False, 0.0


class GoldenCrossStrategy(ScreeningStrategy):
    """ゴールデンクロス戦略"""

    @property
    def condition_name(self) -> str:
        return "GOLDEN_CROSS"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if "SMA_20" not in indicators.columns or "SMA_50" not in indicators.columns:
                return False, 0.0

            sma20 = indicators["SMA_20"].iloc[-2:]
            sma50 = indicators["SMA_50"].iloc[-2:]

            if (
                len(sma20) >= 2
                and not sma20.isna().any()
                and not sma50.isna().any()
                and sma20.iloc[-2] <= sma50.iloc[-2]
                and sma20.iloc[-1] > sma50.iloc[-1]
            ):
                # 20日線が50日線を上抜け
                cross_strength = (
                    (sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1] * 100
                )
                score = min(cross_strength * 50, 100)
                return True, score

        except Exception as e:
            logger.debug(f"ゴールデンクロス評価エラー: {e}")

        return False, 0.0


class DeadCrossStrategy(ScreeningStrategy):
    """デッドクロス戦略"""

    @property
    def condition_name(self) -> str:
        return "DEAD_CROSS"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if "SMA_20" not in indicators.columns or "SMA_50" not in indicators.columns:
                return False, 0.0

            sma20 = indicators["SMA_20"].iloc[-2:]
            sma50 = indicators["SMA_50"].iloc[-2:]

            if (
                len(sma20) >= 2
                and not sma20.isna().any()
                and not sma50.isna().any()
                and sma20.iloc[-2] >= sma50.iloc[-2]
                and sma20.iloc[-1] < sma50.iloc[-1]
            ):
                # 20日線が50日線を下抜け
                cross_strength = (
                    (sma50.iloc[-1] - sma20.iloc[-1]) / sma50.iloc[-1] * 100
                )
                score = min(cross_strength * 50, 100)
                return True, score

        except Exception as e:
            logger.debug(f"デッドクロス評価エラー: {e}")

        return False, 0.0


class VolumeSpikeStrategy(ScreeningStrategy):
    """出来高急増戦略"""

    @property
    def condition_name(self) -> str:
        return "VOLUME_SPIKE"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 2.0,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            recent_volume = df["Volume"].iloc[-1]
            avg_volume = df["Volume"].iloc[-lookback_days:-1].mean()

            if recent_volume > avg_volume * threshold:
                volume_ratio = recent_volume / avg_volume
                score = min((volume_ratio - threshold) * 20 + 50, 100)
                return True, score

        except Exception as e:
            logger.debug(f"出来高急増評価エラー: {e}")

        return False, 0.0


class StrongMomentumStrategy(ScreeningStrategy):
    """強いモメンタム戦略"""

    @property
    def condition_name(self) -> str:
        return "STRONG_MOMENTUM"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 0.05,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if len(df) >= lookback_days:
                current_price = df["Close"].iloc[-1]
                past_price = df["Close"].iloc[-lookback_days]
                momentum = (current_price - past_price) / past_price

                if momentum >= threshold:
                    score = min(momentum * 100, 100)
                    return True, score

        except Exception as e:
            logger.debug(f"強いモメンタム評価エラー: {e}")

        return False, 0.0


class BollingerBreakoutStrategy(ScreeningStrategy):
    """ボリンジャーバンド突破戦略"""

    @property
    def condition_name(self) -> str:
        return "BOLLINGER_BREAKOUT"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if (
                "BB_Upper" not in indicators.columns
                or "BB_Lower" not in indicators.columns
            ):
                return False, 0.0

            current_price = df["Close"].iloc[-1]
            bb_upper = indicators["BB_Upper"].iloc[-1]
            bb_lower = indicators["BB_Lower"].iloc[-1]

            if pd.notna(bb_upper) and pd.notna(bb_lower):
                # 上限突破
                if current_price > bb_upper:
                    breakout_strength = (current_price - bb_upper) / bb_upper * 100
                    score = min(breakout_strength * 50, 100)
                    return True, score
                # 下限突破（買いシグナルとして）
                elif current_price < bb_lower:
                    breakout_strength = (bb_lower - current_price) / bb_lower * 100
                    score = min(breakout_strength * 50, 100)
                    return True, score

        except Exception as e:
            logger.debug(f"ボリンジャーバンド突破評価エラー: {e}")

        return False, 0.0


class BollingerSqueezeStrategy(ScreeningStrategy):
    """ボリンジャーバンドスクイーズ戦略"""

    @property
    def condition_name(self) -> str:
        return "BOLLINGER_SQUEEZE"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 0.02,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            if (
                "BB_Upper" not in indicators.columns
                or "BB_Lower" not in indicators.columns
            ):
                return False, 0.0

            bb_upper = indicators["BB_Upper"].iloc[-1]
            bb_lower = indicators["BB_Lower"].iloc[-1]
            current_price = df["Close"].iloc[-1]

            if pd.notna(bb_upper) and pd.notna(bb_lower):
                band_width = (bb_upper - bb_lower) / current_price

                if band_width <= threshold:
                    # バンド幅が狭いほど高スコア
                    score = min((threshold - band_width) / threshold * 100, 100)
                    return True, score

        except Exception as e:
            logger.debug(f"ボリンジャーバンドスクイーズ評価エラー: {e}")

        return False, 0.0


class PriceNearSupportStrategy(ScreeningStrategy):
    """サポートライン付近戦略"""

    @property
    def condition_name(self) -> str:
        return "PRICE_NEAR_SUPPORT"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 0.03,  # 3%以内
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            current_price = df["Close"].iloc[-1]
            # 過去のローを基にサポートレベルを推定
            recent_lows = df["Low"].iloc[-lookback_days:].nsmallest(3)
            support_level = recent_lows.mean()

            distance_to_support = abs(current_price - support_level) / support_level

            if distance_to_support <= threshold:
                # サポートに近いほど高スコア
                score = min((threshold - distance_to_support) / threshold * 100, 100)
                return True, score

        except Exception as e:
            logger.debug(f"サポートライン付近評価エラー: {e}")

        return False, 0.0


class PriceNearResistanceStrategy(ScreeningStrategy):
    """レジスタンスライン付近戦略"""

    @property
    def condition_name(self) -> str:
        return "PRICE_NEAR_RESISTANCE"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = 0.03,  # 3%以内
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            current_price = df["Close"].iloc[-1]
            # 過去のハイを基にレジスタンスレベルを推定
            recent_highs = df["High"].iloc[-lookback_days:].nlargest(3)
            resistance_level = recent_highs.mean()

            distance_to_resistance = (
                abs(current_price - resistance_level) / resistance_level
            )

            if distance_to_resistance <= threshold:
                # レジスタンスに近いほど高スコア
                score = min((threshold - distance_to_resistance) / threshold * 100, 100)
                return True, score

        except Exception as e:
            logger.debug(f"レジスタンスライン付近評価エラー: {e}")

        return False, 0.0


class ReversalPatternStrategy(ScreeningStrategy):
    """反転パターン戦略"""

    @property
    def condition_name(self) -> str:
        return "REVERSAL_PATTERN"

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        threshold: float = None,
        lookback_days: int = 20,
        **kwargs,
    ) -> Tuple[bool, float]:
        try:
            # 簡単な反転パターンの検出
            # RSIの反転 + 価格の反転
            if "RSI" not in indicators.columns:
                return False, 0.0

            rsi_values = indicators["RSI"].iloc[-5:]  # 過去5日
            price_values = df["Close"].iloc[-5:]

            if len(rsi_values) >= 5 and len(price_values) >= 5:
                # RSIが底を打って上昇し、価格も底を打って上昇
                rsi_trend = (
                    rsi_values.iloc[-1] > rsi_values.iloc[-3] > rsi_values.iloc[-5]
                )
                price_low = price_values.min()
                current_price = price_values.iloc[-1]

                if rsi_trend and current_price > price_low * 1.02:  # 2%以上の上昇
                    rsi_strength = rsi_values.iloc[-1] - rsi_values.min()
                    score = min(rsi_strength, 100)
                    return True, score

        except Exception as e:
            logger.debug(f"反転パターン評価エラー: {e}")

        return False, 0.0


class ScreeningStrategyFactory:
    """スクリーニング戦略ファクトリー"""

    def __init__(self):
        self._strategies = {
            "rsi_oversold": RSIOversoldStrategy(),
            "rsi_overbought": RSIOverboughtStrategy(),
            "macd_bullish": MACDBullishStrategy(),
            "macd_bearish": MACDBearishStrategy(),
            "golden_cross": GoldenCrossStrategy(),
            "dead_cross": DeadCrossStrategy(),
            "volume_spike": VolumeSpikeStrategy(),
            "strong_momentum": StrongMomentumStrategy(),
            "bollinger_breakout": BollingerBreakoutStrategy(),
            "bollinger_squeeze": BollingerSqueezeStrategy(),
            "price_near_support": PriceNearSupportStrategy(),
            "price_near_resistance": PriceNearResistanceStrategy(),
            "reversal_pattern": ReversalPatternStrategy(),
        }

    def get_strategy(self, condition_name: str) -> ScreeningStrategy:
        """指定された条件名に対応する戦略を取得"""
        return self._strategies.get(condition_name)

    def get_all_strategies(self) -> Dict[str, ScreeningStrategy]:
        """すべての戦略を取得"""
        return self._strategies.copy()

    def register_strategy(self, condition_name: str, strategy: ScreeningStrategy):
        """カスタム戦略を登録"""
        self._strategies[condition_name] = strategy
