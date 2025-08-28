"""
テクニカル指標ルール

RSI、MACD、ボリンジャーバンドなどのテクニカル指標に基づくシグナルルール
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from ...utils.logging_config import get_context_logger
from .base import SignalRule
from .config import SignalRulesConfig, get_shared_config

logger = get_context_logger(__name__)


class RSIOversoldRule(SignalRule):
    """RSI過売りルール"""

    def __init__(
        self,
        threshold: float = 30,
        weight: float = 1.0,
        confidence_multiplier: float = 2.0,
    ):
        super().__init__("RSI Oversold", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #649対応: 最適化されたconfig処理
        config = self._get_config_with_fallback(config)

        threshold = config.get_rsi_thresholds().get("oversold", self.threshold)
        confidence_multiplier = config.get_confidence_multiplier(
            "rsi_oversold", self.confidence_multiplier
        )

        if "RSI" not in indicators.columns or indicators["RSI"].empty:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi < threshold:
            # RSIが低いほど信頼度が高い
            confidence = (threshold - latest_rsi) / threshold * 100
            return True, min(confidence * confidence_multiplier, 100)

        return False, 0.0


class RSIOverboughtRule(SignalRule):
    """RSI過買いルール"""

    def __init__(
        self,
        threshold: float = 70,
        weight: float = 1.0,
        confidence_multiplier: float = 2.0,
    ):
        super().__init__("RSI Overbought", weight)
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #649対応: 最適化されたconfig処理
        config = self._get_config_with_fallback(config)

        threshold = config.get_rsi_thresholds().get("overbought", self.threshold)
        confidence_multiplier = config.get_confidence_multiplier(
            "rsi_overbought", self.confidence_multiplier
        )

        if "RSI" not in indicators.columns or indicators["RSI"].empty:
            return False, 0.0

        latest_rsi = indicators["RSI"].iloc[-1]
        if pd.isna(latest_rsi):
            return False, 0.0

        if latest_rsi > threshold:
            # RSIが高いほど信頼度が高い
            confidence = (latest_rsi - threshold) / (100 - threshold) * 100
            return True, min(confidence * confidence_multiplier, 100)

        return False, 0.0


class MACDCrossoverRule(SignalRule):
    """MACDクロスオーバールール"""

    def __init__(
        self, lookback: int = 2, weight: float = 1.5, angle_multiplier: float = 20.0
    ):
        super().__init__("MACD Crossover", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #649対応: 最適化されたconfig処理
        config = self._get_config_with_fallback(config)

        lookback = config.get_macd_settings().get("lookback_period", self.lookback)
        angle_multiplier = config.get_confidence_multiplier(
            "macd_angle", self.angle_multiplier
        )

        # Issue #650対応: lookback期間妥当性チェック強化
        if lookback < 1:
            logger.warning(f"MACDCrossoverRule: Invalid lookback period {lookback}, using default")
            lookback = self.lookback

        if (
            "MACD" not in indicators.columns
            or "MACD_Signal" not in indicators.columns
            or indicators["MACD"].empty
            or indicators["MACD_Signal"].empty
        ):
            return False, 0.0

        current_length = len(indicators)
        if current_length < lookback:
            logger.debug(f"MACDCrossoverRule: Insufficient data length {current_length} for lookback {lookback}")
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        # Issue #650: MACD計算に必要な最小データ期間を検証
        min_macd_period = 26 + lookback  # MACD計算には26期間が必要
        if len(indicators) < min_macd_period or len(macd) < lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # ゴールデンクロスをチェック
        crossed_above = (macd.iloc[-2] <= signal.iloc[-2]) and (
            macd.iloc[-1] > signal.iloc[-1]
        )

        if crossed_above:
            # クロスの角度から信頼度を計算
            angle = abs(macd.iloc[-1] - signal.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class MACDDeathCrossRule(SignalRule):
    """MACDデスクロスルール"""

    def __init__(
        self, lookback: int = 2, weight: float = 1.5, angle_multiplier: float = 20.0
    ):
        super().__init__("MACD Death Cross", weight)
        self.lookback = lookback
        self.angle_multiplier = angle_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #649対応: 最適化されたconfig処理
        config = self._get_config_with_fallback(config)

        lookback = config.get_macd_settings().get("lookback_period", self.lookback)
        angle_multiplier = config.get_confidence_multiplier(
            "macd_angle", self.angle_multiplier
        )

        # Issue #650対応: lookback期間妥当性チェック強化
        if lookback < 1:
            logger.warning(f"MACDDeathCrossRule: Invalid lookback period {lookback}, using default")
            lookback = self.lookback

        if (
            "MACD" not in indicators.columns
            or "MACD_Signal" not in indicators.columns
            or indicators["MACD"].empty
            or indicators["MACD_Signal"].empty
        ):
            return False, 0.0

        current_length = len(indicators)
        if current_length < lookback:
            logger.debug(f"MACDDeathCrossRule: Insufficient data length {current_length} for lookback {lookback}")
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        # Issue #650: MACD計算に必要な最小データ期間を検証
        min_macd_period = 26 + lookback  # MACD計算には26期間が必要
        if len(indicators) < min_macd_period or len(macd) < lookback or macd.isna().any() or signal.isna().any():
            return False, 0.0

        # デスクロスをチェック
        crossed_below = (macd.iloc[-2] >= signal.iloc[-2]) and (
            macd.iloc[-1] < signal.iloc[-1]
        )

        if crossed_below:
            # クロスの角度から信頼度を計算
            angle = abs(signal.iloc[-1] - macd.iloc[-1]) / df["Close"].iloc[-1] * 100
            confidence = min(angle * angle_multiplier, 100)
            return True, confidence

        return False, 0.0


class BollingerBandRule(SignalRule):
    """ボリンジャーバンドルール"""

    def __init__(
        self,
        position: str = "lower",
        weight: float = 1.0,
        deviation_multiplier: float = 10.0,
    ):
        """
        Args:
            position: "lower" (買いシグナル) or "upper" (売りシグナル)
            deviation_multiplier: 乖離率に対する信頼度乗数
        """
        super().__init__(f"Bollinger Band {position}", weight)
        self.position = position
        self.deviation_multiplier = deviation_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #649対応: 最適化されたconfig処理
        config = self._get_config_with_fallback(config)

        deviation_multiplier = config.get_confidence_multiplier(
            "bollinger_deviation", self.deviation_multiplier
        )

        # Issue #651対応: 入力データの堅牢性チェック強化
        if (
            "BB_Upper" not in indicators.columns
            or "BB_Lower" not in indicators.columns
            or "BB_Middle" not in indicators.columns
            or indicators["BB_Upper"].empty
            or indicators["BB_Lower"].empty
            or indicators["BB_Middle"].empty
        ):
            logger.debug("BollingerBandRule: Required Bollinger Band columns missing or empty")
            return False, 0.0

        if df.empty or "Close" not in df.columns:
            logger.debug("BollingerBandRule: Close price data missing")
            return False, 0.0

        close_price = df["Close"].iloc[-1]

        # NaN値チェック
        if pd.isna(close_price):
            logger.debug("BollingerBandRule: Close price is NaN")
            return False, 0.0

        if self.position == "lower":
            bb_lower = indicators["BB_Lower"].iloc[-1]
            if pd.isna(bb_lower):
                return False, 0.0

            if close_price <= bb_lower:
                # Issue #651: 乖離率計算の堅牢性を改善
                if close_price > 0:  # ゼロ除算防止
                    deviation = (bb_lower - close_price) / close_price * 100
                    # 異常な乖離率を制限
                    deviation = min(deviation, 50.0)  # 最大50%に制限
                    confidence = min(deviation * deviation_multiplier, 100)
                    return True, max(confidence, 1.0)  # 最小信頼度1%
                return True, 1.0  # フォールバック信頼度

        elif self.position == "upper":
            bb_upper = indicators["BB_Upper"].iloc[-1]
            if pd.isna(bb_upper):
                return False, 0.0

            if close_price >= bb_upper:
                # Issue #651: 乖離率計算の堅牢性を改善
                if close_price > 0:  # ゼロ除算防止
                    deviation = (close_price - bb_upper) / close_price * 100
                    # 異常な乖離率を制限
                    deviation = min(deviation, 50.0)  # 最大50%に制限
                    confidence = min(deviation * deviation_multiplier, 100)
                    return True, max(confidence, 1.0)  # 最小信頼度1%
                return True, 1.0  # フォールバック信頼度

        return False, 0.0


class VolumeSpikeBuyRule(SignalRule):
    """出来高急増買いルール"""

    def __init__(
        self,
        volume_factor: Optional[float] = None,
        price_change: Optional[float] = None,
        weight: float = 1.5,
        confidence_base: Optional[float] = None,
        confidence_multiplier: Optional[float] = None,
        config: Optional[SignalRulesConfig] = None,
    ):
        super().__init__("Volume Spike Buy", weight)

        # 設定ファイルからデフォルト値を取得
        if config is None:
            config = get_shared_config()

        volume_settings = config.get_volume_spike_settings()

        self.volume_factor = (
            volume_factor
            if volume_factor is not None
            else volume_settings["volume_factor"]
        )
        self.price_change = (
            price_change
            if price_change is not None
            else volume_settings["price_change_threshold"]
        )
        self.confidence_base = (
            confidence_base
            if confidence_base is not None
            else volume_settings["confidence_base"]
        )
        self.confidence_multiplier = (
            confidence_multiplier
            if confidence_multiplier is not None
            else volume_settings["confidence_multiplier"]
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        if config is None:
            config = get_shared_config()

        volume_settings = config.get_volume_spike_settings()
        volume_factor = volume_settings["volume_factor"]
        price_change_threshold = volume_settings["price_change_threshold"]
        confidence_base = volume_settings["confidence_base"]
        confidence_multiplier = volume_settings["confidence_multiplier"]

        min_data = config.get_min_data_for_generation()
        volume_period = config.get_volume_calculation_period()

        if len(df) < min_data:
            return False, 0.0

        # 平均出来高（設定値に基づく期間）
        avg_volume = df["Volume"].iloc[-volume_period:-1].mean()
        latest_volume = df["Volume"].iloc[-1]

        # 価格変化
        price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[
            -2
        ]

        # 出来高が平均の指定倍数以上かつ価格が指定%以上上昇
        if (
            latest_volume > avg_volume * volume_factor
            and price_change > price_change_threshold
        ):
            volume_ratio = latest_volume / avg_volume
            confidence = min(
                (volume_ratio - volume_factor) * confidence_multiplier
                + confidence_base,
                100,
            )
            return True, confidence

        return False, 0.0