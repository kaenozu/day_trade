"""
MACD関連のシグナルルール
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .base_rules import SignalRule
from .config import SignalRulesConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


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
            logger.warning(
                f"MACDCrossoverRule: Invalid lookback period {lookback}, using default"
            )
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
            logger.debug(
                f"MACDCrossoverRule: Insufficient data length {current_length} for lookback {lookback}"
            )
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        # Issue #650: MACD計算に必要な最小データ期間を検証
        min_macd_period = 26 + lookback  # MACD計算には26期間が必要
        if (
            len(indicators) < min_macd_period
            or len(macd) < lookback
            or macd.isna().any()
            or signal.isna().any()
        ):
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
            logger.warning(
                f"MACDDeathCrossRule: Invalid lookback period {lookback}, using default"
            )
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
            logger.debug(
                f"MACDDeathCrossRule: Insufficient data length {current_length} for lookback {lookback}"
            )
            return False, 0.0

        macd = indicators["MACD"].iloc[-lookback:]
        signal = indicators["MACD_Signal"].iloc[-lookback:]

        # Issue #650: MACD計算に必要な最小データ期間を検証
        min_macd_period = 26 + lookback  # MACD計算には26期間が必要
        if (
            len(indicators) < min_macd_period
            or len(macd) < lookback
            or macd.isna().any()
            or signal.isna().any()
        ):
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