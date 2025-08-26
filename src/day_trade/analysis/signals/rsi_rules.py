"""
RSI関連のシグナルルール
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .base_rules import SignalRule
from .config import SignalRulesConfig


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