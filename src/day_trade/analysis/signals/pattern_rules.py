"""
パターンベースルール

チャートパターンに基づくシグナルルール（ブレイクアウト、ゴールデンクロス、デッドクロス）
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from ...utils.logging_config import get_context_logger
from .base import SignalRule
from .config import SignalRulesConfig

logger = get_context_logger(__name__)


class PatternBreakoutRule(SignalRule):
    """パターンブレイクアウトルール"""

    def __init__(self, direction: str = "upward", weight: float = 2.0):
        """
        Args:
            direction: "upward" or "downward"
        """
        super().__init__(f"{direction.capitalize()} Breakout", weight)
        self.direction = direction

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        # Issue #653: パターンデータ一貫性の強化
        breakouts = patterns.get("breakouts")

        # より厳密なデータ検証
        if breakouts is None:
            return False, 0.0

        if not isinstance(breakouts, pd.DataFrame):
            logger.warning(f"PatternBreakoutRule: breakouts is not DataFrame, got {type(breakouts)}")
            return False, 0.0

        if breakouts.empty:
            return False, 0.0

        if self.direction == "upward":
            if (
                "Upward_Breakout" in breakouts.columns
                and "Upward_Confidence" in breakouts.columns
            ):
                latest_breakout = breakouts["Upward_Breakout"].iloc[-1]
                confidence = breakouts["Upward_Confidence"].iloc[-1]

                if latest_breakout and confidence > 0:
                    return True, confidence

        elif (
            self.direction == "downward"
            and "Downward_Breakout" in breakouts.columns
            and "Downward_Confidence" in breakouts.columns
        ):
            latest_breakout = breakouts["Downward_Breakout"].iloc[-1]
            confidence = breakouts["Downward_Confidence"].iloc[-1]

            if latest_breakout and confidence > 0:
                return True, confidence

        return False, 0.0


class GoldenCrossRule(SignalRule):
    """ゴールデンクロスルール"""

    def __init__(self, weight: float = 2.0):
        super().__init__("Golden Cross", weight)

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        crosses = patterns.get("crosses", pd.DataFrame())

        if not isinstance(crosses, pd.DataFrame) or crosses.empty:
            logger.debug(
                f"GoldenCrossRule: 'crosses' is not a DataFrame or is empty. type: {type(crosses)}"
            )
            return False, 0.0

        if "Golden_Cross" in crosses.columns and "Golden_Confidence" in crosses.columns:
            # Look for the most recent golden cross (within last periods from config)
            if config is None:
                config = SignalRulesConfig()
            recent_signal_lookback = config.get_signal_settings().get(
                "recent_signal_lookback", 5
            )
            # Issue #654: lookback_window計算の堅牢性を改善
            if len(crosses) == 0:
                return False, 0.0
            lookback_window = max(1, min(recent_signal_lookback, len(crosses)))
            recent_crosses = crosses["Golden_Cross"].iloc[-lookback_window:]
            recent_confidences = crosses["Golden_Confidence"].iloc[-lookback_window:]

            for i in range(len(recent_crosses) - 1, -1, -1):
                if recent_crosses.iloc[i] and recent_confidences.iloc[i] > 0:
                    return True, recent_confidences.iloc[i]
        return False, 0.0


class DeadCrossRule(SignalRule):
    """デッドクロスルール"""

    def __init__(self, weight: float = 2.0):
        super().__init__("Dead Cross", weight)

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        crosses = patterns.get("crosses", pd.DataFrame())

        if not isinstance(crosses, pd.DataFrame) or crosses.empty:
            logger.debug(
                f"DeadCrossRule: 'crosses' is not a DataFrame or is empty. type: {type(crosses)}"
            )
            return False, 0.0

        if "Dead_Cross" in crosses.columns and "Dead_Confidence" in crosses.columns:
            # Look for the most recent dead cross (within last periods from config)
            if config is None:
                config = SignalRulesConfig()
            recent_signal_lookback = config.get_signal_settings().get(
                "recent_signal_lookback", 5
            )
            # Issue #654: lookback_window計算の堅牢性を改善
            if len(crosses) == 0:
                return False, 0.0
            lookback_window = max(1, min(recent_signal_lookback, len(crosses)))
            recent_crosses = crosses["Dead_Cross"].iloc[-lookback_window:]
            recent_confidences = crosses["Dead_Confidence"].iloc[-lookback_window:]

            for i in range(len(recent_crosses) - 1, -1, -1):
                if recent_crosses.iloc[i] and recent_confidences.iloc[i] > 0:
                    return True, recent_confidences.iloc[i]
        return False, 0.0