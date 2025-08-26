"""
ボリンジャーバンド関連のシグナルルール
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .base_rules import SignalRule
from .config import SignalRulesConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


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