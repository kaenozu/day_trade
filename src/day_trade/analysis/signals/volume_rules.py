"""
出来高関連のシグナルルール
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .base_rules import SignalRule
from .config import SignalRulesConfig, _get_shared_config


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
            config = _get_shared_config()

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
            config = _get_shared_config()

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