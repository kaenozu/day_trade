#!/usr / bin / env python3
"""
Issue #619対応: 高度テクニカル指標計算（統合システム使用）

旧AdvancedTechnicalIndicatorsクラスの後方互換性を保ちながら、
統合テクニカル指標システムに移行
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger
from .technical_indicators_consolidated import (
    TechnicalIndicatorsManager,
    IndicatorConfig,
    calculate_technical_indicators,
    calculate_sma,
    calculate_rsi,
    calculate_macd
)

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

class AdvancedTechnicalIndicators:
    """
    Issue #619対応: 高度テクニカル指標計算（統合システム委譲）

    移動平均線、RSI、MACD、ボリンジャーバンドなど、
    様々なテクニカル指標を計算・提供し、市場分析を支援

    注意: このクラスは後方互換性のために保持されています。
    新しいコードでは technical_indicators_consolidated.TechnicalIndicatorsManager を使用してください。
    """

    def __init__(self) -> None:
    """__init__関数"""
        # Issue #619対応: 統合システムに委譲
        self._manager = TechnicalIndicatorsManager(IndicatorConfig())
        logger.info("高度テクニカル指標計算初期化完了 (統合システム使用)")
        logger.warning("AdvancedTechnicalIndicatorsは非推奨です。TechnicalIndicatorsManagerの使用を推奨します")

    def calculate_sma(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """単純移動平均計算（統合システム委譲）"""
        return calculate_sma(data, period)

    def calculate_ema(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """指数移動平均計算（統合システム委譲）"""
        result = self._manager.calculate_indicators(data, ["ema"], period = period)
        return list(result.values())[0][0].values

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """RSI計算（統合システム委譲）"""
        return calculate_rsi(data, period)

    def calculate_macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, np.ndarray]:
        """MACD計算（統合システム委譲）"""
        return calculate_macd(data, fast, slow, signal)

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """ボリンジャーバンド計算（統合システム委譲）"""
        result = self._manager.calculate_indicators(
            data, ["bollinger_bands"],
            period = period, std_dev = std_dev
        )
        return list(result.values())[0][0].values

    def calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, np.ndarray]:
        """ストキャスティクス計算（統合システム委譲）"""
        result = self._manager.calculate_indicators(
            data, ["stochastic"],
            k_period = k_period, d_period = d_period
        )
        return list(result.values())[0][0].values

    def calculate_ichimoku(
        self,
        data: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        leading_span_b_period: int = 52,
        lagging_span_period: int = 26
    ) -> Dict[str, np.ndarray]:
        """一目均衡表計算（統合システム委譲）"""
        result = self._manager.calculate_indicators(
            data, ["ichimoku"],
            conversion_period = conversion_period,
            base_period = base_period,
            leading_span_b_period = leading_span_b_period,
            lagging_span_period = lagging_span_period
        )
        return list(result.values())[0][0].values

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要取得（統合システム委譲）"""
        return self._manager.get_performance_summary()

# Issue #619対応: 直接関数として利用可能にする（後方互換性）

def calculate_advanced_sma(data: pd.DataFrame, period: int = 20) -> np.ndarray:
    """高度SMA計算（統合システム使用）"""
    return calculate_sma(data, period)

def calculate_advanced_rsi(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    """高度RSI計算（統合システム使用）"""
    return calculate_rsi(data, period)

def calculate_advanced_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict[str, np.ndarray]:
    """高度MACD計算（統合システム使用）"""
    return calculate_macd(data, fast, slow, signal)
