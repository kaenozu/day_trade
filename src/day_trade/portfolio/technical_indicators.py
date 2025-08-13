#!/usr/bin/env python3
"""
Issue #619対応: ポートフォリオ用テクニカル指標（統合システム使用）

ポートフォリオ分析に特化したテクニカル指標計算
統合テクニカル指標システムに委譲し、重複コードを除去しながら、
100+指標の機能を統合システムで提供
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger
from ..analysis.technical_indicators_consolidated import (
    TechnicalIndicatorsManager,
    IndicatorConfig,
    IndicatorCategory,
    IndicatorResult,
    SignalStrength,
    calculate_technical_indicators
)

logger = get_context_logger(__name__)


# Issue #619対応: 後方互換性のため旧エイリアスを保持
class IndicatorCategory(Enum):
    """指標カテゴリ（後方互換性）"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    CYCLE = "cycle"
    CUSTOM = "custom"


class SignalStrength(Enum):
    """シグナル強度（後方互換性）"""
    VERY_STRONG_BUY = 5
    STRONG_BUY = 4
    BUY = 3
    NEUTRAL = 0
    SELL = -3
    STRONG_SELL = -4
    VERY_STRONG_SELL = -5


@dataclass
class IndicatorConfig:
    """指標設定（後方互換性）"""
    enabled_categories: List[IndicatorCategory] = None
    timeframes: List[str] = None
    lookback_periods: int = 252
    smoothing_factor: float = 0.1
    signal_threshold: float = 0.7
    combine_signals: bool = True
    parallel_computation: bool = True
    cache_results: bool = True

    def __post_init__(self):
        if self.enabled_categories is None:
            self.enabled_categories = list(IndicatorCategory)
        if self.timeframes is None:
            self.timeframes = ["1D"]


@dataclass
class IndicatorResult:
    """指標結果（後方互換性）"""
    name: str
    category: IndicatorCategory
    values: np.ndarray
    signals: np.ndarray
    signal_strength: SignalStrength
    confidence: float
    timeframe: str
    calculation_time: float
    metadata: Dict[str, Any]


class TechnicalIndicatorEngine:
    """Issue #619対応: 100+テクニカル指標分析エンジン（統合システム委譲）"""

    def __init__(self, config: IndicatorConfig = None):
        # Issue #619対応: 統合システムへの委譲
        from ..analysis.technical_indicators_consolidated import IndicatorConfig as ConsolidatedConfig
        
        consolidated_config = ConsolidatedConfig(
            enabled_categories=config.enabled_categories if config else None,
            timeframes=config.timeframes if config else None,
            lookback_periods=config.lookback_periods if config else 252,
            smoothing_factor=config.smoothing_factor if config else 0.1,
            signal_threshold=config.signal_threshold if config else 0.7,
            combine_signals=config.combine_signals if config else True,
            parallel_computation=config.parallel_computation if config else True,
            cache_results=config.cache_results if config else True
        )
        
        self._manager = TechnicalIndicatorsManager(consolidated_config)
        self.config = config or IndicatorConfig()
        
        logger.info("100+テクニカル指標エンジン初期化完了 (統合システム使用)")
        logger.warning("TechnicalIndicatorEngineは統合システムに移行しました")

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        symbols: List[str] = None,
        indicators: List[str] = None,
        timeframe: str = "1D",
    ) -> Dict[str, List[IndicatorResult]]:
        """指標計算実行（統合システム委譲）"""
        
        if indicators is None:
            indicators = self.get_available_indicators()
        
        # 統合システムで計算実行
        results = self._manager.calculate_indicators(
            data=data,
            indicators=indicators,
            symbols=symbols,
            timeframe=timeframe
        )
        
        # Issue #619対応: 結果を旧形式に変換（後方互換性）
        converted_results = {}
        for symbol, symbol_results in results.items():
            converted_results[symbol] = []
            for result in symbol_results:
                # 新しい統合システムの結果を旧形式に変換
                converted_result = IndicatorResult(
                    name=result.name,
                    category=self._convert_category(result.category),
                    values=result.values if isinstance(result.values, np.ndarray) else np.array(list(result.values.values())[0]) if isinstance(result.values, dict) else np.array([result.values]),
                    signals=result.signals,
                    signal_strength=self._convert_signal_strength(result.signal_strength),
                    confidence=result.confidence,
                    timeframe=result.timeframe,
                    calculation_time=result.calculation_time,
                    metadata=result.metadata
                )
                converted_results[symbol].append(converted_result)
        
        return converted_results

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標一覧（統合システム委譲）"""
        return self._manager.get_available_indicators()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要（統合システム委譲）"""
        summary = self._manager.get_performance_summary()
        # Issue #619対応: 統合システム使用情報を追加
        summary["consolidation_status"] = "統合システム使用中"
        summary["legacy_compatibility"] = True
        return summary

    def _convert_category(self, consolidated_category) -> IndicatorCategory:
        """カテゴリ変換（統合システム→旧システム）"""
        category_mapping = {
            "trend": IndicatorCategory.TREND,
            "momentum": IndicatorCategory.MOMENTUM,
            "volatility": IndicatorCategory.VOLATILITY,
            "volume": IndicatorCategory.VOLUME,
            "support_resistance": IndicatorCategory.SUPPORT_RESISTANCE,
            "pattern": IndicatorCategory.PATTERN,
            "cycle": IndicatorCategory.CYCLE
        }
        return category_mapping.get(consolidated_category.value, IndicatorCategory.CUSTOM)

    def _convert_signal_strength(self, consolidated_strength) -> SignalStrength:
        """シグナル強度変換（統合システム→旧システム）"""
        strength_mapping = {
            5: SignalStrength.VERY_STRONG_BUY,
            4: SignalStrength.STRONG_BUY,
            3: SignalStrength.BUY,
            0: SignalStrength.NEUTRAL,
            -3: SignalStrength.SELL,
            -4: SignalStrength.STRONG_SELL,
            -5: SignalStrength.VERY_STRONG_SELL
        }
        return strength_mapping.get(consolidated_strength.value, SignalStrength.NEUTRAL)


# Issue #619対応: 主要な指標計算関数を統合システムに委譲

def calculate_sma(data: pd.DataFrame, period: int = 20) -> np.ndarray:
    """SMA計算（統合システム委譲）"""
    from ..analysis.technical_indicators_consolidated import calculate_sma
    return calculate_sma(data, period)

def calculate_ema(data: pd.DataFrame, period: int = 20) -> np.ndarray:
    """EMA計算（統合システム委譲）"""
    manager = TechnicalIndicatorsManager()
    result = manager.calculate_indicators(data, ["ema"], period=period)
    return list(result.values())[0][0].values

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    """RSI計算（統合システム委譲）"""
    from ..analysis.technical_indicators_consolidated import calculate_rsi
    return calculate_rsi(data, period)

def calculate_macd(
    data: pd.DataFrame, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Dict[str, np.ndarray]:
    """MACD計算（統合システム委譲）"""
    from ..analysis.technical_indicators_consolidated import calculate_macd
    return calculate_macd(data, fast, slow, signal)

def calculate_bollinger_bands(
    data: pd.DataFrame, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Dict[str, np.ndarray]:
    """ボリンジャーバンド計算（統合システム委譲）"""
    manager = TechnicalIndicatorsManager()
    result = manager.calculate_indicators(
        data, ["bollinger_bands"], 
        period=period, std_dev=std_dev
    )
    return list(result.values())[0][0].values

def calculate_stochastic(
    data: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3
) -> Dict[str, np.ndarray]:
    """ストキャスティクス計算（統合システム委譲）"""
    manager = TechnicalIndicatorsManager()
    result = manager.calculate_indicators(
        data, ["stochastic"], 
        k_period=k_period, d_period=d_period
    )
    return list(result.values())[0][0].values

def calculate_ichimoku(
    data: pd.DataFrame,
    conversion_period: int = 9,
    base_period: int = 26,
    leading_span_b_period: int = 52,
    lagging_span_period: int = 26
) -> Dict[str, np.ndarray]:
    """一目均衡表計算（統合システム委譲）"""
    manager = TechnicalIndicatorsManager()
    result = manager.calculate_indicators(
        data, ["ichimoku"],
        conversion_period=conversion_period,
        base_period=base_period,
        leading_span_b_period=leading_span_b_period,
        lagging_span_period=lagging_span_period
    )
    return list(result.values())[0][0].values


# グローバルインスタンス（後方互換性）
_indicator_engine = None


def get_indicator_engine(config: IndicatorConfig = None) -> TechnicalIndicatorEngine:
    """テクニカル指標エンジン取得（統合システム使用）"""
    global _indicator_engine
    if _indicator_engine is None:
        _indicator_engine = TechnicalIndicatorEngine(config)
    return _indicator_engine


# Issue #619対応: 統合システムへの移行を示すメタ情報
CONSOLIDATION_INFO = {
    "status": "統合システム移行完了",
    "legacy_support": True,
    "consolidated_module": "technical_indicators_consolidated",
    "migration_date": "2025-08-13",
    "issue_number": "#619"
}


def get_consolidation_info() -> Dict[str, Any]:
    """統合システム情報取得"""
    return CONSOLIDATION_INFO.copy()