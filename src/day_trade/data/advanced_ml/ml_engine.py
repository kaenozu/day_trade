#!/usr/bin/env python3
"""
Advanced ML Engine Main Module

メインのML予測エンジンクラス（統合版）
"""

from typing import Any, Dict, Optional

import pandas as pd

from ...core.optimization_strategy import OptimizationConfig
from ...utils.logging_config import get_context_logger
from .config import ModelConfig
from .core_engine import AdvancedMLEngineCore
from .technical_indicators import TechnicalIndicatorCalculator

logger = get_context_logger(__name__)


class AdvancedMLEngine(AdvancedMLEngineCore):
    """Advanced ML Engine - 次世代AI予測システム"""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        # コアエンジン初期化
        super().__init__(config, optimization_config)
        
        # テクニカル指標計算機能追加
        self.technical_calculator = TechnicalIndicatorCalculator()
        
        logger.info("Advanced ML Engine 初期化完了")

    def calculate_advanced_technical_indicators(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, float]:
        """
        高度テクニカル指標計算（ML拡張版）

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            テクニカル指標スコア辞書
        """
        return self.technical_calculator.calculate_advanced_technical_indicators(
            data, symbol
        )


# PyTorchが利用できない場合のフォールバック
try:
    # PyTorchの可用性を確認
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ImportError("PyTorch not available")
except ImportError:
    logger.warning("PyTorch 未インストール - Advanced ML Engine は制限モードで動作")

    class AdvancedMLEngine:
        """フォールバック版 - 基本機能のみ"""

        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch未インストールのため、基本機能のみ利用可能")
            self.model = None
            self.performance_history = []
            self.technical_calculator = TechnicalIndicatorCalculator()

        def prepare_data(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def train_model(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def predict(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def calculate_advanced_technical_indicators(
            self, data: pd.DataFrame, symbol: str = "UNKNOWN"
        ) -> Dict[str, float]:
            """テクニカル指標計算は利用可能"""
            return self.technical_calculator.calculate_advanced_technical_indicators(
                data, symbol
            )

        def get_model_summary(self):
            return {"status": "PyTorch未インストール", "features": "制限モード"}