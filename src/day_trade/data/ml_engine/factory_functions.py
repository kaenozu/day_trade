#!/usr/bin/env python3
"""
Advanced ML Engine Factory Functions
ファクトリ関数群

エンジンインスタンス作成用の便利関数
"""

from typing import Dict, Optional

from .advanced_config import ModelConfig
from .advanced_engine import AdvancedMLEngine
from .next_gen_trading_engine import NextGenAITradingEngine
from ...ml.hybrid_lstm_transformer import HybridModelConfig


def create_advanced_ml_engine(config_dict: Optional[Dict] = None) -> AdvancedMLEngine:
    """Advanced ML Engine インスタンス作成"""
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()

    return AdvancedMLEngine(config)


def create_next_gen_engine(config: Optional[Dict] = None) -> NextGenAITradingEngine:
    """次世代AIエンジンファクトリ関数"""
    if config:
        hybrid_config = HybridModelConfig(**config)
    else:
        hybrid_config = HybridModelConfig()

    return NextGenAITradingEngine(hybrid_config)