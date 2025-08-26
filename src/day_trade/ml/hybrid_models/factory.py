#!/usr/bin/env python3
"""
ハイブリッドモデル作成ファクトリ関数
"""

from typing import Optional

from .config import HybridModelConfig
from .engine import HybridLSTMTransformerEngine


def create_hybrid_model(config: Optional[HybridModelConfig] = None) -> HybridLSTMTransformerEngine:
    """
    ハイブリッドモデル作成ファクトリ関数

    Args:
        config: ハイブリッドモデル設定（Noneの場合はデフォルト設定を使用）

    Returns:
        HybridLSTMTransformerEngine: 初期化済みハイブリッドモデルエンジン
    """
    if config is None:
        config = HybridModelConfig()

    return HybridLSTMTransformerEngine(config)