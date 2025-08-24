#!/usr/bin/env python3
"""
深層学習統合システム - ユーティリティ関数
Phase F: 次世代機能拡張フェーズ

共通のユーティリティ関数とヘルパー
"""

from typing import Any, Dict, Union

from .model_types import ModelConfig, ModelType, TransformerConfig, LSTMConfig, CNNConfig
try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Issue #528対応: 設定変換ヘルパー関数
def convert_legacy_config(legacy_config: Dict[str, Any]) -> ModelConfig:
    """
    Issue #528対応: レガシー設定辞書からModelConfigへの変換

    Args:
        legacy_config: 古い形式の設定辞書

    Returns:
        ModelConfig: 統合された設定オブジェクト
    """
    # hidden_dim -> hidden_size の統一
    if 'hidden_dim' in legacy_config:
        legacy_config['hidden_size'] = legacy_config.pop('hidden_dim')

    # d_model のデフォルト値設定
    if 'hidden_size' in legacy_config and 'd_model' not in legacy_config:
        legacy_config['d_model'] = legacy_config['hidden_size']

    # モデルタイプの処理
    if 'model_type' in legacy_config and isinstance(legacy_config['model_type'], str):
        legacy_config['model_type'] = ModelType(legacy_config['model_type'])

    return ModelConfig(**legacy_config)


def create_model_config(
    model_type: ModelType,
    **kwargs
) -> Union[ModelConfig, TransformerConfig, LSTMConfig, CNNConfig]:
    """
    Issue #528対応: モデルタイプに応じた最適な設定オブジェクト作成

    Args:
        model_type: 作成するモデルのタイプ
        **kwargs: 設定パラメータ

    Returns:
        モデルタイプに最適化された設定オブジェクト
    """
    # 基本設定から開始
    config = ModelConfig(model_type=model_type, **kwargs)

    # 特化設定への変換
    return config.to_specialized_config()


def calculate_accuracy(y_true, y_pred) -> float:
    """
    精度計算（MAPE）
    
    Args:
        y_true: 実際の値
        y_pred: 予測値
        
    Returns:
        float: 精度（0-1範囲）
    """
    import numpy as np
    
    try:
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8)))
            * 100
        )
        accuracy = max(0, 100 - mape)  # MAPEから精度に変換
        return min(accuracy, 100) / 100  # 0-1範囲に正規化
    except Exception:
        return 0.0


def sigmoid(x):
    """
    シグモイド関数
    
    Args:
        x: 入力値
        
    Returns:
        シグモイド変換された値
    """
    import numpy as np
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    """
    Softmax関数
    
    Args:
        x: 入力配列
        
    Returns:
        Softmax変換された配列
    """
    import numpy as np
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)