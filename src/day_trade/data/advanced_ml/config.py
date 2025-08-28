#!/usr/bin/env python3
"""
Advanced ML Engine Configuration Module

ML エンジンの設定とデータクラスを定義
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelConfig:
    """ML モデル設定"""

    # LSTM設定
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # Transformer設定
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dim_feedforward: int = 2048
    transformer_dropout: float = 0.1

    # 入力・出力設定
    sequence_length: int = 1000
    num_features: int = 50
    prediction_horizon: int = 1

    # 学習設定
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 20

    # GPU設定
    use_gpu: bool = True
    mixed_precision: bool = True


@dataclass
class PredictionResult:
    """予測結果データクラス"""

    predictions: np.ndarray
    confidence: np.ndarray
    probabilities: Optional[np.ndarray]
    model_version: str
    timestamp: float
    features_used: List[str]
    performance_metrics: Dict[str, float]