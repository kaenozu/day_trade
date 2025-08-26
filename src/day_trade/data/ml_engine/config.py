#!/usr/bin/env python3
"""
ML Engine Configuration Module
ML エンジン設定モジュール

このモジュールは機械学習エンジンの設定クラスと
データクラスを提供します。

Classes:
    ModelConfig: MLモデル設定
    PredictionResult: 予測結果データクラス
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelConfig:
    """
    ML モデル設定
    
    機械学習モデルの訓練と推論に使用される
    パラメータを定義します。
    
    Attributes:
        lstm_hidden_size: LSTMの隠れ層サイズ
        lstm_num_layers: LSTMの層数
        lstm_dropout: LSTMのドロップアウト率
        transformer_d_model: Transformerのモデル次元
        transformer_nhead: Transformerのアテンションヘッド数
        transformer_num_layers: Transformerの層数
        transformer_dim_feedforward: Transformerのフィードフォワード次元
        transformer_dropout: Transformerのドロップアウト率
        sequence_length: 入力シーケンス長
        num_features: 特徴量数
        prediction_horizon: 予測期間
        batch_size: バッチサイズ
        learning_rate: 学習率
        num_epochs: エポック数
        patience: 早期終了の忍耐度
        use_gpu: GPU使用フラグ
        mixed_precision: 混合精度使用フラグ
    """

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


class PredictionResult:
    """
    予測結果データクラス
    
    モデルの予測結果と関連する情報を格納します。
    
    Attributes:
        predictions: 予測値配列
        confidence: 信頼度配列
        probabilities: 確率配列（オプション）
        model_version: モデルバージョン
        timestamp: タイムスタンプ
        features_used: 使用された特徴量リスト
        performance_metrics: パフォーマンスメトリクス辞書
    """

    def __init__(
        self,
        predictions: np.ndarray,
        confidence: np.ndarray,
        model_version: str,
        probabilities: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        features_used: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ):
        self.predictions = predictions
        self.confidence = confidence
        self.model_version = model_version
        self.probabilities = probabilities
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.features_used = features_used if features_used is not None else []
        self.performance_metrics = performance_metrics if performance_metrics is not None else {}


def create_default_config() -> ModelConfig:
    """
    デフォルト設定を作成
    
    Returns:
        ModelConfig: デフォルト設定
    """
    return ModelConfig()


def create_lightweight_config() -> ModelConfig:
    """
    軽量設定を作成（CI/テスト用）
    
    Returns:
        ModelConfig: 軽量設定
    """
    return ModelConfig(
        lstm_hidden_size=64,
        lstm_num_layers=1,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_feedforward=256,
        sequence_length=100,
        num_features=10,
        batch_size=8,
        num_epochs=10,
        patience=5,
        use_gpu=False,
        mixed_precision=False
    )


def validate_config(config: ModelConfig) -> bool:
    """
    設定の妥当性を検証
    
    Args:
        config: 検証する設定
        
    Returns:
        bool: 妥当性（True: 妥当, False: 不正）
    """
    try:
        # 必須パラメータの検証
        if config.lstm_hidden_size <= 0:
            return False
        if config.lstm_num_layers <= 0:
            return False
        if config.transformer_d_model <= 0:
            return False
        if config.transformer_nhead <= 0:
            return False
        if config.transformer_num_layers <= 0:
            return False
        if config.sequence_length <= 0:
            return False
        if config.num_features <= 0:
            return False
        if config.prediction_horizon <= 0:
            return False
        if config.batch_size <= 0:
            return False
        if config.learning_rate <= 0:
            return False
        if config.num_epochs <= 0:
            return False
        if config.patience <= 0:
            return False
        
        # ドロップアウト率の検証
        if not (0 <= config.lstm_dropout <= 1):
            return False
        if not (0 <= config.transformer_dropout <= 1):
            return False
        
        # Transformerの制約検証
        if config.transformer_d_model % config.transformer_nhead != 0:
            return False
        
        return True
        
    except Exception:
        return False