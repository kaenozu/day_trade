#!/usr/bin/env python3
"""
深層学習統合システム - 基本型定義
Phase F: 次世代機能拡張フェーズ

基本型、Enum、データクラスの定義
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ModelType(Enum):
    """深層学習モデルタイプ"""

    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    HYBRID = "hybrid"
    HYBRID_LSTM_TRANSFORMER = "hybrid_lstm_transformer"
    ENSEMBLE = "ensemble"


@dataclass
class BaseModelConfig:
    """
    Issue #528対応: 統合された基本モデル設定

    すべての深層学習モデルの共通設定基盤
    """
    # データ設定
    sequence_length: int = 60
    prediction_horizon: int = 1
    validation_split: float = 0.2

    # 学習設定
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10

    # モデル設定
    hidden_size: int = 128  # hidden_dim と統一
    num_layers: int = 2
    dropout_rate: float = 0.2

    # 実行環境設定
    use_pytorch: bool = False
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass
class TransformerConfig(BaseModelConfig):
    """
    Issue #528対応: Transformer特化設定
    """
    # Transformer固有設定
    num_heads: int = 8
    d_model: int = 128
    d_ff: int = 512  # フィードフォワード次元
    max_position_encoding: int = 1000

    # デフォルト値の調整
    hidden_size: int = 128  # d_modelと同期


@dataclass
class LSTMConfig(BaseModelConfig):
    """
    Issue #528対応: LSTM特化設定
    """
    # LSTM固有設定
    bidirectional: bool = False
    layer_norm: bool = True


@dataclass
class CNNConfig(BaseModelConfig):
    """
    Issue #528対応: CNN特化設定
    """
    # CNN固有設定
    kernel_size: int = 3
    num_filters: int = 64
    pool_size: int = 2


# Issue #528対応: 後方互換性のための統合設定クラス
@dataclass
class ModelConfig(BaseModelConfig):
    """
    Issue #528対応: 後方互換性維持のための統合設定クラス

    既存のModelConfigを継承し、必要な追加フィールドを含む
    """
    model_type: ModelType = ModelType.LSTM

    # Transformer用（TransformerConfigから継承）
    num_heads: int = 8
    d_model: int = 128

    # CNN用（CNNConfigから継承）
    kernel_size: int = 3
    num_filters: int = 64

    def to_specialized_config(self) -> Union[TransformerConfig, LSTMConfig, CNNConfig]:
        """
        Issue #528対応: モデルタイプに応じた特化設定への変換

        Returns:
            モデルタイプに適した特化設定インスタンス
        """
        base_kwargs = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'validation_split': self.validation_split,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_pytorch': self.use_pytorch,
        }

        if self.model_type in [ModelType.TRANSFORMER, ModelType.HYBRID_LSTM_TRANSFORMER]:
            return TransformerConfig(
                **base_kwargs,
                num_heads=self.num_heads,
                d_model=self.d_model,
            )
        elif self.model_type in [ModelType.LSTM, ModelType.GRU]:
            return LSTMConfig(**base_kwargs)
        elif self.model_type == ModelType.CNN:
            return CNNConfig(
                **base_kwargs,
                kernel_size=self.kernel_size,
                num_filters=self.num_filters,
            )
        else:
            return self  # デフォルトはBaseModelConfig


# Issue #528対応: 後方互換性のためのエイリアス
DeepLearningConfig = ModelConfig


@dataclass
class TrainingResult:
    """訓練結果"""

    model_type: ModelType
    training_loss: float
    validation_loss: float
    training_accuracy: float
    validation_accuracy: float
    training_time: float
    epochs_trained: int
    model_parameters: int
    convergence_achieved: bool


@dataclass
class ModelTrainingResult:
    """モデル訓練結果（API用）"""

    final_loss: float
    best_loss: float
    epochs_run: int
    training_time: float
    validation_metrics: Dict[str, float]
    convergence_achieved: bool = True


@dataclass
class UncertaintyEstimate:
    """不確実性推定"""

    mean: float
    std: float
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    epistemic: float  # モデル不確実性
    aleatoric: float  # データ不確実性


@dataclass
class PredictionResult:
    """予測結果"""

    predictions: np.ndarray
    confidence: np.ndarray
    uncertainty: Optional[UncertaintyEstimate] = None
    prediction_time: float = 0.0
    model_used: str = ""
    feature_importance: Optional[Dict[str, float]] = None
    metrics: Dict[str, float] = None
    model_weights: Dict[str, float] = None