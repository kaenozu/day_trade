#!/usr/bin/env python3
"""
深層学習統合システム
Phase F: 次世代機能拡張フェーズ

Transformer・LSTM・CNN による高精度予測システム

このパッケージは元々の deep_learning_models.py を機能別に分割したものです。
バックワード互換性のため、元のインポートパスも維持します。
"""

# 基本型のインポート
from .model_types import (
    ModelType,
    BaseModelConfig,
    TransformerConfig,
    LSTMConfig,
    CNNConfig,
    ModelConfig,
    DeepLearningConfig,
    TrainingResult,
    ModelTrainingResult,
    UncertaintyEstimate,
    PredictionResult,
)

# ユーティリティ関数のインポート
from .utils import (
    convert_legacy_config,
    create_model_config,
    calculate_accuracy,
    sigmoid,
    softmax,
)

# データ準備とMixinクラス
from .data_preparation import DataPreparationMixin
from .parallel_processing import ParallelProcessingMixin
from .transformer_components import TransformerBuilderMixin

# モデルクラスのインポート
from .base_model import BaseDeepLearningModel
from .transformer_model import TransformerModel
from .lstm_model import LSTMModel

# モデル管理のインポート（条件的インポート）
try:
    from .model_manager import DeepLearningModelManager
    _MODEL_MANAGER_AVAILABLE = True
except ImportError:
    _MODEL_MANAGER_AVAILABLE = False

# 最適化戦略のインポート（条件的インポート）
try:
    from .optimization_strategy import DeepLearningAcceleratedModels
    _OPTIMIZATION_AVAILABLE = True
except ImportError:
    _OPTIMIZATION_AVAILABLE = False

# バックワード互換性のための統合エクスポート
__all__ = [
    # Types
    "ModelType",
    "BaseModelConfig",
    "TransformerConfig", 
    "LSTMConfig",
    "CNNConfig",
    "ModelConfig",
    "DeepLearningConfig",
    "TrainingResult",
    "ModelTrainingResult",
    "UncertaintyEstimate",
    "PredictionResult",
    
    # Utils
    "convert_legacy_config",
    "create_model_config",
    "calculate_accuracy",
    "sigmoid",
    "softmax",
    
    # Mixins
    "DataPreparationMixin",
    "ParallelProcessingMixin",
    "TransformerBuilderMixin",
    
    # Models
    "BaseDeepLearningModel",
    "TransformerModel",
    "LSTMModel",
]

# 条件的に管理クラスを追加
if _MODEL_MANAGER_AVAILABLE:
    __all__.append("DeepLearningModelManager")

if _OPTIMIZATION_AVAILABLE:
    __all__.append("DeepLearningAcceleratedModels")

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "深層学習統合システム - 機能別分割版"