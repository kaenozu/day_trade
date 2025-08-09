#!/usr/bin/env python3
"""
深層学習統合システム
Phase F: 次世代機能拡張フェーズ

Transformer, LSTM, CNNベースの時系列予測システム
"""

from .deep_learning_models import (
    BaseDeepLearningModel,
    TransformerModel,
    LSTMModel,
    DeepLearningModelManager,
    DeepLearningConfig,
    ModelTrainingResult,
    PredictionResult,
    UncertaintyEstimate
)

__all__ = [
    # 基底クラス
    'BaseDeepLearningModel',

    # 具体的なモデル実装
    'TransformerModel',
    'LSTMModel',

    # 管理システム
    'DeepLearningModelManager',

    # 設定・結果クラス
    'DeepLearningConfig',
    'ModelTrainingResult',
    'PredictionResult',
    'UncertaintyEstimate'
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade ML Team"
__description__ = "深層学習統合システム - GPU加速対応時系列予測エンジン"
