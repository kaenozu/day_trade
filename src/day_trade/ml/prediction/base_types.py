#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本型定義・データクラス・設定クラス

ML予測システムで使用される基本的なデータ型、設定クラス、結果クラスを定義します。
循環依存を避けるため、他のモジュールに依存しない基底層として設計されています。
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np

from src.day_trade.ml.core_types import (
    MLPredictionError,
    DataPreparationError,
    ModelTrainingError,
    ModelMetadataError,
    PredictionError,
    ModelType,
    PredictionTask,
    DataQuality,
    ModelMetadata,
    ModelPerformance,
    DataProvider,
    BaseModelTrainer
)


@dataclass
class TrainingConfig:
    """訓練設定（強化版）"""
    # データ分割
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

    # クロスバリデーション
    cv_folds: int = 5
    enable_cross_validation: bool = True

    # 特徴量選択
    feature_selection: bool = False
    feature_selection_method: str = "SelectKBest"
    max_features: Optional[int] = None

    # モデル保存
    save_model: bool = True
    save_metadata: bool = True

    # ハイパーパラメータ最適化
    use_optimized_params: bool = True
    optimization_method: str = "grid_search"
    optimization_budget: int = 50

    # 前処理
    enable_scaling: bool = True
    handle_missing_values: bool = True
    outlier_detection: bool = False

    # 品質管理
    min_data_quality: DataQuality = DataQuality.FAIR
    performance_threshold: float = 0.6

    # その他
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class PredictionResult:
    """予測結果（強化版）"""
    symbol: str
    timestamp: datetime
    model_type: ModelType
    task: PredictionTask
    model_version: str

    # 予測結果
    prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 詳細情報
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_importance_contribution: Dict[str, float] = field(default_factory=dict)

    # メタ情報
    model_performance_history: Dict[str, float] = field(default_factory=dict)
    data_quality_assessment: Optional[DataQuality] = None
    explanation: str = ""

    # 品質指標
    prediction_stability_score: float = 0.0
    confidence_calibration_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['task'] = self.task.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.data_quality_assessment:
            result['data_quality_assessment'] = self.data_quality_assessment.value
        return result


@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果（強化版）"""
    symbol: str
    timestamp: datetime

    # 最終予測
    final_prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 個別モデル情報
    model_predictions: Dict[str, Any] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_quality_scores: Dict[str, float] = field(default_factory=dict)

    # アンサンブル品質
    consensus_strength: float = 0.0
    disagreement_score: float = 0.0
    prediction_stability: float = 0.0
    diversity_score: float = 0.0

    # メタ情報
    total_models_used: int = 0
    excluded_models: List[str] = field(default_factory=list)
    ensemble_method: str = "weighted_average"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class FeatureEngineringConfig:
    """特徴量エンジニアリング設定"""
    # 基本特徴量
    enable_price_features: bool = True
    enable_volume_features: bool = True
    enable_technical_indicators: bool = True
    
    # 移動平均期間
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    
    # ボラティリティ期間
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # テクニカル指標
    enable_rsi: bool = True
    enable_macd: bool = True
    enable_bollinger_bands: bool = True
    
    # RSI設定
    rsi_period: int = 14
    
    # ボリンジャーバンド設定
    bb_period: int = 20
    bb_std: float = 2.0
    
    # 外れ値処理
    outlier_detection: bool = False
    outlier_method: str = "IQR"  # IQR, zscore
    outlier_threshold: float = 1.5


@dataclass
class DataQualityReport:
    """データ品質レポート"""
    symbol: str
    timestamp: datetime
    total_samples: int
    missing_values_count: int
    missing_values_rate: float
    duplicate_rows: int
    date_gaps_count: int
    price_anomalies: int
    volume_anomalies: int
    ohlc_inconsistencies: int
    quality_score: float
    quality_level: DataQuality
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['quality_level'] = self.quality_level.value
        return result


@dataclass
class ModelTrainingResult:
    """モデル訓練結果"""
    model_id: str
    symbol: str
    model_type: ModelType
    task: PredictionTask
    training_start: datetime
    training_end: datetime
    
    # 訓練データ情報
    training_samples: int
    test_samples: int
    feature_count: int
    
    # 性能メトリクス
    train_score: float
    test_score: float
    cross_val_mean: float
    cross_val_std: float
    
    # 詳細メトリクス（分類）
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # 詳細メトリクス（回帰）
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    # モデル情報
    feature_importance: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0
    
    # 品質情報
    data_quality: DataQuality = DataQuality.FAIR
    training_successful: bool = True
    error_message: str = ""

    @property
    def training_duration(self) -> float:
        """訓練時間（秒）"""
        return (self.training_end - self.training_start).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['task'] = self.task.value
        result['training_start'] = self.training_start.isoformat()
        result['training_end'] = self.training_end.isoformat()
        result['data_quality'] = self.data_quality.value
        result['training_duration'] = self.training_duration
        return result


# 定数定義
DEFAULT_MODEL_CONFIGS = {
    ModelType.RANDOM_FOREST: {
        'classifier_params': {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1, 'class_weight': 'balanced'
        },
        'regressor_params': {
            'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
            'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
            'n_jobs': -1
        }
    },
    ModelType.XGBOOST: {
        'classifier_params': {
            'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'n_jobs': -1, 'eval_metric': 'mlogloss'
        },
        'regressor_params': {
            'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'n_jobs': -1, 'eval_metric': 'rmse'
        }
    },
    ModelType.LIGHTGBM: {
        'classifier_params': {
            'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        },
        'regressor_params': {
            'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
    }
}

# データ品質閾値
DATA_QUALITY_THRESHOLDS = {
    'missing_rate': {
        DataQuality.EXCELLENT: 0.01,
        DataQuality.GOOD: 0.05,
        DataQuality.FAIR: 0.10,
        DataQuality.POOR: 0.20
    },
    'anomaly_rate': {
        DataQuality.EXCELLENT: 0.005,
        DataQuality.GOOD: 0.02,
        DataQuality.FAIR: 0.05,
        DataQuality.POOR: 0.10
    },
    'min_samples': {
        DataQuality.EXCELLENT: 1000,
        DataQuality.GOOD: 500,
        DataQuality.FAIR: 100,
        DataQuality.POOR: 30
    }
}

# 性能閾値
PERFORMANCE_THRESHOLDS = {
    PredictionTask.PRICE_DIRECTION: {
        'excellent': 0.75,
        'good': 0.65,
        'fair': 0.55,
        'poor': 0.45
    },
    PredictionTask.PRICE_REGRESSION: {
        'excellent': 0.7,
        'good': 0.5,
        'fair': 0.3,
        'poor': 0.1
    },
    PredictionTask.VOLATILITY: {
        'excellent': 0.6,
        'good': 0.4,
        'fair': 0.2,
        'poor': 0.0
    },
    PredictionTask.TREND_STRENGTH: {
        'excellent': 0.6,
        'good': 0.4,
        'fair': 0.2,
        'poor': 0.0
    }
}