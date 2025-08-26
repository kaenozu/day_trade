#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ構造定義モジュール - ML予測システム用データクラス

Issue #850対応: データ構造の分離と管理
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
    DataQuality
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
class ModelPerformance:
    """モデル性能情報"""
    model_id: str
    symbol: str
    task: PredictionTask
    model_type: ModelType
    
    # 分類メトリクス
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # 回帰メトリクス
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    # クロスバリデーション
    cross_val_mean: float = 0.0
    cross_val_std: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)
    
    # その他
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0


@dataclass 
class ModelMetadata:
    """モデルメタデータ（強化版）"""
    model_id: str
    model_type: ModelType
    task: PredictionTask
    symbol: str
    version: str
    created_at: datetime
    updated_at: datetime
    
    # データ情報
    feature_columns: List[str]
    target_info: Dict[str, Any]
    training_samples: int
    training_period: str
    data_quality: DataQuality
    
    # モデル設定
    hyperparameters: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    feature_selection_config: Dict[str, Any]
    
    # 性能情報
    performance_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]
    
    # システム情報
    is_classifier: bool
    model_size_mb: float
    training_time_seconds: float
    python_version: str
    sklearn_version: str
    framework_versions: Dict[str, str]
    
    # ステータス
    validation_status: str = "pending"
    deployment_status: str = "development"
    performance_threshold_met: bool = False
    data_drift_detected: bool = False