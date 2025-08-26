#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ構造とEnum定義 - ML Prediction Models Data Types

ML予測モデルで使用されるデータ構造と型定義を提供します。
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.day_trade.ml.core_types import (
    DataQuality,
    ModelType,
    PredictionTask,
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
class ModelConfiguration:
    """モデル設定"""
    model_type: ModelType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Optional[TrainingConfig] = None
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    feature_selection_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        if self.training_config:
            result['training_config'] = asdict(self.training_config)
        return result


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    quality_score: float
    quality_level: DataQuality
    message: str
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['quality_level'] = self.quality_level.value
        return result


@dataclass
class PredictionRequest:
    """予測リクエスト"""
    symbol: str
    timestamp: datetime
    features: Dict[str, float]
    model_types: List[ModelType] = field(default_factory=list)
    tasks: List[PredictionTask] = field(default_factory=list)
    require_explanation: bool = False
    require_confidence_interval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['model_types'] = [mt.value for mt in self.model_types]
        result['tasks'] = [t.value for t in self.tasks]
        return result


@dataclass
class ModelStatus:
    """モデル状態"""
    model_id: str
    model_type: ModelType
    task: PredictionTask
    symbol: str
    is_available: bool
    last_updated: datetime
    performance_score: float
    quality_level: DataQuality
    training_samples: int
    version: str
    status_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['task'] = self.task.value
        result['last_updated'] = self.last_updated.isoformat()
        result['quality_level'] = self.quality_level.value
        return result


@dataclass
class EnsembleWeights:
    """アンサンブル重み"""
    symbol: str
    task: PredictionTask
    weights: Dict[ModelType, float]
    last_updated: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    calculation_method: str = "dynamic_weighted"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['task'] = self.task.value
        result['weights'] = {mt.value: w for mt, w in self.weights.items()}
        result['last_updated'] = self.last_updated.isoformat()
        return result

    def normalize_weights(self) -> None:
        """重みを正規化"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {mt: w / total for mt, w in self.weights.items()}


class PredictionMetrics:
    """予測メトリクス計算ユーティリティ"""

    @staticmethod
    def calculate_ensemble_consensus(predictions: Dict[str, Any]) -> float:
        """アンサンブル予測のコンセンサス強度計算"""
        if len(predictions) <= 1:
            return 1.0

        pred_values = list(predictions.values())
        
        if all(isinstance(p, (int, float)) for p in pred_values):
            # 数値予測の場合
            mean_pred = np.mean(pred_values)
            std_pred = np.std(pred_values)
            if mean_pred != 0:
                cv = std_pred / abs(mean_pred)
                return max(0.0, 1.0 - cv)
            else:
                return max(0.0, 1.0 - std_pred)
        else:
            # カテゴリ予測の場合
            unique_count = len(set(str(p) for p in pred_values))
            return 1.0 - (unique_count - 1) / max(1, len(pred_values) - 1)

    @staticmethod
    def calculate_prediction_stability(
        predictions: List[Dict[str, Any]]
    ) -> float:
        """予測安定性計算（時系列での予測一貫性）"""
        if len(predictions) <= 1:
            return 1.0

        stability_scores = []
        for i in range(1, len(predictions)):
            prev_pred = predictions[i-1]
            curr_pred = predictions[i]
            
            if isinstance(prev_pred, dict) and isinstance(curr_pred, dict):
                # 共通キーでの予測値比較
                common_keys = set(prev_pred.keys()) & set(curr_pred.keys())
                if common_keys:
                    key_similarities = []
                    for key in common_keys:
                        if isinstance(prev_pred[key], (int, float)) and \
                           isinstance(curr_pred[key], (int, float)):
                            if prev_pred[key] != 0:
                                rel_diff = abs(curr_pred[key] - prev_pred[key]) / abs(prev_pred[key])
                                similarity = max(0.0, 1.0 - rel_diff)
                            else:
                                similarity = 1.0 if curr_pred[key] == 0 else 0.0
                        else:
                            similarity = 1.0 if str(prev_pred[key]) == str(curr_pred[key]) else 0.0
                        key_similarities.append(similarity)
                    
                    if key_similarities:
                        stability_scores.append(np.mean(key_similarities))

        return np.mean(stability_scores) if stability_scores else 0.5

    @staticmethod
    def calculate_diversity_score(predictions: Dict[str, Any]) -> float:
        """予測多様性スコア計算"""
        if len(predictions) <= 1:
            return 0.0

        pred_values = list(predictions.values())
        
        if all(isinstance(p, (int, float)) for p in pred_values):
            # 数値予測の多様性
            if len(set(pred_values)) == 1:
                return 0.0
            normalized_std = np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
            return min(1.0, normalized_std)
        else:
            # カテゴリ予測の多様性
            unique_count = len(set(str(p) for p in pred_values))
            return (unique_count - 1) / max(1, len(pred_values) - 1)


# 定数定義
DEFAULT_TRAINING_CONFIG = TrainingConfig()

# 品質閾値
QUALITY_THRESHOLDS = {
    DataQuality.EXCELLENT: 0.9,
    DataQuality.GOOD: 0.75,
    DataQuality.FAIR: 0.6,
    DataQuality.POOR: 0.4,
    DataQuality.INSUFFICIENT: 0.0
}

# モデルタイプのデフォルト設定
DEFAULT_MODEL_CONFIGS = {
    ModelType.RANDOM_FOREST: {
        'classifier_params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        },
        'regressor_params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    }
}