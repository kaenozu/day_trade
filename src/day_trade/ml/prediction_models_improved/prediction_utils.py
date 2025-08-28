#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測ユーティリティ - ML Prediction Models Prediction Utils

アンサンブル予測のユーティリティ関数を提供します。
"""

import logging
from collections import Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.day_trade.ml.core_types import (
    DataQuality,
    ModelMetadata,
)


class PredictionUtils:
    """予測ユーティリティクラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_model_quality_score(
        self, 
        model_key: str, 
        metadata: Optional[ModelMetadata]
    ) -> float:
        """モデル品質スコア計算（強化版）"""
        try:
            if metadata:
                # メタデータから品質スコア計算
                performance_metrics = metadata.performance_metrics

                # 複数メトリクスの重み付き平均
                if metadata.is_classifier:
                    score = (0.4 * performance_metrics.get('accuracy', 0.5) +
                            0.3 * performance_metrics.get('f1_score', 0.5) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5))
                else:
                    score = (0.5 * max(0, performance_metrics.get('r2_score', 0.0)) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5) +
                            0.2 * (1.0 - min(1.0, performance_metrics.get('rmse', 1.0))))

                # データ品質による調整
                quality_multiplier = {
                    DataQuality.EXCELLENT: 1.0,
                    DataQuality.GOOD: 0.9,
                    DataQuality.FAIR: 0.8,
                    DataQuality.POOR: 0.6
                }.get(metadata.data_quality, 0.5)

                return np.clip(score * quality_multiplier, 0.0, 1.0)
            else:
                return 0.6  # デフォルト値

        except Exception as e:
            self.logger.error(f"品質スコア計算エラー: {e}")
            return 0.5

    def calculate_classification_confidence(
        self, 
        pred_proba: np.ndarray,
        quality_score: float,
        metadata: Optional[ModelMetadata]
    ) -> float:
        """強化された分類信頼度計算"""
        # 確率の最大値
        max_prob = np.max(pred_proba)

        # エントロピーベースの不確実性
        entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-15))
        normalized_entropy = entropy / np.log(len(pred_proba))
        certainty = 1.0 - normalized_entropy

        # メタデータベースの調整
        metadata_adjustment = 1.0
        if metadata:
            cv_std = metadata.performance_metrics.get('cross_val_std', 0.1)
            metadata_adjustment = max(0.5, 1.0 - cv_std)  # CV標準偏差が低いほど信頼度高

        # 最終信頼度計算
        confidence = (0.4 * max_prob + 0.3 * certainty + 0.2 * quality_score + 0.1 * metadata_adjustment)

        return np.clip(confidence, 0.1, 0.95)

    def calculate_regression_confidence(
        self, 
        model, 
        features: pd.DataFrame,
        prediction: float, 
        quality_score: float,
        metadata: Optional[ModelMetadata]
    ) -> float:
        """強化された回帰信頼度計算"""
        try:
            # アンサンブルモデルの場合、個別予測のばらつきから信頼度推定
            if hasattr(model, 'estimators_'):
                individual_predictions = []
                for estimator in model.estimators_[:min(10, len(model.estimators_))]:
                    try:
                        pred = estimator.predict(features)[0]
                        individual_predictions.append(pred)
                    except:
                        continue

                if individual_predictions:
                    pred_std = np.std(individual_predictions)
                    pred_mean = np.mean(individual_predictions)
                    if pred_mean != 0:
                        cv = abs(pred_std / pred_mean)
                        prediction_consistency = max(0.1, 1.0 - cv)
                    else:
                        prediction_consistency = 0.5
                else:
                    prediction_consistency = 0.5

            else:
                prediction_consistency = 0.7  # デフォルト値

            # メタデータベースの調整
            metadata_adjustment = 1.0
            if metadata:
                r2_score = metadata.performance_metrics.get('r2_score', 0.5)
                metadata_adjustment = max(0.3, r2_score)

            # 最終信頼度計算
            confidence = (0.4 * quality_score + 0.3 * prediction_consistency + 0.3 * metadata_adjustment)

            return np.clip(confidence, 0.1, 0.95)

        except Exception as e:
            self.logger.error(f"回帰信頼度計算エラー: {e}")
            return quality_score * 0.8

    def calculate_dynamic_weights(
        self, 
        base_weights: Dict,
        quality_scores: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """強化された動的重み計算"""
        try:
            dynamic_weights = {}
            total_score = 0

            for model_name in quality_scores.keys():
                # 各要素の重み
                quality = quality_scores[model_name]
                confidence = confidences[model_name]
                base_weight = base_weights.get(model_name, 1.0)

                # 時間減衰ファクター（新しいモデルほど重視）
                time_decay = 1.0  # 実装可能: モデルの新しさに基づく重み

                # 動的重み計算
                dynamic_weight = (0.3 * quality + 0.3 * confidence + 0.2 * base_weight + 0.2 * time_decay)
                dynamic_weights[model_name] = dynamic_weight
                total_score += dynamic_weight

            # 正規化
            if total_score > 0:
                for model_name in dynamic_weights:
                    dynamic_weights[model_name] /= total_score

            return dynamic_weights

        except Exception as e:
            self.logger.error(f"動的重み計算エラー: {e}")
            # フォールバック: 均等重み
            num_models = len(quality_scores)
            return {name: 1.0/num_models for name in quality_scores.keys()}

    def ensemble_classification(
        self, 
        predictions: Dict[str, any],
        confidences: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, any]:
        """強化された分類アンサンブル"""
        weighted_votes = {}
        total_weight = 0
        prediction_values = list(predictions.values())

        # 重み付き投票
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            vote_strength = weight * confidence
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += vote_strength
            total_weight += vote_strength

        # 最終予測
        final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]

        # アンサンブル信頼度
        if total_weight > 0:
            ensemble_confidence = weighted_votes[final_prediction] / total_weight
        else:
            ensemble_confidence = 0.5

        # コンセンサス強度
        unique_predictions = len(set(prediction_values))
        consensus_strength = 1.0 - (unique_predictions - 1) / max(1, len(predictions) - 1)

        # 予測安定性
        prediction_stability = self.calculate_prediction_stability_classification(predictions)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': prediction_stability
        }

    def ensemble_regression(
        self, 
        predictions: Dict[str, float],
        confidences: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, any]:
        """強化された回帰アンサンブル"""
        weighted_sum = 0
        total_weight = 0
        pred_values = list(predictions.values())

        # 重み付き平均
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            adjusted_weight = weight * confidence
            weighted_sum += prediction * adjusted_weight
            total_weight += adjusted_weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(pred_values)

        # 予測区間推定
        prediction_std = np.std(pred_values)
        prediction_interval = (final_prediction - 1.96 * prediction_std,
                             final_prediction + 1.96 * prediction_std)

        # アンサンブル信頼度
        avg_confidence = np.mean(list(confidences.values()))
        prediction_variance = np.var(pred_values)
        prediction_mean = np.mean(pred_values)

        if prediction_mean != 0:
            cv = prediction_variance / abs(prediction_mean)
            consistency_factor = max(0.1, 1.0 - np.tanh(cv))
        else:
            consistency_factor = 0.5

        ensemble_confidence = avg_confidence * consistency_factor

        # コンセンサス強度
        if prediction_mean != 0:
            consensus_strength = max(0.0, 1.0 - (prediction_std / abs(prediction_mean)))
        else:
            consensus_strength = max(0.0, 1.0 - prediction_std)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'prediction_interval': prediction_interval,
            'consensus_strength': np.clip(consensus_strength, 0.0, 1.0),
            'disagreement_score': 1.0 - np.clip(consensus_strength, 0.0, 1.0),
            'prediction_stability': self.calculate_prediction_stability_regression(predictions)
        }

    def calculate_diversity_score(self, predictions: Dict[str, any]) -> float:
        """予測多様性スコア計算"""
        try:
            pred_values = list(predictions.values())
            if len(set(str(p) for p in pred_values)) == 1:
                return 0.0  # 完全一致

            if all(isinstance(p, (int, float)) for p in pred_values):
                # 数値予測の多様性
                normalized_std = np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
                return min(1.0, normalized_std)
            else:
                # カテゴリ予測の多様性
                unique_count = len(set(str(p) for p in pred_values))
                return (unique_count - 1) / max(1, len(pred_values) - 1)

        except Exception:
            return 0.5

    def calculate_prediction_stability_classification(self, predictions: Dict[str, any]) -> float:
        """分類予測安定性計算"""
        pred_values = list(predictions.values())
        if len(set(str(p) for p in pred_values)) == 1:
            return 1.0  # 完全一致

        # 最も多い予測の割合
        counts = Counter(str(p) for p in pred_values)
        most_common_count = counts.most_common(1)[0][1]
        stability = most_common_count / len(pred_values)

        return stability

    def calculate_prediction_stability_regression(self, predictions: Dict[str, float]) -> float:
        """回帰予測安定性計算"""
        pred_values = list(predictions.values())
        if len(pred_values) <= 1:
            return 1.0

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)

        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            stability = max(0.0, 1.0 - cv)
        else:
            stability = max(0.0, 1.0 - std_pred)

        return np.clip(stability, 0.0, 1.0)

    def validate_prediction_inputs(
        self, 
        symbol: str, 
        features: pd.DataFrame
    ) -> tuple[bool, str]:
        """予測入力データ検証"""
        try:
            if not symbol:
                return False, "シンボルが空です"

            if features.empty:
                return False, "特徴量データが空です"

            if features.isnull().all().any():
                return False, "全て欠損値の特徴量があります"

            # 無限値チェック
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                return False, f"無限値が含まれています: {inf_count}個"

            return True, "検証成功"

        except Exception as e:
            return False, f"検証エラー: {e}"