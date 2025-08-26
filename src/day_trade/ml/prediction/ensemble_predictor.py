#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アンサンブル予測システム

複数の機械学習モデルの予測を統合し、信頼性の高いアンサンブル予測を生成するシステムです。
動的重み付け、品質評価、予測安定性評価などの高度な機能を提供します。
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base_types import EnsemblePrediction
from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
    DataQuality,
    ModelMetadata
)

# 循環依存回避のための型チェック時のみインポート
if TYPE_CHECKING:
    from .ml_models import MLPredictionModels


class EnhancedEnsemblePredictor:
    """強化されたアンサンブル予測システム"""

    def __init__(self, ml_models: 'MLPredictionModels'):
        self.ml_models = ml_models
        self.logger = logging.getLogger(__name__)

    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（強化版）"""
        predictions = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            try:
                ensemble_pred = await self._make_enhanced_ensemble_prediction(symbol, features, task)
                if ensemble_pred:
                    predictions[task] = ensemble_pred
            except Exception as e:
                self.logger.error(f"予測失敗 {task.value}: {e}")

        return predictions

    async def _make_enhanced_ensemble_prediction(self, symbol: str, features: pd.DataFrame,
                                               task: PredictionTask) -> Optional[EnsemblePrediction]:
        """強化されたアンサンブル予測"""
        model_predictions = {}
        model_confidences = {}
        model_quality_scores = {}
        excluded_models = []

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{model_type.value}_{task.value}"

            if model_key not in self.ml_models.trained_models:
                excluded_models.append(f"{model_type.value} (not trained)")
                continue

            try:
                model = self.ml_models.trained_models[model_key]
                metadata = self.ml_models.model_metadata.get(model_key)

                # モデル品質スコア
                quality_score = await self._get_model_quality_score(model_key, metadata)
                model_quality_scores[model_type.value] = quality_score

                # 品質が低すぎる場合は除外
                if quality_score < 0.3:
                    excluded_models.append(f"{model_type.value} (low quality: {quality_score:.2f})")
                    continue

                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類予測
                    pred_proba = model.predict_proba(features)
                    pred_class = model.predict(features)[0]

                    # ラベルエンコーダーで逆変換
                    if model_key in self.ml_models.label_encoders:
                        le = self.ml_models.label_encoders[model_key]
                        pred_class = le.inverse_transform([pred_class])[0]

                    confidence = self._calculate_enhanced_classification_confidence(
                        pred_proba[0], quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # 回帰予測
                    pred_value = model.predict(features)[0]
                    confidence = self._calculate_enhanced_regression_confidence(
                        model, features, pred_value, quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = confidence

            except Exception as e:
                self.logger.error(f"予測失敗 {model_key}: {e}")
                excluded_models.append(f"{model_type.value} (error: {str(e)[:50]})")

        if not model_predictions:
            return None

        # 動的重み計算
        dynamic_weights = await self._calculate_enhanced_dynamic_weights(
            symbol, task, model_quality_scores, model_confidences
        )

        # アンサンブル統合
        if task == PredictionTask.PRICE_DIRECTION:
            ensemble_result = self._enhanced_ensemble_classification(
                model_predictions, model_confidences, dynamic_weights
            )
        else:
            ensemble_result = self._enhanced_ensemble_regression(
                model_predictions, model_confidences, dynamic_weights
            )

        # 品質メトリクス計算
        diversity_score = self._calculate_diversity_score(model_predictions)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            prediction_interval=ensemble_result.get('prediction_interval'),
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_weights={k: v for k, v in dynamic_weights.items()},
            model_quality_scores=model_quality_scores,
            consensus_strength=ensemble_result['consensus_strength'],
            disagreement_score=ensemble_result['disagreement_score'],
            prediction_stability=ensemble_result.get('prediction_stability', 0.0),
            diversity_score=diversity_score,
            total_models_used=len(model_predictions),
            excluded_models=excluded_models,
            ensemble_method="enhanced_weighted_ensemble"
        )

    async def _get_model_quality_score(self, model_key: str, metadata: Optional[ModelMetadata]) -> float:
        """モデル品質スコア取得（強化版）"""
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

    def _calculate_enhanced_classification_confidence(self, pred_proba: np.ndarray,
                                                    quality_score: float,
                                                    metadata: Optional[ModelMetadata]) -> float:
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

    def _calculate_enhanced_regression_confidence(self, model, features: pd.DataFrame,
                                                prediction: float, quality_score: float,
                                                metadata: Optional[ModelMetadata]) -> float:
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

    async def _calculate_enhanced_dynamic_weights(self, symbol: str, task: PredictionTask,
                                                quality_scores: Dict[str, float],
                                                confidences: Dict[str, float]) -> Dict[str, float]:
        """強化された動的重み計算"""
        try:
            # 基本重み（過去の性能ベース）
            base_weights = self.ml_models.ensemble_weights.get(symbol, {}).get(task, {})

            dynamic_weights = {}
            total_score = 0

            for model_name in quality_scores.keys():
                try:
                    model_type = ModelType(model_name)
                except ValueError:
                    continue

                # 各要素の重み
                quality = quality_scores[model_name]
                confidence = confidences[model_name]
                base_weight = base_weights.get(model_type, 1.0)

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

    def _enhanced_ensemble_classification(self, predictions: Dict[str, Any],
                                        confidences: Dict[str, float],
                                        weights: Dict[str, float]) -> Dict[str, Any]:
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
        prediction_stability = self._calculate_prediction_stability_classification(predictions)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': prediction_stability
        }

    def _enhanced_ensemble_regression(self, predictions: Dict[str, float],
                                    confidences: Dict[str, float],
                                    weights: Dict[str, float]) -> Dict[str, Any]:
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
            'prediction_stability': self._calculate_prediction_stability_regression(predictions)
        }

    def _calculate_diversity_score(self, predictions: Dict[str, Any]) -> float:
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

    def _calculate_prediction_stability_classification(self, predictions: Dict[str, Any]) -> float:
        """分類予測安定性計算"""
        pred_values = list(predictions.values())
        if len(set(str(p) for p in pred_values)) == 1:
            return 1.0  # 完全一致

        # 最も多い予測の割合
        counts = Counter(str(p) for p in pred_values)
        most_common_count = counts.most_common(1)[0][1]
        stability = most_common_count / len(pred_values)

        return stability

    def _calculate_prediction_stability_regression(self, predictions: Dict[str, float]) -> float:
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

    def get_ensemble_statistics(self, symbol: str) -> Dict[str, Any]:
        """アンサンブル統計情報取得"""
        try:
            stats = {
                'symbol': symbol,
                'available_models': {},
                'ensemble_weights': {},
                'recent_predictions': []
            }

            # 利用可能なモデルの確認
            for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                task_models = []
                for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                    model_key = f"{symbol}_{model_type.value}_{task.value}"
                    if model_key in self.ml_models.trained_models:
                        metadata = self.ml_models.model_metadata.get(model_key)
                        task_models.append({
                            'type': model_type.value,
                            'version': metadata.version if metadata else 'unknown',
                            'performance': metadata.performance_metrics if metadata else {}
                        })

                stats['available_models'][task.value] = task_models

            # アンサンブル重み
            if symbol in self.ml_models.ensemble_weights:
                for task, weights in self.ml_models.ensemble_weights[symbol].items():
                    stats['ensemble_weights'][task.value] = {
                        k.value if hasattr(k, 'value') else str(k): v
                        for k, v in weights.items()
                    }

            return stats

        except Exception as e:
            self.logger.error(f"アンサンブル統計取得エラー: {e}")
            return {'error': str(e)}

    def validate_prediction_quality(self, prediction: EnsemblePrediction) -> Dict[str, Any]:
        """予測品質検証"""
        validation_result = {
            'is_valid': True,
            'quality_level': 'good',
            'issues': [],
            'recommendations': []
        }

        try:
            # 基本的な検証
            if prediction.total_models_used < 2:
                validation_result['issues'].append("使用モデル数が少なすぎます")
                validation_result['quality_level'] = 'poor'

            if prediction.confidence < 0.3:
                validation_result['issues'].append("信頼度が低すぎます")
                validation_result['quality_level'] = 'poor'

            if prediction.consensus_strength < 0.5:
                validation_result['issues'].append("モデル間の合意が弱いです")
                validation_result['recommendations'].append("予測の解釈に注意してください")

            if prediction.diversity_score < 0.1:
                validation_result['issues'].append("モデルの多様性が不足しています")
                validation_result['recommendations'].append("追加のモデル種類を検討してください")

            # 品質レベルの最終判定
            if len(validation_result['issues']) == 0:
                validation_result['quality_level'] = 'excellent'
            elif len(validation_result['issues']) <= 1:
                validation_result['quality_level'] = 'good'
            elif len(validation_result['issues']) <= 2:
                validation_result['quality_level'] = 'fair'
            else:
                validation_result['quality_level'] = 'poor'
                validation_result['is_valid'] = False

            return validation_result

        except Exception as e:
            self.logger.error(f"予測品質検証エラー: {e}")
            return {
                'is_valid': False,
                'quality_level': 'unknown',
                'issues': [f"検証エラー: {e}"],
                'recommendations': ["予測品質検証処理を確認してください"]
            }