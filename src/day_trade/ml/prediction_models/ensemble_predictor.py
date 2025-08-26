#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アンサンブル予測システム

Issue #850-4: 強化されたアンサンブル予測機能
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.day_trade.ml.core_types import ModelType, PredictionTask
from .data_structures import EnsemblePrediction, ModelMetadata
from .confidence_calculator import ConfidenceCalculator


class EnhancedEnsemblePredictor:
    """強化されたアンサンブル予測システム"""

    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.logger = logging.getLogger(__name__)
        self.confidence_calculator = ConfidenceCalculator()

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
                quality_score = self.confidence_calculator.calculate_model_quality_score(model_key, metadata)
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

                    confidence = self.confidence_calculator.calculate_classification_confidence(
                        pred_proba[0], quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # 回帰予測
                    pred_value = model.predict(features)[0]
                    confidence = self.confidence_calculator.calculate_regression_confidence(
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
        diversity_score = self.confidence_calculator.calculate_diversity_score(model_predictions)

        # アンサンブル予測結果の保存
        await self._save_ensemble_prediction(symbol, task, ensemble_result, model_predictions,
                                           model_confidences, dynamic_weights, model_quality_scores,
                                           excluded_models, diversity_score)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            prediction_interval=ensemble_result.get('prediction_interval'),
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_weights={ModelType(k): v for k, v in dynamic_weights.items()},
            model_quality_scores=model_quality_scores,
            consensus_strength=ensemble_result['consensus_strength'],
            disagreement_score=ensemble_result['disagreement_score'],
            prediction_stability=ensemble_result.get('prediction_stability', 0.0),
            diversity_score=diversity_score,
            total_models_used=len(model_predictions),
            excluded_models=excluded_models,
            ensemble_method="enhanced_weighted_ensemble"
        )

    async def _calculate_enhanced_dynamic_weights(self, symbol: str, task: PredictionTask,
                                                quality_scores: Dict[str, float],
                                                confidences: Dict[str, float]) -> Dict[str, float]:
        """強化された動的重み計算"""
        try:
            # 基本重み（過去の性能ベース）
            base_weights = getattr(self.ml_models, 'ensemble_weights', {}).get(symbol, {}).get(task, {})

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

    def _enhanced_ensemble_classification(self, predictions: Dict[str, any],
                                        confidences: Dict[str, float],
                                        weights: Dict[str, float]) -> Dict[str, any]:
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
        consensus_strength = self.confidence_calculator.calculate_consensus_strength(predictions)

        # 予測安定性
        prediction_stability = self.confidence_calculator.calculate_prediction_stability(predictions)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': prediction_stability
        }

    def _enhanced_ensemble_regression(self, predictions: Dict[str, float],
                                    confidences: Dict[str, float],
                                    weights: Dict[str, float]) -> Dict[str, any]:
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
        ensemble_confidence = self.confidence_calculator.calculate_ensemble_confidence_regression(
            predictions, confidences, weights
        )

        # コンセンサス強度
        consensus_strength = self.confidence_calculator.calculate_consensus_strength(predictions)

        return {
            'prediction': final_prediction,
            'confidence': ensemble_confidence,
            'prediction_interval': prediction_interval,
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': self.confidence_calculator.calculate_prediction_stability(predictions)
        }

    async def _save_ensemble_prediction(self, symbol: str, task: PredictionTask, 
                                      ensemble_result: Dict, model_predictions: Dict,
                                      model_confidences: Dict, dynamic_weights: Dict,
                                      model_quality_scores: Dict, excluded_models: List,
                                      diversity_score: float):
        """アンサンブル予測結果の保存"""
        try:
            ensemble_data = {
                'final_prediction': ensemble_result['prediction'],
                'confidence': ensemble_result['confidence'],
                'consensus_strength': ensemble_result['consensus_strength'],
                'disagreement_score': ensemble_result['disagreement_score'],
                'prediction_stability': ensemble_result.get('prediction_stability', 0.0),
                'diversity_score': diversity_score,
                'total_models_used': len(model_predictions),
                'model_predictions': model_predictions,
                'model_weights': dynamic_weights,
                'model_quality_scores': model_quality_scores,
                'excluded_models': excluded_models,
                'ensemble_method': 'enhanced_weighted_ensemble'
            }

            # メタデータマネージャーに保存
            if hasattr(self.ml_models, 'metadata_manager'):
                await self.ml_models.metadata_manager.save_ensemble_prediction(
                    symbol, datetime.now(), task, ensemble_data
                )

        except Exception as e:
            self.logger.error(f"アンサンブル予測保存エラー: {e}")

    def get_prediction_explanation(self, prediction: EnsemblePrediction) -> str:
        """予測結果の説明生成"""
        try:
            explanation_parts = []
            
            # 基本情報
            explanation_parts.append(f"銘柄 {prediction.symbol} の予測結果:")
            explanation_parts.append(f"最終予測: {prediction.final_prediction}")
            explanation_parts.append(f"信頼度: {prediction.confidence:.1%}")
            
            # モデル情報
            explanation_parts.append(f"使用モデル数: {prediction.total_models_used}")
            if prediction.excluded_models:
                explanation_parts.append(f"除外モデル: {', '.join(prediction.excluded_models)}")
            
            # 品質指標
            explanation_parts.append(f"コンセンサス強度: {prediction.consensus_strength:.1%}")
            explanation_parts.append(f"予測安定性: {prediction.prediction_stability:.1%}")
            explanation_parts.append(f"多様性スコア: {prediction.diversity_score:.1%}")
            
            # 個別モデル予測
            if prediction.model_predictions:
                explanation_parts.append("個別モデル予測:")
                for model, pred in prediction.model_predictions.items():
                    confidence = prediction.model_confidences.get(model, 0.0)
                    weight = prediction.model_weights.get(model, 0.0)
                    explanation_parts.append(f"  {model}: {pred} (信頼度: {confidence:.1%}, 重み: {weight:.1%})")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            self.logger.error(f"予測説明生成エラー: {e}")
            return f"予測: {prediction.final_prediction}, 信頼度: {prediction.confidence:.1%}"