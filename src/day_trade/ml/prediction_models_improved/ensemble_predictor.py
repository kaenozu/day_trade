#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
アンサンブル予測システム - ML Prediction Models Ensemble Predictor

ML予測モデルのアンサンブル予測とモデル統合を行います。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
)
from .data_types import EnsemblePrediction
from .prediction_utils import PredictionUtils


class EnhancedEnsemblePredictor:
    """強化されたアンサンブル予測システム"""

    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.logger = logging.getLogger(__name__)
        self.prediction_utils = PredictionUtils()

    async def predict(
        self, 
        symbol: str, 
        features: pd.DataFrame
    ) -> Dict[PredictionTask, EnsemblePrediction]:
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

    async def _make_enhanced_ensemble_prediction(
        self, 
        symbol: str, 
        features: pd.DataFrame,
        task: PredictionTask
    ) -> Optional[EnsemblePrediction]:
        """強化されたアンサンブル予測"""

        model_predictions = {}
        model_confidences = {}
        model_quality_scores = {}
        excluded_models = []

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{model_type.value}_{task.value}"

            if not self.ml_models.has_trained_model(symbol, model_type, task):
                excluded_models.append(f"{model_type.value} (not trained)")
                continue

            try:
                model = self.ml_models.get_model_by_key(model_key)
                metadata = self.ml_models.get_metadata_by_key(model_key)

                if model is None:
                    excluded_models.append(f"{model_type.value} (not loaded)")
                    continue

                # モデル品質スコア
                quality_score = self.prediction_utils.calculate_model_quality_score(model_key, metadata)
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
                    label_encoder = self.ml_models.get_label_encoder_by_key(model_key)
                    if label_encoder:
                        pred_class = label_encoder.inverse_transform([pred_class])[0]

                    confidence = self.prediction_utils.calculate_classification_confidence(
                        pred_proba[0], quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # 回帰予測
                    pred_value = model.predict(features)[0]
                    confidence = self.prediction_utils.calculate_regression_confidence(
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
        base_weights = self.ml_models.ensemble_weights.get(symbol, {}).get(task, {})
        dynamic_weights = self.prediction_utils.calculate_dynamic_weights(
            base_weights, model_quality_scores, model_confidences
        )

        # アンサンブル統合
        if task == PredictionTask.PRICE_DIRECTION:
            ensemble_result = self.prediction_utils.ensemble_classification(
                model_predictions, model_confidences, dynamic_weights
            )
        else:
            ensemble_result = self.prediction_utils.ensemble_regression(
                model_predictions, model_confidences, dynamic_weights
            )

        # 品質メトリクス計算
        diversity_score = self.prediction_utils.calculate_diversity_score(model_predictions)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            prediction_interval=ensemble_result.get('prediction_interval'),
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_weights={ModelType(k).value: v for k, v in dynamic_weights.items()},
            model_quality_scores=model_quality_scores,
            consensus_strength=ensemble_result['consensus_strength'],
            disagreement_score=ensemble_result['disagreement_score'],
            prediction_stability=ensemble_result.get('prediction_stability', 0.0),
            diversity_score=diversity_score,
            total_models_used=len(model_predictions),
            excluded_models=excluded_models,
            ensemble_method="enhanced_weighted_ensemble"
        )

    def predict_single_task(
        self, 
        symbol: str, 
        features: pd.DataFrame, 
        task: PredictionTask
    ) -> Optional[EnsemblePrediction]:
        """単一タスクの予測（同期版）"""
        try:
            # 非同期メソッドを同期で実行
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._make_enhanced_ensemble_prediction(symbol, features, task)
            )
        except Exception as e:
            self.logger.error(f"単一タスク予測エラー: {e}")
            return None

    def validate_prediction_inputs(
        self, 
        symbol: str, 
        features: pd.DataFrame
    ) -> tuple[bool, str]:
        """予測入力データ検証"""
        return self.prediction_utils.validate_prediction_inputs(symbol, features)

    def get_prediction_explanation(
        self, 
        ensemble_prediction: EnsemblePrediction
    ) -> str:
        """予測結果の説明文生成"""
        try:
            explanation_parts = []
            
            # 基本情報
            explanation_parts.append(f"銘柄: {ensemble_prediction.symbol}")
            explanation_parts.append(f"予測値: {ensemble_prediction.final_prediction}")
            explanation_parts.append(f"信頼度: {ensemble_prediction.confidence:.2%}")
            
            # アンサンブル情報
            explanation_parts.append(f"使用モデル数: {ensemble_prediction.total_models_used}")
            explanation_parts.append(f"コンセンサス強度: {ensemble_prediction.consensus_strength:.2%}")
            
            # 除外されたモデル
            if ensemble_prediction.excluded_models:
                explanation_parts.append(f"除外モデル: {', '.join(ensemble_prediction.excluded_models)}")
            
            # 個別予測
            if ensemble_prediction.model_predictions:
                pred_strs = [f"{k}: {v}" for k, v in ensemble_prediction.model_predictions.items()]
                explanation_parts.append(f"個別予測: {', '.join(pred_strs)}")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            return f"説明生成エラー: {e}"

    def get_ensemble_statistics(
        self, 
        ensemble_prediction: EnsemblePrediction
    ) -> Dict[str, float]:
        """アンサンブル統計取得"""
        try:
            stats = {
                'total_models': ensemble_prediction.total_models_used,
                'consensus_strength': ensemble_prediction.consensus_strength,
                'disagreement_score': ensemble_prediction.disagreement_score,
                'prediction_stability': ensemble_prediction.prediction_stability,
                'diversity_score': ensemble_prediction.diversity_score,
                'confidence': ensemble_prediction.confidence,
                'excluded_models_count': len(ensemble_prediction.excluded_models)
            }

            # 重みの統計
            if ensemble_prediction.model_weights:
                weights = list(ensemble_prediction.model_weights.values())
                stats.update({
                    'weight_mean': np.mean(weights),
                    'weight_std': np.std(weights),
                    'weight_max': np.max(weights),
                    'weight_min': np.min(weights)
                })

            # 信頼度の統計
            if ensemble_prediction.model_confidences:
                confidences = list(ensemble_prediction.model_confidences.values())
                stats.update({
                    'confidence_mean': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'confidence_max': np.max(confidences),
                    'confidence_min': np.min(confidences)
                })

            return stats

        except Exception as e:
            self.logger.error(f"アンサンブル統計取得エラー: {e}")
            return {'error': str(e)}

    def compare_predictions(
        self, 
        predictions: List[EnsemblePrediction]
    ) -> Dict[str, any]:
        """複数の予測結果を比較"""
        try:
            if not predictions:
                return {'error': '予測結果がありません'}

            comparison = {
                'count': len(predictions),
                'symbols': [p.symbol for p in predictions],
                'timestamps': [p.timestamp for p in predictions],
                'final_predictions': [p.final_prediction for p in predictions],
                'confidences': [p.confidence for p in predictions],
                'consensus_strengths': [p.consensus_strength for p in predictions]
            }

            # 統計値
            confidences = comparison['confidences']
            consensus_strengths = comparison['consensus_strengths']

            comparison['statistics'] = {
                'confidence_mean': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'consensus_mean': np.mean(consensus_strengths),
                'consensus_std': np.std(consensus_strengths)
            }

            return comparison

        except Exception as e:
            return {'error': f'比較処理エラー: {e}'}

    def get_model_contribution_analysis(
        self, 
        ensemble_prediction: EnsemblePrediction
    ) -> Dict[str, Dict]:
        """モデル貢献度分析"""
        try:
            analysis = {}

            for model_name in ensemble_prediction.model_predictions.keys():
                model_analysis = {
                    'prediction': ensemble_prediction.model_predictions.get(model_name),
                    'confidence': ensemble_prediction.model_confidences.get(model_name, 0.0),
                    'weight': ensemble_prediction.model_weights.get(model_name, 0.0),
                    'quality_score': ensemble_prediction.model_quality_scores.get(model_name, 0.0)
                }

                # 貢献度計算
                contribution = (model_analysis['confidence'] * model_analysis['weight'] * 
                              model_analysis['quality_score'])
                model_analysis['contribution'] = contribution

                analysis[model_name] = model_analysis

            return analysis

        except Exception as e:
            self.logger.error(f"モデル貢献度分析エラー: {e}")
            return {'error': str(e)}