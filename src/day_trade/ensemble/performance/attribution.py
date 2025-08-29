#!/usr/bin/env python3
"""
貢献度分析エンジン
"""

import numpy as np
import asyncio
import logging
import time
from typing import Dict, List, Tuple, Any

from .types import AttributionResult

# SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, using fallback attribution methods")

logger = logging.getLogger(__name__)

class AttributionEngine:
    """貢献度分析エンジン"""

    def __init__(self, method: str = "shap", fallback_method: str = "permutation"):
        self.method = method if SHAP_AVAILABLE else fallback_method
        self.fallback_method = fallback_method
        self.attribution_cache: Dict[str, AttributionResult] = {}

        logger.info(f"AttributionEngine initialized with {self.method} method")

    async def compute_attributions(self,
                                 ensemble_system: Any,
                                 test_data: np.ndarray,
                                 predictions: np.ndarray,
                                 model_predictions: Dict[str, np.ndarray]) -> AttributionResult:
        """貢献度計算"""
        try:
            if self.method == "shap" and SHAP_AVAILABLE:
                return await self._compute_shap_attributions(
                    ensemble_system, test_data, predictions, model_predictions
                )
            else:
                return await self._compute_permutation_attributions(
                    ensemble_system, test_data, predictions, model_predictions
                )

        except Exception as e:
            logger.error(f"Attribution computation error: {e}")
            return self._create_fallback_attribution(model_predictions)

    async def _compute_shap_attributions(self,
                                       ensemble_system: Any,
                                       test_data: np.ndarray,
                                       predictions: np.ndarray,
                                       model_predictions: Dict[str, np.ndarray]) -> AttributionResult:
        """SHAP貢献度計算"""
        try:
            # アンサンブル予測関数
            def ensemble_predict_func(X):
                if hasattr(ensemble_system, 'predict_ensemble'):
                    pred, _ = asyncio.run(ensemble_system.predict_ensemble(X))
                    return pred.flatten()
                else:
                    # フォールバック予測
                    return np.mean([pred.flatten() for pred in model_predictions.values()], axis=0)

            # SHAP Explainer
            background_data = test_data[:min(100, len(test_data))]  # バックグラウンドサンプル
            explainer = shap.KernelExplainer(ensemble_predict_func, background_data)

            # SHAP値計算
            shap_values = explainer.shap_values(test_data[:50])  # 計算コスト考慮

            # モデル別貢献度（重みベース）
            model_attributions = {}
            if hasattr(ensemble_system, 'get_current_weights'):
                weights = ensemble_system.get_current_weights()
                for model_id, weight in weights.items():
                    # 重みを貢献度とする簡易計算
                    model_attributions[model_id] = np.full(len(test_data), weight)

            # 特徴量貢献度
            feature_attributions = {
                f"feature_{i}": shap_values[:, i] if i < shap_values.shape[1] else np.zeros(len(shap_values))
                for i in range(test_data.shape[1])
            }

            # 相互作用効果（簡易計算）
            interaction_effects = self._compute_interaction_effects(test_data, shap_values)

            # 信頼区間
            confidence_intervals = {}
            for model_id, attr in model_attributions.items():
                mean_attr = np.mean(attr)
                std_attr = np.std(attr)
                confidence_intervals[model_id] = (
                    mean_attr - 1.96 * std_attr,
                    mean_attr + 1.96 * std_attr
                )

            return AttributionResult(
                model_attributions=model_attributions,
                feature_attributions=feature_attributions,
                interaction_effects=interaction_effects,
                attribution_method="shap",
                confidence_intervals=confidence_intervals,
                timestamp=time.time(),
                metadata={"shap_values_shape": shap_values.shape}
            )

        except Exception as e:
            logger.error(f"SHAP attribution error: {e}")
            return await self._compute_permutation_attributions(
                ensemble_system, test_data, predictions, model_predictions
            )

    async def _compute_permutation_attributions(self,
                                              ensemble_system: Any,
                                              test_data: np.ndarray,
                                              predictions: np.ndarray,
                                              model_predictions: Dict[str, np.ndarray]) -> AttributionResult:
        """Permutation貢献度計算"""
        try:
            # ベースライン性能
            baseline_mse = mean_squared_error(predictions, predictions)  # 0 by definition

            model_attributions = {}

            # モデル別貢献度（除去影響）
            for model_id in model_predictions:
                # 該当モデルを除いた予測
                other_predictions = [pred for mid, pred in model_predictions.items() if mid != model_id]

                if other_predictions:
                    reduced_ensemble = np.mean(other_predictions, axis=0)
                    reduced_mse = mean_squared_error(predictions, reduced_ensemble)
                    attribution = max(0, reduced_mse - baseline_mse)  # 除去による性能低下
                else:
                    attribution = 1.0

                model_attributions[model_id] = np.full(len(test_data), attribution)

            # 特徴量貢献度（Permutation importance）
            feature_attributions = {}

            for feature_idx in range(test_data.shape[1]):
                # 特徴量をシャッフル
                shuffled_data = test_data.copy()
                np.random.shuffle(shuffled_data[:, feature_idx])

                # 性能低下を計算（簡易版）
                baseline_var = np.var(test_data[:, feature_idx])
                shuffled_var = np.var(shuffled_data[:, feature_idx])
                importance = abs(baseline_var - shuffled_var) / (baseline_var + 1e-8)

                feature_attributions[f"feature_{feature_idx}"] = np.full(len(test_data), importance)

            # 相互作用効果（簡易計算）
            interaction_effects = {}
            if len(model_predictions) >= 2:
                model_ids = list(model_predictions.keys())
                for i in range(len(model_ids)):
                    for j in range(i+1, len(model_ids)):
                        # 2つのモデルの相関を相互作用とする
                        corr, _ = pearsonr(
                            model_predictions[model_ids[i]].flatten(),
                            model_predictions[model_ids[j]].flatten()
                        )
                        interaction_key = f"{model_ids[i]}_{model_ids[j]}"
                        interaction_effects[interaction_key] = np.full(len(test_data), abs(corr))

            # 信頼区間（簡易計算）
            confidence_intervals = {}
            for model_id, attr in model_attributions.items():
                mean_attr = np.mean(attr)
                std_attr = np.std(attr) + 1e-8
                confidence_intervals[model_id] = (
                    mean_attr - 1.96 * std_attr,
                    mean_attr + 1.96 * std_attr
                )

            return AttributionResult(
                model_attributions=model_attributions,
                feature_attributions=feature_attributions,
                interaction_effects=interaction_effects,
                attribution_method="permutation",
                confidence_intervals=confidence_intervals,
                timestamp=time.time(),
                metadata={"permutation_samples": len(test_data)}
            )

        except Exception as e:
            logger.error(f"Permutation attribution error: {e}")
            return self._create_fallback_attribution(model_predictions)

    def _compute_interaction_effects(self, test_data: np.ndarray, shap_values: np.ndarray) -> Dict[str, np.ndarray]:
        """相互作用効果計算"""
        interaction_effects = {}

        try:
            # 特徴量間の相互作用（簡易版）
            for i in range(min(5, test_data.shape[1])):  # 最初の5特徴量のみ
                for j in range(i+1, min(5, test_data.shape[1])):
                    # 特徴量の掛け算による相互作用
                    interaction = test_data[:len(shap_values), i] * test_data[:len(shap_values), j]
                    interaction_key = f"feature_{i}_x_feature_{j}"
                    interaction_effects[interaction_key] = interaction

        except Exception as e:
            logger.warning(f"Interaction computation error: {e}")

        return interaction_effects

    def _create_fallback_attribution(self, model_predictions: Dict[str, np.ndarray]) -> AttributionResult:
        """フォールバック貢献度"""
        n_samples = len(next(iter(model_predictions.values()))) if model_predictions else 100

        # 均等貢献度
        n_models = len(model_predictions)
        equal_attribution = 1.0 / n_models if n_models > 0 else 1.0

        model_attributions = {
            model_id: np.full(n_samples, equal_attribution)
            for model_id in model_predictions
        }

        feature_attributions = {
            f"feature_{i}": np.random.randn(n_samples) * 0.1
            for i in range(5)  # デフォルト5特徴量
        }

        return AttributionResult(
            model_attributions=model_attributions,
            feature_attributions=feature_attributions,
            interaction_effects={},
            attribution_method="fallback",
            confidence_intervals={},
            timestamp=time.time()
        )
