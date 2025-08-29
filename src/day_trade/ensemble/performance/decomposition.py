#!/usr/bin/env python3
"""
パフォーマンス分解器
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any

from .types import PerformanceDecomposition, AttributionResult

logger = logging.getLogger(__name__)

class PerformanceDecomposer:
    """パフォーマンス分解器"""

    def __init__(self, method: str = "variance_decomposition"):
        self.method = method
        self.decomposition_history: List[PerformanceDecomposition] = []

        logger.info(f"PerformanceDecomposer initialized with {method} method")

    async def decompose(self,
                      predictions: np.ndarray,
                      model_predictions: Dict[str, np.ndarray],
                      attributions: AttributionResult) -> PerformanceDecomposition:
        """パフォーマンス分解"""
        try:
            if self.method == "variance_decomposition":
                return await self._variance_decomposition(predictions, model_predictions, attributions)
            elif self.method == "additive_decomposition":
                return await self._additive_decomposition(predictions, model_predictions, attributions)
            else:
                return await self._simple_decomposition(predictions, model_predictions, attributions)

        except Exception as e:
            logger.error(f"Performance decomposition error: {e}")
            return self._create_fallback_decomposition(model_predictions)

    async def _variance_decomposition(self,
                                    predictions: np.ndarray,
                                    model_predictions: Dict[str, np.ndarray],
                                    attributions: AttributionResult) -> PerformanceDecomposition:
        """分散分解"""
        try:
            # 全体パフォーマンス（分散で測定）
            total_performance = np.var(predictions)

            # 個別モデル貢献度
            individual_contributions = {}
            total_individual = 0.0

            for model_id, model_pred in model_predictions.items():
                # モデルの分散貢献
                model_var = np.var(model_pred)
                contribution = model_var / (total_performance + 1e-8)
                individual_contributions[model_id] = contribution
                total_individual += contribution

            # 正規化
            if total_individual > 0:
                for model_id in individual_contributions:
                    individual_contributions[model_id] /= total_individual

            # 相互作用項
            interaction_terms = {}
            model_ids = list(model_predictions.keys())

            for i in range(len(model_ids)):
                for j in range(i+1, len(model_ids)):
                    model_i = model_predictions[model_ids[i]]
                    model_j = model_predictions[model_ids[j]]

                    # 共分散を相互作用とする
                    covariance = np.cov(model_i.flatten(), model_j.flatten())[0, 1]
                    interaction_terms[f"{model_ids[i]}_{model_ids[j]}"] = abs(covariance) / (total_performance + 1e-8)

            # 残差パフォーマンス
            explained_variance = sum(individual_contributions.values()) + sum(interaction_terms.values())
            residual_performance = max(0, 1.0 - explained_variance)

            # コンポーネントランキング
            all_components = {**individual_contributions, **interaction_terms}
            component_rankings = sorted(all_components.items(), key=lambda x: x[1], reverse=True)

            decomposition = PerformanceDecomposition(
                total_performance=total_performance,
                individual_contributions=individual_contributions,
                interaction_terms=interaction_terms,
                residual_performance=residual_performance,
                decomposition_method="variance_decomposition",
                variance_explained=explained_variance,
                component_rankings=component_rankings
            )

            self.decomposition_history.append(decomposition)
            return decomposition

        except Exception as e:
            logger.error(f"Variance decomposition error: {e}")
            return self._create_fallback_decomposition(model_predictions)

    async def _additive_decomposition(self,
                                    predictions: np.ndarray,
                                    model_predictions: Dict[str, np.ndarray],
                                    attributions: AttributionResult) -> PerformanceDecomposition:
        """加算分解"""
        try:
            # アンサンブル予測の加算分解
            total_performance = 1.0  # 正規化ベース

            # 重み取得（attributionsから）
            individual_contributions = {}

            for model_id in model_predictions:
                if model_id in attributions.model_attributions:
                    contribution = np.mean(attributions.model_attributions[model_id])
                    individual_contributions[model_id] = contribution
                else:
                    individual_contributions[model_id] = 1.0 / len(model_predictions)

            # 正規化
            total_contrib = sum(individual_contributions.values())
            if total_contrib > 0:
                individual_contributions = {
                    k: v / total_contrib for k, v in individual_contributions.items()
                }

            # 相互作用項（attributionsから）
            interaction_terms = {}
            for interaction_key, interaction_values in attributions.interaction_effects.items():
                interaction_terms[interaction_key] = np.mean(interaction_values)

            # 残差
            explained = sum(individual_contributions.values()) + sum(interaction_terms.values())
            residual_performance = max(0, total_performance - explained)

            # ランキング
            all_components = {**individual_contributions, **interaction_terms}
            component_rankings = sorted(all_components.items(), key=lambda x: x[1], reverse=True)

            return PerformanceDecomposition(
                total_performance=total_performance,
                individual_contributions=individual_contributions,
                interaction_terms=interaction_terms,
                residual_performance=residual_performance,
                decomposition_method="additive_decomposition",
                variance_explained=explained,
                component_rankings=component_rankings
            )

        except Exception as e:
            logger.error(f"Additive decomposition error: {e}")
            return self._create_fallback_decomposition(model_predictions)

    async def _simple_decomposition(self,
                                  predictions: np.ndarray,
                                  model_predictions: Dict[str, np.ndarray],
                                  attributions: AttributionResult) -> PerformanceDecomposition:
        """簡易分解"""
        n_models = len(model_predictions)
        uniform_contribution = 1.0 / n_models if n_models > 0 else 1.0

        individual_contributions = {
            model_id: uniform_contribution for model_id in model_predictions
        }

        return PerformanceDecomposition(
            total_performance=1.0,
            individual_contributions=individual_contributions,
            interaction_terms={},
            residual_performance=0.0,
            decomposition_method="simple",
            variance_explained=1.0,
            component_rankings=list(individual_contributions.items())
        )

    def _create_fallback_decomposition(self, model_predictions: Dict[str, np.ndarray]) -> PerformanceDecomposition:
        """フォールバック分解"""
        return self._simple_decomposition(np.array([0]), model_predictions, None)
