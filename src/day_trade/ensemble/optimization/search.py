#!/usr/bin/env python3
"""
アーキテクチャ探索
"""

import numpy as np
import asyncio
import logging
import copy
from typing import List, Any, Dict

from .types import EnsembleConfiguration, ModelCandidate
from .objectives import ObjectiveFunction

logger = logging.getLogger(__name__)

class ArchitectureSearch:
    """アーキテクチャ探索"""

    def __init__(self,
                 model_pool: List[ModelCandidate],
                 search_space_size: int = 1000):
        self.model_pool = model_pool
        self.search_space_size = search_space_size
        self.search_history: List[EnsembleConfiguration] = []

        logger.info(f"ArchitectureSearch initialized with {len(self.model_pool)} model candidates")

    def generate_random_architecture(self) -> EnsembleConfiguration:
        """ランダムアーキテクチャ生成"""
        # モデル選択（2-5個をランダム選択）
        enabled_models = [m for m in self.model_pool if m.is_enabled]
        n_models = np.random.randint(2, min(6, len(enabled_models) + 1))
        selected_models = np.random.choice(
            [m.model_id for m in enabled_models],
            size=min(n_models, len(enabled_models)),
            replace=False
        ).tolist()

        # 重み生成（Dirichlet分布）
        weights_array = np.random.dirichlet([1.0] * len(selected_models))
        weights = dict(zip(selected_models, weights_array))

        # ハイパーパラメータ（簡易版）
        hyperparameters = {}
        for model_id in selected_models:
            model_candidate = next((m for m in self.model_pool if m.model_id == model_id), None)
            if model_candidate:
                hyperparameters[model_id] = self._sample_hyperparameters(model_candidate)

        return EnsembleConfiguration(
            model_selection=selected_models,
            weights=weights,
            hyperparameters=hyperparameters,
            aggregation_method="weighted_average"
        )

    def _sample_hyperparameters(self, model_candidate: ModelCandidate) -> Dict[str, Any]:
        """ハイパーパラメータサンプリング"""
        sampled = {}

        for param_name, param_config in model_candidate.hyperparameter_space.items():
            param_type = param_config.get('type', 'float')

            if param_type == 'float':
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                sampled[param_name] = np.random.uniform(low, high)

            elif param_type == 'int':
                low = param_config.get('low', 1)
                high = param_config.get('high', 100)
                sampled[param_name] = np.random.randint(low, high + 1)

            elif param_type == 'categorical':
                choices = param_config.get('choices', ['default'])
                sampled[param_name] = np.random.choice(choices)

            else:
                sampled[param_name] = param_config.get('default', 1.0)

        return sampled

    async def search(self,
                   objective_function: ObjectiveFunction,
                   max_evaluations: int = 100) -> EnsembleConfiguration:
        """アーキテクチャ探索"""
        best_config = None
        best_score = float('-inf')

        logger.info(f"Starting architecture search with {max_evaluations} evaluations")

        for i in range(max_evaluations):
            # ランダム構成生成
            config = self.generate_random_architecture()

            # 評価
            score = objective_function.evaluate(config)

            # 最良更新
            if score > best_score:
                best_score = score
                best_config = copy.deepcopy(config)
                logger.debug(f"New best configuration found: score={score:.4f}")

            # 履歴保存
            config.optimization_history.append({
                'evaluation': i,
                'score': score,
                'is_best': score == best_score
            })
            self.search_history.append(config)

        logger.info(f"Architecture search completed. Best score: {best_score:.4f}")
        return best_config if best_config else self.generate_random_architecture()
