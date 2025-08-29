#!/usr/bin/env python3
"""
AutoML統合
"""

import numpy as np
import time
import logging
import copy
from typing import List, Any, Dict, Tuple

from .types import EnsembleConfiguration, ModelCandidate
from .objectives import MultiObjectiveFunction

logger = logging.getLogger(__name__)

class AutoMLIntegration:
    """AutoML統合"""

    def __init__(self, framework: str = "custom"):
        self.framework = framework
        self.auto_configs: List[EnsembleConfiguration] = []

        logger.info(f"AutoMLIntegration initialized with {framework} framework")

    async def auto_configure(self,
                           training_data: Tuple[np.ndarray, np.ndarray],
                           model_pool: List[ModelCandidate],
                           time_budget_minutes: int = 30) -> List[EnsembleConfiguration]:
        """自動構成"""
        try:
            # 簡易AutoML実装
            configs = []

            # データ特性分析
            X, y = training_data
            data_characteristics = self._analyze_data(X, y)

            # データ特性に基づく推奨構成生成
            recommended_configs = self._generate_recommended_configs(
                data_characteristics, model_pool
            )

            configs.extend(recommended_configs)

            # 時間予算内で追加探索
            time_per_config = max(1, (time_budget_minutes * 60) // len(recommended_configs))

            for config in recommended_configs:
                # 軽量な最適化
                optimized_config = await self._lightweight_optimization(
                    config, training_data, time_per_config
                )
                configs.append(optimized_config)

            self.auto_configs = configs
            return configs

        except Exception as e:
            logger.error(f"Auto-configuration error: {e}")
            return []

    def _analyze_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """データ特性分析"""
        characteristics = {}

        # 基本統計
        characteristics['n_samples'] = X.shape[0]
        characteristics['n_features'] = X.shape[1]
        characteristics['feature_mean'] = np.mean(X)
        characteristics['feature_std'] = np.std(X)
        characteristics['target_mean'] = np.mean(y)
        characteristics['target_std'] = np.std(y)

        # 複雑性指標
        characteristics['data_complexity'] = X.shape[1] / X.shape[0]
        characteristics['target_variability'] = np.std(y) / (np.abs(np.mean(y)) + 1e-8)

        # 推奨モデルタイプ
        if X.shape[0] < 1000:
            characteristics['recommended_models'] = ['linear', 'tree']
        elif X.shape[1] > 50:
            characteristics['recommended_models'] = ['ensemble', 'neural']
        else:
            characteristics['recommended_models'] = ['ensemble', 'tree', 'linear']

        return characteristics

    def _generate_recommended_configs(self,
                                    data_characteristics: Dict[str, Any],
                                    model_pool: List[ModelCandidate]) -> List[EnsembleConfiguration]:
        """推奨構成生成"""
        configs = []

        recommended_model_types = data_characteristics.get('recommended_models', [])

        # 推奨モデルタイプごとに構成作成
        for model_type in recommended_model_types:
            matching_models = [
                m for m in model_pool
                if model_type in m.model_class.lower() and m.is_enabled
            ]

            if matching_models:
                # 2-3個のモデルで構成
                selected_models = matching_models[:min(3, len(matching_models))]

                # 均等重み
                weights = {m.model_id: 1.0/len(selected_models) for m in selected_models}

                # デフォルトハイパーパラメータ
                hyperparameters = {}
                for model in selected_models:
                    hyperparameters[model.model_id] = {
                        param: config.get('default', 1.0)
                        for param, config in model.hyperparameter_space.items()
                    }

                config = EnsembleConfiguration(
                    model_selection=[m.model_id for m in selected_models],
                    weights=weights,
                    hyperparameters=hyperparameters,
                    aggregation_method="weighted_average"
                )

                configs.append(config)

        return configs

    async def _lightweight_optimization(self,
                                      config: EnsembleConfiguration,
                                      training_data: Tuple[np.ndarray, np.ndarray],
                                      time_budget_seconds: int) -> EnsembleConfiguration:
        """軽量最適化"""
        start_time = time.time()
        best_config = copy.deepcopy(config)

        # 簡易目的関数
        objective = MultiObjectiveFunction(training_data)
        best_score = objective.evaluate(best_config)

        # 制限時間内で重み最適化
        while time.time() - start_time < time_budget_seconds:
            # 重みの軽微な調整
            test_config = copy.deepcopy(best_config)

            # 重みに小さなノイズ追加
            total_weight = 0.0
            for model_id in test_config.weights:
                noise = np.random.normal(0, 0.05)
                test_config.weights[model_id] = max(0.01, test_config.weights[model_id] + noise)
                total_weight += test_config.weights[model_id]

            # 正規化
            for model_id in test_config.weights:
                test_config.weights[model_id] /= total_weight

            # 評価
            score = objective.evaluate(test_config)

            if score > best_score:
                best_score = score
                best_config = test_config

        return best_config
