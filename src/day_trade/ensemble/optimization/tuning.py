#!/usr/bin/env python3
"""
ハイパーパラメータ調整
"""

import numpy as np
import asyncio
import logging
import copy
import warnings
from typing import List, Any, Dict, Callable

import optuna

from .types import EnsembleConfiguration
from .objectives import ObjectiveFunction

logger = logging.getLogger(__name__)

class HyperparameterTuning:
    """ハイパーパラメータ調整"""

    def __init__(self, method: str = "bayesian", n_trials: int = 50):
        self.method = method
        self.n_trials = n_trials
        self.tuning_history: List[Dict[str, Any]] = []

        logger.info(f"HyperparameterTuning initialized with {method} method")

    def _create_optuna_objective(self,
                                configuration: EnsembleConfiguration,
                                objective_function: ObjectiveFunction) -> Callable:
        """Optuna目的関数作成"""
        def objective(trial):
            # ハイパーパラメータ提案
            tuned_config = copy.deepcopy(configuration)

            for model_id, hyperparams in tuned_config.hyperparameters.items():
                tuned_hyperparams = {}

                for param_name, param_value in hyperparams.items():
                    if isinstance(param_value, float):
                        # 現在値の±50%範囲で提案
                        low = max(0.001, param_value * 0.5)
                        high = param_value * 1.5
                        tuned_hyperparams[param_name] = trial.suggest_float(
                            f"{model_id}_{param_name}", low, high
                        )
                    elif isinstance(param_value, int):
                        low = max(1, int(param_value * 0.7))
                        high = int(param_value * 1.3)
                        tuned_hyperparams[param_name] = trial.suggest_int(
                            f"{model_id}_{param_name}", low, high
                        )
                    else:
                        tuned_hyperparams[param_name] = param_value

                tuned_config.hyperparameters[model_id] = tuned_hyperparams

            # 評価
            score = objective_function.evaluate(tuned_config)
            return score

        return objective

    async def tune(self,
                 configuration: EnsembleConfiguration,
                 objective_function: ObjectiveFunction) -> EnsembleConfiguration:
        """ハイパーパラメータ調整"""
        try:
            if self.method == "bayesian" and optuna is not None:
                return await self._bayesian_tuning(configuration, objective_function)
            elif self.method == "random":
                return await self._random_tuning(configuration, objective_function)
            else:
                logger.warning(f"Unknown tuning method: {self.method}")
                return configuration

        except Exception as e:
            logger.error(f"Hyperparameter tuning error: {e}")
            return configuration

    async def _bayesian_tuning(self,
                             configuration: EnsembleConfiguration,
                             objective_function: ObjectiveFunction) -> EnsembleConfiguration:
        """ベイズ最適化調整"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            study = optuna.create_study(direction='maximize')
            objective = self._create_optuna_objective(configuration, objective_function)

            study.optimize(objective, n_trials=self.n_trials)

            # 最良パラメータ適用
            best_params = study.best_params
            tuned_config = copy.deepcopy(configuration)

            for param_name, param_value in best_params.items():
                parts = param_name.split('_', 1)
                if len(parts) == 2:
                    model_id, hyperparam_name = parts
                    if model_id in tuned_config.hyperparameters:
                        tuned_config.hyperparameters[model_id][hyperparam_name] = param_value

            # 最終評価
            final_score = objective_function.evaluate(tuned_config)
            tuned_config.estimated_performance = final_score

            logger.info(f"Bayesian tuning completed. Score improved: {final_score:.4f}")
            return tuned_config

    async def _random_tuning(self,
                           configuration: EnsembleConfiguration,
                           objective_function: ObjectiveFunction) -> EnsembleConfiguration:
        """ランダム調整"""
        best_config = copy.deepcopy(configuration)
        best_score = objective_function.evaluate(best_config)

        for trial in range(self.n_trials):
            # ランダム摂動
            perturbed_config = copy.deepcopy(configuration)

            for model_id, hyperparams in perturbed_config.hyperparameters.items():
                for param_name, param_value in hyperparams.items():
                    if isinstance(param_value, float):
                        # ±20%の範囲でランダム摂動
                        noise_factor = np.random.uniform(0.8, 1.2)
                        hyperparams[param_name] = param_value * noise_factor
                    elif isinstance(param_value, int):
                        # ±1の範囲で摂動
                        hyperparams[param_name] = max(1, param_value + np.random.randint(-1, 2))

            # 評価
            score = objective_function.evaluate(perturbed_config)

            if score > best_score:
                best_score = score
                best_config = copy.deepcopy(perturbed_config)

        logger.info(f"Random tuning completed. Score improved: {best_score:.4f}")
        return best_config
