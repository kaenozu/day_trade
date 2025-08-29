#!/usr/bin/env python3
"""
最適化のための目的関数
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from abc import abstractmethod

from sklearn.metrics import mean_squared_error

from .types import EnsembleConfiguration

logger = logging.getLogger(__name__)

class ObjectiveFunction:
    """目的関数基底クラス"""

    def __init__(self,
                 training_data: Tuple[np.ndarray, np.ndarray],
                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 cv_folds: int = 5):
        self.training_data = training_data
        self.validation_data = validation_data
        self.cv_folds = cv_folds
        self.evaluation_count = 0

    @abstractmethod
    def evaluate(self, configuration: EnsembleConfiguration) -> float:
        """設定評価"""
        pass

class MultiObjectiveFunction(ObjectiveFunction):
    """多目的最適化関数"""

    def __init__(self,
                 training_data: Tuple[np.ndarray, np.ndarray],
                 validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 objectives: List[str] = None,
                 weights: List[float] = None):
        super().__init__(training_data, validation_data)

        self.objectives = objectives or ["accuracy", "diversity", "efficiency"]
        self.objective_weights = weights or [0.6, 0.2, 0.2]

        if len(self.objective_weights) != len(self.objectives):
            self.objective_weights = [1.0 / len(self.objectives)] * len(self.objectives)

    def evaluate(self, configuration: EnsembleConfiguration) -> float:
        """多目的評価"""
        self.evaluation_count += 1

        scores = {}

        # 精度評価
        if "accuracy" in self.objectives:
            scores["accuracy"] = self._evaluate_accuracy(configuration)

        # 多様性評価
        if "diversity" in self.objectives:
            scores["diversity"] = self._evaluate_diversity(configuration)

        # 効率性評価
        if "efficiency" in self.objectives:
            scores["efficiency"] = self._evaluate_efficiency(configuration)

        # 重み付き総合スコア
        total_score = 0.0
        for objective, weight in zip(self.objectives, self.objective_weights):
            if objective in scores:
                total_score += weight * scores[objective]

        # 設定にメトリクス保存
        configuration.performance_metrics.update(scores)
        configuration.estimated_performance = total_score

        return total_score

    def _evaluate_accuracy(self, configuration: EnsembleConfiguration) -> float:
        """精度評価"""
        try:
            X_train, y_train = self.training_data

            if self.validation_data:
                X_val, y_val = self.validation_data
            else:
                # クロスバリデーション
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

            # アンサンブル予測（簡易実装）
            predictions = self._ensemble_predict(configuration, X_val, X_train, y_train)

            # MSE計算 (小さいほど良い -> 大きいほど良いに変換)
            mse = mean_squared_error(y_val, predictions)
            accuracy_score = 1.0 / (1.0 + mse)

            return accuracy_score

        except Exception as e:
            logger.warning(f"Accuracy evaluation error: {e}")
            return 0.1

    def _ensemble_predict(self,
                         configuration: EnsembleConfiguration,
                         X_test: np.ndarray,
                         X_train: np.ndarray,
                         y_train: np.ndarray) -> np.ndarray:
        """アンサンブル予測（簡易版）"""
        predictions = []
        total_weight = 0.0

        for model_id in configuration.model_selection:
            weight = configuration.weights.get(model_id, 0.0)
            if weight <= 0:
                continue

            # 簡易モデル予測（実際の実装では実際のモデルを使用）
            if "rf" in model_id.lower():
                # Random Forest風
                pred = np.mean(X_test, axis=1, keepdims=True) + np.random.randn(X_test.shape[0], 1) * 0.1
            elif "linear" in model_id.lower():
                # Linear風
                pred = np.sum(X_test * 0.1, axis=1, keepdims=True)
            else:
                # デフォルト
                pred = np.random.randn(X_test.shape[0], 1) * 0.1

            if len(predictions) == 0:
                predictions = pred * weight
            else:
                predictions += pred * weight

            total_weight += weight

        if total_weight > 0:
            predictions /= total_weight
        else:
            predictions = np.zeros((X_test.shape[0], 1))

        return predictions

    def _evaluate_diversity(self, configuration: EnsembleConfiguration) -> float:
        """多様性評価"""
        try:
            selected_models = configuration.model_selection

            if len(selected_models) <= 1:
                return 0.0

            # モデル間の多様性を重みの分散で評価
            weights = [configuration.weights.get(model_id, 0.0) for model_id in selected_models]
            weight_variance = np.var(weights) if len(weights) > 1 else 0.0

            # モデル数ボーナス
            model_count_bonus = min(len(selected_models) / 10.0, 1.0)

            # 多様性スコア
            diversity_score = weight_variance + model_count_bonus

            return diversity_score

        except Exception as e:
            logger.warning(f"Diversity evaluation error: {e}")
            return 0.0

    def _evaluate_efficiency(self, configuration: EnsembleConfiguration) -> float:
        """効率性評価"""
        try:
            # モデル数ペナルティ（少ないほど良い）
            model_count = len(configuration.model_selection)
            count_penalty = 1.0 / (1.0 + model_count * 0.1)

            # 重み集中度（集中しているほど効率的）
            weights = list(configuration.weights.values())
            if weights:
                weight_concentration = max(weights) if weights else 0.0
            else:
                weight_concentration = 0.0

            # 効率性スコア
            efficiency_score = count_penalty * 0.5 + weight_concentration * 0.5

            return efficiency_score

        except Exception as e:
            logger.warning(f"Efficiency evaluation error: {e}")
            return 0.1
