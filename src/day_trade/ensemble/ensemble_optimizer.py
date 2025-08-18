#!/usr/bin/env python3
"""
アンサンブル最適化エンジン
Ensemble Optimization Engine

Issue #762: 高度なアンサンブル予測システムの強化 - Phase 3
"""

import numpy as np
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import copy
import warnings

# 最適化ライブラリ
from scipy.optimize import minimize, differential_evolution

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    gp_minimize = None
    Real = None
    Integer = None
    Categorical = None
    use_named_args = None
    SKOPT_AVAILABLE = False

# 機械学習
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
import optuna

# 遺伝的アルゴリズム（オプショナル）
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    base = creator = tools = algorithms = None

# ログ設定
logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfiguration:
    """アンサンブル構成"""
    model_selection: List[str]
    weights: Dict[str, float]
    hyperparameters: Dict[str, Dict[str, Any]]
    aggregation_method: str = "weighted_average"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    estimated_performance: float = 0.0
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationResult:
    """最適化結果"""
    best_configuration: EnsembleConfiguration
    optimization_score: float
    convergence_history: List[float]
    total_evaluations: int
    optimization_time: float
    method_used: str
    cross_validation_scores: Optional[List[float]] = None

@dataclass
class ModelCandidate:
    """モデル候補"""
    model_id: str
    model_class: str
    hyperparameter_space: Dict[str, Any]
    base_performance: float = 0.0
    complexity_score: float = 1.0
    is_enabled: bool = True

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

class ArchitectureSearch:
    """アーキテクチャ探索"""

    def __init__(self,
                 model_pool: List[ModelCandidate],
                 search_space_size: int = 1000):
        self.model_pool = model_pool
        self.search_space_size = search_space_size
        self.search_history: List[EnsembleConfiguration] = []

        logger.info(f"ArchitectureSearch initialized with {len(model_pool)} model candidates")

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

class EnsembleOptimizer:
    """アンサンブル最適化器"""

    def __init__(self,
                 model_pool: List[ModelCandidate],
                 optimization_budget: int = 100,
                 methods: List[str] = None):
        self.model_pool = model_pool
        self.optimization_budget = optimization_budget
        self.methods = methods or ["architecture_search", "bayesian", "genetic"]

        # コンポーネント初期化
        self.architecture_search = ArchitectureSearch(model_pool)
        self.hyperparameter_tuning = HyperparameterTuning()
        self.automl_integration = AutoMLIntegration()

        # 結果履歴
        self.optimization_history: List[OptimizationResult] = []

        logger.info(f"EnsembleOptimizer initialized with {len(model_pool)} models")

    async def optimize_ensemble(self,
                              training_data: Tuple[np.ndarray, np.ndarray],
                              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              objectives: List[str] = None) -> OptimizationResult:
        """アンサンブル最適化"""
        start_time = time.time()

        try:
            # 目的関数作成
            objective_function = MultiObjectiveFunction(
                training_data, validation_data, objectives or ["accuracy", "diversity", "efficiency"]
            )

            best_configuration = None
            best_score = float('-inf')
            convergence_history = []

            # 予算分配
            budget_per_method = self.optimization_budget // len(self.methods)

            for method in self.methods:
                logger.info(f"Running optimization with {method}")

                if method == "architecture_search":
                    config = await self._optimize_with_architecture_search(
                        objective_function, budget_per_method
                    )
                elif method == "bayesian":
                    config = await self._optimize_with_bayesian(
                        objective_function, budget_per_method
                    )
                elif method == "genetic":
                    config = await self._optimize_with_genetic(
                        objective_function, budget_per_method
                    )
                else:
                    logger.warning(f"Unknown optimization method: {method}")
                    continue

                # 評価
                score = objective_function.evaluate(config)
                convergence_history.append(score)

                if score > best_score:
                    best_score = score
                    best_configuration = copy.deepcopy(config)

            # ハイパーパラメータ最適化
            if best_configuration:
                best_configuration = await self.hyperparameter_tuning.tune(
                    best_configuration, objective_function
                )
                final_score = objective_function.evaluate(best_configuration)
                convergence_history.append(final_score)
            else:
                # フォールバック構成
                best_configuration = self._create_fallback_configuration()
                final_score = objective_function.evaluate(best_configuration)

            # 結果作成
            optimization_time = time.time() - start_time

            result = OptimizationResult(
                best_configuration=best_configuration,
                optimization_score=final_score,
                convergence_history=convergence_history,
                total_evaluations=objective_function.evaluation_count,
                optimization_time=optimization_time,
                method_used="+".join(self.methods)
            )

            self.optimization_history.append(result)

            logger.info(f"Optimization completed in {optimization_time:.2f}s. Score: {final_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"Ensemble optimization error: {e}")
            # エラー時のフォールバック
            fallback_config = self._create_fallback_configuration()
            return OptimizationResult(
                best_configuration=fallback_config,
                optimization_score=0.0,
                convergence_history=[],
                total_evaluations=0,
                optimization_time=time.time() - start_time,
                method_used="fallback"
            )

    async def _optimize_with_architecture_search(self,
                                                objective_function: ObjectiveFunction,
                                                budget: int) -> EnsembleConfiguration:
        """アーキテクチャ探索最適化"""
        return await self.architecture_search.search(objective_function, budget)

    async def _optimize_with_bayesian(self,
                                    objective_function: ObjectiveFunction,
                                    budget: int) -> EnsembleConfiguration:
        """ベイズ最適化"""
        # ベイズ最適化用の設定空間定義
        dimensions = []

        # モデル選択（バイナリ変数）
        for model in self.model_pool[:5]:  # 最初の5モデルのみ
            dimensions.append(Integer(0, 1, name=f"select_{model.model_id}"))

        # 重み（連続変数）
        for model in self.model_pool[:5]:
            dimensions.append(Real(0.0, 1.0, name=f"weight_{model.model_id}"))

        @use_named_args(dimensions)
        def objective(**params):
            # パラメータから構成作成
            selected_models = []
            weights = {}

            for model in self.model_pool[:5]:
                if params.get(f"select_{model.model_id}", 0) > 0:
                    selected_models.append(model.model_id)
                    weights[model.model_id] = params.get(f"weight_{model.model_id}", 0.0)

            if not selected_models:
                return -1.0  # ペナルティ

            # 重み正規化
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}

            # 構成作成
            config = EnsembleConfiguration(
                model_selection=selected_models,
                weights=weights,
                hyperparameters={model_id: {} for model_id in selected_models}
            )

            # 評価（最大化問題に変換）
            return objective_function.evaluate(config)

        # 最適化実行
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = gp_minimize(
                func=lambda x: -objective(x),  # 最小化に変換
                dimensions=dimensions,
                n_calls=budget,
                random_state=42
            )

        # 最良結果から構成作成
        best_params = dict(zip([d.name for d in dimensions], result.x))

        selected_models = []
        weights = {}
        for model in self.model_pool[:5]:
            if best_params.get(f"select_{model.model_id}", 0) > 0:
                selected_models.append(model.model_id)
                weights[model.model_id] = best_params.get(f"weight_{model.model_id}", 0.0)

        if not selected_models:
            return self._create_fallback_configuration()

        # 重み正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return EnsembleConfiguration(
            model_selection=selected_models,
            weights=weights,
            hyperparameters={model_id: {} for model_id in selected_models}
        )

    async def _optimize_with_genetic(self,
                                   objective_function: ObjectiveFunction,
                                   budget: int) -> EnsembleConfiguration:
        """遺伝的アルゴリズム最適化"""
        # 遺伝的アルゴリズムのセットアップ
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # 個体表現：[model_selections..., weights...]
        n_models = min(5, len(self.model_pool))

        # 遺伝子：モデル選択(0/1) + 重み(0-1)
        toolbox.register("select_gene", np.random.randint, 0, 2)
        toolbox.register("weight_gene", np.random.random)

        def create_individual():
            individual = []
            # モデル選択遺伝子
            for _ in range(n_models):
                individual.append(toolbox.select_gene())
            # 重み遺伝子
            for _ in range(n_models):
                individual.append(toolbox.weight_gene())
            return creator.Individual(individual)

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            # 個体から構成作成
            selections = individual[:n_models]
            weights_raw = individual[n_models:]

            selected_models = []
            weights = {}

            for i, (select, weight) in enumerate(zip(selections, weights_raw)):
                if select > 0 and i < len(self.model_pool):
                    model_id = self.model_pool[i].model_id
                    selected_models.append(model_id)
                    weights[model_id] = weight

            if not selected_models:
                return (0.0,)  # フィットネス値はタプル

            # 重み正規化
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}

            # 構成作成・評価
            config = EnsembleConfiguration(
                model_selection=selected_models,
                weights=weights,
                hyperparameters={model_id: {} for model_id in selected_models}
            )

            score = objective_function.evaluate(config)
            return (score,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 進化実行
        population = toolbox.population(n=20)

        # 評価
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 進化ループ
        NGEN = budget // 20
        for gen in range(NGEN):
            # 選択
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 突然変異
            for mutant in offspring:
                if np.random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 評価
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

        # 最良個体選択
        best_ind = tools.selBest(population, 1)[0]

        # 最良個体から構成作成
        selections = best_ind[:n_models]
        weights_raw = best_ind[n_models:]

        selected_models = []
        weights = {}

        for i, (select, weight) in enumerate(zip(selections, weights_raw)):
            if select > 0 and i < len(self.model_pool):
                model_id = self.model_pool[i].model_id
                selected_models.append(model_id)
                weights[model_id] = weight

        if not selected_models:
            return self._create_fallback_configuration()

        # 重み正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return EnsembleConfiguration(
            model_selection=selected_models,
            weights=weights,
            hyperparameters={model_id: {} for model_id in selected_models}
        )

    def _create_fallback_configuration(self) -> EnsembleConfiguration:
        """フォールバック構成作成"""
        # 最初の2つのモデルを均等重みで使用
        available_models = [m for m in self.model_pool if m.is_enabled][:2]

        if not available_models:
            # モデルが無い場合のダミー構成
            return EnsembleConfiguration(
                model_selection=["dummy_model"],
                weights={"dummy_model": 1.0},
                hyperparameters={"dummy_model": {}}
            )

        model_ids = [m.model_id for m in available_models]
        uniform_weight = 1.0 / len(model_ids)
        weights = {mid: uniform_weight for mid in model_ids}

        hyperparameters = {}
        for model in available_models:
            hyperparameters[model.model_id] = {
                param: config.get('default', 1.0)
                for param, config in model.hyperparameter_space.items()
            }

        return EnsembleConfiguration(
            model_selection=model_ids,
            weights=weights,
            hyperparameters=hyperparameters
        )

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """最適化統計取得"""
        if not self.optimization_history:
            return {}

        recent_results = self.optimization_history[-10:]

        return {
            'total_optimizations': len(self.optimization_history),
            'avg_optimization_time': np.mean([r.optimization_time for r in recent_results]),
            'avg_score': np.mean([r.optimization_score for r in recent_results]),
            'best_score': max([r.optimization_score for r in self.optimization_history]),
            'avg_evaluations': np.mean([r.total_evaluations for r in recent_results]),
            'methods_used': list(set([r.method_used for r in recent_results]))
        }

# 便利関数
def create_ensemble_optimizer(
    model_pool: List[ModelCandidate],
    config: Optional[Dict[str, Any]] = None
) -> EnsembleOptimizer:
    """アンサンブル最適化器作成"""
    if config is None:
        config = {}

    return EnsembleOptimizer(
        model_pool=model_pool,
        optimization_budget=config.get('optimization_budget', 100),
        methods=config.get('methods', ["architecture_search", "bayesian"])
    )

def create_sample_model_pool() -> List[ModelCandidate]:
    """サンプルモデルプール作成"""
    return [
        ModelCandidate(
            model_id="rf_model",
            model_class="RandomForest",
            hyperparameter_space={
                'n_estimators': {'type': 'int', 'low': 10, 'high': 200, 'default': 100},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20, 'default': 10}
            },
            base_performance=0.8,
            complexity_score=0.5
        ),
        ModelCandidate(
            model_id="linear_model",
            model_class="LinearRegression",
            hyperparameter_space={
                'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'default': 1.0}
            },
            base_performance=0.6,
            complexity_score=0.2
        ),
        ModelCandidate(
            model_id="neural_model",
            model_class="NeuralNetwork",
            hyperparameter_space={
                'hidden_size': {'type': 'int', 'low': 16, 'high': 128, 'default': 64},
                'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'default': 0.01}
            },
            base_performance=0.75,
            complexity_score=0.8
        )
    ]

async def demo_ensemble_optimization():
    """アンサンブル最適化デモ"""
    # サンプルデータ
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.sum(X[:, :3], axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1

    training_data = (X[:150], y[:150])
    validation_data = (X[150:], y[150:])

    # モデルプール
    model_pool = create_sample_model_pool()

    # 最適化器作成
    optimizer = create_ensemble_optimizer(model_pool)

    print("Starting ensemble optimization...")

    # 最適化実行
    result = await optimizer.optimize_ensemble(training_data, validation_data)

    print(f"Optimization completed!")
    print(f"Best score: {result.optimization_score:.4f}")
    print(f"Selected models: {result.best_configuration.model_selection}")
    print(f"Weights: {result.best_configuration.weights}")
    print(f"Optimization time: {result.optimization_time:.2f}s")

    # 統計表示
    stats = optimizer.get_optimization_statistics()
    print("Optimization Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_ensemble_optimization())