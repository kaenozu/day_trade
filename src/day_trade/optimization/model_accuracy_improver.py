#!/usr/bin/env python3
"""
Model Accuracy Improver
モデル精度改善システム

This module implements advanced techniques to improve model accuracy
through automated hyperparameter tuning, ensemble methods, and deep learning optimization.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import numpy as np
import pandas as pd
import logging
import time
import joblib
from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, validation_curve,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor, BaggingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LinearRegression,
    BayesianRidge, SGDRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

from ..utils.error_handling import TradingResult


class OptimizationMethod(Enum):
    """最適化手法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    OPTUNA = "optuna"
    HYPEROPT = "hyperopt"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"


class EnsembleMethod(Enum):
    """アンサンブル手法"""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    DYNAMIC_SELECTION = "dynamic_selection"


class ValidationMethod(Enum):
    """検証手法"""
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"
    PURGED_CROSS_VALIDATION = "purged_cross_validation"
    BLOCKED_CROSS_VALIDATION = "blocked_cross_validation"
    MONTE_CARLO = "monte_carlo"


@dataclass
class ModelConfiguration:
    """モデル設定"""
    model_name: str
    model_class: Any
    param_space: Dict[str, Any]
    optimization_method: OptimizationMethod
    validation_method: ValidationMethod
    scoring_metric: str
    early_stopping_rounds: Optional[int] = None
    feature_importance_method: Optional[str] = None


@dataclass
class OptimizationResult:
    """最適化結果"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    optimization_time: float
    n_trials: int
    improvement_percent: float
    feature_importance: Dict[str, float]
    model_artifact: Any


@dataclass
class EnsembleResult:
    """アンサンブル結果"""
    ensemble_id: str
    ensemble_method: EnsembleMethod
    component_models: List[str]
    ensemble_score: float
    component_scores: List[float]
    weights: Optional[List[float]]
    ensemble_model: Any
    improvement_over_best_single: float


@dataclass
class AccuracyImprovement:
    """精度改善結果"""
    improvement_id: str
    baseline_score: float
    final_score: float
    improvement_percent: float
    methods_applied: List[str]
    optimization_results: List[OptimizationResult]
    ensemble_result: Optional[EnsembleResult]
    total_training_time: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedHyperparameterOptimizer:
    """高度なハイパーパラメータ最適化"""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=10000)
        self.best_params_cache = {}
        self.logger = logging.getLogger(__name__)
        
    async def optimize_with_optuna(self, model_config: ModelConfiguration, 
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 n_trials: int = 100) -> OptimizationResult:
        """Optunaを使用したベイズ最適化"""
        try:
            def objective(trial):
                # パラメータサンプリング
                params = {}
                for param_name, param_config in model_config.param_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                
                # モデル作成と評価
                model = model_config.model_class(**params)
                
                # Cross validation
                cv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring=model_config.scoring_metric,
                    n_jobs=-1
                )
                
                return cv_scores.mean()
            
            start_time = time.time()
            
            # Optunaの最適化実行
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials)
            
            optimization_time = time.time() - start_time
            
            # 最良モデルで特徴量重要度計算
            best_model = model_config.model_class(**study.best_params)
            best_model.fit(X_train, y_train)
            
            feature_importance = self._get_feature_importance(best_model, X_train.columns)
            
            return OptimizationResult(
                model_name=model_config.model_name,
                best_params=study.best_params,
                best_score=study.best_value,
                cv_scores=[],  # Optunaでは個別のCV scoreは保存されない
                optimization_time=optimization_time,
                n_trials=len(study.trials),
                improvement_percent=0.0,  # ベースラインとの比較で計算
                feature_importance=feature_importance,
                model_artifact=best_model
            )
            
        except Exception as e:
            self.logger.error(f"Optuna optimization failed for {model_config.model_name}: {e}")
            raise e
    
    async def optimize_with_bayesian(self, model_config: ModelConfiguration,
                                   X_train: pd.DataFrame, y_train: pd.Series,
                                   n_iter: int = 50) -> OptimizationResult:
        """ベイズ最適化"""
        try:
            # パラメータ範囲をBayesian Optimizationの形式に変換
            pbounds = {}
            for param_name, param_config in model_config.param_space.items():
                if param_config['type'] in ['int', 'float']:
                    pbounds[param_name] = (param_config['low'], param_config['high'])
            
            def objective_function(**params):
                try:
                    # 整数パラメータの調整
                    adjusted_params = {}
                    for param_name, param_config in model_config.param_space.items():
                        if param_name in params:
                            if param_config['type'] == 'int':
                                adjusted_params[param_name] = int(round(params[param_name]))
                            else:
                                adjusted_params[param_name] = params[param_name]
                        elif param_config['type'] == 'categorical':
                            # カテゴリカルパラメータはデフォルト値を使用
                            adjusted_params[param_name] = param_config['choices'][0]
                    
                    model = model_config.model_class(**adjusted_params)
                    
                    cv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv, scoring=model_config.scoring_metric,
                        n_jobs=-1
                    )
                    
                    return cv_scores.mean()
                    
                except Exception as e:
                    self.logger.error(f"Objective function error: {e}")
                    return -np.inf
            
            start_time = time.time()
            
            optimizer = BayesianOptimization(
                f=objective_function,
                pbounds=pbounds,
                random_state=42,
                verbose=0
            )
            
            optimizer.maximize(
                init_points=min(10, n_iter // 5),
                n_iter=n_iter
            )
            
            optimization_time = time.time() - start_time
            
            # 最良パラメータの調整
            best_params = {}
            for param_name, param_config in model_config.param_space.items():
                if param_name in optimizer.max['params']:
                    if param_config['type'] == 'int':
                        best_params[param_name] = int(round(optimizer.max['params'][param_name]))
                    else:
                        best_params[param_name] = optimizer.max['params'][param_name]
                elif param_config['type'] == 'categorical':
                    best_params[param_name] = param_config['choices'][0]
            
            # 最良モデルで特徴量重要度計算
            best_model = model_config.model_class(**best_params)
            best_model.fit(X_train, y_train)
            
            feature_importance = self._get_feature_importance(best_model, X_train.columns)
            
            return OptimizationResult(
                model_name=model_config.model_name,
                best_params=best_params,
                best_score=optimizer.max['target'],
                cv_scores=[],
                optimization_time=optimization_time,
                n_trials=len(optimizer.space),
                improvement_percent=0.0,
                feature_importance=feature_importance,
                model_artifact=best_model
            )
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed for {model_config.model_name}: {e}")
            raise e
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度の取得"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                return dict(zip(feature_names, importances))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Feature importance extraction failed: {e}")
            return {}


class AdvancedEnsembleOptimizer:
    """高度なアンサンブル最適化"""
    
    def __init__(self):
        self.ensemble_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    async def create_stacking_ensemble(self, base_models: List[Any], 
                                     meta_model: Any,
                                     X_train: pd.DataFrame, y_train: pd.Series,
                                     X_val: pd.DataFrame, y_val: pd.Series) -> EnsembleResult:
        """スタッキングアンサンブルの作成"""
        try:
            start_time = time.time()
            
            # メタ特徴量の生成（Out-of-fold predictions）
            meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
            meta_features_val = np.zeros((X_val.shape[0], len(base_models)))
            
            cv = TimeSeriesSplit(n_splits=5)
            
            for i, model in enumerate(base_models):
                oof_predictions = np.zeros(X_train.shape[0])
                val_predictions = []
                
                for train_idx, valid_idx in cv.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_valid = X_train.iloc[valid_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    oof_predictions[valid_idx] = model.predict(X_fold_valid)
                    val_predictions.append(model.predict(X_val))
                
                meta_features_train[:, i] = oof_predictions
                meta_features_val[:, i] = np.mean(val_predictions, axis=0)
            
            # メタモデルの訓練
            meta_model.fit(meta_features_train, y_train)
            
            # アンサンブルの予測
            ensemble_pred_val = meta_model.predict(meta_features_val)
            ensemble_score = r2_score(y_val, ensemble_pred_val)
            
            # 個別モデルのスコア
            component_scores = []
            for model in base_models:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred)
                component_scores.append(score)
            
            best_single_score = max(component_scores)
            improvement = ((ensemble_score - best_single_score) / best_single_score * 100) if best_single_score > 0 else 0
            
            # スタッキングモデルの作成
            stacking_model = StackingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                final_estimator=meta_model,
                cv=cv
            )
            stacking_model.fit(X_train, y_train)
            
            return EnsembleResult(
                ensemble_id=f"stacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ensemble_method=EnsembleMethod.STACKING,
                component_models=[type(model).__name__ for model in base_models],
                ensemble_score=ensemble_score,
                component_scores=component_scores,
                weights=None,  # スタッキングでは明示的な重みなし
                ensemble_model=stacking_model,
                improvement_over_best_single=improvement
            )
            
        except Exception as e:
            self.logger.error(f"Stacking ensemble creation failed: {e}")
            raise e
    
    async def create_weighted_ensemble(self, models: List[Any], 
                                     X_train: pd.DataFrame, y_train: pd.Series,
                                     X_val: pd.DataFrame, y_val: pd.Series,
                                     optimization_method: str = 'optuna') -> EnsembleResult:
        """重み付きアンサンブルの作成"""
        try:
            # 各モデルの予測を取得
            predictions = []
            component_scores = []
            
            for model in models:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                predictions.append(pred)
                score = r2_score(y_val, pred)
                component_scores.append(score)
            
            predictions = np.array(predictions).T
            
            if optimization_method == 'optuna':
                weights = await self._optimize_weights_optuna(predictions, y_val)
            else:
                weights = await self._optimize_weights_scipy(predictions, y_val)
            
            # 重み付きアンサンブルの予測
            ensemble_pred = np.sum(predictions * weights, axis=1)
            ensemble_score = r2_score(y_val, ensemble_pred)
            
            best_single_score = max(component_scores)
            improvement = ((ensemble_score - best_single_score) / best_single_score * 100) if best_single_score > 0 else 0
            
            # 重み付きVotingRegressorの作成
            weighted_ensemble = VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                weights=weights
            )
            weighted_ensemble.fit(X_train, y_train)
            
            return EnsembleResult(
                ensemble_id=f"weighted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ensemble_method=EnsembleMethod.VOTING,
                component_models=[type(model).__name__ for model in models],
                ensemble_score=ensemble_score,
                component_scores=component_scores,
                weights=weights.tolist(),
                ensemble_model=weighted_ensemble,
                improvement_over_best_single=improvement
            )
            
        except Exception as e:
            self.logger.error(f"Weighted ensemble creation failed: {e}")
            raise e
    
    async def _optimize_weights_optuna(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Optunaによる重み最適化"""
        n_models = predictions.shape[1]
        
        def objective(trial):
            weights = []
            for i in range(n_models):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # 正規化
            
            ensemble_pred = np.sum(predictions * weights, axis=1)
            score = r2_score(y_true, ensemble_pred)
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200)
        
        # 最適な重みを取得
        optimal_weights = []
        for i in range(n_models):
            optimal_weights.append(study.best_params[f'weight_{i}'])
        
        optimal_weights = np.array(optimal_weights)
        return optimal_weights / np.sum(optimal_weights)
    
    async def _optimize_weights_scipy(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """scipyによる重み最適化"""
        from scipy.optimize import minimize
        
        n_models = predictions.shape[1]
        
        def objective(weights):
            weights = weights / np.sum(weights)  # 正規化
            ensemble_pred = np.sum(predictions * weights, axis=1)
            mse = mean_squared_error(y_true, ensemble_pred)
            return mse
        
        # 初期重み（等重み）
        initial_weights = np.ones(n_models) / n_models
        
        # 制約（重みの合計が1）
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x


class ModelAccuracyImprover:
    """モデル精度改善システム"""
    
    def __init__(self):
        self.hyperparameter_optimizer = AdvancedHyperparameterOptimizer()
        self.ensemble_optimizer = AdvancedEnsembleOptimizer()
        self.improvement_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    async def comprehensive_accuracy_improvement(self, 
                                               X_train: pd.DataFrame, y_train: pd.Series,
                                               X_val: pd.DataFrame, y_val: pd.Series,
                                               baseline_model: Any = None,
                                               improvement_target: float = 0.1) -> TradingResult[AccuracyImprovement]:
        """包括的精度改善"""
        try:
            self.logger.info("Starting comprehensive accuracy improvement...")
            
            start_time = time.time()
            
            # ベースライン性能の計算
            if baseline_model is None:
                baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_val)
            baseline_score = r2_score(y_val, baseline_pred)
            
            self.logger.info(f"Baseline R² score: {baseline_score:.4f}")
            
            # モデル設定の定義
            model_configs = self._create_model_configurations()
            
            # 個別モデルの最適化
            optimization_results = []
            optimized_models = []
            
            for config in model_configs:
                self.logger.info(f"Optimizing {config.model_name}...")
                
                try:
                    if config.optimization_method == OptimizationMethod.OPTUNA:
                        result = await self.hyperparameter_optimizer.optimize_with_optuna(
                            config, X_train, y_train, n_trials=50
                        )
                    else:
                        result = await self.hyperparameter_optimizer.optimize_with_bayesian(
                            config, X_train, y_train, n_iter=30
                        )
                    
                    # 改善率の計算
                    result.improvement_percent = ((result.best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
                    
                    optimization_results.append(result)
                    optimized_models.append(result.model_artifact)
                    
                    self.logger.info(f"{config.model_name} optimized: R² = {result.best_score:.4f} ({result.improvement_percent:+.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Optimization failed for {config.model_name}: {e}")
                    continue
            
            # 上位モデルを選択してアンサンブル
            top_models = sorted(optimization_results, key=lambda x: x.best_score, reverse=True)[:5]
            top_model_objects = [result.model_artifact for result in top_models]
            
            # アンサンブル最適化
            ensemble_result = None
            if len(top_model_objects) >= 2:
                self.logger.info("Creating optimized ensemble...")
                
                try:
                    # スタッキングアンサンブル
                    from sklearn.linear_model import Ridge
                    meta_model = Ridge(alpha=1.0)
                    
                    ensemble_result = await self.ensemble_optimizer.create_stacking_ensemble(
                        top_model_objects, meta_model, X_train, y_train, X_val, y_val
                    )
                    
                    self.logger.info(f"Ensemble R² score: {ensemble_result.ensemble_score:.4f} ({ensemble_result.improvement_over_best_single:+.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Ensemble creation failed: {e}")
            
            # 最終スコア計算
            final_score = ensemble_result.ensemble_score if ensemble_result else max(result.best_score for result in optimization_results)
            improvement_percent = ((final_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            total_training_time = time.time() - start_time
            
            # 推奨事項の生成
            recommendations = self._generate_recommendations(
                optimization_results, ensemble_result, improvement_percent, improvement_target
            )
            
            # 適用手法のリスト
            methods_applied = [
                'hyperparameter_optimization',
                'cross_validation',
                'feature_importance_analysis'
            ]
            
            if ensemble_result:
                methods_applied.append('ensemble_modeling')
            
            # 結果の作成
            improvement = AccuracyImprovement(
                improvement_id=f"improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                baseline_score=baseline_score,
                final_score=final_score,
                improvement_percent=improvement_percent,
                methods_applied=methods_applied,
                optimization_results=optimization_results,
                ensemble_result=ensemble_result,
                total_training_time=total_training_time,
                recommendations=recommendations
            )
            
            # 履歴に記録
            self.improvement_history.append(improvement)
            
            self.logger.info(f"Accuracy improvement completed: {improvement_percent:+.1f}% improvement")
            
            return TradingResult.success(improvement)
            
        except Exception as e:
            self.logger.error(f"Comprehensive accuracy improvement failed: {e}")
            return TradingResult.failure(f"Accuracy improvement error: {e}")
    
    def _create_model_configurations(self) -> List[ModelConfiguration]:
        """モデル設定の作成"""
        return [
            # Random Forest
            ModelConfiguration(
                model_name="random_forest",
                model_class=RandomForestRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
                },
                optimization_method=OptimizationMethod.OPTUNA,
                validation_method=ValidationMethod.TIME_SERIES_SPLIT,
                scoring_metric='r2'
            ),
            
            # XGBoost
            ModelConfiguration(
                model_name="xgboost",
                model_class=xgb.XGBRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 10.0},
                    'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0}
                },
                optimization_method=OptimizationMethod.OPTUNA,
                validation_method=ValidationMethod.TIME_SERIES_SPLIT,
                scoring_metric='r2'
            ),
            
            # LightGBM
            ModelConfiguration(
                model_name="lightgbm",
                model_class=lgb.LGBMRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 10.0},
                    'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 10.0}
                },
                optimization_method=OptimizationMethod.OPTUNA,
                validation_method=ValidationMethod.TIME_SERIES_SPLIT,
                scoring_metric='r2'
            ),
            
            # Gradient Boosting
            ModelConfiguration(
                model_name="gradient_boosting",
                model_class=GradientBoostingRegressor,
                param_space={
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
                },
                optimization_method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
                validation_method=ValidationMethod.TIME_SERIES_SPLIT,
                scoring_metric='r2'
            ),
            
            # Support Vector Regression
            ModelConfiguration(
                model_name="svr",
                model_class=SVR,
                param_space={
                    'C': {'type': 'float', 'low': 0.1, 'high': 1000.0, 'log': True},
                    'gamma': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
                    'epsilon': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
                    'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']}
                },
                optimization_method=OptimizationMethod.OPTUNA,
                validation_method=ValidationMethod.TIME_SERIES_SPLIT,
                scoring_metric='r2'
            )
        ]
    
    def _generate_recommendations(self, optimization_results: List[OptimizationResult],
                                ensemble_result: Optional[EnsembleResult],
                                improvement_percent: float,
                                target_percent: float) -> List[str]:
        """推奨事項の生成"""
        recommendations = []
        
        if improvement_percent < target_percent:
            recommendations.append(f"Target improvement of {target_percent:.1f}% not reached. Consider additional feature engineering.")
        
        if len(optimization_results) > 0:
            best_model = max(optimization_results, key=lambda x: x.best_score)
            recommendations.append(f"Best single model: {best_model.model_name} with R² = {best_model.best_score:.4f}")
        
        if ensemble_result and ensemble_result.improvement_over_best_single > 2.0:
            recommendations.append("Ensemble modeling shows significant improvement. Consider using ensemble in production.")
        elif ensemble_result:
            recommendations.append("Ensemble improvement is marginal. Single model might be sufficient.")
        
        # 特徴量重要度に基づく推奨事項
        if optimization_results:
            all_features = set()
            for result in optimization_results:
                all_features.update(result.feature_importance.keys())
            
            if len(all_features) > 50:
                recommendations.append("Consider feature selection to reduce model complexity.")
        
        # 一般的な推奨事項
        recommendations.extend([
            "Regular model retraining recommended as new data becomes available.",
            "Monitor model performance in production for concept drift.",
            "Consider additional data sources for potential feature enhancement."
        ])
        
        return recommendations
    
    async def get_improvement_summary(self) -> Dict[str, Any]:
        """改善サマリーの取得"""
        try:
            if not self.improvement_history:
                return {'status': 'no_improvements_recorded'}
            
            recent_improvements = list(self.improvement_history)[-10:]  # 直近10件
            
            avg_improvement = np.mean([imp.improvement_percent for imp in recent_improvements])
            best_improvement = max([imp.improvement_percent for imp in recent_improvements])
            total_training_time = sum([imp.total_training_time for imp in recent_improvements])
            
            # 成功した手法の集計
            method_counts = defaultdict(int)
            for improvement in recent_improvements:
                for method in improvement.methods_applied:
                    method_counts[method] += 1
            
            return {
                'total_improvements': len(self.improvement_history),
                'recent_improvements_count': len(recent_improvements),
                'average_improvement_percent': avg_improvement,
                'best_improvement_percent': best_improvement,
                'total_training_time_hours': total_training_time / 3600,
                'most_successful_methods': dict(method_counts),
                'last_improvement_date': recent_improvements[-1].timestamp if recent_improvements else None
            }
            
        except Exception as e:
            self.logger.error(f"Improvement summary generation failed: {e}")
            return {'error': str(e)}


# Global instance
model_accuracy_improver = ModelAccuracyImprover()


async def improve_model_accuracy(X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               baseline_model: Any = None,
                               target_improvement: float = 0.1) -> TradingResult[AccuracyImprovement]:
    """モデル精度改善の実行"""
    return await model_accuracy_improver.comprehensive_accuracy_improvement(
        X_train, y_train, X_val, y_val, baseline_model, target_improvement
    )


async def get_accuracy_improvement_summary() -> Dict[str, Any]:
    """精度改善サマリーの取得"""
    return await model_accuracy_improver.get_improvement_summary()