#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Hyperparameter Optimizer - 改善版ハイパーパラメータ最適化システム
Issue #856対応：最適化戦略と設定の柔軟性強化

主要改善点:
1. 最適化手法の選択とフォールバックロジックの明確化
2. ハイパーパラメータ空間定義の外部化
3. 設定の動的管理と拡張性
4. エラーハンドリングとロギングの強化
5. 最適化結果の詳細分析
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import warnings
from enum import Enum
warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 最適化ライブラリ
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from scipy.stats import randint, uniform, loguniform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ModelType(Enum):
    """サポートされるモデル種別"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class PredictionTask(Enum):
    """予測タスク種別"""
    PRICE_DIRECTION = "classification"
    PRICE_REGRESSION = "regression"


class OptimizationMethod(Enum):
    """最適化手法"""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    ADAPTIVE = "adaptive"  # 新機能：適応的最適化


@dataclass
class OptimizationConfig:
    """最適化設定"""
    method: OptimizationMethod
    cv_folds: int = 5
    n_jobs: int = -1
    max_iterations: int = 100
    random_state: int = 42
    scoring_timeout: int = 300  # 秒
    early_stopping_rounds: int = 10
    n_iter_random: int = 50  # RandomizedSearchCV用
    verbose: int = 1

    # 新機能：適応的設定
    adaptive_budget: int = 100  # 適応的最適化の予算
    exploration_ratio: float = 0.3  # 探索の割合


@dataclass
class OptimizationResult:
    """最適化結果"""
    model_type: str
    task: str
    method: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    optimization_time: float
    improvement: float
    param_importance: Dict[str, float]
    convergence_curve: List[float] = field(default_factory=list)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)


class HyperparameterSpaceManager:
    """ハイパーパラメータ空間管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/hyperparameter_spaces.yaml")
        self.spaces = self._load_hyperparameter_spaces()

    def _load_hyperparameter_spaces(self) -> Dict[str, Dict]:
        """外部設定ファイルからハイパーパラメータ空間を読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    spaces = yaml.safe_load(f)
                self.logger.info(f"Loaded hyperparameter spaces from {self.config_path}")
                return spaces
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using default spaces")
                return self._get_default_spaces()
        except Exception as e:
            self.logger.error(f"Error loading hyperparameter spaces: {e}")
            return self._get_default_spaces()

    def _get_default_spaces(self) -> Dict[str, Dict]:
        """デフォルトのハイパーパラメータ空間"""
        return {
            "random_forest": {
                "classifier": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [10, 15, 20, 25, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.3, 0.5],
                    "class_weight": ["balanced", "balanced_subsample", None],
                    "bootstrap": [True, False]
                },
                "regressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [10, 15, 20, 25, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.3, 0.5],
                    "bootstrap": [True, False]
                }
            },
            "xgboost": {
                "classifier": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "gamma": [0, 0.1, 0.2, 0.5],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0]
                },
                "regressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "gamma": [0, 0.1, 0.2, 0.5],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0]
                }
            },
            "lightgbm": {
                "classifier": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10, -1],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0],
                    "num_leaves": [15, 31, 63, 127]
                },
                "regressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10, -1],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5, 1.0],
                    "reg_lambda": [0, 0.1, 0.5, 1.0],
                    "num_leaves": [15, 31, 63, 127]
                }
            }
        }

    def get_param_space(self, model_type: ModelType, task: PredictionTask) -> Dict[str, Any]:
        """指定されたモデルとタスクのパラメータ空間を取得"""
        task_key = "classifier" if task == PredictionTask.PRICE_DIRECTION else "regressor"
        return self.spaces.get(model_type.value, {}).get(task_key, {})

    def save_spaces(self):
        """現在のパラメータ空間を設定ファイルに保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.spaces, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Saved hyperparameter spaces to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving hyperparameter spaces: {e}")


class ImprovedHyperparameterOptimizer:
    """改善版ハイパーパラメータ最適化システム"""

    def __init__(self, config_path: Optional[Path] = None,
                 results_db_path: str = "data/optimization_results.db"):
        self.logger = logging.getLogger(__name__)

        # ハイパーパラメータ空間管理
        self.space_manager = HyperparameterSpaceManager(config_path)

        # 最適化設定
        self.optimization_configs = self._load_optimization_configs()

        # 結果保存用データベース
        self.results_db_path = results_db_path
        self._init_results_database()

        # キャッシュ
        self.optimized_params = {}
        self.optimization_history = []

        self.logger.info("Improved Hyperparameter Optimizer initialized")

    def _load_optimization_configs(self) -> Dict[str, OptimizationConfig]:
        """最適化設定の読み込み"""
        return {
            'random': OptimizationConfig(
                method=OptimizationMethod.RANDOM,
                cv_folds=5,
                n_jobs=-1,
                max_iterations=100,
                n_iter_random=50,
                random_state=42
            ),
            'grid': OptimizationConfig(
                method=OptimizationMethod.GRID,
                cv_folds=5,
                n_jobs=-1,
                max_iterations=50,
                random_state=42
            ),
            'bayesian': OptimizationConfig(
                method=OptimizationMethod.BAYESIAN,
                cv_folds=5,
                n_jobs=-1,
                max_iterations=100,
                random_state=42
            ),
            'adaptive': OptimizationConfig(
                method=OptimizationMethod.ADAPTIVE,
                cv_folds=5,
                n_jobs=-1,
                max_iterations=150,
                adaptive_budget=100,
                exploration_ratio=0.3,
                random_state=42
            )
        }

    def _init_results_database(self):
        """結果保存用データベース初期化"""
        try:
            Path(self.results_db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.results_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        model_type TEXT,
                        task TEXT,
                        method TEXT,
                        best_params TEXT,
                        best_score REAL,
                        improvement REAL,
                        optimization_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        optimization_id INTEGER,
                        iteration INTEGER,
                        params TEXT,
                        score REAL,
                        FOREIGN KEY (optimization_id) REFERENCES optimization_results (id)
                    )
                ''')

            self.logger.info("Optimization results database initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize results database: {e}")

    async def optimize_model(self, symbol: str, model_type: ModelType, task: PredictionTask,
                           X: pd.DataFrame, y: pd.Series,
                           baseline_score: float = 0.5,
                           method: str = 'random') -> OptimizationResult:
        """改善版モデル最適化実行"""

        self.logger.info(f"Starting optimization: {model_type.value} for {task.value} using {method}")

        start_time = datetime.now()

        try:
            # 最適化設定取得
            config = self.optimization_configs.get(method)
            if not config:
                self.logger.warning(f"Unknown method '{method}', using 'random'")
                config = self.optimization_configs['random']
                method = 'random'

            # モデル初期化
            model = self._create_model(model_type, task)

            # パラメータ空間取得
            param_space = self.space_manager.get_param_space(model_type, task)
            if not param_space:
                raise ValueError(f"No parameter space defined for {model_type.value} {task.value}")

            # スコアリング設定
            scoring = self._get_scoring_function(task)

            # 最適化実行
            optimizer = self._create_optimizer(method, model, param_space, config, scoring)

            # 最適化実行とモニタリング
            optimization_result = await self._run_optimization_with_monitoring(
                optimizer, X, y, symbol, model_type, task, method, baseline_score, start_time
            )

            self.logger.info(f"Optimization completed: {optimization_result.best_score:.4f} "
                           f"(improvement: {optimization_result.improvement:.2f}%)")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Optimization failed for {symbol}: {e}")
            raise

    def _create_model(self, model_type: ModelType, task: PredictionTask):
        """モデル作成"""
        if model_type == ModelType.RANDOM_FOREST:
            if task == PredictionTask.PRICE_DIRECTION:
                return RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                return RandomForestRegressor(random_state=42, n_jobs=-1)

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            if task == PredictionTask.PRICE_DIRECTION:
                return xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
            else:
                return xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')

        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            if task == PredictionTask.PRICE_DIRECTION:
                return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            else:
                return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _get_scoring_function(self, task: PredictionTask) -> str:
        """スコアリング関数取得"""
        if task == PredictionTask.PRICE_DIRECTION:
            return 'accuracy'
        else:
            return 'r2'

    def _create_optimizer(self, method: str, model, param_space: Dict,
                         config: OptimizationConfig, scoring: str):
        """最適化器作成 - 明確なフォールバックロジック実装"""

        cv = TimeSeriesSplit(n_splits=config.cv_folds)

        if method == 'grid':
            # Grid Search - パラメータ空間を制限
            limited_space = self._limit_param_space_for_grid(param_space, max_combinations=1000)
            return GridSearchCV(
                model, limited_space, cv=cv, scoring=scoring,
                n_jobs=config.n_jobs, verbose=config.verbose
            )

        elif method == 'random':
            # Random Search - 明確にRandomizedSearchCVを使用
            if SCIPY_AVAILABLE:
                # scipyが利用可能な場合は分布を使用
                random_space = self._convert_to_random_distributions(param_space)
                self.logger.info("Using RandomizedSearchCV with scipy distributions")
            else:
                # scipyが利用不可能な場合はリストベース
                random_space = param_space
                self.logger.info("Using RandomizedSearchCV with list-based parameters (scipy not available)")

            return RandomizedSearchCV(
                model, random_space, n_iter=config.n_iter_random,
                cv=cv, scoring=scoring, n_jobs=config.n_jobs,
                random_state=config.random_state, verbose=config.verbose
            )

        elif method == 'bayesian' and BAYESIAN_AVAILABLE:
            # Bayesian Optimization
            bayesian_space = self._convert_to_bayesian_space(param_space)
            return BayesSearchCV(
                model, bayesian_space, n_iter=config.max_iterations,
                cv=cv, scoring=scoring, n_jobs=config.n_jobs,
                random_state=config.random_state, verbose=config.verbose
            )

        elif method == 'adaptive':
            # 適応的最適化（Random -> Bayesian）
            return self._create_adaptive_optimizer(
                model, param_space, config, scoring, cv
            )

        else:
            # フォールバック: Random Search
            self.logger.warning(f"Method '{method}' not available, falling back to random search")
            random_space = param_space
            return RandomizedSearchCV(
                model, random_space, n_iter=config.n_iter_random,
                cv=cv, scoring=scoring, n_jobs=config.n_jobs,
                random_state=config.random_state, verbose=config.verbose
            )

    def _convert_to_random_distributions(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """scipyの分布を使ったRandom Search用パラメータ空間変換"""
        random_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                # 数値のみの場合
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]

                if len(numeric_values) > 2 and all(isinstance(v, int) for v in numeric_values):
                    # 整数パラメータ - 範囲指定
                    random_space[param] = randint(min(numeric_values), max(numeric_values) + 1)
                elif len(numeric_values) > 2 and all(isinstance(v, float) for v in numeric_values):
                    # 浮動小数点パラメータ - 範囲指定
                    min_val, max_val = min(numeric_values), max(numeric_values)
                    random_space[param] = uniform(min_val, max_val - min_val)
                else:
                    # カテゴリカルまたは少数の値
                    random_space[param] = values
            else:
                random_space[param] = values

        return random_space

    def _convert_to_bayesian_space(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """Bayesian Optimization用パラメータ空間変換"""
        bayesian_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]

                if len(numeric_values) > 2 and all(isinstance(v, int) for v in numeric_values):
                    # 整数パラメータ
                    bayesian_space[param] = Integer(min(numeric_values), max(numeric_values))
                elif len(numeric_values) > 2 and all(isinstance(v, float) for v in numeric_values):
                    # 浮動小数点パラメータ
                    bayesian_space[param] = Real(min(numeric_values), max(numeric_values))
                else:
                    # カテゴリカルパラメータ
                    bayesian_space[param] = Categorical(values)
            else:
                bayesian_space[param] = Categorical([values])

        return bayesian_space

    def _limit_param_space_for_grid(self, param_space: Dict[str, List],
                                   max_combinations: int = 1000) -> Dict[str, List]:
        """Grid Search用パラメータ空間制限"""
        total_combinations = 1
        for values in param_space.values():
            if isinstance(values, list):
                total_combinations *= len(values)

        if total_combinations <= max_combinations:
            return param_space

        # 組み合わせが多すぎる場合は各パラメータの値を制限
        limited_space = {}
        reduction_factor = (max_combinations / total_combinations) ** (1 / len(param_space))

        for param, values in param_space.items():
            if isinstance(values, list):
                max_values = max(2, int(len(values) * reduction_factor))
                if len(values) > max_values:
                    step = len(values) // max_values
                    limited_space[param] = values[::step][:max_values]
                else:
                    limited_space[param] = values
            else:
                limited_space[param] = values

        return limited_space

    def _create_adaptive_optimizer(self, model, param_space: Dict, config: OptimizationConfig,
                                 scoring: str, cv):
        """適応的最適化器作成（Random -> Bayesian）"""
        # まずRandom Searchで初期探索
        exploration_iters = int(config.adaptive_budget * config.exploration_ratio)

        if BAYESIAN_AVAILABLE:
            bayesian_space = self._convert_to_bayesian_space(param_space)
            optimizer = BayesSearchCV(
                model, bayesian_space, n_iter=config.adaptive_budget,
                cv=cv, scoring=scoring, n_jobs=config.n_jobs,
                random_state=config.random_state, verbose=config.verbose
            )
            # BayesSearchCVは内部でランダム探索から始まるため
            return optimizer
        else:
            # Bayesianが利用不可能な場合はRandom Search
            random_space = self._convert_to_random_distributions(param_space) if SCIPY_AVAILABLE else param_space
            return RandomizedSearchCV(
                model, random_space, n_iter=config.adaptive_budget,
                cv=cv, scoring=scoring, n_jobs=config.n_jobs,
                random_state=config.random_state, verbose=config.verbose
            )

    async def _run_optimization_with_monitoring(self, optimizer, X: pd.DataFrame, y: pd.Series,
                                              symbol: str, model_type: ModelType, task: PredictionTask,
                                              method: str, baseline_score: float, start_time: datetime) -> OptimizationResult:
        """最適化実行とモニタリング"""

        # 最適化実行
        optimizer.fit(X, y)

        # 結果分析
        best_score = optimizer.best_score_
        best_params = optimizer.best_params_
        cv_scores = list(optimizer.cv_results_['mean_test_score'])

        # 改善率計算
        improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

        # パラメータ重要度計算
        param_importance = self._calculate_param_importance(optimizer.cv_results_)

        # 最適化時間
        optimization_time = (datetime.now() - start_time).total_seconds()

        # 最適化履歴
        optimization_history = self._extract_optimization_history(optimizer.cv_results_)

        # 収束曲線
        convergence_curve = self._calculate_convergence_curve(cv_scores)

        # 検証スコア
        validation_scores = self._calculate_validation_scores(optimizer, X, y, task)

        # 結果作成
        result = OptimizationResult(
            model_type=model_type.value,
            task=task.value,
            method=method,
            best_params=best_params,
            best_score=best_score,
            cv_scores=cv_scores,
            optimization_time=optimization_time,
            improvement=improvement,
            param_importance=param_importance,
            convergence_curve=convergence_curve,
            validation_scores=validation_scores,
            optimization_history=optimization_history
        )

        # 結果保存
        await self._save_optimization_result(result, symbol)

        return result

    def _calculate_param_importance(self, cv_results: Dict) -> Dict[str, float]:
        """パラメータ重要度計算"""
        try:
            param_importance = {}

            # 各パラメータの値とスコアの相関を計算
            for param_name in cv_results['params'][0].keys():
                param_values = []
                scores = []

                for i, params in enumerate(cv_results['params']):
                    param_val = params.get(param_name)
                    if param_val is not None:
                        # 数値パラメータのみ相関計算
                        if isinstance(param_val, (int, float)):
                            param_values.append(param_val)
                            scores.append(cv_results['mean_test_score'][i])

                if len(param_values) > 1:
                    correlation = np.corrcoef(param_values, scores)[0, 1]
                    param_importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    param_importance[param_name] = 0.0

            return param_importance

        except Exception as e:
            self.logger.warning(f"Failed to calculate parameter importance: {e}")
            return {}

    def _extract_optimization_history(self, cv_results: Dict) -> List[Dict]:
        """最適化履歴抽出"""
        history = []
        for i, (params, score) in enumerate(zip(cv_results['params'], cv_results['mean_test_score'])):
            history.append({
                'iteration': i + 1,
                'params': params,
                'score': score,
                'std_score': cv_results.get('std_test_score', [0] * len(cv_results['params']))[i]
            })
        return history

    def _calculate_convergence_curve(self, cv_scores: List[float]) -> List[float]:
        """収束曲線計算"""
        convergence = []
        best_so_far = float('-inf')

        for score in cv_scores:
            if score > best_so_far:
                best_so_far = score
            convergence.append(best_so_far)

        return convergence

    def _calculate_validation_scores(self, optimizer, X: pd.DataFrame, y: pd.Series,
                                   task: PredictionTask) -> Dict[str, float]:
        """検証スコア計算"""
        try:
            best_model = optimizer.best_estimator_

            # 訓練データでの予測
            y_pred = best_model.predict(X)

            if task == PredictionTask.PRICE_DIRECTION:
                # 分類タスク
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                return {
                    'train_accuracy': accuracy_score(y, y_pred),
                    'train_precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                    'train_recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                    'train_f1': f1_score(y, y_pred, average='weighted', zero_division=0)
                }
            else:
                # 回帰タスク
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                return {
                    'train_r2': r2_score(y, y_pred),
                    'train_mse': mean_squared_error(y, y_pred),
                    'train_mae': mean_absolute_error(y, y_pred)
                }
        except Exception as e:
            self.logger.warning(f"Failed to calculate validation scores: {e}")
            return {}

    async def _save_optimization_result(self, result: OptimizationResult, symbol: str):
        """最適化結果保存"""
        try:
            with sqlite3.connect(self.results_db_path) as conn:
                cursor = conn.cursor()

                # メイン結果保存
                cursor.execute('''
                    INSERT INTO optimization_results
                    (symbol, model_type, task, method, best_params, best_score, improvement, optimization_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, result.model_type, result.task, result.method,
                    json.dumps(result.best_params), result.best_score,
                    result.improvement, result.optimization_time
                ))

                optimization_id = cursor.lastrowid

                # 最適化履歴保存
                for history_item in result.optimization_history:
                    cursor.execute('''
                        INSERT INTO optimization_history
                        (optimization_id, iteration, params, score)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        optimization_id, history_item['iteration'],
                        json.dumps(history_item['params']), history_item['score']
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to save optimization result: {e}")

    async def get_optimization_results(self, symbol: Optional[str] = None,
                                     model_type: Optional[str] = None,
                                     limit: int = 100) -> List[Dict]:
        """最適化結果取得"""
        try:
            with sqlite3.connect(self.results_db_path) as conn:
                query = "SELECT * FROM optimization_results WHERE 1=1"
                params = []

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                if model_type:
                    query += " AND model_type = ?"
                    params.append(model_type)

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor = conn.cursor()
                cursor.execute(query, params)

                columns = [description[0] for description in cursor.description]
                results = []

                for row in cursor.fetchall():
                    result_dict = dict(zip(columns, row))
                    result_dict['best_params'] = json.loads(result_dict['best_params'])
                    results.append(result_dict)

                return results

        except Exception as e:
            self.logger.error(f"Failed to get optimization results: {e}")
            return []

    def get_best_params(self, symbol: str, model_type: ModelType, task: PredictionTask) -> Optional[Dict[str, Any]]:
        """最適パラメータ取得"""
        cache_key = f"{symbol}_{model_type.value}_{task.value}"
        return self.optimized_params.get(cache_key)


# テスト関数
async def test_improved_hyperparameter_optimizer():
    """改善版ハイパーパラメータ最適化システムのテスト"""
    print("=== Improved Hyperparameter Optimizer Test ===")

    try:
        # オプティマイザー初期化
        optimizer = ImprovedHyperparameterOptimizer()
        print("✓ Optimizer initialized")

        # テストデータ生成
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = pd.DataFrame(np.random.randn(n_samples, n_features),
                        columns=[f'feature_{i}' for i in range(n_features)])
        y_class = pd.Series(np.random.choice([0, 1], n_samples))
        y_reg = pd.Series(np.random.randn(n_samples))

        print(f"✓ Test data generated: {X.shape}")

        # 分類タスクの最適化テスト
        print("\n--- Random Forest Classification Test ---")
        result_class = await optimizer.optimize_model(
            symbol="TEST_SYMBOL",
            model_type=ModelType.RANDOM_FOREST,
            task=PredictionTask.PRICE_DIRECTION,
            X=X, y=y_class,
            baseline_score=0.5,
            method='random'
        )

        print(f"Best score: {result_class.best_score:.4f}")
        print(f"Improvement: {result_class.improvement:.2f}%")
        print(f"Optimization time: {result_class.optimization_time:.2f}s")
        print(f"Parameter importance: {list(result_class.param_importance.keys())}")

        # 回帰タスクの最適化テスト
        print("\n--- XGBoost Regression Test ---")
        if XGBOOST_AVAILABLE:
            result_reg = await optimizer.optimize_model(
                symbol="TEST_SYMBOL",
                model_type=ModelType.XGBOOST,
                task=PredictionTask.PRICE_REGRESSION,
                X=X, y=y_reg,
                baseline_score=0.0,
                method='random'
            )

            print(f"Best score: {result_reg.best_score:.4f}")
            print(f"Improvement: {result_reg.improvement:.2f}%")
            print(f"Optimization time: {result_reg.optimization_time:.2f}s")
        else:
            print("XGBoost not available, skipping regression test")

        # 結果取得テスト
        print("\n--- Results Retrieval Test ---")
        results = await optimizer.get_optimization_results(symbol="TEST_SYMBOL", limit=5)
        print(f"✓ Retrieved {len(results)} optimization results")

        for result in results:
            print(f"  - {result['model_type']} {result['task']}: score={result['best_score']:.4f}")

        print("\n✓ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(test_improved_hyperparameter_optimizer())