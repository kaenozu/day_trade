#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimizer - ハイパーパラメータ最適化システム（改善版）

Issue #856対応：最適化戦略と設定の柔軟性強化
主な改善点：
1. 最適化手法の選択とフォールバックロジックの明確化
2. ハイパーパラメータ空間定義の外部化
3. パラメータ重要度計算の改善
4. baseline_scoreの調整
5. Windows環境対策コードの集約
6. テストコードの分離
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import warnings
from enum import Enum
warnings.filterwarnings('ignore')

# 共通ユーティリティ
try:
    from src.day_trade.utils.encoding_utils import setup_windows_encoding
    setup_windows_encoding()
except ImportError:
    # フォールバック: 簡易Windows対応
    import sys
    import os
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

# 最適化ライブラリ
try:
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # Bayesian Optimization (オプション)
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# ロギング設定
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """モデルタイプ"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class PredictionTask(Enum):
    """予測タスク"""
    PRICE_DIRECTION = "price_direction"  # 分類
    PRICE_REGRESSION = "price_regression"  # 回帰


class OptimizationMethod(Enum):
    """最適化手法"""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class OptimizationConfig:
    """最適化設定"""
    method: str
    cv_folds: int
    max_iterations: int
    scoring: str
    n_jobs: int
    random_state: int
    n_iter: Optional[int] = None  # RandomizedSearchCV用


@dataclass
class OptimizationResult:
    """最適化結果"""
    model_type: str
    task: str
    optimization_method: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    optimization_time: float
    improvement: float  # ベースラインからの改善率
    param_importance: Dict[str, float]
    baseline_score: float = 0.0
    total_combinations: int = 0


class HyperparameterOptimizer:
    """ハイパーパラメータ最適化システム（改善版）"""

    def __init__(self, config_path: Optional[Path] = None,
                 hyperparameter_config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is required for hyperparameter optimization"
            )

        # 設定ファイルパス
        self.config_path = config_path or Path("config/optimization_config.yaml")
        self.hyperparameter_config_path = hyperparameter_config_path or Path("config/hyperparameter_spaces.yaml")

        # データディレクトリ
        self.data_dir = Path("hyperparameter_optimization")
        self.data_dir.mkdir(exist_ok=True)

        # 設定読み込み
        self.config = {}
        self.hyperparameter_spaces = {}
        self._load_configuration()

        # データベース初期化
        db_path = self.config.get('storage', {}).get('database_path',
                                                  'hyperparameter_optimization/optimization_results.db')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # 最適化設定
        self.optimization_configs = self._load_optimization_configs()

        # 最適化履歴
        self.optimization_history = {}
        self.optimized_params = {}

        self.logger.info("Hyperparameter optimizer initialized with external config")

    def _load_configuration(self):
        """設定ファイル読み込み"""
        try:
            # 最適化設定読み込み
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"最適化設定を読み込みました: {self.config_path}")
            else:
                self.logger.warning(f"最適化設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

            # ハイパーパラメータ空間読み込み
            if self.hyperparameter_config_path.exists():
                with open(self.hyperparameter_config_path, 'r', encoding='utf-8') as f:
                    hyperparameter_config = yaml.safe_load(f)
                    self.hyperparameter_spaces = hyperparameter_config
                self.logger.info(f"ハイパーパラメータ空間を読み込みました: {self.hyperparameter_config_path}")
            else:
                self.logger.warning(f"ハイパーパラメータ設定ファイルが見つかりません: {self.hyperparameter_config_path}")
                self.hyperparameter_spaces = self._define_default_hyperparameter_spaces()

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            self._load_default_settings()

    def _create_default_config(self):
        """デフォルト設定ファイル作成"""
        # デフォルト設定はすでにconfig/optimization_config.yamlで作成済み
        self._load_default_settings()

    def _load_default_settings(self):
        """デフォルト設定読み込み"""
        self.config = {
            'optimization_methods': {
                'grid_search': {
                    'method': 'grid', 'cv_folds': 3, 'max_iterations': 50,
                    'scoring': 'accuracy', 'n_jobs': -1, 'random_state': 42, 'enabled': True
                },
                'random_search': {
                    'method': 'random', 'cv_folds': 3, 'max_iterations': 100,
                    'scoring': 'accuracy', 'n_jobs': -1, 'random_state': 42, 'enabled': True
                }
            },
            'storage': {'database_path': 'hyperparameter_optimization/optimization_results.db'}
        }
        self.hyperparameter_spaces = self._define_default_hyperparameter_spaces()

    def _load_optimization_configs(self) -> Dict[str, OptimizationConfig]:
        """最適化設定をOptimizationConfigオブジェクトに変換"""
        configs = {}

        optimization_methods = self.config.get('optimization_methods', {})

        for method_name, method_config in optimization_methods.items():
            if method_config.get('enabled', True):
                configs[method_name] = OptimizationConfig(
                    method=method_config.get('method', 'random'),
                    cv_folds=method_config.get('cv_folds', 3),
                    max_iterations=method_config.get('max_iterations', 100),
                    scoring=method_config.get('scoring', 'accuracy'),
                    n_jobs=method_config.get('n_jobs', -1),
                    random_state=method_config.get('random_state', 42),
                    n_iter=method_config.get('max_iterations', 100)  # RandomizedSearchCV用
                )

        return configs

    def _define_default_hyperparameter_spaces(self) -> Dict[str, Any]:
        """デフォルトハイパーパラメータ空間定義"""
        return {
            'random_forest_classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            },
            'random_forest_regressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            },
            'xgboost_classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'xgboost_regressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 改善版最適化結果テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    optimization_method TEXT NOT NULL,
                    best_score REAL NOT NULL,
                    baseline_score REAL NOT NULL,
                    improvement REAL NOT NULL,
                    optimization_time REAL NOT NULL,
                    best_params TEXT NOT NULL,
                    cv_scores TEXT NOT NULL,
                    total_combinations INTEGER,
                    config_snapshot TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 改善版パラメータ重要度テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_parameter_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_run_id INTEGER NOT NULL,
                    parameter_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    importance_method TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (optimization_run_id) REFERENCES enhanced_optimization_results (id)
                )
            """)

    def _get_baseline_score(self, task: PredictionTask, scoring: str) -> float:
        """タスクと評価指標に応じた適切なベースラインスコア取得"""
        baseline_scores = self.hyperparameter_spaces.get('baseline_scores', {})

        if task == PredictionTask.PRICE_DIRECTION:
            # 分類タスク
            classification_scores = baseline_scores.get('classification', {})
            return classification_scores.get(scoring, 0.5)
        else:
            # 回帰タスク
            regression_scores = baseline_scores.get('regression', {})
            if scoring == 'r2':
                return regression_scores.get('r2', 0.0)
            else:
                return regression_scores.get(scoring, 1.0)

    async def optimize_model(self, symbol: str, model_type: ModelType, task: PredictionTask,
                           X: pd.DataFrame, y: pd.Series,
                           baseline_score: Optional[float] = None,
                           method: str = 'random') -> OptimizationResult:
        """モデル最適化実行（改善版）"""

        self.logger.info(f"Optimizing {model_type.value} for {task.value} using {method}")
        start_time = datetime.now()

        # 最適化設定取得
        config = self.optimization_configs.get(f'{method}_search')
        if not config:
            self.logger.warning(f"設定が見つかりません: {method}_search. random_searchを使用します")
            config = self.optimization_configs.get('random_search')
            if not config:
                raise ValueError("利用可能な最適化設定がありません")

        # モデルとパラメータ空間選択
        scoring = self._get_scoring_for_task(task)

        if model_type == ModelType.RANDOM_FOREST:
            if task == PredictionTask.PRICE_DIRECTION:
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_space = self.hyperparameter_spaces.get('random_forest_classifier', {})
            else:  # PRICE_REGRESSION
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_space = self.hyperparameter_spaces.get('random_forest_regressor', {})

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            if task == PredictionTask.PRICE_DIRECTION:
                model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
                param_space = self.hyperparameter_spaces.get('xgboost_classifier', {})
            else:  # PRICE_REGRESSION
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
                param_space = self.hyperparameter_spaces.get('xgboost_regressor', {})
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not param_space:
            raise ValueError(f"パラメータ空間が定義されていません: {model_type.value}")

        # ベースラインスコア設定
        if baseline_score is None:
            baseline_score = self._get_baseline_score(task, scoring)

        # 最適化実行
        try:
            if method == 'grid':
                # Grid Search（パラメータ数を制限）
                max_combinations = self.config.get('advanced', {}).get('parameter_space_limiting', {}).get('max_combinations_grid', 100)
                limited_param_space = self._limit_param_space(param_space, max_combinations)

                optimizer = GridSearchCV(
                    model,
                    limited_param_space,
                    cv=TimeSeriesSplit(n_splits=config.cv_folds),
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    verbose=1
                )

            elif method == 'random' and SCIPY_AVAILABLE:
                # RandomizedSearchCV（正しい実装）
                random_param_space = self._convert_to_random_space(param_space)

                optimizer = RandomizedSearchCV(
                    model,
                    random_param_space,
                    n_iter=config.n_iter or config.max_iterations,
                    cv=TimeSeriesSplit(n_splits=config.cv_folds),
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    random_state=config.random_state,
                    verbose=1
                )

            elif method == 'bayesian' and BAYESIAN_AVAILABLE:
                # Bayesian Optimization
                bayesian_space = self._convert_to_bayesian_space(param_space)
                optimizer = BayesSearchCV(
                    model,
                    bayesian_space,
                    cv=TimeSeriesSplit(n_splits=config.cv_folds),
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    n_iter=config.max_iterations,
                    random_state=config.random_state,
                    verbose=1
                )

            else:  # フォールバック: Grid Search
                self.logger.warning(f"メソッド'{method}'が利用できません。Grid Searchにフォールバックします")
                limited_param_space = self._limit_param_space(param_space, 2)
                optimizer = GridSearchCV(
                    model,
                    limited_param_space,
                    cv=TimeSeriesSplit(n_splits=config.cv_folds),
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    verbose=1
                )

            # 最適化実行
            optimizer.fit(X, y)

            # 結果分析
            best_score = optimizer.best_score_
            best_params = optimizer.best_params_
            cv_scores = list(optimizer.cv_results_['mean_test_score'])

            # 改善率計算（適切なベースライン使用）
            if baseline_score != 0:
                improvement = ((best_score - baseline_score) / abs(baseline_score) * 100)
            else:
                improvement = best_score * 100

            # パラメータ重要度計算（改善版）
            param_importance = self._calculate_param_importance_improved(optimizer.cv_results_)

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationResult(
                model_type=model_type.value,
                task=task.value,
                optimization_method=method,
                best_params=best_params,
                best_score=best_score,
                cv_scores=cv_scores,
                optimization_time=optimization_time,
                improvement=improvement,
                param_importance=param_importance,
                baseline_score=baseline_score
            )

            self.logger.info(f"Optimization completed: {best_score:.4f} (improvement: {improvement:.2f}%)")

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

    def _get_scoring_for_task(self, task: PredictionTask) -> str:
        """タスクに応じたスコアリング指標取得"""
        task_scoring = self.config.get('task_scoring', {})

        if task == PredictionTask.PRICE_DIRECTION:
            return task_scoring.get('price_direction', {}).get('primary', 'accuracy')
        else:
            return task_scoring.get('price_regression', {}).get('primary', 'r2')

    def _convert_to_random_space(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """Random Search用のパラメータ分布に変換"""
        random_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                random_space[param] = values
            else:
                random_space[param] = values

        return random_space

    def _limit_param_space(self, param_space: Dict[str, List], max_combinations: int) -> Dict[str, List]:
        """パラメータ空間を制限"""
        limited_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                # 最大組み合わせ数を考慮して制限
                max_values = min(len(values), 3)  # 各パラメータ最大3値
                limited_space[param] = values[:max_values]
            else:
                limited_space[param] = values

        return limited_space

    def _convert_to_bayesian_space(self, param_space: Dict) -> Dict:
        """Bayesian最適化用のパラメータ空間に変換"""
        bayesian_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, float)) and v is not None for v in values):
                    # 数値の場合
                    if all(isinstance(v, int) for v in values):
                        bayesian_space[param] = Integer(min(values), max(values))
                    else:
                        bayesian_space[param] = Real(min(values), max(values))
                else:
                    # カテゴリカルの場合
                    bayesian_space[param] = Categorical(values)
            else:
                bayesian_space[param] = values

        return bayesian_space

    def _calculate_param_importance_improved(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """パラメータ重要度計算（改善版）"""
        param_importance = {}

        try:
            params = cv_results['params']
            scores = cv_results['mean_test_score']

            if len(params) < 2:
                return {}

            # パラメータごとの統計分析
            param_effects = {}

            for param_dict, score in zip(params, scores):
                for param_name, param_value in param_dict.items():
                    if param_name not in param_effects:
                        param_effects[param_name] = {'values': [], 'scores': []}

                    param_effects[param_name]['values'].append(param_value)
                    param_effects[param_name]['scores'].append(score)

            # 各パラメータの重要度計算
            for param_name, data in param_effects.items():
                values = data['values']
                scores = data['scores']

                if len(set(values)) < 2:  # パラメータ値のバリエーションが少ない
                    param_importance[param_name] = 0.0
                    continue

                try:
                    # 数値パラメータの場合：相関係数を使用
                    numeric_values = []
                    numeric_scores = []

                    for val, score in zip(values, scores):
                        if isinstance(val, (int, float)) and val is not None:
                            numeric_values.append(float(val))
                            numeric_scores.append(score)

                    if len(numeric_values) >= 2 and len(set(numeric_values)) >= 2:
                        correlation = abs(np.corrcoef(numeric_values, numeric_scores)[0, 1])
                        if not np.isnan(correlation):
                            param_importance[param_name] = correlation
                        else:
                            param_importance[param_name] = 0.0
                    else:
                        # カテゴリカルパラメータ：値ごとの平均スコア分散
                        value_groups = {}
                        for val, score in zip(values, scores):
                            val_key = str(val)
                            if val_key not in value_groups:
                                value_groups[val_key] = []
                            value_groups[val_key].append(score)

                        group_means = [np.mean(group_scores) for group_scores in value_groups.values()]

                        if len(group_means) > 1:
                            overall_mean = np.mean(scores)
                            # グループ間分散を重要度として使用
                            variance = np.var(group_means)
                            param_importance[param_name] = variance / (np.var(scores) + 1e-8)
                        else:
                            param_importance[param_name] = 0.0

                except Exception as e:
                    self.logger.warning(f"パラメータ{param_name}の重要度計算でエラー: {e}")
                    param_importance[param_name] = 0.0

            # 上位N個のパラメータのみ保持（ノイズ除去）
            if len(param_importance) > 5:
                sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
                param_importance = dict(sorted_params[:5])

        except Exception as e:
            self.logger.warning(f"パラメータ重要度計算エラー: {e}")
            return {}

        return param_importance

    def _calculate_param_importance(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """パラメータ重要度計算（後方互換性のため）"""
        return self._calculate_param_importance_improved(cv_results)

    async def optimize_all_models(self, symbol: str, X: pd.DataFrame, y_dict: Dict[PredictionTask, pd.Series],
                                baseline_scores: Dict[str, float] = None) -> Dict[str, OptimizationResult]:
        """全モデルの最適化実行"""

        if baseline_scores is None:
            baseline_scores = {}

        results = {}

        # Random Forest最適化
        for task, y in y_dict.items():
            if task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                try:
                    baseline_key = f"RandomForest_{task.value}"
                    baseline = baseline_scores.get(baseline_key, 0.5)

                    rf_result = await self.optimize_model(
                        symbol, ModelType.RANDOM_FOREST, task, X, y,
                        baseline_score=baseline, method='random'
                    )
                    results[f"RandomForest_{task.value}"] = rf_result

                except Exception as e:
                    self.logger.error(f"Random Forest optimization failed for {task.value}: {e}")

        # XGBoost最適化
        if XGBOOST_AVAILABLE:
            for task, y in y_dict.items():
                if task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                    try:
                        baseline_key = f"XGBoost_{task.value}"
                        baseline = baseline_scores.get(baseline_key, 0.5)

                        xgb_result = await self.optimize_model(
                            symbol, ModelType.XGBOOST, task, X, y,
                            baseline_score=baseline, method='random'
                        )
                        results[f"XGBoost_{task.value}"] = xgb_result

                    except Exception as e:
                        self.logger.error(f"XGBoost optimization failed for {task.value}: {e}")

        return results

    async def _save_optimization_result(self, result: OptimizationResult, symbol: str, method: str):
        """最適化結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 設定スナップショット
                config_snapshot = yaml.dump(self.config)

                # 最適化結果の記録
                cursor.execute("""
                    INSERT INTO enhanced_optimization_results
                    (model_type, task, optimization_method, best_score, baseline_score,
                     improvement, optimization_time, best_params, cv_scores,
                     total_combinations, config_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.model_type,
                    result.task,
                    result.optimization_method,
                    result.best_score,
                    result.baseline_score,
                    result.improvement,
                    result.optimization_time,
                    json.dumps(result.best_params),
                    json.dumps(result.cv_scores),
                    result.total_combinations,
                    config_snapshot
                ))

                optimization_run_id = cursor.lastrowid

                # パラメータ重要度の記録
                for param_name, importance_score in result.param_importance.items():
                    cursor.execute("""
                        INSERT INTO enhanced_parameter_importance
                        (optimization_run_id, parameter_name, importance_score, importance_method)
                        VALUES (?, ?, ?, ?)
                    """, (optimization_run_id, param_name, importance_score, 'enhanced_combined'))

                conn.commit()
                self.logger.info(f"最適化結果をデータベースに記録しました (ID: {optimization_run_id})")

        except Exception as e:
            self.logger.error(f"最適化結果記録エラー: {e}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化結果サマリーを取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 最新の最適化結果
                cursor.execute("""
                    SELECT model_type, task, optimization_method, best_score,
                           improvement, optimization_time, created_at
                    FROM enhanced_optimization_results
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_results = cursor.fetchall()

                # パラメータ重要度統計
                cursor.execute("""
                    SELECT parameter_name, AVG(importance_score) as avg_importance,
                           COUNT(*) as frequency
                    FROM enhanced_parameter_importance
                    GROUP BY parameter_name
                    ORDER BY avg_importance DESC
                    LIMIT 10
                """)
                param_importance_stats = cursor.fetchall()

                return {
                    'config_path': str(self.config_path),
                    'recent_optimizations': recent_results,
                    'parameter_importance_stats': param_importance_stats,
                    'available_methods': [method.value for method in OptimizationMethod],
                    'integrations': {
                        'sklearn': SKLEARN_AVAILABLE,
                        'xgboost': XGBOOST_AVAILABLE,
                        'scipy': SCIPY_AVAILABLE,
                        'bayesian': BAYESIAN_AVAILABLE
                    }
                }

        except Exception as e:
            self.logger.error(f"サマリー情報取得エラー: {e}")
            return {'error': str(e)}


# ユーティリティ関数
def create_enhanced_hyperparameter_optimizer(
    config_path: Optional[str] = None,
    hyperparameter_config_path: Optional[str] = None
) -> HyperparameterOptimizer:
    """
    EnhancedHyperparameterOptimizerインスタンスの作成

    Args:
        config_path: 最適化設定ファイルパス
        hyperparameter_config_path: ハイパーパラメータ空間設定ファイルパス

    Returns:
        EnhancedHyperparameterOptimizerインスタンス
    """
    path = Path(config_path) if config_path else None
    hyperparameter_path_obj = Path(hyperparameter_config_path) if hyperparameter_config_path else None

    return HyperparameterOptimizer(
        config_path=path,
        hyperparameter_config_path=hyperparameter_path_obj
    )

# グローバルインスタンス（後方互換性のため）
try:
    hyperparameter_optimizer = HyperparameterOptimizer()
except Exception:
    # 設定ファイルがない場合のフォールバック
    hyperparameter_optimizer = None

if __name__ == "__main__":
    # 基本動作確認は別ファイルに分離済み
    pass
