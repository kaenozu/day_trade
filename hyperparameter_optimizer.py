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

# 共通ユーティリティのインポート
from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

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

# Windows環境対応
apply_windows_encoding_fix()

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
    method: OptimizationMethod
    cv_folds: int
    max_iterations: int
    n_iter: int  # RandomizedSearchCV用
    scoring: str
    n_jobs: int
    random_state: int
    verbose: int = 1


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
    baseline_score: float
    total_combinations: int = 0


@dataclass
class HyperparameterSpaceConfig:
    """外部設定からのハイパーパラメータ空間"""
    random_forest_classifier: Dict[str, Any] = field(default_factory=dict)
    random_forest_regressor: Dict[str, Any] = field(default_factory=dict)
    xgboost_classifier: Dict[str, Any] = field(default_factory=dict)
    xgboost_regressor: Dict[str, Any] = field(default_factory=dict)


class EnhancedHyperparameterOptimizer:
    """
    改善版ハイパーパラメータ最適化システム

    主な機能:
    - 外部YAML設定による柔軟なパラメータ空間定義
    - 最適化手法の明確な選択ロジック
    - タスクに応じた適切なベースラインスコア
    - 改善されたパラメータ重要度分析
    - Windows環境対応統合
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn is required for hyperparameter optimization"
            )

        # 設定ファイルパス
        self.config_path = config_path or Path("config/hyperparameter_spaces.yaml")

        # データディレクトリ
        self.data_dir = Path("hyperparameter_optimization")
        self.data_dir.mkdir(exist_ok=True)

        # 設定読み込み
        self.hyperparameter_spaces = HyperparameterSpaceConfig()
        self.optimization_configs = {}
        self.baseline_configs = {}
        self._load_configuration()

        # データベース初期化
        db_path = self.config.get('storage', {}).get('database_path',
                                                  'hyperparameter_optimization/optimization_results.db')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # 最適化履歴
        self.optimization_history = {}
        self.optimized_params = {}

        self.logger.info("Enhanced Hyperparameter Optimizer initialized")

    def _load_configuration(self):
        """外部設定ファイルの読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                self.logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

            # ハイパーパラメータ空間の設定
            self._load_hyperparameter_spaces()

            # 最適化設定の読み込み
            self._load_optimization_configs()

            # ベースラインスコア設定の読み込み
            self._load_baseline_configs()

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            self._load_default_settings()

    def _create_default_config(self):
        """デフォルト設定ファイルの作成"""
        # 既にconfig/hyperparameter_spaces.yamlが作成されているので、それを読み込む
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                self._load_default_settings()
        except Exception as e:
            self.logger.error(f"デフォルト設定作成エラー: {e}")
            self._load_default_settings()

    def _load_default_settings(self):
        """デフォルト設定の読み込み"""
        self.config = {
            'random_forest_classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'random_forest_regressor': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'optimization_settings': {
                'random_search': {'n_iter': 50},
                'grid_search': {'max_iterations': 50},
                'bayesian_optimization': {'n_iter': 30}
            },
            'baseline_scores': {
                'classification': {'accuracy': 0.5},
                'regression': {'r2': 0.0}
            }
        }
        self.logger.info("デフォルト設定を使用します")

    def _load_hyperparameter_spaces(self):
        """ハイパーパラメータ空間の読み込み"""
        spaces = {}
        for model_key in ['random_forest_classifier', 'random_forest_regressor',
                          'xgboost_classifier', 'xgboost_regressor']:
            spaces[model_key] = self.config.get(model_key, {})

        self.hyperparameter_spaces = HyperparameterSpaceConfig(**spaces)

    def _load_optimization_configs(self):
        """最適化設定の読み込み"""
        opt_settings = self.config.get('optimization_settings', {})

        self.optimization_configs = {
            OptimizationMethod.GRID: OptimizationConfig(
                method=OptimizationMethod.GRID,
                cv_folds=opt_settings.get('grid_search', {}).get('cv_folds', 3),
                max_iterations=opt_settings.get('grid_search', {}).get('max_iterations', 50),
                n_iter=0,  # Grid Searchでは使用しない
                scoring='accuracy',
                n_jobs=opt_settings.get('grid_search', {}).get('n_jobs', -1),
                random_state=42,
                verbose=opt_settings.get('grid_search', {}).get('verbose', 1)
            ),
            OptimizationMethod.RANDOM: OptimizationConfig(
                method=OptimizationMethod.RANDOM,
                cv_folds=opt_settings.get('random_search', {}).get('cv_folds', 3),
                max_iterations=opt_settings.get('random_search', {}).get('max_iterations', 100),
                n_iter=opt_settings.get('random_search', {}).get('n_iter', 50),
                scoring='accuracy',
                n_jobs=opt_settings.get('random_search', {}).get('n_jobs', -1),
                random_state=42,
                verbose=opt_settings.get('random_search', {}).get('verbose', 1)
            ),
            OptimizationMethod.BAYESIAN: OptimizationConfig(
                method=OptimizationMethod.BAYESIAN,
                cv_folds=opt_settings.get('bayesian_optimization', {}).get('cv_folds', 3),
                max_iterations=opt_settings.get('bayesian_optimization', {}).get('max_iterations', 50),
                n_iter=opt_settings.get('bayesian_optimization', {}).get('n_iter', 30),
                scoring='accuracy',
                n_jobs=opt_settings.get('bayesian_optimization', {}).get('n_jobs', -1),
                random_state=42,
                verbose=opt_settings.get('bayesian_optimization', {}).get('verbose', 1)
            )
        }

    def _load_baseline_configs(self):
        """ベースラインスコア設定の読み込み"""
        baseline_settings = self.config.get('baseline_scores', {})

        self.baseline_configs = {
            'classification': baseline_settings.get('classification', {'accuracy': 0.5}),
            'regression': baseline_settings.get('regression', {'r2': 0.0})
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

    def get_appropriate_baseline_score(self, task: PredictionTask,
                                       X: pd.DataFrame, y: pd.Series) -> float:
        """タスクに応じた適切なベースラインスコアを計算"""
        try:
            if task == PredictionTask.PRICE_DIRECTION:
                # 分類タスク：最頻値での予測精度
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy='most_frequent')
                dummy.fit(X, y)
                baseline = dummy.score(X, y)
                self.logger.info(f"分類ベースラインスコア (最頻値予測): {baseline:.3f}")
                return baseline

            else:  # 回帰タスク
                # 回帰タスク：平均値での予測のR2スコア
                from sklearn.dummy import DummyRegressor
                dummy = DummyRegressor(strategy='mean')
                dummy.fit(X, y)
                baseline = dummy.score(X, y)  # R2スコア
                self.logger.info(f"回帰ベースラインスコア (平均値予測): {baseline:.3f}")
                return baseline

        except Exception as e:
            self.logger.warning(f"ベースラインスコア計算エラー: {e}")
            # フォールバック値
            if task == PredictionTask.PRICE_DIRECTION:
                return self.baseline_configs['classification']['accuracy']
            else:
                return self.baseline_configs['regression']['r2']

    async def optimize_model(self, symbol: str, model_type: ModelType, task: PredictionTask,
                             X: pd.DataFrame, y: pd.Series,
                             method: OptimizationMethod = OptimizationMethod.RANDOM) -> OptimizationResult:
        """
        改善版モデル最適化実行
        """

        self.logger.info(f"Optimizing {model_type.value} for {task.value} using {method.value}")
        start_time = datetime.now()

        # 適切なベースラインスコア計算
        baseline_score = self.get_appropriate_baseline_score(task, X, y)

        # 最適化設定取得
        config = self.optimization_configs[method]

        # モデルとパラメータ空間選択
        model, param_space, scoring = self._get_model_and_params(model_type, task)

        # 最適化実行
        try:
            optimizer = self._create_optimizer(method, config, model, param_space, scoring)

            # 最適化実行
            optimizer.fit(X, y)

            # 結果分析
            best_score = optimizer.best_score_
            best_params = optimizer.best_params_
            cv_scores = list(optimizer.cv_results_['mean_test_score'])

            # 改善率計算（修正版）
            if baseline_score != 0:
                improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100
            else:
                improvement = best_score * 100

            # 改善されたパラメータ重要度計算
            param_importance = await self._calculate_enhanced_param_importance(
                optimizer, X, y, best_params
            )

            optimization_time = (datetime.now() - start_time).total_seconds()

            # 組み合わせ数計算
            total_combinations = self._calculate_total_combinations(param_space, method, config)

            result = OptimizationResult(
                model_type=model_type.value,
                task=task.value,
                optimization_method=method.value,
                best_params=best_params,
                best_score=best_score,
                cv_scores=cv_scores,
                optimization_time=optimization_time,
                improvement=improvement,
                param_importance=param_importance,
                baseline_score=baseline_score,
                total_combinations=total_combinations
            )

            # 結果をデータベースに記録
            await self._record_optimization_result(result)

            self.logger.info(
                f"最適化完了 - ベストスコア: {best_score:.4f}, "
                f"改善率: {improvement:.2f}%, 所要時間: {optimization_time:.1f}秒"
            )

            return result

        except Exception as e:
            self.logger.error(f"最適化実行エラー: {e}")
            raise

    def _get_model_and_params(self, model_type: ModelType, task: PredictionTask) -> Tuple[Any, Dict, str]:
        """モデルとパラメータ空間、評価指標を取得"""

        if model_type == ModelType.RANDOM_FOREST:
            if task == PredictionTask.PRICE_DIRECTION:
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_space = self.hyperparameter_spaces.random_forest_classifier
                scoring = 'accuracy'
            else:  # PRICE_REGRESSION
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_space = self.hyperparameter_spaces.random_forest_regressor
                scoring = 'r2'

        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            if task == PredictionTask.PRICE_DIRECTION:
                model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
                param_space = self.hyperparameter_spaces.xgboost_classifier
                scoring = 'accuracy'
            else:  # PRICE_REGRESSION
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
                param_space = self.hyperparameter_spaces.xgboost_regressor
                scoring = 'r2'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not param_space:
            raise ValueError(f"パラメータ空間が定義されていません: {model_type.value}")

        return model, param_space, scoring

    def _create_optimizer(self, method: OptimizationMethod, config: OptimizationConfig,
                          model: Any, param_space: Dict, scoring: str) -> Any:
        """最適化器を作成（Issue #856修正：RandomizedSearchCVを正しく使用）"""

        cv = TimeSeriesSplit(n_splits=config.cv_folds)

        if method == OptimizationMethod.GRID:
            # Grid Search
            limited_param_space = self._limit_param_space_for_grid(param_space, config.max_iterations)
            return GridSearchCV(
                model,
                limited_param_space,
                cv=cv,
                scoring=scoring,
                n_jobs=config.n_jobs,
                verbose=config.verbose
            )

        elif method == OptimizationMethod.RANDOM:
            # Random Search（修正：RandomizedSearchCVを正しく使用）
            if not SCIPY_AVAILABLE:
                self.logger.warning("scipy not available, falling back to Grid Search")
                limited_param_space = self._limit_param_space_for_grid(param_space, 20)
                return GridSearchCV(
                    model,
                    limited_param_space,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    verbose=config.verbose
                )
            else:
                return RandomizedSearchCV(
                    model,
                    param_space,
                    n_iter=config.n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=config.n_jobs,
                    random_state=config.random_state,
                    verbose=config.verbose
                )

        elif method == OptimizationMethod.BAYESIAN and BAYESIAN_AVAILABLE:
            # Bayesian Optimization
            bayesian_space = self._convert_to_bayesian_space(param_space)
            return BayesSearchCV(
                model,
                bayesian_space,
                n_iter=config.n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
                verbose=config.verbose
            )
        else:
            # フォールバック：Grid Search
            self.logger.warning(f"Method {method.value} not available, falling back to Grid Search")
            limited_param_space = self._limit_param_space_for_grid(param_space, 20)
            return GridSearchCV(
                model,
                limited_param_space,
                cv=cv,
                scoring=scoring,
                n_jobs=config.n_jobs,
                verbose=config.verbose
            )

    def _limit_param_space_for_grid(self, param_space: Dict, max_combinations: int) -> Dict:
        """Grid Search用にパラメータ空間を制限"""
        limited_space = {}
        total_combinations = 1

        for param, values in param_space.items():
            if isinstance(values, list):
                # 組み合わせ数を考慮してサイズを制限
                max_values = min(len(values), max(2, max_combinations // total_combinations))
                limited_space[param] = values[:max_values]
                total_combinations *= len(limited_space[param])
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

    async def _calculate_enhanced_param_importance(self, optimizer: Any, X: pd.DataFrame,
                                                   y: pd.Series, best_params: Dict) -> Dict[str, float]:
        """
        改善されたパラメータ重要度計算

        Issue #856対応：
        1. CV結果の分散分析
        2. パラメータと性能の相関分析
        3. より統計的に意味のある重要度計算
        """
        importance_scores = {}

        try:
            # 1. CV結果の分散による重要度
            cv_variance_importance = self._calculate_cv_variance_importance(optimizer)

            # 2. パラメータ-性能相関による重要度
            correlation_importance = self._calculate_correlation_importance(optimizer)

            # 3. 最適パラメータの偏差による重要度
            deviation_importance = self._calculate_deviation_importance(optimizer, best_params)

            # 重要度スコアの統合
            all_params = set(cv_variance_importance.keys()) | set(correlation_importance.keys()) | set(deviation_importance.keys())

            for param in all_params:
                cv_score = cv_variance_importance.get(param, 0.0)
                corr_score = correlation_importance.get(param, 0.0)
                dev_score = deviation_importance.get(param, 0.0)

                # 重み付き平均
                combined_score = (cv_score * 0.4 + corr_score * 0.4 + dev_score * 0.2)
                importance_scores[param] = combined_score

            # 上位N個のパラメータのみ保持（ノイズ除去）
            if len(importance_scores) > 5:
                sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                importance_scores = dict(sorted_params[:5])

        except Exception as e:
            self.logger.warning(f"パラメータ重要度計算エラー: {e}")
            # フォールバック：単純な分散計算
            importance_scores = self._simple_param_importance(optimizer)

        return importance_scores

    def _calculate_cv_variance_importance(self, optimizer: Any) -> Dict[str, float]:
        """CV結果の分散による重要度計算"""
        importance = {}

        try:
            cv_results = optimizer.cv_results_
            params_keys = [key for key in cv_results.keys() if key.startswith('param_')]

            for param_key in params_keys:
                param_name = param_key.replace('param_', '')

                # パラメータごとのスコア分散を計算
                param_scores = {}
                for i, param_value in enumerate(cv_results[param_key]):
                    if param_value not in param_scores:
                        param_scores[param_value] = []
                    param_scores[param_value].append(cv_results['mean_test_score'][i])

                # 各パラメータ値でのスコア平均値の分散
                if len(param_scores) > 1:
                    mean_scores = [np.mean(scores) for scores in param_scores.values()]
                    variance = np.var(mean_scores)
                    importance[param_name] = variance
                else:
                    importance[param_name] = 0.0

        except Exception as e:
            self.logger.debug(f"CV分散計算エラー: {e}")

        return importance

    def _calculate_correlation_importance(self, optimizer: Any) -> Dict[str, float]:
        """パラメータと性能の相関による重要度計算"""
        importance = {}

        try:
            cv_results = optimizer.cv_results_
            scores = cv_results['mean_test_score']

            params_keys = [key for key in cv_results.keys() if key.startswith('param_')]

            for param_key in params_keys:
                param_name = param_key.replace('param_', '')
                param_values = cv_results[param_key]

                # 数値パラメータのみ相関計算
                try:
                    numeric_values = []
                    numeric_scores = []

                    for i, value in enumerate(param_values):
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            numeric_values.append(float(value))
                            numeric_scores.append(scores[i])

                    if len(numeric_values) > 5 and len(set(numeric_values)) > 1:
                        correlation, _ = pearsonr(numeric_values, numeric_scores)
                        importance[param_name] = abs(correlation)
                    else:
                        importance[param_name] = 0.0

                except Exception:
                    importance[param_name] = 0.0

        except Exception as e:
            self.logger.debug(f"相関計算エラー: {e}")

        return importance

    def _calculate_deviation_importance(self, optimizer: Any, best_params: Dict) -> Dict[str, float]:
        """最適パラメータの偏差による重要度計算"""
        importance = {}

        try:
            cv_results = optimizer.cv_results_

            for param_name, best_value in best_params.items():
                param_key = f'param_{param_name}'
                if param_key in cv_results:
                    param_values = cv_results[param_key]

                    # 数値パラメータの場合
                    if isinstance(best_value, (int, float)):
                        numeric_values = [v for v in param_values if isinstance(v, (int, float))]
                        if len(numeric_values) > 1:
                            param_std = np.std(numeric_values)
                            param_mean = np.mean(numeric_values)
                            if param_std > 0:
                                # 標準化された偏差
                                deviation = abs(best_value - param_mean) / param_std
                                importance[param_name] = min(deviation, 3.0)  # 上限設定
                            else:
                                importance[param_name] = 0.0
                        else:
                            importance[param_name] = 0.0
                    else:
                        # カテゴリカルパラメータの場合
                        unique_values = list(set(param_values))
                        if len(unique_values) > 1:
                            importance[param_name] = 1.0 / len(unique_values)
                        else:
                            importance[param_name] = 0.0

        except Exception as e:
            self.logger.debug(f"偏差計算エラー: {e}")

        return importance

    def _simple_param_importance(self, optimizer: Any) -> Dict[str, float]:
        """シンプルなパラメータ重要度計算（フォールバック）"""
        importance = {}

        try:
            cv_results = optimizer.cv_results_

            for key in cv_results.keys():
                if key.startswith('param_'):
                    param_name = key.replace('param_', '')
                    # 単純に最高スコアとの差を重要度とする
                    importance[param_name] = 1.0

        except Exception:
            pass

        return importance

    def _calculate_total_combinations(self, param_space: Dict, method: OptimizationMethod,
                                      config: OptimizationConfig) -> int:
        """パラメータ組み合わせ総数を計算"""
        if method == OptimizationMethod.GRID:
            total = 1
            for values in param_space.values():
                if isinstance(values, list):
                    total *= len(values)
            return min(total, config.max_iterations)
        elif method == OptimizationMethod.RANDOM:
            return config.n_iter
        elif method == OptimizationMethod.BAYESIAN:
            return config.n_iter
        else:
            return 1

    async def _record_optimization_result(self, result: OptimizationResult):
        """最適化結果をデータベースに記録"""
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
                ")
                recent_results = cursor.fetchall()

                # パラメータ重要度統計
                cursor.execute("""
                    SELECT parameter_name, AVG(importance_score) as avg_importance,
                           COUNT(*) as frequency
                    FROM enhanced_parameter_importance
                    GROUP BY parameter_name
                    ORDER BY avg_importance DESC
                    LIMIT 10
                ")
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
    config_path: Optional[str] = None
) -> EnhancedHyperparameterOptimizer:
    """EnhancedHyperparameterOptimizerインスタンスの作成"""
    path = Path(config_path) if config_path else None
    return EnhancedHyperparameterOptimizer(config_path=path)


if __name__ == "__main__":
    # 基本的な動作確認（テストコードは別ファイルに分離予定）
    async def main():
        logger.info("Enhanced Hyperparameter Optimizer テスト開始")

        optimizer = create_enhanced_hyperparameter_optimizer()
        summary = optimizer.get_optimization_summary()

        logger.info("最適化システム初期化完了")
        logger.info(f"利用可能な最適化手法: {summary.get('available_methods', [])}")
        logger.info(f"統合状況: {summary.get('integrations', {})}")

    # テスト実行
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())