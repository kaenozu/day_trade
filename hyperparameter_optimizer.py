#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimizer - ハイパーパラメータ最適化システム

GridSearch・RandomSearch・Bayesian最適化による予測精度向上
Issue #796-3実装：ハイパーパラメータ最適化
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import sqlite3
import warnings
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
    from sklearn.metrics import accuracy_score, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # Bayesian Optimization (オプション)
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# 既存システム
try:
    from ml_prediction_models import MLPredictionModels, ModelType, PredictionTask
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

@dataclass
class OptimizationConfig:
    """最適化設定"""
    method: str  # 'grid', 'random', 'bayesian'
    cv_folds: int
    max_iterations: int
    scoring: str
    n_jobs: int
    random_state: int

@dataclass
class OptimizationResult:
    """最適化結果"""
    model_type: str
    task: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    optimization_time: float
    improvement: float  # 最適化前との改善率
    param_importance: Dict[str, float]

@dataclass
class HyperparameterSpace:
    """ハイパーパラメータ空間定義"""
    random_forest_classifier: Dict[str, Any]
    random_forest_regressor: Dict[str, Any]
    xgboost_classifier: Dict[str, Any]
    xgboost_regressor: Dict[str, Any]

class HyperparameterOptimizer:
    """ハイパーパラメータ最適化システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required")

        # データディレクトリ
        self.data_dir = Path("hyperparameter_optimization")
        self.data_dir.mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.data_dir / "optimization_results.db"
        self._init_database()

        # 最適化設定
        self.optimization_configs = {
            'grid_search': OptimizationConfig(
                method='grid',
                cv_folds=3,  # 時系列データでは少なめに
                max_iterations=50,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            ),
            'random_search': OptimizationConfig(
                method='random',
                cv_folds=3,
                max_iterations=100,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            ),
            'bayesian_optimization': OptimizationConfig(
                method='bayesian',
                cv_folds=3,
                max_iterations=50,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
        }

        # ハイパーパラメータ空間定義
        self.hyperparameter_spaces = self._define_hyperparameter_spaces()

        # 最適化履歴
        self.optimization_history = {}

        # 最適化済みパラメータ
        self.optimized_params = {}

        self.logger.info("Hyperparameter optimizer initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 最適化結果テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    optimization_method TEXT,
                    best_score REAL,
                    improvement REAL,
                    optimization_time REAL,
                    best_params TEXT,
                    cv_scores TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # パラメータ重要度テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    parameter_name TEXT,
                    importance_score REAL,
                    optimization_run_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _define_hyperparameter_spaces(self) -> HyperparameterSpace:
        """ハイパーパラメータ空間定義"""

        return HyperparameterSpace(
            # Random Forest分類
            random_forest_classifier={
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'bootstrap': [True, False]
            },

            # Random Forest回帰
            random_forest_regressor={
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'bootstrap': [True, False]
            },

            # XGBoost分類
            xgboost_classifier={
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.5],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            },

            # XGBoost回帰
            xgboost_regressor={
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.5],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            }
        )

    async def optimize_model(self, symbol: str, model_type: ModelType, task: PredictionTask,
                           X: pd.DataFrame, y: pd.Series,
                           baseline_score: float = 0.5,
                           method: str = 'random') -> OptimizationResult:
        """モデル最適化実行"""

        self.logger.info(f"Optimizing {model_type.value} for {task.value} using {method}")

        start_time = datetime.now()

        # 最適化設定取得
        config = self.optimization_configs.get(f'{method}_search')
        if not config:
            config = self.optimization_configs['random_search']

        # モデルとパラメータ空間選択
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

        # 最適化実行
        try:
            if method == 'grid':
                # Grid Search（パラメータ数を制限）
                limited_param_space = self._limit_param_space(param_space, 3)  # 組み合わせ数制限
                optimizer = GridSearchCV(
                    model,
                    limited_param_space,
                    cv=TimeSeriesSplit(n_splits=config.cv_folds),
                    scoring=scoring,
                    n_jobs=config.n_jobs,
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

            else:  # Grid Search (fallback from random due to type issues)
                # パラメータ数を制限してGrid Searchを実行
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

            # 改善率計算
            improvement = ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0

            # パラメータ重要度計算（最適化の影響度）
            param_importance = self._calculate_param_importance(optimizer.cv_results_)

            optimization_time = (datetime.now() - start_time).total_seconds()

            result = OptimizationResult(
                model_type=model_type.value,
                task=task.value,
                best_params=best_params,
                best_score=best_score,
                cv_scores=cv_scores,
                optimization_time=optimization_time,
                improvement=improvement,
                param_importance=param_importance
            )

            # 結果保存
            await self._save_optimization_result(result, symbol, method)

            # キャッシュに保存
            cache_key = f"{symbol}_{model_type.value}_{task.value}"
            self.optimized_params[cache_key] = best_params

            self.logger.info(f"Optimization completed: {best_score:.4f} (improvement: {improvement:.2f}%)")

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

    def _convert_to_random_space(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """Random Search用のパラメータ分布に変換"""

        random_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                # 数値のみの場合（Noneを除く）
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
                if numeric_values and all(isinstance(v, int) for v in numeric_values):
                    # 整数パラメータ
                    if SCIPY_AVAILABLE:
                        random_space[param] = randint(min(numeric_values), max(numeric_values) + 1)
                    else:
                        random_space[param] = numeric_values
                elif numeric_values and all(isinstance(v, float) for v in numeric_values):
                    # 浮動小数点パラメータ
                    if SCIPY_AVAILABLE:
                        random_space[param] = uniform(min(numeric_values), max(numeric_values) - min(numeric_values))
                    else:
                        random_space[param] = numeric_values
                else:
                    # カテゴリカルパラメータ（混合型含む）
                    random_space[param] = values
            else:
                random_space[param] = values

        return random_space

    def _limit_param_space(self, param_space: Dict[str, List], max_values_per_param: int = 3) -> Dict[str, List]:
        """パラメータ空間を制限（計算時間短縮のため）"""

        limited_space = {}

        for param, values in param_space.items():
            if isinstance(values, list) and len(values) > max_values_per_param:
                # 値を均等にサンプリング
                step = len(values) // max_values_per_param
                limited_values = values[::step][:max_values_per_param]
                limited_space[param] = limited_values
            else:
                limited_space[param] = values

        return limited_space

    def _convert_to_bayesian_space(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """Bayesian Optimization用の空間に変換"""

        if not BAYESIAN_AVAILABLE:
            return param_space

        bayesian_space = {}

        for param, values in param_space.items():
            if isinstance(values, list):
                if all(isinstance(v, int) for v in values if v is not None):
                    # 整数パラメータ
                    int_values = [v for v in values if v is not None]
                    if int_values:
                        bayesian_space[param] = Integer(min(int_values), max(int_values))
                elif all(isinstance(v, float) for v in values if v is not None):
                    # 浮動小数点パラメータ
                    float_values = [v for v in values if v is not None]
                    if float_values:
                        bayesian_space[param] = Real(min(float_values), max(float_values))
                else:
                    # カテゴリカルパラメータ（リストのまま）
                    bayesian_space[param] = values
            else:
                bayesian_space[param] = values

        return bayesian_space

    def _calculate_param_importance(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """パラメータ重要度計算"""

        param_importance = {}

        try:
            # パラメータごとのスコア分散を計算
            params = cv_results['params']
            scores = cv_results['mean_test_score']

            # 各パラメータがスコアに与える影響を推定
            for param_dict, score in zip(params, scores):
                for param_name, param_value in param_dict.items():
                    if param_name not in param_importance:
                        param_importance[param_name] = []
                    param_importance[param_name].append(score)

            # 各パラメータの分散を重要度として使用
            for param_name, score_list in param_importance.items():
                if len(score_list) > 1:
                    param_importance[param_name] = np.var(score_list)
                else:
                    param_importance[param_name] = 0.0

            # 正規化
            max_importance = max(param_importance.values()) if param_importance else 1.0
            if max_importance > 0:
                param_importance = {k: v/max_importance for k, v in param_importance.items()}

        except Exception as e:
            self.logger.error(f"Failed to calculate parameter importance: {e}")
            param_importance = {}

        return param_importance

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
                # メイン結果保存
                cursor = conn.execute("""
                    INSERT INTO optimization_results
                    (model_type, task, optimization_method, best_score, improvement,
                     optimization_time, best_params, cv_scores)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.model_type,
                    result.task,
                    method,
                    result.best_score,
                    result.improvement,
                    result.optimization_time,
                    json.dumps(result.best_params),
                    json.dumps(result.cv_scores)
                ))

                optimization_run_id = cursor.lastrowid

                # パラメータ重要度保存
                for param_name, importance in result.param_importance.items():
                    conn.execute("""
                        INSERT INTO parameter_importance
                        (model_type, task, parameter_name, importance_score, optimization_run_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        result.model_type,
                        result.task,
                        param_name,
                        importance,
                        optimization_run_id
                    ))

        except Exception as e:
            self.logger.error(f"Failed to save optimization result: {e}")

    def get_optimized_params(self, symbol: str, model_type: ModelType, task: PredictionTask) -> Dict[str, Any]:
        """最適化済みパラメータ取得"""

        cache_key = f"{symbol}_{model_type.value}_{task.value}"
        return self.optimized_params.get(cache_key, {})

    async def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリー取得"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新の最適化結果
                cursor = conn.execute("""
                    SELECT model_type, task, best_score, improvement, optimization_time
                    FROM optimization_results
                    ORDER BY created_at DESC
                    LIMIT 20
                """)

                recent_optimizations = cursor.fetchall()

                # 最大改善率
                cursor = conn.execute("""
                    SELECT MAX(improvement) as max_improvement,
                           AVG(improvement) as avg_improvement
                    FROM optimization_results
                """)

                improvement_stats = cursor.fetchone()

                # パラメータ重要度統計
                cursor = conn.execute("""
                    SELECT parameter_name, AVG(importance_score) as avg_importance
                    FROM parameter_importance
                    GROUP BY parameter_name
                    ORDER BY avg_importance DESC
                    LIMIT 10
                """)

                param_importance = cursor.fetchall()

                return {
                    'optimization_count': len(self.optimized_params),
                    'recent_optimizations': [
                        {
                            'model': r[0],
                            'task': r[1],
                            'score': r[2],
                            'improvement': r[3],
                            'time': r[4]
                        } for r in recent_optimizations
                    ],
                    'max_improvement': improvement_stats[0] if improvement_stats else 0,
                    'avg_improvement': improvement_stats[1] if improvement_stats else 0,
                    'important_params': [
                        {
                            'parameter': p[0],
                            'importance': p[1]
                        } for p in param_importance
                    ]
                }

        except Exception as e:
            self.logger.error(f"Failed to get optimization summary: {e}")
            return {
                'optimization_count': 0,
                'recent_optimizations': [],
                'max_improvement': 0,
                'avg_improvement': 0,
                'important_params': []
            }

# グローバルインスタンス
hyperparameter_optimizer = HyperparameterOptimizer()

# テスト関数
async def test_hyperparameter_optimization():
    """ハイパーパラメータ最適化のテスト"""

    print("=== ハイパーパラメータ最適化 テスト ===")

    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn not available")
        return

    optimizer = HyperparameterOptimizer()

    # MLモデルシステムと連携
    if ML_MODELS_AVAILABLE:
        print("\n[ MLモデルシステムとの統合テスト ]")

        try:
            from ml_prediction_models import ml_prediction_models

            # テスト銘柄
            symbol = "7203"

            # 訓練データ準備
            features, targets = await ml_prediction_models.prepare_training_data(symbol, "3mo")

            print(f"データ準備完了: 特徴量{features.shape}, ターゲット{len(targets)}")

            # ベースライン性能取得（最適化前）
            baseline_performances = await ml_prediction_models.train_models(symbol, "3mo")
            baseline_scores = {}

            for model_type, task_perfs in baseline_performances.items():
                for task, perf in task_perfs.items():
                    key = f"{model_type.value}_{task.value}"
                    baseline_scores[key] = perf.accuracy

            print(f"ベースライン性能: {len(baseline_scores)}モデル")
            for model_task, score in baseline_scores.items():
                print(f"  {model_task}: {score:.4f}")

            # ハイパーパラメータ最適化実行
            print(f"\n[ ハイパーパラメータ最適化実行 ]")

            # 有効なターゲット準備
            valid_idx = features.index[:-1]  # 最後の行は未来の値なので除外
            X = features.loc[valid_idx]

            valid_targets = {}
            for task, target_series in targets.items():
                if task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                    y = target_series.loc[valid_idx].dropna()
                    X_clean = X.loc[y.index]
                    if len(y) >= 50:  # 十分なサンプル数
                        valid_targets[task] = y

            print(f"最適化対象: {len(valid_targets)}タスク")

            # 最適化実行（1つのタスクでテスト）
            if valid_targets:
                task, y = next(iter(valid_targets.items()))
                X_clean = X.loc[y.index]

                print(f"\n最適化実行: {task.value} (サンプル数: {len(y)})")

                # Random Forest最適化
                if len(y) >= 30:
                    baseline_key = f"RandomForest_{task.value}"
                    baseline_score = baseline_scores.get(baseline_key, 0.5)

                    rf_result = await optimizer.optimize_model(
                        symbol, ModelType.RANDOM_FOREST, task, X_clean, y,
                        baseline_score=baseline_score, method='grid'
                    )

                    print(f"\nRandom Forest最適化結果:")
                    print(f"  最適スコア: {rf_result.best_score:.4f}")
                    print(f"  改善率: {rf_result.improvement:.2f}%")
                    print(f"  最適化時間: {rf_result.optimization_time:.1f}秒")
                    print(f"  最適パラメータ:")
                    for param, value in list(rf_result.best_params.items())[:5]:
                        print(f"    {param}: {value}")

                    # 重要パラメータ
                    if rf_result.param_importance:
                        top_params = sorted(rf_result.param_importance.items(),
                                          key=lambda x: x[1], reverse=True)[:3]
                        print(f"  重要パラメータ:")
                        for param, importance in top_params:
                            print(f"    {param}: {importance:.3f}")

            else:
                print("⚠️ 最適化に十分なデータがありません")

        except Exception as e:
            print(f"❌ 統合テストエラー: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("⚠️ MLモデルシステムが利用できません")

    # 独立テスト（ダミーデータ）
    print(f"\n[ 独立テスト（ダミーデータ）]")

    try:
        from sklearn.datasets import make_classification, make_regression

        # 分類データ
        X_cls, y_cls = make_classification(n_samples=200, n_features=10,
                                          n_informative=5, random_state=42)
        X_cls = pd.DataFrame(X_cls)
        y_cls = pd.Series(y_cls)

        cls_result = await optimizer.optimize_model(
            "TEST", ModelType.RANDOM_FOREST, PredictionTask.PRICE_DIRECTION,
            X_cls, y_cls, baseline_score=0.6, method='grid'
        )

        print(f"分類最適化:")
        print(f"  スコア: {cls_result.best_score:.4f}")
        print(f"  改善率: {cls_result.improvement:.2f}%")

        # システムサマリー
        print(f"\n[ システムサマリー ]")
        summary = await optimizer.get_optimization_summary()

        print(f"最適化実行回数: {summary['optimization_count']}")
        print(f"最大改善率: {summary['max_improvement']:.2f}%")
        print(f"平均改善率: {summary['avg_improvement']:.2f}%")

        if summary['important_params']:
            print(f"重要パラメータ（上位3）:")
            for param in summary['important_params'][:3]:
                print(f"  {param['parameter']}: {param['importance']:.3f}")

    except Exception as e:
        print(f"❌ 独立テストエラー: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== ハイパーパラメータ最適化 テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_hyperparameter_optimization())