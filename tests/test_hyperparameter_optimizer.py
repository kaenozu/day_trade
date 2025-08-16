#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Hyperparameter Optimizer
ハイパーパラメータ最適化システムのテストケース

pytestフレームワークを使用した構造化テスト
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# テスト対象のインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))

from hyperparameter_optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptimizationResult,
    HyperparameterSpace
)

# ML関連のインポート（条件付き）
try:
    from ml_prediction_models import ModelType, PredictionTask
    ML_AVAILABLE = True
except ImportError:
    # テスト用のダミー定義
    class ModelType:
        RANDOM_FOREST = "Random Forest"
        XGBOOST = "XGBoost"

        @property
        def value(self):
            return self

    class PredictionTask:
        PRICE_DIRECTION = "price_direction"
        PRICE_REGRESSION = "price_regression"

        @property
        def value(self):
            return self

    ML_AVAILABLE = False

# sklearn関連のインポート（条件付き）
try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class TestHyperparameterOptimizer:
    """HyperparameterOptimizerのテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリの作成"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def config_files(self, temp_dir):
        """テスト用設定ファイルの作成"""
        # 最適化設定
        config_path = temp_dir / "optimization_config.yaml"
        optimization_config = {
            'optimization_methods': {
                'grid_search': {
                    'method': 'grid',
                    'cv_folds': 2,
                    'max_iterations': 10,
                    'scoring': 'accuracy',
                    'n_jobs': 1,
                    'random_state': 42,
                    'enabled': True
                },
                'random_search': {
                    'method': 'random',
                    'cv_folds': 2,
                    'max_iterations': 20,
                    'scoring': 'accuracy',
                    'n_jobs': 1,
                    'random_state': 42,
                    'enabled': True
                }
            },
            'task_scoring': {
                'price_direction': {'primary': 'accuracy'},
                'price_regression': {'primary': 'r2'}
            },
            'baseline_scores': {
                'classification': {'accuracy': 0.5},
                'regression': {'r2': 0.0}
            },
            'storage': {
                'database_path': str(temp_dir / 'test_optimizer.db')
            },
            'advanced': {
                'parameter_space_limiting': {
                    'max_combinations_grid': 50
                }
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimization_config, f)

        # ハイパーパラメータ空間設定
        hyperparameter_path = temp_dir / "hyperparameter_spaces.yaml"
        hyperparameter_config = {
            'random_forest_classifier': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            },
            'random_forest_regressor': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            },
            'xgboost_classifier': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            },
            'xgboost_regressor': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2]
            }
        }

        with open(hyperparameter_path, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparameter_config, f)

        return config_path, hyperparameter_path

    @pytest.fixture
    def optimizer(self, config_files):
        """テスト用HyperparameterOptimizerインスタンス"""
        config_path, hyperparameter_path = config_files

        # sklearn利用可能性をチェック
        if not SKLEARN_AVAILABLE:
            pytest.skip("Scikit-learn not available")

        return HyperparameterOptimizer(
            config_path=config_path,
            hyperparameter_config_path=hyperparameter_path
        )

    def test_initialization(self, optimizer):
        """初期化テスト"""
        assert optimizer.config is not None
        assert optimizer.hyperparameter_spaces is not None
        assert optimizer.optimization_configs is not None
        assert len(optimizer.optimization_configs) >= 1

    def test_config_loading(self, optimizer):
        """設定読み込みテスト"""
        # 最適化設定の確認
        assert 'grid_search' in optimizer.optimization_configs
        assert 'random_search' in optimizer.optimization_configs

        # 設定内容の確認
        grid_config = optimizer.optimization_configs['grid_search']
        assert grid_config.method == 'grid'
        assert grid_config.cv_folds == 2
        assert grid_config.max_iterations == 10

    def test_hyperparameter_space_loading(self, optimizer):
        """ハイパーパラメータ空間読み込みテスト"""
        spaces = optimizer.hyperparameter_spaces

        assert 'random_forest_classifier' in spaces
        assert 'random_forest_regressor' in spaces
        assert 'xgboost_classifier' in spaces
        assert 'xgboost_regressor' in spaces

        # 内容の確認
        rf_clf_space = spaces['random_forest_classifier']
        assert 'n_estimators' in rf_clf_space
        assert 'max_depth' in rf_clf_space

    def test_get_baseline_score(self, optimizer):
        """ベースラインスコア取得テスト"""
        # 分類タスク
        baseline_cls = optimizer._get_baseline_score(
            PredictionTask.PRICE_DIRECTION, 'accuracy'
        )
        assert baseline_cls == 0.5

        # 回帰タスク
        baseline_reg = optimizer._get_baseline_score(
            PredictionTask.PRICE_REGRESSION, 'r2'
        )
        assert baseline_reg == 0.0

    def test_get_scoring_for_task(self, optimizer):
        """タスク別スコアリング取得テスト"""
        cls_scoring = optimizer._get_scoring_for_task(PredictionTask.PRICE_DIRECTION)
        assert cls_scoring == 'accuracy'

        reg_scoring = optimizer._get_scoring_for_task(PredictionTask.PRICE_REGRESSION)
        assert reg_scoring == 'r2'

    def test_convert_to_random_space(self, optimizer):
        """Random Search用パラメータ変換テスト"""
        param_space = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.2, 0.3],
            'criterion': ['gini', 'entropy']
        }

        random_space = optimizer._convert_to_random_space(param_space)

        assert 'n_estimators' in random_space
        assert 'max_depth' in random_space
        assert 'learning_rate' in random_space
        assert 'criterion' in random_space

    def test_limit_param_space(self, optimizer):
        """パラメータ空間制限テスト"""
        param_space = {
            'param1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'param2': ['a', 'b', 'c', 'd', 'e']
        }

        limited_space = optimizer._limit_param_space(param_space, max_values_per_param=3)

        assert len(limited_space['param1']) <= 3
        assert len(limited_space['param2']) <= 3

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_optimize_model_grid_search(self, optimizer):
        """Grid Search最適化テスト"""
        # テストデータ作成
        X, y = make_classification(n_samples=100, n_features=5,
                                 n_informative=3, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        # 最適化実行
        result = await optimizer.optimize_model(
            symbol="TEST",
            model_type=ModelType.RANDOM_FOREST,
            task=PredictionTask.PRICE_DIRECTION,
            X=X_df,
            y=y_series,
            method='grid'
        )

        # 結果検証
        assert isinstance(result, OptimizationResult)
        assert result.model_type == ModelType.RANDOM_FOREST.value
        assert result.task == PredictionTask.PRICE_DIRECTION.value
        assert result.best_score > 0
        assert isinstance(result.best_params, dict)
        assert len(result.cv_scores) > 0
        assert result.optimization_time > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_optimize_model_random_search(self, optimizer):
        """Random Search最適化テスト"""
        # テストデータ作成
        X, y = make_regression(n_samples=100, n_features=5,
                              noise=0.1, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        # 最適化実行
        with patch('hyperparameter_optimizer.SCIPY_AVAILABLE', True):
            result = await optimizer.optimize_model(
                symbol="TEST",
                model_type=ModelType.RANDOM_FOREST,
                task=PredictionTask.PRICE_REGRESSION,
                X=X_df,
                y=y_series,
                method='random'
            )

        # 結果検証
        assert isinstance(result, OptimizationResult)
        assert result.model_type == ModelType.RANDOM_FOREST.value
        assert result.task == PredictionTask.PRICE_REGRESSION.value
        assert isinstance(result.best_params, dict)
        assert len(result.cv_scores) > 0

    def test_calculate_param_importance_improved(self, optimizer):
        """改善されたパラメータ重要度計算テスト"""
        # モックCVResults作成
        cv_results = {
            'params': [
                {'n_estimators': 10, 'max_depth': 3},
                {'n_estimators': 20, 'max_depth': 3},
                {'n_estimators': 10, 'max_depth': 5},
                {'n_estimators': 20, 'max_depth': 5}
            ],
            'mean_test_score': [0.8, 0.85, 0.82, 0.87]
        }

        importance = optimizer._calculate_param_importance_improved(cv_results)

        assert isinstance(importance, dict)
        # 重要度の値は0-1の範囲
        for param, score in importance.items():
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_save_optimization_result(self, optimizer):
        """最適化結果保存テスト"""
        result = OptimizationResult(
            model_type="Random Forest",
            task="price_direction",
            best_params={'n_estimators': 100},
            best_score=0.85,
            cv_scores=[0.8, 0.85, 0.87],
            optimization_time=120.5,
            improvement=15.2,
            param_importance={'n_estimators': 0.8}
        )

        # 保存実行
        await optimizer._save_optimization_result(result, "TEST", "grid")

        # データベース存在確認
        assert optimizer.db_path.exists()

    def test_get_optimized_params(self, optimizer):
        """最適化済みパラメータ取得テスト"""
        # キャッシュにデータ設定
        cache_key = "TEST_Random Forest_price_direction"
        test_params = {'n_estimators': 200, 'max_depth': 10}
        optimizer.optimized_params[cache_key] = test_params

        # 取得テスト
        retrieved_params = optimizer.get_optimized_params(
            "TEST", ModelType.RANDOM_FOREST, PredictionTask.PRICE_DIRECTION
        )

        assert retrieved_params == test_params

    @pytest.mark.asyncio
    async def test_get_optimization_summary(self, optimizer):
        """最適化サマリー取得テスト"""
        summary = await optimizer.get_optimization_summary()

        assert isinstance(summary, dict)
        assert 'optimization_count' in summary
        assert 'recent_optimizations' in summary
        assert 'max_improvement' in summary
        assert 'avg_improvement' in summary
        assert 'important_params' in summary

class TestOptimizationConfig:
    """OptimizationConfigデータクラスのテスト"""

    def test_optimization_config_creation(self):
        """OptimizationConfig作成テスト"""
        config = OptimizationConfig(
            method='random',
            cv_folds=5,
            max_iterations=100,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            n_iter=50
        )

        assert config.method == 'random'
        assert config.cv_folds == 5
        assert config.max_iterations == 100
        assert config.scoring == 'accuracy'
        assert config.n_jobs == -1
        assert config.random_state == 42
        assert config.n_iter == 50

class TestOptimizationResult:
    """OptimizationResultデータクラスのテスト"""

    def test_optimization_result_creation(self):
        """OptimizationResult作成テスト"""
        result = OptimizationResult(
            model_type="XGBoost",
            task="price_regression",
            best_params={'learning_rate': 0.1, 'n_estimators': 100},
            best_score=0.892,
            cv_scores=[0.88, 0.89, 0.90],
            optimization_time=245.7,
            improvement=12.5,
            param_importance={'learning_rate': 0.95, 'n_estimators': 0.72}
        )

        assert result.model_type == "XGBoost"
        assert result.task == "price_regression"
        assert result.best_score == 0.892
        assert len(result.cv_scores) == 3
        assert result.optimization_time == 245.7
        assert result.improvement == 12.5
        assert len(result.param_importance) == 2

# 統合テスト
class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="Scikit-learn not available")
    async def test_full_optimization_workflow(self, temp_dir):
        """完全な最適化ワークフローテスト"""
        # 設定ファイル作成
        config_path = temp_dir / "integration_config.yaml"
        hyperparameter_path = temp_dir / "integration_hyperparams.yaml"

        # 簡易設定
        config = {
            'optimization_methods': {
                'grid_search': {
                    'method': 'grid', 'cv_folds': 2, 'max_iterations': 4,
                    'scoring': 'accuracy', 'n_jobs': 1, 'random_state': 42, 'enabled': True
                }
            },
            'task_scoring': {
                'price_direction': {'primary': 'accuracy'}
            },
            'baseline_scores': {
                'classification': {'accuracy': 0.5}
            },
            'storage': {'database_path': str(temp_dir / 'integration.db')},
            'advanced': {'parameter_space_limiting': {'max_combinations_grid': 10}}
        }

        hyperparams = {
            'random_forest_classifier': {
                'n_estimators': [10, 20],
                'max_depth': [3, 5]
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        with open(hyperparameter_path, 'w', encoding='utf-8') as f:
            yaml.dump(hyperparams, f)

        # オプティマイザー作成
        optimizer = HyperparameterOptimizer(
            config_path=config_path,
            hyperparameter_config_path=hyperparameter_path
        )

        # テストデータ作成
        X, y = make_classification(n_samples=50, n_features=3,
                                 n_informative=2, random_state=42)
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        # 最適化実行
        result = await optimizer.optimize_model(
            symbol="INTEGRATION_TEST",
            model_type=ModelType.RANDOM_FOREST,
            task=PredictionTask.PRICE_DIRECTION,
            X=X_df,
            y=y_series,
            method='grid'
        )

        # 結果検証
        assert result.best_score > 0
        assert result.optimization_time > 0

        # サマリー確認
        summary = await optimizer.get_optimization_summary()
        assert summary['optimization_count'] >= 0

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])