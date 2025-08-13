#!/usr/bin/env python3
"""
RandomForestModel Parallel Hyperparameter Optimization Tests

Issue #701対応: RandomForestModelハイパーパラメータ最適化並列化テスト
"""

import pytest
import numpy as np
import os
import multiprocessing as mp
from unittest.mock import patch, Mock

import sys
sys.path.append('C:/gemini-desktop/day_trade/src')

from day_trade.ml.base_models.random_forest_model import RandomForestModel


class TestRandomForestParallel:
    """Issue #701: RandomForestModel並列化テスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n_samples, n_features = 50, 5  # 高速テスト用
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :2], axis=1) + 0.1 * np.random.randn(n_samples)
        return X, y

    def test_optimize_gridsearch_parallel_jobs_full_parallel_rf(self):
        """Issue #701: RandomForest完全並列時のGridSearchCV最適化"""
        config = {'n_jobs': -1, 'enable_hyperopt': True}
        model = RandomForestModel(config)

        optimal_jobs = model._optimize_gridsearch_parallel_jobs()
        cpu_count = mp.cpu_count()

        if cpu_count >= 8:
            assert optimal_jobs == max(2, cpu_count // 2)
        elif cpu_count >= 4:
            assert optimal_jobs == 2
        else:
            assert optimal_jobs == 1

    def test_optimize_gridsearch_parallel_jobs_serial_rf(self):
        """Issue #701: RandomForest直列時のGridSearchCV最適化"""
        config = {'n_jobs': 1, 'enable_hyperopt': True}
        model = RandomForestModel(config)

        optimal_jobs = model._optimize_gridsearch_parallel_jobs()

        # RandomForestが直列の場合、GridSearchCVは完全並列化
        assert optimal_jobs == -1

    def test_optimize_gridsearch_parallel_jobs_limited_parallel(self):
        """Issue #701: RandomForest限定並列時のGridSearchCV最適化"""
        config = {'n_jobs': 2, 'enable_hyperopt': True}
        model = RandomForestModel(config)

        optimal_jobs = model._optimize_gridsearch_parallel_jobs()
        cpu_count = mp.cpu_count()
        expected = min(max(1, cpu_count - 2), cpu_count // 2)

        assert optimal_jobs == expected

    def test_environment_variable_override(self):
        """Issue #701: 環境変数による設定オーバーライド"""
        with patch.dict(os.environ, {'GRIDSEARCH_N_JOBS': '4'}):
            config = {'n_jobs': -1}
            model = RandomForestModel(config)

            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == 4

    def test_invalid_environment_variable(self):
        """Issue #701: 無効な環境変数の処理"""
        with patch.dict(os.environ, {'GRIDSEARCH_N_JOBS': 'invalid'}):
            config = {'n_jobs': -1}
            model = RandomForestModel(config)

            # 無効な環境変数は無視され、通常のロジックが実行される
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs > 0  # 有効な値が返される

    def test_error_handling_in_optimization(self):
        """Issue #701: 最適化エラーハンドリング"""
        config = {'n_jobs': -1}
        model = RandomForestModel(config)

        # multiprocessingモジュールを一時的にモック
        with patch('multiprocessing.cpu_count', side_effect=Exception("Test error")):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            # エラー時はデフォルト値（1）が返される
            assert optimal_jobs == 1

    @patch('day_trade.ml.base_models.random_forest_model.GridSearchCV')
    def test_gridsearch_parallel_configuration(self, mock_gridsearch, sample_data):
        """Issue #701: GridSearchCV並列設定の確認"""
        X, y = sample_data

        # モックGridSearchCVを設定
        mock_gs_instance = Mock()
        mock_gs_instance.fit.return_value = None
        mock_gs_instance.best_params_ = {'n_estimators': 100, 'max_depth': 10}
        mock_gs_instance.best_score_ = -0.5
        mock_gs_instance.best_estimator_ = Mock()
        mock_gridsearch.return_value = mock_gs_instance

        config = {'n_jobs': -1, 'enable_hyperopt': True, 'verbose': False}
        model = RandomForestModel(config)

        # ハイパーパラメータ最適化実行
        model.fit(X, y)

        # GridSearchCVが適切な並列設定で呼ばれたかチェック
        mock_gridsearch.assert_called_once()
        call_args = mock_gridsearch.call_args

        # n_jobsパラメータの確認
        assert 'n_jobs' in call_args.kwargs
        n_jobs_used = call_args.kwargs['n_jobs']
        assert n_jobs_used != 1  # 並列化されている（1ではない）
        assert isinstance(n_jobs_used, int)

    def test_parallel_vs_serial_performance_concept(self, sample_data):
        """Issue #701: 並列vs直列パフォーマンス概念テスト"""
        X, y = sample_data

        # 非常に小さなパラメータ空間でテスト
        base_config = {
            'enable_hyperopt': True,
            'cv_folds': 2,
            'verbose': False
        }

        # 直列設定
        config_serial = {**base_config, 'n_jobs': 1}
        model_serial = RandomForestModel(config_serial)
        serial_jobs = model_serial._optimize_gridsearch_parallel_jobs()

        # 並列設定
        config_parallel = {**base_config, 'n_jobs': -1}
        model_parallel = RandomForestModel(config_parallel)
        parallel_jobs = model_parallel._optimize_gridsearch_parallel_jobs()

        # 並列設定の方がより多くのジョブを使用するか、全CPU使用
        if serial_jobs != -1 and parallel_jobs != -1:
            # 具体的数値の場合の比較
            assert parallel_jobs >= serial_jobs or parallel_jobs == -1
        else:
            # いずれかが-1（全CPU使用）の場合
            assert serial_jobs == -1 or parallel_jobs >= 1

    def test_cpu_count_edge_cases(self):
        """Issue #701: CPU数エッジケースのテスト"""
        config = {'n_jobs': -1}
        model = RandomForestModel(config)

        # 低CPU数環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=1):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == 1  # 単一CPUでは直列実行

        # 高CPU数環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=32):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == 16  # CPU数の50%

        # 中程度CPU数環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=6):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == 2  # 標準マシン設定

    def test_config_validation(self):
        """Issue #701: 設定の妥当性確認"""
        # 基本設定
        config = {'n_jobs': -1, 'enable_hyperopt': True}
        model = RandomForestModel(config)

        # 設定が適切に保存されているか確認
        assert model.config['n_jobs'] == -1
        assert model.config['enable_hyperopt'] == True

        # 最適化メソッドが正常に動作するか
        optimal_jobs = model._optimize_gridsearch_parallel_jobs()
        assert isinstance(optimal_jobs, int)
        assert optimal_jobs >= -1  # -1 or 正の整数


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])