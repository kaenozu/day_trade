#!/usr/bin/env python3
"""
SVRModel Parallel Hyperparameter Optimization Tests

Issue #699対応: SVRModelハイパーパラメータ最適化並列化テスト
"""

import pytest
import numpy as np
import os
import multiprocessing as mp
from unittest.mock import patch, Mock

import sys
sys.path.append('C:/gemini-desktop/day_trade/src')

from day_trade.ml.base_models.svr_model import SVRModel


class TestSVRParallel:
    """Issue #699: SVRModel並列化テスト"""
    
    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n_samples, n_features = 100, 5  # 高速テスト用
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :2], axis=1) + 0.1 * np.random.randn(n_samples)
        return X, y
    
    def test_optimize_gridsearch_parallel_jobs_high_performance(self):
        """Issue #699: 高性能マシンでの並列化最適化"""
        config = {'enable_hyperopt': True, 'verbose': False}
        model = SVRModel(config)
        
        # 高性能マシン（16コア）をシミュレート
        with patch('multiprocessing.cpu_count', return_value=16):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected_jobs = max(4, int(16 * 0.75))  # 75%使用
            assert optimal_jobs == expected_jobs
    
    def test_optimize_gridsearch_parallel_jobs_standard_performance(self):
        """Issue #699: 標準マシンでの並列化最適化"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 標準マシン（8コア）をシミュレート
        with patch('multiprocessing.cpu_count', return_value=8):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected_jobs = max(2, int(8 * 0.6))  # 60%使用
            assert optimal_jobs == expected_jobs
    
    def test_optimize_gridsearch_parallel_jobs_medium_performance(self):
        """Issue #699: 中性能マシンでの並列化最適化"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 中性能マシン（4コア）をシミュレート
        with patch('multiprocessing.cpu_count', return_value=4):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected_jobs = max(2, 4 // 2)  # 50%使用
            assert optimal_jobs == expected_jobs
    
    def test_optimize_gridsearch_parallel_jobs_low_performance(self):
        """Issue #699: 低性能マシンでの並列化最適化"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 低性能マシン（2コア）をシミュレート
        with patch('multiprocessing.cpu_count', return_value=2):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == -1  # 全CPU使用
    
    def test_environment_variable_override(self):
        """Issue #699: 環境変数による設定オーバーライド"""
        with patch.dict(os.environ, {'SVR_GRIDSEARCH_N_JOBS': '6'}):
            config = {'enable_hyperopt': True}
            model = SVRModel(config)
            
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == 6
    
    def test_invalid_environment_variable(self):
        """Issue #699: 無効な環境変数の処理"""
        with patch.dict(os.environ, {'SVR_GRIDSEARCH_N_JOBS': 'invalid'}):
            config = {'enable_hyperopt': True}
            model = SVRModel(config)
            
            # 無効な環境変数は無視され、通常のロジックが実行される
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs > 0  # 有効な値が返される
    
    def test_cache_size_impact_on_parallelization(self):
        """Issue #699: キャッシュサイズが並列化に与える影響"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 大きなキャッシュサイズを設定
        model._optimal_cache_size = 1500  # 1.5GB
        
        with patch('multiprocessing.cpu_count', return_value=12):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            # 大きなキャッシュサイズのため並列度が制限される
            expected_max = max(2, 12 // 3)  # CPU数の1/3
            assert optimal_jobs <= expected_max
    
    def test_small_cache_size_no_restriction(self):
        """Issue #699: 小さなキャッシュサイズでは並列度制限なし"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 小さなキャッシュサイズを設定
        model._optimal_cache_size = 200  # 200MB
        
        with patch('multiprocessing.cpu_count', return_value=8):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected_jobs = max(2, int(8 * 0.6))  # 通常の60%使用
            assert optimal_jobs == expected_jobs
    
    def test_error_handling_in_optimization(self):
        """Issue #699: 最適化エラーハンドリング"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # multiprocessingモジュールを一時的にモック
        with patch('multiprocessing.cpu_count', side_effect=Exception("Test error")):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            # エラー時はデフォルト値（2）が返される
            assert optimal_jobs == 2
    
    def test_gridsearch_parallel_configuration_direct(self, sample_data):
        """Issue #699: GridSearchCV並列設定の直接確認"""
        X, y = sample_data
        
        config = {'enable_hyperopt': True, 'verbose': True}
        model = SVRModel(config)
        
        # 並列化最適化メソッドの直接テスト
        optimal_jobs = model._optimize_gridsearch_parallel_jobs()
        
        # n_jobsが1以外（並列化されている）であることを確認
        assert optimal_jobs != 1  # 元の設定（1）から変更されている
        assert isinstance(optimal_jobs, int)
        assert optimal_jobs >= -1  # -1 or 正の整数
        
        # システムCPU数に応じた適切な値であることを確認
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        
        if cpu_count >= 16:
            expected_range = range(4, int(cpu_count * 0.75) + 1)
        elif cpu_count >= 8:
            expected_range = range(2, int(cpu_count * 0.6) + 1)
        elif cpu_count >= 4:
            expected_range = range(2, cpu_count // 2 + 1)
        else:
            expected_range = [-1]  # 全CPU使用
        
        assert optimal_jobs in expected_range or optimal_jobs == -1
    
    def test_parallel_vs_serial_performance_concept(self):
        """Issue #699: 並列vs直列パフォーマンス概念テスト"""
        config_base = {'enable_hyperopt': True, 'verbose': False}
        
        # 高性能マシンでのテスト
        with patch('multiprocessing.cpu_count', return_value=8):
            model_high = SVRModel(config_base)
            parallel_jobs = model_high._optimize_gridsearch_parallel_jobs()
            
            # 並列化が有効になっている
            assert parallel_jobs > 1 or parallel_jobs == -1
            
        # 低性能マシンでのテスト
        with patch('multiprocessing.cpu_count', return_value=2):
            model_low = SVRModel(config_base)
            low_perf_jobs = model_low._optimize_gridsearch_parallel_jobs()
            
            # 低性能でも並列化される（全CPU使用）
            assert low_perf_jobs == -1
    
    def test_cpu_count_edge_cases(self):
        """Issue #699: CPU数エッジケースのテスト"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 単一CPU環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=1):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            assert optimal_jobs == -1  # 全CPU使用（=1コア）
        
        # 超高性能環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=64):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected = max(4, int(64 * 0.75))  # 75%使用
            assert optimal_jobs == expected
        
        # 中程度CPU数環境をシミュレート
        with patch('multiprocessing.cpu_count', return_value=6):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            expected = max(2, 6 // 2)  # 50%使用
            assert optimal_jobs == expected
    
    def test_configuration_validation(self):
        """Issue #699: 設定の妥当性確認"""
        # 基本設定
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 設定が適切に保存されているか確認
        assert model.config['enable_hyperopt'] == True
        
        # 最適化メソッドが正常に動作するか
        optimal_jobs = model._optimize_gridsearch_parallel_jobs()
        assert isinstance(optimal_jobs, int)
        assert optimal_jobs >= -1  # -1 or 正の整数
    
    def test_integration_parallel_hyperopt_in_training(self, sample_data):
        """Issue #699: 学習統合での並列ハイパーパラメータ最適化"""
        X, y = sample_data
        
        config = {
            'enable_hyperopt': True,
            'cv_folds': 2,  # 高速化
            'verbose': False
        }
        model = SVRModel(config)
        
        # システムCPU数を制御して一定の結果を保証
        with patch('multiprocessing.cpu_count', return_value=4):
            # 学習実行
            results = model.fit(X, y)
            
            # 正常に学習が完了している
            assert model.is_trained
            assert 'training_time' in results
            assert results['training_time'] > 0
            
            # 最適化が実行され、最適パラメータが設定されている
            assert hasattr(model, 'best_params')
            assert len(model.best_params) > 0
    
    def test_memory_efficiency_consideration(self):
        """Issue #699: メモリ効率性の考慮"""
        config = {'enable_hyperopt': True}
        model = SVRModel(config)
        
        # 大きなキャッシュサイズと高CPU数の組み合わせ
        model._optimal_cache_size = 2000  # 2GB
        
        with patch('multiprocessing.cpu_count', return_value=24):
            optimal_jobs = model._optimize_gridsearch_parallel_jobs()
            
            # メモリ制約により並列度が制限される
            max_allowed = max(2, 24 // 3)  # CPU数の1/3
            assert optimal_jobs <= max_allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])