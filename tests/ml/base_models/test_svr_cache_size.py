#!/usr/bin/env python3
"""
SVRModel Dynamic Cache Size Tests

Issue #700対応: SVRModelの動的キャッシュサイズ調整テスト
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

import sys
sys.path.append('C:/gemini-desktop/day_trade/src')

from day_trade.ml.base_models.svr_model import SVRModel


class TestSVRCacheSize:
    """Issue #700: SVRModel動的キャッシュサイズテスト"""
    
    @pytest.fixture
    def small_dataset(self):
        """小規模テストデータ"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
        return X, y
    
    @pytest.fixture
    def large_dataset(self):
        """大規模テストデータ"""
        np.random.seed(42)
        X = np.random.randn(5000, 20)
        y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(5000)
        return X, y
    
    def test_default_auto_cache_size_enabled(self):
        """Issue #700: デフォルトで自動キャッシュサイズが有効"""
        model = SVRModel()
        assert model.config['auto_cache_size'] == True
        assert model.config['cache_size'] is None
    
    def test_manual_cache_size_override(self):
        """Issue #700: 手動キャッシュサイズ設定のオーバーライド"""
        config = {'cache_size': 500, 'auto_cache_size': False}
        model = SVRModel(config)
        
        # 手動設定が優先される
        assert model.config['cache_size'] == 500
        assert model.config['auto_cache_size'] == False
    
    def test_calculate_optimal_cache_size_rbf_kernel(self, large_dataset):
        """Issue #700: RBFカーネル用の最適キャッシュサイズ計算"""
        X, y = large_dataset
        config = {'kernel': 'rbf', 'auto_cache_size': True}
        model = SVRModel(config)
        
        cache_size = model._calculate_optimal_cache_size(X)
        
        # RBFカーネルは大きなキャッシュを使用
        assert cache_size >= model.config['min_cache_size']
        assert cache_size <= model.config['max_cache_size']
    
    def test_calculate_optimal_cache_size_linear_kernel(self, large_dataset):
        """Issue #700: リニアカーネル用の最適キャッシュサイズ計算"""
        X, y = large_dataset
        config = {'kernel': 'linear', 'auto_cache_size': True}
        model = SVRModel(config)
        
        cache_size_linear = model._calculate_optimal_cache_size(X)
        
        # リニアカーネルは小さなキャッシュで十分
        config_rbf = {'kernel': 'rbf', 'auto_cache_size': True}
        model_rbf = SVRModel(config_rbf)
        cache_size_rbf = model_rbf._calculate_optimal_cache_size(X)
        
        # リニアカーネルのキャッシュサイズ <= RBFカーネル
        assert cache_size_linear <= cache_size_rbf
    
    def test_data_size_impact_on_cache_size(self, small_dataset, large_dataset):
        """Issue #700: データサイズがキャッシュサイズに与える影響"""
        config = {'kernel': 'rbf', 'auto_cache_size': True}
        model = SVRModel(config)
        
        X_small, _ = small_dataset
        X_large, _ = large_dataset
        
        cache_small = model._calculate_optimal_cache_size(X_small)
        cache_large = model._calculate_optimal_cache_size(X_large)
        
        # 大規模データセットは同じまたはより大きなキャッシュを使用
        assert cache_large >= cache_small
    
    @patch('psutil.virtual_memory')
    def test_memory_constraint_application(self, mock_memory, large_dataset):
        """Issue #700: メモリ制約の適用"""
        X, y = large_dataset
        
        # 限られたメモリ環境をシミュレート
        mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB
        
        config = {'kernel': 'rbf', 'cache_memory_ratio': 0.2}  # 20%使用
        model = SVRModel(config)
        
        cache_size = model._calculate_optimal_cache_size(X)
        expected_max = 1024 * 0.2  # 204.8MB
        
        # メモリ制約が適用されている
        assert cache_size <= expected_max + 50  # 多少の余裕を持たせる
    
    def test_cache_size_bounds_enforcement(self, large_dataset):
        """Issue #700: キャッシュサイズ境界の強制"""
        X, y = large_dataset
        
        config = {
            'kernel': 'rbf',
            'min_cache_size': 150,
            'max_cache_size': 300,
            'auto_cache_size': True
        }
        model = SVRModel(config)
        
        cache_size = model._calculate_optimal_cache_size(X)
        
        # 境界が守られている
        assert cache_size >= 150
        assert cache_size <= 300
    
    def test_auto_cache_size_disabled(self, large_dataset):
        """Issue #700: 自動キャッシュサイズ無効時の動作"""
        X, y = large_dataset
        
        config = {'auto_cache_size': False, 'cache_size': 400}
        model = SVRModel(config)
        
        cache_size = model._calculate_optimal_cache_size(X)
        
        # 固定値が返される
        assert cache_size == 400
    
    @patch('psutil.virtual_memory', side_effect=ImportError("psutil not available"))
    def test_psutil_fallback(self, mock_psutil, large_dataset):
        """Issue #700: psutil未使用環境でのフォールバック"""
        X, y = large_dataset
        
        model = SVRModel({'kernel': 'rbf'})
        cache_size = model._calculate_optimal_cache_size(X)
        
        # フォールバック動作（エラーなく完了）
        assert isinstance(cache_size, float)
        assert cache_size >= 100  # 最小値以上
    
    def test_get_svr_params_with_dynamic_cache(self, large_dataset):
        """Issue #700: SVRパラメータ取得での動的キャッシュサイズ"""
        X, y = large_dataset
        
        config = {'cache_size': None, 'auto_cache_size': True, 'kernel': 'rbf'}
        model = SVRModel(config)
        
        params = model._get_svr_params(X)
        
        # 動的に計算されたキャッシュサイズが設定されている
        assert 'cache_size' in params
        assert params['cache_size'] >= model.config['min_cache_size']
        assert params['cache_size'] <= model.config['max_cache_size']
        
        # 計算結果がキャッシュされている
        assert model._optimal_cache_size is not None
        assert model._optimal_cache_size == params['cache_size']
    
    def test_get_svr_params_with_fixed_cache(self, large_dataset):
        """Issue #700: 固定キャッシュサイズでのSVRパラメータ取得"""
        X, y = large_dataset
        
        config = {'cache_size': 250, 'auto_cache_size': True}
        model = SVRModel(config)
        
        params = model._get_svr_params(X)
        
        # 設定済みキャッシュサイズが使用される
        assert params['cache_size'] == 250
    
    def test_error_handling_in_cache_calculation(self, large_dataset):
        """Issue #700: キャッシュサイズ計算でのエラーハンドリング"""
        X, y = large_dataset
        
        model = SVRModel({'kernel': 'rbf'})
        
        # 異常なデータでテスト
        with patch.object(X, 'shape', side_effect=Exception("Test error")):
            cache_size = model._calculate_optimal_cache_size(X)
            
            # エラー時はフォールバック値が返される
            assert cache_size == model.config['min_cache_size']
    
    def test_integration_cache_size_in_training(self, small_dataset):
        """Issue #700: 学習統合でのキャッシュサイズ使用"""
        X, y = small_dataset
        
        config = {
            'auto_cache_size': True,
            'kernel': 'rbf', 
            'enable_hyperopt': False,  # 高速化
            'verbose': False
        }
        model = SVRModel(config)
        
        # 学習実行
        results = model.fit(X, y)
        
        # 正常に学習が完了し、動的キャッシュサイズが使用された
        assert model.is_trained
        assert model._optimal_cache_size is not None
        assert model._optimal_cache_size >= model.config['min_cache_size']
    
    def test_kernel_specific_cache_optimization(self, large_dataset):
        """Issue #700: カーネル別キャッシュ最適化"""
        X, y = large_dataset
        
        # 各カーネルタイプでテスト
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        cache_sizes = {}
        
        for kernel in kernels:
            config = {'kernel': kernel, 'auto_cache_size': True}
            model = SVRModel(config)
            cache_sizes[kernel] = model._calculate_optimal_cache_size(X)
        
        # リニアカーネルが最小、非線形カーネルがより大きい
        assert cache_sizes['linear'] <= cache_sizes['rbf']
        assert cache_sizes['linear'] <= cache_sizes['poly']
        assert cache_sizes['linear'] <= cache_sizes['sigmoid']
        
        # すべてのカーネルで有効な値が得られる
        for kernel, cache_size in cache_sizes.items():
            assert cache_size >= 100  # 最小値
            assert cache_size <= 2000  # 最大値
    
    def test_configuration_validation(self):
        """Issue #700: 設定の妥当性確認"""
        # デフォルト設定
        model = SVRModel()
        assert model.config['min_cache_size'] == 100
        assert model.config['max_cache_size'] == 2000
        assert model.config['cache_memory_ratio'] == 0.1
        assert model.config['auto_cache_size'] == True
        
        # カスタム設定
        config = {
            'min_cache_size': 50,
            'max_cache_size': 1000,
            'cache_memory_ratio': 0.05
        }
        model_custom = SVRModel(config)
        assert model_custom.config['min_cache_size'] == 50
        assert model_custom.config['max_cache_size'] == 1000
        assert model_custom.config['cache_memory_ratio'] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])