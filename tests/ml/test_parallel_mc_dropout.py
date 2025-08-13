#!/usr/bin/env python3
"""
Monte Carlo Dropout and Permutation Importance Parallelization Tests

Issue #695対応: DeepLearningModelsMC Dropout並列化テスト
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch, MagicMock
import sys

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.deep_learning_models import (
        TransformerModel,
        LSTMModel,
        ModelConfig,
        JOBLIB_AVAILABLE
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestMCDropoutParallelization:
    """Issue #695: Monte Carlo Dropout並列化テスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return ModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            epochs=1,
            batch_size=4
        )
    
    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        data = pd.DataFrame({
            'Open': np.random.randn(50),
            'High': np.random.randn(50),
            'Low': np.random.randn(50),
            'Close': np.random.randn(50),
            'Volume': np.random.randn(50)
        })
        return data
    
    def test_mc_dropout_parallel_jobs_optimization(self, config):
        """Issue #695: Monte Carlo Dropout並列ジョブ数最適化テスト"""
        model = TransformerModel(config)
        
        # 少数サンプル
        assert model._optimize_mc_dropout_parallel_jobs(5) == 1
        
        # 中程度サンプル
        jobs_50 = model._optimize_mc_dropout_parallel_jobs(50)
        assert 1 <= jobs_50 <= 4
        
        # 大量サンプル
        jobs_500 = model._optimize_mc_dropout_parallel_jobs(500)
        assert jobs_500 >= 2
        
        # ユーザー指定並列数
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=1) == 1
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=-1) > 1
    
    def test_permutation_parallel_jobs_optimization(self, config):
        """Issue #695: Permutation Importance並列ジョブ数最適化テスト"""
        model = TransformerModel(config)
        
        # 少数特徴量
        assert model._optimize_permutation_parallel_jobs(2) == 1
        
        # 中程度特徴量
        jobs_5 = model._optimize_permutation_parallel_jobs(5)
        assert 1 <= jobs_5 <= 5
        
        # ユーザー指定並列数
        assert model._optimize_permutation_parallel_jobs(5, n_jobs=1) == 1
        assert model._optimize_permutation_parallel_jobs(5, n_jobs=3) <= 3
    
    @pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib not available")
    def test_parallel_monte_carlo_dropout_execution(self, config, sample_data):
        """Issue #695: 並列Monte Carlo Dropout実行テスト"""
        model = TransformerModel(config)
        
        # 簡易モデルを設定（テスト用）
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        # 並列実行テスト
        num_samples = 10
        n_jobs = 2
        
        predictions_list = model._parallel_monte_carlo_dropout(sample_data, num_samples, n_jobs)
        
        # 結果検証
        assert len(predictions_list) == num_samples
        assert all(isinstance(pred, np.ndarray) for pred in predictions_list)
        
        # 予測形状の一貫性確認
        first_shape = predictions_list[0].shape
        assert all(pred.shape == first_shape for pred in predictions_list)
    
    @pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib not available")
    def test_parallel_permutation_importance_execution(self, config, sample_data):
        """Issue #695: 並列Permutation Importance実行テスト"""
        model = TransformerModel(config)
        
        # 簡易モデルを設定
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        # テストデータ準備
        X, y = model.prepare_data(sample_data)
        baseline_pred = model._predict_internal(X)
        baseline_error = np.mean((baseline_pred - y) ** 2)
        
        feature_names = ["Open", "High", "Low", "Close", "Volume"]
        n_jobs = 2
        
        # 並列実行テスト
        importance_dict = model._parallel_permutation_importance(
            X, baseline_pred, baseline_error, feature_names, n_jobs
        )
        
        # 結果検証
        assert len(importance_dict) == len(feature_names)
        assert all(name in importance_dict for name in feature_names)
        assert all(isinstance(imp, float) for imp in importance_dict.values())
        assert all(imp >= 0 for imp in importance_dict.values())
    
    def test_predict_with_uncertainty_parallel_interface(self, config, sample_data):
        """Issue #695: predict_with_uncertainty並列インターフェーステスト"""
        model = TransformerModel(config)
        
        # 簡易モデル設定
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        # デフォルト並列化テスト
        with patch.object(model, '_parallel_monte_carlo_dropout') as mock_parallel:
            with patch.object(model, '_optimize_mc_dropout_parallel_jobs', return_value=2):
                mock_parallel.return_value = [np.array([1.0]), np.array([1.1])]
                
                result = model.predict_with_uncertainty(sample_data, num_samples=2)
                
                # 並列化メソッドが呼ばれることを確認
                if JOBLIB_AVAILABLE:
                    mock_parallel.assert_called_once()
                
                # 結果の形状確認
                assert hasattr(result, 'predictions')
                assert hasattr(result, 'confidence')
                assert hasattr(result, 'uncertainty')
    
    def test_get_feature_importance_parallel_interface(self, config, sample_data):
        """Issue #695: get_feature_importance並列インターフェーステスト"""
        model = TransformerModel(config)
        
        # 簡易モデル設定
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        # 並列化テスト
        with patch.object(model, '_parallel_permutation_importance') as mock_parallel:
            with patch.object(model, '_optimize_permutation_parallel_jobs', return_value=2):
                mock_parallel.return_value = {
                    'Open': 0.3, 'High': 0.2, 'Low': 0.2, 'Close': 0.2, 'Volume': 0.1
                }
                
                importance = model.get_feature_importance(sample_data)
                
                # 並列化メソッドが呼ばれることを確認
                if JOBLIB_AVAILABLE:
                    mock_parallel.assert_called_once()
                
                # 結果の正規化確認
                assert abs(sum(importance.values()) - 1.0) < 1e-6
    
    def test_parallel_fallback_to_serial(self, config, sample_data):
        """Issue #695: 並列化フォールバック（直列実行）テスト"""
        model = TransformerModel(config)
        
        # 簡易モデル設定
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        # joblib無効時のフォールバック
        with patch('day_trade.ml.deep_learning_models.JOBLIB_AVAILABLE', False):
            with patch('day_trade.ml.deep_learning_models.logger') as mock_logger:
                result = model.predict_with_uncertainty(sample_data, num_samples=5, n_jobs=2)
                
                # フォールバック警告が出ることを確認
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "直列実行にフォールバック" in warning_msg
                
                # 結果は正常に返される
                assert hasattr(result, 'predictions')
    
    def test_performance_improvement_monte_carlo(self, config, sample_data):
        """Issue #695: Monte Carlo Dropout性能向上テスト"""
        model = TransformerModel(config)
        
        # 簡易モデル設定
        model.model = {"fallback_weights": np.random.randn(500) * 0.1}
        model.is_trained = True
        
        num_samples = 20
        
        # 直列実行時間測定
        start_time = time.time()
        result_serial = model.predict_with_uncertainty(sample_data, num_samples=num_samples, n_jobs=1)
        serial_time = time.time() - start_time
        
        # 並列実行時間測定（joblib利用可能時）
        if JOBLIB_AVAILABLE:
            start_time = time.time()
            result_parallel = model.predict_with_uncertainty(sample_data, num_samples=num_samples, n_jobs=-1)
            parallel_time = time.time() - start_time
            
            # 並列化の効果確認（必ずしも高速化するとは限らないが、極端に遅くならないことを確認）
            assert parallel_time <= serial_time * 2.0
            
            # 結果の一致性確認（完全一致は期待しないが、形状は同じ）
            assert result_serial.predictions.shape == result_parallel.predictions.shape


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestParallelizationEdgeCases:
    """Issue #695: 並列化エッジケーステスト"""
    
    def test_single_sample_optimization(self):
        """Issue #695: 単一サンプル時の最適化テスト"""
        config = ModelConfig(epochs=1)
        model = TransformerModel(config)
        
        # 単一サンプルは直列実行になることを確認
        assert model._optimize_mc_dropout_parallel_jobs(1) == 1
        assert model._optimize_permutation_parallel_jobs(1) == 1
    
    def test_error_handling_in_optimization(self):
        """Issue #695: 最適化エラーハンドリングテスト"""
        config = ModelConfig(epochs=1)
        model = TransformerModel(config)
        
        # multiprocessing.cpu_count()エラー時のフォールバック
        with patch('day_trade.ml.deep_learning_models.mp.cpu_count', side_effect=Exception("Test error")):
            with patch('day_trade.ml.deep_learning_models.logger') as mock_logger:
                jobs = model._optimize_mc_dropout_parallel_jobs(100)
                
                # エラーログが出力される
                mock_logger.warning.assert_called()
                # フォールバック値（1）が返される
                assert jobs == 1
    
    def test_n_jobs_validation(self):
        """Issue #695: n_jobs引数バリデーションテスト"""
        config = ModelConfig(epochs=1)
        model = TransformerModel(config)
        
        # 不正なn_jobs値のハンドリング
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=0) == 1
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=-2) == 1
        
        # 正常なn_jobs値
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=2) == 2
        assert model._optimize_mc_dropout_parallel_jobs(100, n_jobs=-1) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])