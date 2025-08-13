#!/usr/bin/env python3
"""
Issue #475対応: DynamicWeightingSystem予測・実績処理改善テスト

統一的なデータ処理と冗長性の排除の検証
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import time

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.dynamic_weighting_system import (
        DynamicWeightingSystem,
        DynamicWeightingConfig,
        MarketRegime
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingPredictionHandling:
    """Issue #475: 予測・実績処理改善テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["model_a", "model_b", "model_c"]
    
    @pytest.fixture
    def dws_system(self, model_names):
        """基本的なシステム"""
        return DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
    
    def test_normalize_to_array_single_values(self, dws_system):
        """Issue #475: 単一値の正規化テスト"""
        # float
        result = dws_system._normalize_to_array(42.5, "test_float")
        assert result.shape == (1,)
        assert result[0] == 42.5
        
        # int
        result = dws_system._normalize_to_array(42, "test_int")
        assert result.shape == (1,)
        assert result[0] == 42.0
        
        # numpy scalar
        result = dws_system._normalize_to_array(np.float64(42.5), "test_numpy_scalar")
        assert result.shape == (1,)
        assert result[0] == 42.5
    
    def test_normalize_to_array_sequences(self, dws_system):
        """Issue #475: 配列・リストの正規化テスト"""
        # list
        result = dws_system._normalize_to_array([1.1, 2.2, 3.3], "test_list")
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [1.1, 2.2, 3.3])
        
        # tuple
        result = dws_system._normalize_to_array((4.4, 5.5), "test_tuple")
        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, [4.4, 5.5])
        
        # numpy array
        result = dws_system._normalize_to_array(np.array([6.6, 7.7, 8.8]), "test_array")
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [6.6, 7.7, 8.8])
        
        # 2D array -> flatten to 1D
        result = dws_system._normalize_to_array(np.array([[1, 2], [3, 4]]), "test_2d")
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4])
    
    def test_normalize_to_array_invalid_data(self, dws_system):
        """Issue #475: 無効データの拒否テスト"""
        # None
        with pytest.raises(ValueError, match="None"):
            dws_system._normalize_to_array(None, "test_none")
        
        # NaN in single value
        with pytest.raises(ValueError, match="無効な値"):
            dws_system._normalize_to_array(float('nan'), "test_nan")
        
        # Inf in single value
        with pytest.raises(ValueError, match="無効な値"):
            dws_system._normalize_to_array(float('inf'), "test_inf")
        
        # NaN in array
        with pytest.raises(ValueError, match="無効な値"):
            dws_system._normalize_to_array([1.0, float('nan'), 3.0], "test_array_nan")
        
        # Empty list
        with pytest.raises(ValueError, match="空"):
            dws_system._normalize_to_array([], "test_empty")
        
        # Empty array
        with pytest.raises(ValueError, match="空"):
            dws_system._normalize_to_array(np.array([]), "test_empty_array")
    
    def test_update_performance_unified_processing(self, dws_system):
        """Issue #475: 統一的な処理テスト"""
        # 各種データ型の組み合わせテスト
        test_cases = [
            # (predictions, actuals, description)
            ({"model_a": 100.5, "model_b": 101, "model_c": 99.8}, 100.2, "全て単一値"),
            ({"model_a": [100.1, 100.2], "model_b": [101.1, 101.2], "model_c": [99.1, 99.2]}, [100.0, 100.1], "全てリスト"),
            ({"model_a": np.array([100.3]), "model_b": np.array([101.3]), "model_c": np.array([99.3])}, np.array([100.25]), "全てNumPy"),
            ({"model_a": 100.4, "model_b": [101.4, 101.5], "model_c": np.array([99.4])}, 100.35, "混合型"),
        ]
        
        for predictions, actuals, desc in test_cases:
            # エラーが発生しないことを確認
            dws_system.update_performance(predictions, actuals, int(time.time()))
            # データが蓄積されていることを確認
            assert len(dws_system.recent_actuals) > 0
            for model_name in predictions.keys():
                assert len(dws_system.recent_predictions[model_name]) > 0
    
    def test_update_performance_dimension_handling(self, dws_system):
        """Issue #475: 次元不一致処理テスト"""
        # 単一予測値 vs 複数実際値 -> 予測値を複製
        dws_system.update_performance(
            {"model_a": 100.0, "model_b": 101.0, "model_c": 99.0}, 
            [100.1, 100.2], 
            int(time.time())
        )
        
        # 複数予測値 vs 単一実際値 -> 実際値に合わせて予測値の最初を使用
        dws_system.update_performance(
            {"model_a": [100.1, 100.2, 100.3], "model_b": [101.1, 101.2, 101.3], "model_c": [99.1, 99.2, 99.3]}, 
            100.15, 
            int(time.time())
        )
        
        # データが正常に処理されることを確認
        assert len(dws_system.recent_actuals) > 0
    
    def test_update_performance_dimension_mismatch_error(self, dws_system):
        """Issue #475: 次元不一致エラーテスト"""
        # 異なるサイズの配列同士（自動調整不可）
        # モデル間で異なる次元の予測値は個別に処理されるため、
        # 一部のモデルでエラーが出ても処理は継続される
        dws_system.update_performance(
            {"model_a": [100.1, 100.2, 100.3], "model_b": [101.1, 101.2], "model_c": [99.1]}, 
            [100.0, 100.1], 
            int(time.time())
        )
        # エラーが発生したモデルを除いて処理が継続されることを確認
        # 有効なデータは処理される
        assert len(dws_system.recent_actuals) > 0
    
    def test_update_performance_batch(self, dws_system):
        """Issue #475: バッチ処理テスト"""
        batch_size = 5
        batch_predictions = [
            {"model_a": 100+i*0.1, "model_b": 101+i*0.1, "model_c": 99+i*0.1}
            for i in range(batch_size)
        ]
        batch_actuals = [100.05+i*0.1 for i in range(batch_size)]
        batch_timestamps = [int(time.time())+i for i in range(batch_size)]
        
        # バッチ処理実行
        dws_system.update_performance_batch(batch_predictions, batch_actuals, batch_timestamps)
        
        # データが正しく蓄積されることを確認
        assert len(dws_system.recent_actuals) == batch_size
        for model_name in batch_predictions[0].keys():
            assert len(dws_system.recent_predictions[model_name]) == batch_size
    
    def test_update_performance_batch_validation(self, dws_system):
        """Issue #475: バッチ処理検証テスト"""
        # バッチサイズ不一致
        with pytest.raises(ValueError, match="バッチサイズ不一致"):
            dws_system.update_performance_batch(
                [{"model_a": 100}], 
                [100, 101], 
                None
            )
        
        # タイムスタンプサイズ不一致
        with pytest.raises(ValueError, match="タイムスタンプのサイズ不一致"):
            dws_system.update_performance_batch(
                [{"model_a": 100}, {"model_a": 101}], 
                [100, 101], 
                [1]
            )
    
    def test_validate_input_data(self, dws_system):
        """Issue #475: データ検証テスト"""
        # 有効なデータ
        valid_predictions = {"model_a": 100.5, "model_b": 101.2, "model_c": 99.8}
        valid_actuals = 100.1
        
        report = dws_system.validate_input_data(valid_predictions, valid_actuals)
        assert report['valid'] is True
        assert len(report['errors']) == 0
        assert 'model_a' in report['model_stats']
        assert 'actuals' in report['data_shape']
    
    def test_validate_input_data_invalid(self, dws_system):
        """Issue #475: 無効データ検証テスト"""
        # 無効な予測データ
        invalid_predictions = {"model_a": float('nan'), "model_b": 101.2, "model_c": 99.8}
        valid_actuals = 100.1
        
        report = dws_system.validate_input_data(invalid_predictions, valid_actuals)
        assert report['valid'] is False
        assert len(report['errors']) > 0
        assert any('model_a' in error for error in report['errors'])
    
    def test_get_data_statistics(self, dws_system):
        """Issue #475: データ統計テスト"""
        # 初期状態
        stats = dws_system.get_data_statistics()
        assert stats['total_samples'] == 0
        assert len(stats['models']) == 0
        assert not stats['data_health']['sufficient_samples']
        
        # データを追加
        for i in range(15):
            dws_system.update_performance(
                {"model_a": 100+i*0.1, "model_b": 101+i*0.1, "model_c": 99+i*0.1}, 
                100.05+i*0.1, 
                int(time.time())+i
            )
        
        # 統計を確認
        stats = dws_system.get_data_statistics()
        assert stats['total_samples'] == 15
        assert len(stats['models']) == 3
        assert 'mean' in stats['actuals_stats']
        assert 'correlation' in stats['models']['model_a']
        assert stats['data_health']['sufficient_samples']
        assert stats['data_health']['all_models_active']
    
    def test_error_recovery_and_partial_processing(self, dws_system):
        """Issue #475: エラー回復と部分処理テスト"""
        # 一部のモデルデータが無効な場合の処理
        mixed_predictions = {
            "model_a": 100.5,           # 有効
            "model_b": float('nan'),    # 無効
            "model_c": [99.8, 99.9]     # 有効
        }
        valid_actuals = [100.1, 100.2]
        
        # エラーが発生しても処理が継続されることを確認
        dws_system.update_performance(mixed_predictions, valid_actuals, int(time.time()))
        
        # 有効なデータは処理されることを確認
        assert len(dws_system.recent_predictions["model_a"]) > 0
        assert len(dws_system.recent_predictions["model_c"]) > 0
        assert len(dws_system.recent_actuals) > 0


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingDataTypeIntegration:
    """Issue #475: データ型統合テスト"""
    
    def test_comprehensive_data_type_workflow(self):
        """Issue #475: 包括的データ型ワークフローテスト"""
        model_names = ["neural_net", "random_forest", "svm"]
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(
            min_samples_for_update=5,
            update_frequency=3,
            verbose=False
        ))
        
        # Step 1: 様々なデータ型での更新
        data_scenarios = [
            # シナリオ1: 全て単一値
            ({"neural_net": 100.1, "random_forest": 100.2, "svm": 100.0}, 100.05),
            # シナリオ2: 全てリスト
            ({"neural_net": [101.1, 101.2], "random_forest": [101.3, 101.4], "svm": [101.0, 101.1]}, [101.15, 101.25]),
            # シナリオ3: 混合型
            ({"neural_net": 102.1, "random_forest": [102.2], "svm": np.array([102.0])}, 102.05),
            # シナリオ4: 次元調整が必要なケース
            ({"neural_net": [103.1, 103.2, 103.3], "random_forest": 103.4, "svm": 103.0}, 103.15),
        ]
        
        for i, (predictions, actuals) in enumerate(data_scenarios):
            dws.update_performance(predictions, actuals, int(time.time())+i)
        
        # Step 2: データ統計確認
        stats = dws.get_data_statistics()
        assert stats['total_samples'] > 0
        assert len(stats['models']) == 3
        assert all(model in stats['models'] for model in model_names)
        
        # Step 3: バッチ処理テスト
        batch_predictions = [
            {"neural_net": 104+i*0.1, "random_forest": 104.1+i*0.1, "svm": 103.9+i*0.1}
            for i in range(5)
        ]
        batch_actuals = [104.05+i*0.1 for i in range(5)]
        
        dws.update_performance_batch(batch_predictions, batch_actuals)
        
        # Step 4: 最終確認
        final_stats = dws.get_data_statistics()
        assert final_stats['total_samples'] > stats['total_samples']
        assert final_stats['data_health']['all_models_active']
        assert final_stats['data_health']['sufficient_samples']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])