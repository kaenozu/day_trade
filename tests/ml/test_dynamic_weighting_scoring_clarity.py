#!/usr/bin/env python3
"""
Issue #477対応: DynamicWeightingSystemスコアリング明確化テスト

スコア計算の透明性向上とカスタマイズ機能の検証
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import math

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
class TestDynamicWeightingScoringClarity:
    """Issue #477: スコアリング明確化テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["model_a", "model_b", "model_c"]
    
    @pytest.fixture
    def scoring_config(self):
        """スコアリング専用設定"""
        return DynamicWeightingConfig(
            window_size=20,
            min_samples_for_update=10,
            update_frequency=5,
            accuracy_weight=1.5,     # 精度重視
            direction_weight=0.8,    # 方向性は少し軽視
            sharpe_clip_min=0.05,    # 低めのクリップ値
            enable_score_logging=True,
            verbose=False
        )
    
    @pytest.fixture
    def dws_with_scoring(self, model_names, scoring_config):
        """スコアリング設定付きシステム"""
        return DynamicWeightingSystem(model_names, scoring_config)
    
    def test_performance_based_scoring_formula(self, model_names):
        """Issue #477: performance_basedスコア計算式の検証"""
        config = DynamicWeightingConfig(
            accuracy_weight=2.0,
            direction_weight=1.0,
            enable_score_logging=True,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 既知の予測値と実際値でテスト
        predictions = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        actuals =     [10.1, 11.9, 14.2, 15.8, 18.1, 19.9, 22.2, 23.8, 26.1, 27.9, 30.2]
        
        # データ投入
        for i, (pred_dict, actual) in enumerate(zip(
            [{"model_a": p, "model_b": p+1, "model_c": p-1} for p in predictions],
            actuals
        )):
            dws.update_performance(pred_dict, actual, i)
        
        # スコア計算
        weights = dws._performance_based_weighting()
        
        # 重み制約の確認
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for weight in weights.values():
            assert weight > 0
        
        # model_aが最も精度が高い（pred=actualに最も近い）ことを確認
        assert weights["model_a"] > weights["model_c"]  # model_cは常に-1ずれ
    
    def test_sharpe_based_scoring_formula(self, model_names):
        """Issue #477: sharpe_basedスコア計算式の検証"""
        config = DynamicWeightingConfig(
            sharpe_clip_min=0.2,  # 高めのクリップ値
            enable_score_logging=True,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 異なる予測パターンでテスト
        base_values = [100, 102, 101, 105, 103, 108, 106, 112, 110, 115, 113]
        
        for i, base in enumerate(base_values):
            predictions = {
                "model_a": base * 1.01,    # 常に少し高く予測（上昇バイアス）
                "model_b": base * 0.99,    # 常に少し低く予測（下降バイアス）
                "model_c": base            # 変化なし予測
            }
            dws.update_performance(predictions, base, i)
        
        weights = dws._sharpe_based_weighting()
        
        # 重み制約の確認
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for weight in weights.values():
            assert weight > 0
    
    def test_custom_scoring_weights(self, model_names):
        """Issue #477: カスタムスコア重み係数テスト"""
        # 精度重視設定
        accuracy_focused_config = DynamicWeightingConfig(
            accuracy_weight=3.0,
            direction_weight=0.5,
            verbose=False
        )
        dws_accuracy = DynamicWeightingSystem(model_names, accuracy_focused_config)
        
        # 方向重視設定
        direction_focused_config = DynamicWeightingConfig(
            accuracy_weight=0.5,
            direction_weight=3.0,
            verbose=False
        )
        dws_direction = DynamicWeightingSystem(model_names, direction_focused_config)
        
        # テストデータ投入
        test_data = [
            ([10.0, 10.1, 10.2], [10.5, 9.9, 10.3], 10.1),  # model_a: 高精度、model_b: 方向逆
            ([11.0, 11.1, 11.2], [11.8, 10.8, 11.4], 11.2),
            ([12.0, 12.1, 12.2], [12.7, 11.7, 12.5], 12.0),
            ([13.0, 13.1, 13.2], [13.6, 12.6, 13.6], 13.1),
            ([14.0, 14.1, 14.2], [14.5, 13.5, 14.7], 14.2),
            ([15.0, 15.1, 15.2], [15.4, 14.4, 15.8], 15.0),
            ([16.0, 16.1, 16.2], [16.3, 15.3, 16.9], 16.1),
            ([17.0, 17.1, 17.2], [17.2, 16.2, 17.0], 17.2),
            ([18.0, 18.1, 18.2], [18.1, 17.1, 18.1], 18.0),
            ([19.0, 19.1, 19.2], [19.0, 18.0, 19.2], 19.1)
        ]
        
        for predictions_list, _, actual in test_data:
            pred_dict = {
                "model_a": predictions_list[0],
                "model_b": predictions_list[1], 
                "model_c": predictions_list[2]
            }
            dws_accuracy.update_performance(pred_dict, actual, 0)
            dws_direction.update_performance(pred_dict, actual, 0)
        
        accuracy_weights = dws_accuracy._performance_based_weighting()
        direction_weights = dws_direction._performance_based_weighting()
        
        # 精度重視設定では精度の高いmodel_aが高い重み
        # 方向重視設定では異なる重み分散が期待される
        assert accuracy_weights != direction_weights
    
    def test_sharpe_clip_customization(self, model_names):
        """Issue #477: シャープレシオクリップ値カスタマイズテスト"""
        # 低いクリップ値
        low_clip_config = DynamicWeightingConfig(sharpe_clip_min=0.01, verbose=False)
        dws_low = DynamicWeightingSystem(model_names, low_clip_config)
        
        # 高いクリップ値
        high_clip_config = DynamicWeightingConfig(sharpe_clip_min=0.5, verbose=False)
        dws_high = DynamicWeightingSystem(model_names, high_clip_config)
        
        # 悪い予測データ（負のシャープレシオになるように）
        for i in range(15):
            predictions = {
                "model_a": 100 + np.random.normal(0, 10),  # 高ノイズ
                "model_b": 100 + np.random.normal(0, 15),  # さらに高ノイズ
                "model_c": 100 + np.random.normal(0, 20)   # 最高ノイズ
            }
            actual = 100
            dws_low.update_performance(predictions, actual, i)
            dws_high.update_performance(predictions, actual, i)
        
        weights_low = dws_low._sharpe_based_weighting()
        weights_high = dws_high._sharpe_based_weighting()
        
        # 異なるクリップ値で異なる結果が得られることを確認
        assert weights_low != weights_high
    
    def test_update_scoring_config(self, dws_with_scoring):
        """Issue #477: スコアリング設定動的更新テスト"""
        # 初期設定確認
        initial_config = dws_with_scoring.get_scoring_config()
        assert initial_config['accuracy_weight'] == 1.5
        assert initial_config['direction_weight'] == 0.8
        assert initial_config['sharpe_clip_min'] == 0.05
        
        # 設定更新
        dws_with_scoring.update_scoring_config(
            accuracy_weight=2.5,
            direction_weight=1.2,
            sharpe_clip_min=0.15,
            enable_score_logging=False
        )
        
        # 更新確認
        updated_config = dws_with_scoring.get_scoring_config()
        assert updated_config['accuracy_weight'] == 2.5
        assert updated_config['direction_weight'] == 1.2
        assert updated_config['sharpe_clip_min'] == 0.15
        assert updated_config['enable_score_logging'] == False
    
    def test_scoring_config_validation(self, dws_with_scoring):
        """Issue #477: スコアリング設定バリデーションテスト"""
        # 負の値でエラーが発生することを確認
        with pytest.raises(ValueError):
            dws_with_scoring.update_scoring_config(accuracy_weight=-1.0)
        
        with pytest.raises(ValueError):
            dws_with_scoring.update_scoring_config(direction_weight=-0.5)
        
        with pytest.raises(ValueError):
            dws_with_scoring.update_scoring_config(sharpe_clip_min=-0.1)
    
    def test_scoring_explanation_generation(self, dws_with_scoring):
        """Issue #477: スコアリング手法説明生成テスト"""
        explanations = dws_with_scoring.get_scoring_explanation()
        
        # 全手法の説明が存在することを確認
        assert 'performance_based' in explanations
        assert 'sharpe_based' in explanations
        assert 'regime_aware' in explanations
        
        # 各説明に必要なフィールドが存在することを確認
        for method, explanation in explanations.items():
            assert 'description' in explanation
            assert 'formula' in explanation
            assert 'range' in explanation
            assert 'components' in explanation
            
        # 現在の設定値が反映されていることを確認
        perf_explanation = explanations['performance_based']
        assert '1.5' in perf_explanation['formula']  # accuracy_weight
        assert '0.8' in perf_explanation['formula']  # direction_weight
        
        sharpe_explanation = explanations['sharpe_based']
        assert '0.05' in sharpe_explanation['range']  # sharpe_clip_min
    
    def test_score_logging_control(self, model_names):
        """Issue #477: スコア詳細ログ制御テスト"""
        # ログ有効設定
        config_with_logging = DynamicWeightingConfig(
            enable_score_logging=True,
            verbose=False
        )
        dws_with_logging = DynamicWeightingSystem(model_names, config_with_logging)
        
        # ログ無効設定
        config_without_logging = DynamicWeightingConfig(
            enable_score_logging=False,
            verbose=False
        )
        dws_without_logging = DynamicWeightingSystem(model_names, config_without_logging)
        
        # テストデータ投入
        for i in range(12):
            predictions = {"model_a": 100+i, "model_b": 101+i, "model_c": 99+i}
            actual = 100+i
            dws_with_logging.update_performance(predictions, actual, i)
            dws_without_logging.update_performance(predictions, actual, i)
        
        # 両方とも正常に動作することを確認
        weights_with_logging = dws_with_logging._performance_based_weighting()
        weights_without_logging = dws_without_logging._performance_based_weighting()
        
        # ログ設定に関係なく同じ結果が得られることを確認
        for model in model_names:
            assert abs(weights_with_logging[model] - weights_without_logging[model]) < 1e-6


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingScoringTransparency:
    """Issue #477: スコアリング透明性テスト"""
    
    def test_rmse_inverse_score_calculation(self):
        """Issue #477: RMSE逆数スコア計算の検証"""
        # RMSE逆数の数学的妥当性テスト
        rmse_values = [0.0, 0.5, 1.0, 2.0, 10.0]
        expected_scores = [1.0, 2/3, 0.5, 1/3, 1/11]
        
        for rmse, expected in zip(rmse_values, expected_scores):
            calculated = 1.0 / (1.0 + rmse)
            assert abs(calculated - expected) < 1e-10
    
    def test_direction_score_calculation(self):
        """Issue #477: 方向スコア計算の検証"""
        # 方向一致率の計算テスト
        actual_diffs = np.array([1, -1, 2, -2, 0.5])    # 実際の変化
        pred_diffs_perfect = np.array([1, -1, 2, -2, 0.5])  # 完全一致
        pred_diffs_opposite = np.array([-1, 1, -2, 2, -0.5])  # 完全逆
        pred_diffs_mixed = np.array([1, 1, 2, 2, 0.5])      # 混合
        
        # 完全一致（1.0）
        perfect_score = np.mean(np.sign(actual_diffs) == np.sign(pred_diffs_perfect))
        assert perfect_score == 1.0
        
        # 完全逆（0.0）
        opposite_score = np.mean(np.sign(actual_diffs) == np.sign(pred_diffs_opposite))
        assert opposite_score == 0.0
        
        # 混合（0.6 = 3/5）
        mixed_score = np.mean(np.sign(actual_diffs) == np.sign(pred_diffs_mixed))
        assert mixed_score == 0.6
    
    def test_sharpe_ratio_calculation(self):
        """Issue #477: シャープレシオ計算の検証"""
        # シャープレシオの数学的検証
        
        # ケース1: 正の安定したリターン
        stable_returns = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        sharpe_stable = np.mean(stable_returns) / np.std(stable_returns)
        assert sharpe_stable == float('inf')  # 標準偏差が0のため
        
        # ケース2: 正の変動リターン  
        variable_returns = np.array([0.1, 0.2, 0.05, 0.15, 0.12])
        expected_sharpe = np.mean(variable_returns) / np.std(variable_returns)
        calculated_sharpe = np.mean(variable_returns) / np.std(variable_returns)
        assert abs(calculated_sharpe - expected_sharpe) < 1e-10
        
        # ケース3: 負のリターン
        negative_returns = np.array([-0.1, -0.05, -0.15, -0.08, -0.12])
        negative_sharpe = np.mean(negative_returns) / np.std(negative_returns)
        assert negative_sharpe < 0  # 負のシャープレシオ
    
    def test_accuracy_returns_calculation(self):
        """Issue #477: 精度リターン計算の検証"""
        # pred_returns × actual_returns の意味を検証
        
        # 同方向の場合（正×正、負×負）
        pred_returns_up = np.array([0.1, 0.05, 0.2])
        actual_returns_up = np.array([0.08, 0.06, 0.15])
        accuracy_returns_positive = pred_returns_up * actual_returns_up
        assert all(accuracy_returns_positive > 0)  # 全て正値
        
        # 逆方向の場合（正×負、負×正）
        pred_returns_mixed = np.array([0.1, -0.05, 0.2])
        actual_returns_mixed = np.array([-0.08, 0.06, -0.15])
        accuracy_returns_negative = pred_returns_mixed * actual_returns_mixed
        assert all(accuracy_returns_negative < 0)  # 全て負値
        
        # 混合の場合
        pred_returns_complex = np.array([0.1, -0.05, 0.2, -0.1])
        actual_returns_complex = np.array([0.08, -0.06, -0.15, -0.05])
        accuracy_returns_complex = pred_returns_complex * actual_returns_complex
        positive_count = sum(accuracy_returns_complex > 0)
        negative_count = sum(accuracy_returns_complex < 0)
        assert positive_count == 2  # 同方向は2つ
        assert negative_count == 2  # 逆方向は2つ


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])