#!/usr/bin/env python3
"""
Issue #478対応: DynamicWeightingSystemレジーム認識調整外部化テスト

レジーム調整係数のハードコード解消と外部設定対応の検証
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import json
import tempfile
import os

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
class TestDynamicWeightingRegimeExternalization:
    """Issue #478: レジーム認識調整外部化テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["model_a", "model_b", "model_c"]
    
    @pytest.fixture
    def custom_regime_adjustments(self):
        """カスタムレジーム調整設定"""
        return {
            MarketRegime.BULL_MARKET: {"model_a": 1.5, "model_b": 1.0, "model_c": 0.8},
            MarketRegime.BEAR_MARKET: {"model_a": 0.7, "model_b": 1.3, "model_c": 1.2},
            MarketRegime.SIDEWAYS: {"model_a": 1.0, "model_b": 1.0, "model_c": 1.0},
            MarketRegime.HIGH_VOLATILITY: {"model_a": 0.9, "model_b": 0.8, "model_c": 1.4},
            MarketRegime.LOW_VOLATILITY: {"model_a": 1.2, "model_b": 1.1, "model_c": 0.9}
        }
    
    @pytest.fixture
    def config_with_custom_adjustments(self, custom_regime_adjustments):
        """カスタム調整設定付きの設定"""
        return DynamicWeightingConfig(
            window_size=20,
            min_samples_for_update=10,
            update_frequency=5,
            weighting_method="regime_aware",
            regime_adjustments=custom_regime_adjustments,
            verbose=False
        )
    
    def test_default_regime_adjustments_creation(self, model_names):
        """Issue #478: デフォルトレジーム調整設定作成テスト"""
        config = DynamicWeightingConfig(verbose=False)
        dws = DynamicWeightingSystem(model_names, config)
        
        # デフォルト設定が作成されることを確認
        assert dws.config.regime_adjustments is not None
        assert isinstance(dws.config.regime_adjustments, dict)
        
        # 全市場状態の設定が存在することを確認
        for regime in MarketRegime:
            assert regime in dws.config.regime_adjustments
            
        # 全モデルの設定が存在することを確認
        for regime, adjustments in dws.config.regime_adjustments.items():
            for model_name in model_names:
                assert model_name in adjustments
                assert isinstance(adjustments[model_name], (int, float))
                assert adjustments[model_name] > 0
    
    def test_custom_regime_adjustments(self, model_names, config_with_custom_adjustments):
        """Issue #478: カスタムレジーム調整設定テスト"""
        dws = DynamicWeightingSystem(model_names, config_with_custom_adjustments)
        
        # カスタム設定が正しく設定されることを確認
        adjustments = dws.config.regime_adjustments
        assert adjustments[MarketRegime.BULL_MARKET]["model_a"] == 1.5
        assert adjustments[MarketRegime.BEAR_MARKET]["model_b"] == 1.3
        assert adjustments[MarketRegime.HIGH_VOLATILITY]["model_c"] == 1.4
    
    def test_regime_aware_weighting_with_custom_adjustments(self, model_names, config_with_custom_adjustments):
        """Issue #478: カスタム調整でのレジーム認識重み計算テスト"""
        dws = DynamicWeightingSystem(model_names, config_with_custom_adjustments)
        
        # 十分なパフォーマンスデータを追加
        np.random.seed(42)
        for i in range(15):
            predictions = {
                "model_a": 100 + i * 0.1 + np.random.normal(0, 0.5),
                "model_b": 100 + i * 0.1 + np.random.normal(0, 0.8),
                "model_c": 100 + i * 0.1 + np.random.normal(0, 1.2)
            }
            actual = 100 + i * 0.1
            dws.update_performance(predictions, actual, i)
        
        # 強気相場設定でテスト
        dws.current_regime = MarketRegime.BULL_MARKET
        bull_weights = dws._regime_aware_weighting()
        
        # 弱気相場設定でテスト
        dws.current_regime = MarketRegime.BEAR_MARKET
        bear_weights = dws._regime_aware_weighting()
        
        # 重みが異なることを確認（レジーム調整が効いている）
        assert bull_weights != bear_weights
        
        # 重み制約を満たすことを確認
        for weights in [bull_weights, bear_weights]:
            assert abs(sum(weights.values()) - 1.0) < 1e-6
            for weight in weights.values():
                assert weight > 0
    
    def test_update_regime_adjustments(self, model_names):
        """Issue #478: レジーム調整設定動的更新テスト"""
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
        
        # 新しい調整設定
        new_adjustments = {
            MarketRegime.BULL_MARKET: {"model_a": 2.0, "model_b": 0.5, "model_c": 0.8},
            MarketRegime.BEAR_MARKET: {"model_a": 0.6, "model_b": 1.8, "model_c": 1.1}
        }
        
        dws.update_regime_adjustments(new_adjustments)
        
        # 更新が正しく適用されることを確認
        current_adjustments = dws.get_regime_adjustments()
        assert current_adjustments[MarketRegime.BULL_MARKET]["model_a"] == 2.0
        assert current_adjustments[MarketRegime.BEAR_MARKET]["model_b"] == 1.8
    
    def test_update_regime_adjustments_validation(self, model_names):
        """Issue #478: レジーム調整設定更新時のバリデーションテスト"""
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
        
        # 無効な調整係数（負の値）
        invalid_adjustments = {
            MarketRegime.BULL_MARKET: {"model_a": -1.0, "model_b": 1.0, "model_c": 1.0}
        }
        
        with pytest.raises(ValueError):
            dws.update_regime_adjustments(invalid_adjustments)
        
        # 無効な調整係数（ゼロ）
        invalid_adjustments_zero = {
            MarketRegime.BULL_MARKET: {"model_a": 0.0, "model_b": 1.0, "model_c": 1.0}
        }
        
        with pytest.raises(ValueError):
            dws.update_regime_adjustments(invalid_adjustments_zero)
    
    def test_load_regime_adjustments_from_dict(self, model_names):
        """Issue #478: 辞書からのレジーム調整設定読み込みテスト"""
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
        
        # 文字列キーの辞書
        adjustments_dict = {
            "bull": {"model_a": 1.5, "model_b": 1.0, "model_c": 0.8},
            "bear": {"model_a": 0.7, "model_b": 1.3, "model_c": 1.2},
            "sideways": {"model_a": 1.0, "model_b": 1.0, "model_c": 1.0}
        }
        
        dws.load_regime_adjustments_from_dict(adjustments_dict)
        
        # 設定が正しく読み込まれることを確認
        current_adjustments = dws.get_regime_adjustments()
        assert current_adjustments[MarketRegime.BULL_MARKET]["model_a"] == 1.5
        assert current_adjustments[MarketRegime.BEAR_MARKET]["model_b"] == 1.3
        assert current_adjustments[MarketRegime.SIDEWAYS]["model_c"] == 1.0
    
    def test_load_regime_adjustments_unknown_regime(self, model_names):
        """Issue #478: 未知のレジーム名に対する処理テスト"""
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
        
        # 未知のレジーム名を含む辞書
        adjustments_dict = {
            "bull": {"model_a": 1.5, "model_b": 1.0, "model_c": 0.8},
            "unknown_regime": {"model_a": 2.0, "model_b": 2.0, "model_c": 2.0}  # 未知のレジーム
        }
        
        # エラーが発生しないことを確認（警告のみ）
        dws.load_regime_adjustments_from_dict(adjustments_dict)
        
        # 有効なレジームのみが読み込まれることを確認
        current_adjustments = dws.get_regime_adjustments()
        assert MarketRegime.BULL_MARKET in current_adjustments
        # 未知のレジームは無視される
    
    def test_regime_aware_weighting_without_adjustments(self, model_names):
        """Issue #478: 調整設定なしでのレジーム認識重み計算テスト"""
        config = DynamicWeightingConfig(
            weighting_method="regime_aware",
            regime_adjustments=None,  # 明示的にNone設定
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 設定を削除
        dws.config.regime_adjustments = None
        
        # 十分なパフォーマンスデータを追加
        for i in range(15):
            predictions = {"model_a": 100 + i, "model_b": 100 + i, "model_c": 100 + i}
            actual = 100 + i
            dws.update_performance(predictions, actual, i)
        
        # 基本重みが返されることを確認
        weights = dws._regime_aware_weighting()
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_regime_aware_weighting_error_handling(self, model_names):
        """Issue #478: レジーム認識重み計算のエラーハンドリングテスト"""
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(verbose=False))
        
        # パフォーマンスデータなしでの呼び出し
        weights = dws._regime_aware_weighting()
        
        # エラーが発生しても有効な重みが返されることを確認
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_default_adjustments_model_name_mapping(self):
        """Issue #478: デフォルト調整でのモデル名マッピングテスト"""
        # 特定のモデル名パターンをテスト
        model_names_rf = ["random_forest_model", "custom_model_a", "custom_model_b"]
        dws_rf = DynamicWeightingSystem(model_names_rf, DynamicWeightingConfig(verbose=False))
        
        adjustments = dws_rf.config.regime_adjustments
        
        # "random_forest"パターンにマッチするモデルの調整係数をチェック
        bull_adjustments = adjustments[MarketRegime.BULL_MARKET]
        assert bull_adjustments["random_forest_model"] == 1.2  # デフォルトのrandom_forest係数
    
    def test_get_regime_adjustments(self, model_names, custom_regime_adjustments):
        """Issue #478: レジーム調整設定取得テスト"""
        config = DynamicWeightingConfig(
            regime_adjustments=custom_regime_adjustments,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        retrieved_adjustments = dws.get_regime_adjustments()
        
        # 設定が正しく取得できることを確認
        assert retrieved_adjustments == custom_regime_adjustments
        assert retrieved_adjustments is not None
        assert len(retrieved_adjustments) == len(MarketRegime)


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingRegimeIntegration:
    """Issue #478: レジーム外部化統合テスト"""
    
    def test_full_regime_externalization_workflow(self):
        """Issue #478: レジーム外部化の完全なワークフローテスト"""
        model_names = ["aggressive_model", "conservative_model", "balanced_model"]
        
        # Step 1: デフォルト設定でシステム作成
        dws = DynamicWeightingSystem(model_names, DynamicWeightingConfig(
            weighting_method="regime_aware",
            verbose=False
        ))
        
        # Step 2: カスタム設定の読み込み
        custom_adjustments_dict = {
            "bull": {"aggressive_model": 1.8, "conservative_model": 0.6, "balanced_model": 1.0},
            "bear": {"aggressive_model": 0.4, "conservative_model": 1.6, "balanced_model": 1.2},
            "high_vol": {"aggressive_model": 0.7, "conservative_model": 0.8, "balanced_model": 1.5}
        }
        
        dws.load_regime_adjustments_from_dict(custom_adjustments_dict)
        
        # Step 3: パフォーマンスデータ投入
        np.random.seed(42)
        for i in range(20):
            predictions = {
                "aggressive_model": 100 + i * 0.2 + np.random.normal(0, 1.0),
                "conservative_model": 100 + i * 0.1 + np.random.normal(0, 0.3),
                "balanced_model": 100 + i * 0.15 + np.random.normal(0, 0.6)
            }
            actual = 100 + i * 0.15
            dws.update_performance(predictions, actual, i)
        
        # Step 4: 異なるレジームでの重み計算テスト
        original_regime = dws.current_regime
        
        # 強気相場
        dws.current_regime = MarketRegime.BULL_MARKET
        bull_weights = dws._regime_aware_weighting()
        
        # 弱気相場
        dws.current_regime = MarketRegime.BEAR_MARKET
        bear_weights = dws._regime_aware_weighting()
        
        # 高ボラティリティ
        dws.current_regime = MarketRegime.HIGH_VOLATILITY
        high_vol_weights = dws._regime_aware_weighting()
        
        # Step 5: 結果検証
        # 強気相場では攻撃的モデルが高い重み
        assert bull_weights["aggressive_model"] > bull_weights["conservative_model"]
        
        # 弱気相場では保守的モデルが高い重み
        assert bear_weights["conservative_model"] > bear_weights["aggressive_model"]
        
        # 高ボラティリティではバランスモデルが高い重み
        assert high_vol_weights["balanced_model"] > min(
            high_vol_weights["aggressive_model"], 
            high_vol_weights["conservative_model"]
        )
        
        # 全重みが制約を満たす
        for weights in [bull_weights, bear_weights, high_vol_weights]:
            assert abs(sum(weights.values()) - 1.0) < 1e-6
            for weight in weights.values():
                assert weight > 0
        
        # レジームを元に戻す
        dws.current_regime = original_regime


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])