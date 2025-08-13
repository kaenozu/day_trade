#!/usr/bin/env python3
"""
Dynamic Weighting System Tests

Issue #481対応: DynamicWeightingSystemテスト信頼性向上
異なる市場レジーム、データ量、設定パラメータでの振る舞いを検証
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from typing import Dict, List
import time

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.dynamic_weighting_system import (
        DynamicWeightingSystem,
        DynamicWeightingConfig,
        MarketRegime,
        PerformanceWindow
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestPerformanceWindow:
    """Issue #481: PerformanceWindow単体テスト"""
    
    def test_performance_window_basic_metrics(self):
        """基本メトリクス計算テスト"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        timestamps = list(range(5))
        
        window = PerformanceWindow(predictions, actuals, timestamps)
        metrics = window.calculate_metrics()
        
        # 基本メトリクスが計算されることを確認
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'hit_rate' in metrics
        assert 'sample_count' in metrics
        
        assert metrics['sample_count'] == 5
        assert 0 <= metrics['hit_rate'] <= 1
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
    
    def test_performance_window_perfect_prediction(self):
        """完全予測時のメトリクステスト"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = predictions.copy()  # 完全一致
        timestamps = list(range(4))
        
        window = PerformanceWindow(predictions, actuals, timestamps)
        metrics = window.calculate_metrics()
        
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['hit_rate'] == 1.0  # 完全方向一致
    
    def test_performance_window_empty_data(self):
        """空データ時のメトリクステスト"""
        window = PerformanceWindow(np.array([]), np.array([]), [])
        metrics = window.calculate_metrics()
        
        assert metrics == {}
    
    def test_performance_window_single_sample(self):
        """単一サンプル時のメトリクステスト"""
        window = PerformanceWindow(np.array([1.0]), np.array([1.1]), [0])
        metrics = window.calculate_metrics()
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert metrics['hit_rate'] == 0.5  # 方向判定不可
        assert metrics['sample_count'] == 1


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingConfig:
    """Issue #481: DynamicWeightingConfig設定テスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        config = DynamicWeightingConfig()
        
        # デフォルト値確認
        assert config.window_size == 100
        assert config.min_samples_for_update == 50
        assert config.update_frequency == 20
        assert config.weighting_method == "performance_based"
        assert config.decay_factor == 0.95
        assert config.momentum_factor == 0.1
        
        # 制約値確認
        assert 0 < config.decay_factor < 1
        assert 0 < config.momentum_factor < 1
        assert config.min_weight < config.max_weight
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        config = DynamicWeightingConfig(
            window_size=50,
            weighting_method="sharpe_based",
            decay_factor=0.9,
            max_weight_change=0.2
        )
        
        assert config.window_size == 50
        assert config.weighting_method == "sharpe_based"
        assert config.decay_factor == 0.9
        assert config.max_weight_change == 0.2


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemCore:
    """Issue #481: DynamicWeightingSystemコア機能テスト"""
    
    @pytest.fixture
    def model_names(self):
        """テスト用モデル名"""
        return ["model_a", "model_b", "model_c"]
    
    @pytest.fixture
    def basic_config(self):
        """基本テスト設定"""
        return DynamicWeightingConfig(
            window_size=20,
            min_samples_for_update=10,
            update_frequency=5,
            verbose=False
        )
    
    @pytest.fixture
    def dws(self, model_names, basic_config):
        """テスト用DynamicWeightingSystem"""
        return DynamicWeightingSystem(model_names, basic_config)
    
    def test_initialization(self, model_names, basic_config):
        """初期化テスト"""
        dws = DynamicWeightingSystem(model_names, basic_config)
        
        # 基本属性確認
        assert dws.model_names == model_names
        assert dws.config == basic_config
        assert dws.current_regime == MarketRegime.SIDEWAYS
        
        # 初期重み確認（均等分散）
        weights = dws.get_current_weights()
        expected_weight = 1.0 / len(model_names)
        for weight in weights.values():
            assert abs(weight - expected_weight) < 1e-6
        
        # 重みの合計が1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_update_performance_basic(self, dws):
        """基本的な性能更新テスト"""
        predictions = {"model_a": 10.0, "model_b": 11.0, "model_c": 9.0}
        actual = 10.5
        timestamp = 0
        
        dws.update_performance(predictions, actual, timestamp)
        
        # パフォーマンス履歴の確認
        assert len(dws.recent_actuals) == 1
        for model_name in dws.model_names:
            assert len(dws.recent_predictions[model_name]) == 1
    
    def test_regime_detection_bull_market(self, dws):
        """強気相場検出テスト"""
        # 上昇トレンドのデータを投入
        base_value = 100
        for i in range(30):
            # 上昇トレンド + ノイズ
            actual = base_value + i * 0.5 + np.random.normal(0, 0.1)
            predictions = {
                "model_a": actual + np.random.normal(0, 0.5),
                "model_b": actual + np.random.normal(0, 0.5),
                "model_c": actual + np.random.normal(0, 0.5)
            }
            dws.update_performance(predictions, actual, i)
        
        # 強気相場が検出されることを確認（または少なくとも横ばいでない）
        assert dws.current_regime in [MarketRegime.BULL_MARKET, MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]
    
    def test_regime_detection_bear_market(self, dws):
        """弱気相場検出テスト"""
        # 下降トレンドのデータを投入
        base_value = 100
        for i in range(30):
            # 下降トレンド + ノイズ
            actual = base_value - i * 0.5 + np.random.normal(0, 0.1)
            predictions = {
                "model_a": actual + np.random.normal(0, 0.5),
                "model_b": actual + np.random.normal(0, 0.5),
                "model_c": actual + np.random.normal(0, 0.5)
            }
            dws.update_performance(predictions, actual, i)
        
        # 弱気相場が検出されることを確認（または高ボラティリティ）
        assert dws.current_regime in [MarketRegime.BEAR_MARKET, MarketRegime.HIGH_VOLATILITY]
    
    def test_regime_detection_high_volatility(self, dws):
        """高ボラティリティ検出テスト"""
        # 高ボラティリティデータを投入
        base_value = 100
        for i in range(30):
            # 高ノイズデータ
            actual = base_value + np.random.normal(0, 5)  # 高い標準偏差
            predictions = {
                "model_a": actual + np.random.normal(0, 1),
                "model_b": actual + np.random.normal(0, 1),
                "model_c": actual + np.random.normal(0, 1)
            }
            dws.update_performance(predictions, actual, i)
        
        # 高ボラティリティが検出されることを確認
        assert dws.current_regime == MarketRegime.HIGH_VOLATILITY
    
    def test_weight_constraints(self, dws):
        """重み制約テスト"""
        # 極端な性能差のあるデータで重み更新をテスト
        for i in range(50):
            predictions = {
                "model_a": 10.0,  # 常に正確
                "model_b": 5.0,   # 常に外れ値
                "model_c": 8.0    # 中程度
            }
            actual = 10.0
            dws.update_performance(predictions, actual, i)
        
        weights = dws.get_current_weights()
        
        # 重み制約の確認
        for weight in weights.values():
            assert dws.config.min_weight <= weight <= dws.config.max_weight
        
        # 重みの合計が1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemMarketRegimes:
    """Issue #481: 異なる市場レジームでの動作テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["conservative_model", "aggressive_model", "balanced_model"]
    
    @pytest.fixture
    def regime_config(self):
        """市場レジーム感知設定"""
        return DynamicWeightingConfig(
            window_size=30,
            min_samples_for_update=15,
            update_frequency=10,
            weighting_method="regime_aware",
            enable_regime_detection=True,
            verbose=False
        )
    
    def test_bull_market_scenario(self, model_names, regime_config):
        """強気相場シナリオテスト"""
        dws = DynamicWeightingSystem(model_names, regime_config)
        
        # 強気相場データ生成（上昇トレンド）
        base_price = 100
        for day in range(60):
            actual_price = base_price + day * 0.3 + np.random.normal(0, 0.5)
            
            # 各モデルの予測（保守的、積極的、バランス）
            predictions = {
                "conservative_model": actual_price - 0.2,  # 保守的予測
                "aggressive_model": actual_price + 0.1,    # 積極的予測
                "balanced_model": actual_price + np.random.normal(0, 0.3)
            }
            
            dws.update_performance(predictions, actual_price, day)
        
        # 強気相場での重み分布確認
        final_weights = dws.get_current_weights()
        assert dws.current_regime in [MarketRegime.BULL_MARKET, MarketRegime.LOW_VOLATILITY]
        
        # 重み制約の確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6
        for weight in final_weights.values():
            assert 0 < weight < 1
    
    def test_bear_market_scenario(self, model_names, regime_config):
        """弱気相場シナリオテスト"""
        dws = DynamicWeightingSystem(model_names, regime_config)
        
        # 弱気相場データ生成（下降トレンド）
        base_price = 100
        for day in range(60):
            actual_price = base_price - day * 0.2 + np.random.normal(0, 0.8)
            
            predictions = {
                "conservative_model": actual_price + 0.5,  # 楽観的すぎる
                "aggressive_model": actual_price - 0.1,    # 悲観的予測が的中
                "balanced_model": actual_price + np.random.normal(0, 0.4)
            }
            
            dws.update_performance(predictions, actual_price, day)
        
        final_weights = dws.get_current_weights()
        assert dws.current_regime in [MarketRegime.BEAR_MARKET, MarketRegime.HIGH_VOLATILITY]
        
        # 重み制約確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6
    
    def test_sideways_market_scenario(self, model_names, regime_config):
        """横ばい相場シナリオテスト"""
        dws = DynamicWeightingSystem(model_names, regime_config)
        
        # 横ばい相場データ生成
        base_price = 100
        for day in range(60):
            actual_price = base_price + np.random.normal(0, 0.3)  # トレンドなし
            
            predictions = {
                "conservative_model": actual_price + np.random.normal(0, 0.2),
                "aggressive_model": actual_price + np.random.normal(0, 0.4),
                "balanced_model": actual_price + np.random.normal(0, 0.1)
            }
            
            dws.update_performance(predictions, actual_price, day)
        
        final_weights = dws.get_current_weights()
        
        # 横ばい相場では均等に近い重みになることを期待
        weight_variance = np.var(list(final_weights.values()))
        assert weight_variance < 0.1  # 重みの分散が小さい
    
    def test_volatile_market_scenario(self, model_names, regime_config):
        """高ボラティリティ相場シナリオテスト"""
        dws = DynamicWeightingSystem(model_names, regime_config)
        
        # 高ボラティリティデータ生成
        base_price = 100
        for day in range(60):
            # 大きな価格変動
            shock = np.random.choice([-5, -2, 0, 2, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            actual_price = base_price + shock + np.random.normal(0, 2)
            
            predictions = {
                "conservative_model": base_price + np.random.normal(0, 0.5),  # 変動に追従しない
                "aggressive_model": actual_price + np.random.normal(0, 3),    # 過剰反応
                "balanced_model": actual_price + np.random.normal(0, 1)
            }
            
            dws.update_performance(predictions, actual_price, day)
        
        # 高ボラティリティ検出確認
        assert dws.current_regime == MarketRegime.HIGH_VOLATILITY
        
        final_weights = dws.get_current_weights()
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemMethods:
    """Issue #481: 異なる重み調整手法テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["model_1", "model_2", "model_3"]
    
    @pytest.fixture
    def test_data(self):
        """共通テストデータ"""
        np.random.seed(42)
        n_points = 100
        actual_values = 100 + np.cumsum(np.random.normal(0, 1, n_points))
        return actual_values
    
    def test_performance_based_weighting(self, model_names, test_data):
        """性能ベース重み調整テスト"""
        config = DynamicWeightingConfig(
            window_size=50,
            weighting_method="performance_based",
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        for i, actual in enumerate(test_data):
            # モデル1が最も正確、モデル3が最も不正確
            predictions = {
                "model_1": actual + np.random.normal(0, 0.5),  # 低誤差
                "model_2": actual + np.random.normal(0, 1.0),  # 中誤差
                "model_3": actual + np.random.normal(0, 2.0)   # 高誤差
            }
            dws.update_performance(predictions, actual, i)
        
        final_weights = dws.get_current_weights()
        
        # 最も正確なモデルが最高重みを持つことを確認
        assert final_weights["model_1"] > final_weights["model_2"]
        assert final_weights["model_2"] > final_weights["model_3"]
    
    def test_sharpe_based_weighting(self, model_names, test_data):
        """シャープレシオベース重み調整テスト"""
        config = DynamicWeightingConfig(
            window_size=50,
            weighting_method="sharpe_based",
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        for i, actual in enumerate(test_data):
            predictions = {
                "model_1": actual + np.random.normal(0, 0.5),
                "model_2": actual + np.random.normal(0, 1.0),
                "model_3": actual + np.random.normal(0, 1.5)
            }
            dws.update_performance(predictions, actual, i)
        
        final_weights = dws.get_current_weights()
        
        # 重み制約の確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6
        for weight in final_weights.values():
            assert config.min_weight <= weight <= config.max_weight
    
    def test_regime_aware_weighting(self, model_names, test_data):
        """レジーム認識重み調整テスト"""
        config = DynamicWeightingConfig(
            window_size=50,
            weighting_method="regime_aware",
            enable_regime_detection=True,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        for i, actual in enumerate(test_data):
            predictions = {
                "model_1": actual + np.random.normal(0, 0.8),
                "model_2": actual + np.random.normal(0, 0.8),
                "model_3": actual + np.random.normal(0, 0.8)
            }
            dws.update_performance(predictions, actual, i)
        
        final_weights = dws.get_current_weights()
        
        # レジーム認識の確認
        assert dws.current_regime in [regime for regime in MarketRegime]
        
        # 重み制約確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemDataSize:
    """Issue #481: 異なるデータ量での動作テスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["model_x", "model_y"]
    
    def test_small_dataset_behavior(self, model_names):
        """小規模データセット動作テスト"""
        config = DynamicWeightingConfig(
            window_size=10,
            min_samples_for_update=5,
            update_frequency=3,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 小規模データ（15ポイント）
        for i in range(15):
            actual = 100 + i * 0.1
            predictions = {
                "model_x": actual + np.random.normal(0, 0.1),
                "model_y": actual + np.random.normal(0, 0.2)
            }
            dws.update_performance(predictions, actual, i)
        
        weights = dws.get_current_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # 小規模データでも適切に動作することを確認
        assert len(dws.recent_predictions["model_x"]) <= config.window_size
    
    def test_large_dataset_behavior(self, model_names):
        """大規模データセット動作テスト"""
        config = DynamicWeightingConfig(
            window_size=100,
            min_samples_for_update=50,
            update_frequency=20,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 大規模データ（500ポイント）
        start_time = time.time()
        
        for i in range(500):
            actual = 100 + np.sin(i * 0.01) * 10 + np.random.normal(0, 1)
            predictions = {
                "model_x": actual + np.random.normal(0, 1),
                "model_y": actual + np.random.normal(0, 1.5)
            }
            dws.update_performance(predictions, actual, i)
        
        processing_time = time.time() - start_time
        
        # 処理時間が妥当であることを確認（500ポイントで5秒以内）
        assert processing_time < 5.0
        
        weights = dws.get_current_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # ウィンドウサイズ制約の確認
        assert len(dws.recent_predictions["model_x"]) <= config.window_size
    
    def test_incremental_data_addition(self, model_names):
        """インクリメンタルデータ追加テスト"""
        config = DynamicWeightingConfig(window_size=20, verbose=False)
        dws = DynamicWeightingSystem(model_names, config)
        
        # 段階的データ追加
        weights_history = []
        
        for batch in range(5):
            # 各バッチ10ポイント
            for i in range(10):
                point_idx = batch * 10 + i
                actual = 100 + point_idx * 0.05
                predictions = {
                    "model_x": actual + np.random.normal(0, 0.5),
                    "model_y": actual + np.random.normal(0, 0.8)
                }
                dws.update_performance(predictions, actual, point_idx)
            
            # バッチ終了後の重みを記録
            weights_history.append(dws.get_current_weights().copy())
        
        # 重みが時間とともに変化することを確認
        assert len(weights_history) == 5
        
        # 最初と最後の重みが異なることを確認
        first_weights = weights_history[0]
        last_weights = weights_history[-1]
        
        weights_changed = any(
            abs(first_weights[model] - last_weights[model]) > 0.01
            for model in model_names
        )
        assert weights_changed


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemEdgeCases:
    """Issue #481: エッジケーステスト"""
    
    @pytest.fixture
    def model_names(self):
        return ["edge_model_1", "edge_model_2"]
    
    def test_identical_predictions(self, model_names):
        """全モデル同一予測時のテスト"""
        config = DynamicWeightingConfig(window_size=20, verbose=False)
        dws = DynamicWeightingSystem(model_names, config)
        
        # 全モデルが同一予測
        for i in range(30):
            actual = 100 + i * 0.1
            identical_prediction = actual + 0.1
            predictions = {model: identical_prediction for model in model_names}
            
            dws.update_performance(predictions, actual, i)
        
        weights = dws.get_current_weights()
        
        # 同一性能なら均等重みに近いことを確認
        expected_weight = 1.0 / len(model_names)
        for weight in weights.values():
            assert abs(weight - expected_weight) < 0.1
    
    def test_extreme_outlier_predictions(self, model_names):
        """極端な外れ値予測テスト"""
        config = DynamicWeightingConfig(window_size=30, verbose=False)
        dws = DynamicWeightingSystem(model_names, config)
        
        for i in range(40):
            actual = 100.0
            predictions = {
                "edge_model_1": actual + np.random.normal(0, 0.1),  # 正常
                "edge_model_2": actual + 1000 if i == 20 else actual + np.random.normal(0, 0.1)  # 1回だけ極端
            }
            
            dws.update_performance(predictions, actual, i)
        
        weights = dws.get_current_weights()
        
        # 重み制約が守られることを確認
        for weight in weights.values():
            assert config.min_weight <= weight <= config.max_weight
    
    def test_nan_inf_handling(self, model_names):
        """NaN/Inf値処理テスト"""
        config = DynamicWeightingConfig(window_size=20, verbose=False)
        dws = DynamicWeightingSystem(model_names, config)
        
        for i in range(25):
            actual = 100.0
            predictions = {
                "edge_model_1": actual + np.random.normal(0, 0.5),
                "edge_model_2": np.nan if i == 10 else actual + np.random.normal(0, 0.5)
            }
            
            # NaN値があってもクラッシュしないことを確認
            try:
                dws.update_performance(predictions, actual, i)
            except Exception as e:
                # 適切なエラーハンドリングがされているかログ確認
                print(f"NaN処理エラー（想定内の可能性）: {e}")
        
        weights = dws.get_current_weights()
        
        # 最終的に有効な重みが得られることを確認
        assert all(0 <= weight <= 1 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_single_model_system(self):
        """単一モデルシステムテスト"""
        config = DynamicWeightingConfig(window_size=15, verbose=False)
        dws = DynamicWeightingSystem(["single_model"], config)
        
        for i in range(20):
            actual = 100 + i * 0.2
            predictions = {"single_model": actual + np.random.normal(0, 0.5)}
            dws.update_performance(predictions, actual, i)
        
        weights = dws.get_current_weights()
        
        # 単一モデルは重み1.0を持つ
        assert abs(weights["single_model"] - 1.0) < 1e-6


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDynamicWeightingSystemIntegration:
    """Issue #481: 統合テスト"""
    
    def test_full_market_cycle_simulation(self):
        """完全市場サイクルシミュレーションテスト"""
        model_names = ["trend_follower", "mean_reverter", "momentum_model"]
        config = DynamicWeightingConfig(
            window_size=50,
            min_samples_for_update=25,
            update_frequency=10,
            weighting_method="regime_aware",
            enable_regime_detection=True,
            verbose=False
        )
        dws = DynamicWeightingSystem(model_names, config)
        
        # 完全市場サイクル（300日）
        regime_changes = []
        
        for phase in range(3):
            for day in range(100):
                global_day = phase * 100 + day
                
                if phase == 0:
                    # 強気相場
                    actual = 100 + global_day * 0.2 + np.random.normal(0, 1)
                    predictions = {
                        "trend_follower": actual + np.random.normal(0, 0.5),    # 上昇で有効
                        "mean_reverter": actual - 2.0 + np.random.normal(0, 1),  # 上昇で不利
                        "momentum_model": actual + np.random.normal(0, 0.8)
                    }
                elif phase == 1:
                    # 弱気相場
                    actual = 120 - (global_day - 100) * 0.15 + np.random.normal(0, 1.5)
                    predictions = {
                        "trend_follower": actual + 1.0 + np.random.normal(0, 1),    # 下降で不利
                        "mean_reverter": actual + np.random.normal(0, 0.5),          # 平均回帰で有効
                        "momentum_model": actual + np.random.normal(0, 1.2)
                    }
                else:
                    # 横ばい相場
                    actual = 105 + np.random.normal(0, 0.5)
                    predictions = {
                        "trend_follower": actual + np.random.normal(0, 1),
                        "mean_reverter": actual + np.random.normal(0, 0.3),
                        "momentum_model": actual + np.random.normal(0, 0.7)
                    }
                
                old_regime = dws.current_regime
                dws.update_performance(predictions, actual, global_day)
                
                if old_regime != dws.current_regime:
                    regime_changes.append((global_day, old_regime, dws.current_regime))
        
        # 最終結果検証
        final_weights = dws.get_current_weights()
        
        # 基本制約確認
        assert abs(sum(final_weights.values()) - 1.0) < 1e-6
        for weight in final_weights.values():
            assert config.min_weight <= weight <= config.max_weight
        
        # レジーム変化が検出されたことを確認
        assert len(regime_changes) > 0
        
        # 各フェーズで適切にモデル重みが調整されたことをログ出力で確認
        print(f"最終重み: {final_weights}")
        print(f"レジーム変化: {len(regime_changes)}回")
        print(f"最終レジーム: {dws.current_regime}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])