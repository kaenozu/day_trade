#!/usr/bin/env python3
"""
Data Preparation Optimization Tests

Issue #694対応: DeepLearningModelsデータ準備最適化テスト
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
        ModelConfig
    )
    TEST_AVAILABLE = True
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDataPreparationOptimization:
    """Issue #694: データ準備最適化テスト"""

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
    def small_data(self):
        """小規模テスト用データ"""
        np.random.seed(42)
        return pd.DataFrame({
            'Open': np.random.randn(50),
            'High': np.random.randn(50),
            'Low': np.random.randn(50),
            'Close': np.random.randn(50),
            'Volume': np.random.randn(50)
        })

    @pytest.fixture
    def large_data(self):
        """大規模テスト用データ"""
        np.random.seed(42)
        return pd.DataFrame({
            'Open': np.random.randn(10000),
            'High': np.random.randn(10000),
            'Low': np.random.randn(10000),
            'Close': np.random.randn(10000),
            'Volume': np.random.randn(10000)
        })

    def test_optimized_vs_legacy_consistency(self, config, small_data):
        """Issue #694: 最適化版と従来版の結果一致性テスト"""
        model = TransformerModel(config)

        # 最適化版実行
        X_optimized, y_optimized = model.prepare_data(small_data, use_optimized=True)

        # 従来版実行
        X_legacy, y_legacy = model.prepare_data(small_data, use_optimized=False)

        # 形状一致確認
        assert X_optimized.shape == X_legacy.shape
        assert y_optimized.shape == y_legacy.shape

        # 値の近似一致確認（正規化処理で若干の差異は許容）
        np.testing.assert_allclose(X_optimized, X_legacy, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(y_optimized, y_legacy, rtol=1e-6, atol=1e-6)

    def test_sliding_window_view_functionality(self, config):
        """Issue #694: sliding_window_view機能テスト"""
        model = TransformerModel(config)

        # 簡単なテストデータ
        features = np.arange(20).reshape(-1, 1).astype(np.float32)
        sequence_length = 3
        n_sequences = len(features) - sequence_length + 1

        # sliding windowテスト
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            # sliding_window_view利用可能時のテスト
            X = sliding_window_view(features, window_shape=sequence_length, axis=0)
            X = X[:n_sequences]

            # 期待される結果と比較
            expected_first_window = np.array([[[0], [1], [2]]])
            np.testing.assert_array_equal(X[0:1], expected_first_window)

            expected_last_window = np.array([[[17], [18], [19]]])
            np.testing.assert_array_equal(X[-1:], expected_last_window)

        except (ImportError, AttributeError):
            # sliding_window_view未利用環境の場合はスキップ
            pytest.skip("sliding_window_view not available")

    def test_manual_sliding_window_fallback(self, config):
        """Issue #694: 手動sliding windowフォールバックテスト"""
        model = TransformerModel(config)

        # テストデータ
        features = np.arange(15).reshape(-1, 1).astype(np.float32)
        sequence_length = 5
        n_sequences = len(features) - sequence_length + 1

        # 手動sliding window実行
        X = model._manual_sliding_window(features, sequence_length, n_sequences)

        # 結果形状確認
        assert X.shape == (n_sequences, sequence_length, 1)

        # 内容確認
        expected_first = np.array([[0], [1], [2], [3], [4]])
        np.testing.assert_array_equal(X[0], expected_first)

        expected_last = np.array([[10], [11], [12], [13], [14]])
        np.testing.assert_array_equal(X[-1], expected_last)

    def test_performance_improvement(self, config, large_data):
        """Issue #694: 性能向上テスト"""
        model = TransformerModel(config)

        # 従来版実行時間測定
        start_time = time.time()
        X_legacy, y_legacy = model.prepare_data(large_data, use_optimized=False)
        legacy_time = time.time() - start_time

        # 最適化版実行時間測定
        start_time = time.time()
        X_optimized, y_optimized = model.prepare_data(large_data, use_optimized=True)
        optimized_time = time.time() - start_time

        # 性能向上確認（最適化版が大幅に遅くならないことを確認）
        assert optimized_time <= legacy_time * 1.5, f"最適化版が期待より遅い: {optimized_time:.3f}s vs {legacy_time:.3f}s"

        # 結果一致性確認
        assert X_optimized.shape == X_legacy.shape
        assert y_optimized.shape == y_legacy.shape

        print(f"性能比較: 従来版 {legacy_time:.3f}s, 最適化版 {optimized_time:.3f}s")

    def test_memory_efficiency_float32(self, config, small_data):
        """Issue #694: float32メモリ効率化テスト"""
        model = TransformerModel(config)

        X, y = model.prepare_data(small_data)

        # float32型確認
        assert X.dtype == np.float32
        assert y.dtype == np.float32

        # メモリ使用量確認（float32はfloat64の半分）
        expected_x_bytes = X.size * 4  # float32 = 4バイト
        actual_x_bytes = X.nbytes
        assert actual_x_bytes == expected_x_bytes

    def test_vectorized_normalization(self, config, small_data):
        """Issue #694: ベクトル化正規化テスト"""
        model = TransformerModel(config)

        # 正規化前後のデータ確認
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in feature_columns if col in small_data.columns]
        original_features = small_data[available_columns].values

        X, y = model.prepare_data(small_data)

        # 正規化が適用されていることを確認
        # 元データの平均・標準偏差と最終出力の統計を比較
        original_mean = np.mean(original_features, axis=0)
        original_std = np.std(original_features, axis=0)

        # Xから最初のタイムステップの平均値を抽出
        first_timestep_mean = np.mean(X[:, 0, :], axis=0)
        first_timestep_std = np.std(X[:, 0, :], axis=0)

        # 正規化により平均が0に近く、標準偏差が1に近くなることを確認
        np.testing.assert_allclose(first_timestep_mean, np.zeros_like(first_timestep_mean), atol=1e-1)
        np.testing.assert_allclose(first_timestep_std, np.ones_like(first_timestep_std), atol=0.5)

    def test_edge_cases(self, config):
        """Issue #694: エッジケーステスト"""
        model = TransformerModel(config)

        # 不十分なデータサイズ
        tiny_data = pd.DataFrame({
            'Open': [1, 2],
            'Close': [1, 2]
        })

        with pytest.raises(ValueError, match="データが不足"):
            model.prepare_data(tiny_data)

        # 必要な列が存在しない
        invalid_data = pd.DataFrame({
            'invalid_col': [1, 2, 3, 4, 5]
        })

        with pytest.raises(ValueError, match="必要な価格データ列が見つかりません"):
            model.prepare_data(invalid_data)

    def test_multi_step_prediction(self, config, small_data):
        """Issue #694: 複数ステップ予測対応テスト"""
        # 複数ステップ予測設定
        multi_config = ModelConfig(
            sequence_length=10,
            prediction_horizon=3,
            epochs=1
        )

        model = TransformerModel(multi_config)

        X, y = model.prepare_data(small_data)

        # 予測ホライズンが反映されていることを確認
        assert y.shape[1] == 3  # 3ステップ予測

        # 最適化版と従来版の一致確認
        X_opt, y_opt = model.prepare_data(small_data, use_optimized=True)
        X_leg, y_leg = model.prepare_data(small_data, use_optimized=False)

        np.testing.assert_allclose(X_opt, X_leg, rtol=1e-6)
        np.testing.assert_allclose(y_opt, y_leg, rtol=1e-6)

    def test_target_column_handling(self, config, small_data):
        """Issue #694: ターゲット列処理テスト"""
        model = TransformerModel(config)

        # デフォルトターゲット列（Close）
        X1, y1 = model.prepare_data(small_data)

        # 明示的ターゲット列指定
        X2, y2 = model.prepare_data(small_data, target_column="Close")

        # 結果一致確認
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

        # 異なるターゲット列
        X3, y3 = model.prepare_data(small_data, target_column="High")

        # 特徴量は同じだがターゲットは異なる
        np.testing.assert_array_equal(X1, X3)
        assert not np.array_equal(y1, y3)

    def test_sliding_window_view_fallback(self, config, small_data):
        """Issue #694: sliding_window_viewフォールバック動作テスト"""
        model = TransformerModel(config)

        # sliding_window_viewが利用できない状況をシミュレート
        with patch('day_trade.ml.deep_learning_models.sliding_window_view', side_effect=ImportError):
            with patch('day_trade.ml.deep_learning_models.logger') as mock_logger:
                X, y = model.prepare_data(small_data, use_optimized=True)

                # フォールバック警告が出力されることを確認
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "sliding_window_view未利用" in warning_msg
                assert "手動stride実装を使用" in warning_msg

                # 結果は正常に返される
                assert X.shape[0] > 0
                assert y.shape[0] > 0


@pytest.mark.skipif(not TEST_AVAILABLE, reason="Required modules not available")
class TestDataPreparationBenchmark:
    """Issue #694: データ準備ベンチマークテスト"""

    @pytest.fixture
    def benchmark_data(self):
        """ベンチマーク用大規模データ"""
        np.random.seed(123)
        return pd.DataFrame({
            'Open': np.random.randn(50000),
            'High': np.random.randn(50000),
            'Low': np.random.randn(50000),
            'Close': np.random.randn(50000),
            'Volume': np.random.randn(50000)
        })

    def test_large_dataset_preparation(self, benchmark_data):
        """Issue #694: 大規模データセット準備テスト"""
        config = ModelConfig(
            sequence_length=60,
            prediction_horizon=5,
            epochs=1
        )

        model = TransformerModel(config)

        # 大規模データでのデータ準備実行
        start_time = time.time()
        X, y = model.prepare_data(benchmark_data, use_optimized=True)
        preparation_time = time.time() - start_time

        # 結果検証
        expected_sequences = len(benchmark_data) - config.sequence_length - config.prediction_horizon + 1
        assert X.shape[0] == expected_sequences
        assert X.shape[1] == config.sequence_length
        assert X.shape[2] == 5  # OHLCV
        assert y.shape[0] == expected_sequences
        assert y.shape[1] == config.prediction_horizon

        # 準備時間が合理的範囲内であることを確認
        assert preparation_time < 10.0  # 10秒以内

        print(f"大規模データ準備時間: {preparation_time:.3f}秒, データ形状: X{X.shape}, y{y.shape}")

    def test_memory_usage_comparison(self, benchmark_data):
        """Issue #694: メモリ使用量比較テスト"""
        config = ModelConfig(
            sequence_length=30,
            prediction_horizon=1,
            epochs=1
        )

        model = TransformerModel(config)

        # float32とfloat64のメモリ使用量比較
        X_float32, _ = model.prepare_data(benchmark_data, use_optimized=True)

        # 手動でfloat64版を作成
        features = benchmark_data[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float64)

        # メモリ使用量比較
        float32_memory = X_float32.nbytes
        float64_memory_estimate = X_float32.size * 8  # float64 = 8バイト

        # float32がfloat64の半分のメモリ使用量であることを確認
        assert float32_memory == float64_memory_estimate / 2

        print(f"メモリ使用量: float32 {float32_memory / (1024**2):.1f}MB, float64想定 {float64_memory_estimate / (1024**2):.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])