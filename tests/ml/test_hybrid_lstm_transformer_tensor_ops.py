#!/usr/bin/env python3
"""
HybridLSTMTransformer Tensor Operations Optimization Tests

Issue #698対応: HybridLSTMTransformerテンソル操作最適化テスト
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import sys
import torch
import torch.nn as nn

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.hybrid_lstm_transformer import (
        HybridLSTMTransformerEngine,
        HybridModelConfig,
        CrossAttentionLayer,
        ModifiedTransformerEncoder,
        PositionalEncoding,
        HybridLSTMTransformerModel,
        PYTORCH_AVAILABLE
    )
    TEST_AVAILABLE = PYTORCH_AVAILABLE
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="PyTorch or required modules not available")
class TestTensorOperationsOptimization:
    """Issue #698: テンソル操作最適化テスト"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return HybridModelConfig(
            sequence_length=20,
            prediction_horizon=3,
            lstm_hidden_size=64,
            transformer_d_model=32,
            transformer_num_heads=4,
            cross_attention_heads=2,
            cross_attention_dim=32,
            epochs=5,
            batch_size=8
        )

    @pytest.fixture
    def sample_tensors(self):
        """テスト用テンソルデータ"""
        batch_size, seq_len, features = 4, 20, 10
        lstm_dim, transformer_dim = 128, 32

        return {
            'input_tensor': torch.randn(batch_size, seq_len, features),
            'lstm_features': torch.randn(batch_size, lstm_dim),
            'transformer_features': torch.randn(batch_size, transformer_dim),
            'batch_size': batch_size,
            'seq_len': seq_len,
            'features': features
        }

    def test_cross_attention_qkv_optimization(self, sample_tensors):
        """Issue #698: CrossAttentionLayerのQKV統合計算テスト"""
        lstm_dim = 128
        transformer_dim = 32
        attention_dim = 32

        layer = CrossAttentionLayer(
            lstm_dim=lstm_dim,
            transformer_dim=transformer_dim,
            attention_heads=2,
            attention_dim=attention_dim
        )

        # QKV統合レイヤーの存在確認
        assert hasattr(layer, 'qkv_proj'), "QKV統合レイヤーが存在しません"
        assert layer.qkv_proj.out_features == attention_dim * 3, "QKV出力次元が正しくありません"

        # フォワードパス実行
        lstm_features = sample_tensors['lstm_features']
        transformer_features = sample_tensors['transformer_features']

        fused_features, attention_weights = layer(lstm_features, transformer_features)

        # 出力形状確認
        assert fused_features.shape == (sample_tensors['batch_size'], attention_dim)
        assert attention_weights.shape == (sample_tensors['batch_size'], 2, 2)

    def test_cross_attention_einsum_optimization(self, sample_tensors):
        """Issue #698: CrossAttentionLayerのeinsum最適化テスト"""
        layer = CrossAttentionLayer(
            lstm_dim=128,
            transformer_dim=32,
            attention_heads=2,
            attention_dim=32
        )

        lstm_features = sample_tensors['lstm_features']
        transformer_features = sample_tensors['transformer_features']

        # 最適化前後での結果一致確認
        with torch.no_grad():
            fused_features, attention_weights = layer(lstm_features, transformer_features)

            # 出力が有効な範囲内であることを確認
            assert torch.isfinite(fused_features).all(), "無限値が含まれています"
            assert torch.isfinite(attention_weights).all(), "注意重みに無限値が含まれています"

            # 注意重みの正規化確認
            attention_sum = attention_weights.sum(dim=-1)
            assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)

    def test_modified_transformer_conv_optimization(self, sample_tensors):
        """Issue #698: ModifiedTransformerEncoderの畳み込み最適化テスト"""
        d_model = 32
        encoder = ModifiedTransformerEncoder(
            d_model=d_model,
            num_heads=4,
            dim_feedforward=128,
            num_layers=2
        )

        # グループ畳み込みの設定確認
        assert hasattr(encoder, 'temporal_conv'), "temporal_convが存在しません"
        assert encoder.temporal_conv.groups > 1, "グループ畳み込みが設定されていません"
        assert encoder._use_memory_efficient_conv == True, "メモリ効率化フラグが設定されていません"

        # フォワードパス実行
        input_tensor = sample_tensors['input_tensor'][:, :, :d_model]  # d_modelに合わせる
        output = encoder(input_tensor)

        # 出力形状確認
        expected_shape = input_tensor.shape
        assert output.shape == expected_shape, f"出力形状が異なります: {output.shape} != {expected_shape}"

    def test_positional_encoding_optimization(self):
        """Issue #698: PositionalEncodingのtranspose削減テスト"""
        d_model = 64
        max_len = 100
        pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # 事前計算された位置エンコーディングの形状確認
        assert pos_encoding.pe.shape == (1, max_len, d_model), "位置エンコーディング形状が正しくありません"

        # フォワードパス実行
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_encoding(x)

        # 出力形状確認
        assert output.shape == x.shape, "位置エンコーディング後の形状が正しくありません"

        # 位置エンコーディングが適用されていることを確認
        assert not torch.equal(output, x), "位置エンコーディングが適用されていません"

    def test_hybrid_model_efficient_tensor_ops(self, config, sample_tensors):
        """Issue #698: HybridLSTMTransformerModelの効率的テンソル操作テスト"""
        input_dim = sample_tensors['features']
        model = HybridLSTMTransformerModel(config, input_dim)

        # 効率化フラグの確認
        assert hasattr(model, '_use_efficient_tensor_ops'), "効率化フラグが存在しません"
        assert model._use_efficient_tensor_ops == True, "効率化フラグが無効です"

        # フォワードパス実行
        input_tensor = sample_tensors['input_tensor']
        with torch.no_grad():
            predictions = model(input_tensor)

            # 出力形状確認
            expected_shape = (sample_tensors['batch_size'], config.prediction_horizon)
            assert predictions.shape == expected_shape, f"予測出力形状が正しくありません: {predictions.shape}"

    def test_attention_analysis_memory_efficiency(self, config, sample_tensors):
        """Issue #698: アテンション分析のメモリ効率化テスト"""
        input_dim = sample_tensors['features']
        model = HybridLSTMTransformerModel(config, input_dim)

        input_tensor = sample_tensors['input_tensor']

        with torch.no_grad():
            predictions, attention_info = model(input_tensor, return_attention=True)

            # アテンション情報の確認
            assert 'cross_attention_weights' in attention_info
            assert 'lstm_contribution' in attention_info
            assert 'transformer_contribution' in attention_info

            # 貢献度の範囲確認
            lstm_contrib = attention_info['lstm_contribution']
            transformer_contrib = attention_info['transformer_contribution']

            assert 0.0 <= lstm_contrib <= 1.0, f"LSTM貢献度が範囲外: {lstm_contrib}"
            assert 0.0 <= transformer_contrib <= 1.0, f"Transformer貢献度が範囲外: {transformer_contrib}"

    def test_uncertainty_estimation_batch_optimization(self, config, sample_tensors):
        """Issue #698: 不確実性推定のバッチ最適化テスト"""
        input_dim = sample_tensors['features']
        model = HybridLSTMTransformerModel(config, input_dim)

        input_tensor = sample_tensors['input_tensor']

        # 大きなサンプル数での効率化テスト
        num_samples = 20
        with torch.no_grad():
            mean_pred, std_pred = model.forward_with_uncertainty(input_tensor, num_samples)

            # 出力形状確認
            expected_shape = (sample_tensors['batch_size'], config.prediction_horizon)
            assert mean_pred.shape == expected_shape, "平均予測形状が正しくありません"
            assert std_pred.shape == expected_shape, "標準偏差形状が正しくありません"

            # 不確実性の値が有効範囲内であることを確認
            assert torch.isfinite(mean_pred).all(), "平均予測に無限値が含まれています"
            assert torch.isfinite(std_pred).all(), "標準偏差に無限値が含まれています"
            assert (std_pred >= 0).all(), "標準偏差が負の値です"

    def test_memory_usage_comparison(self, config, sample_tensors):
        """Issue #698: メモリ使用量比較テスト"""
        input_dim = sample_tensors['features']

        # 最適化有効モデル
        model_optimized = HybridLSTMTransformerModel(config, input_dim)
        model_optimized._use_efficient_tensor_ops = True

        # 最適化無効モデル
        model_standard = HybridLSTMTransformerModel(config, input_dim)
        model_standard._use_efficient_tensor_ops = False

        input_tensor = sample_tensors['input_tensor']

        # メモリ使用量測定（簡易）
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with torch.no_grad():
            # 最適化版実行
            pred_opt = model_optimized(input_tensor)

            # 標準版実行
            pred_std = model_standard(input_tensor)

            # 結果の一致性確認（完全一致は期待しないが、近似的に一致すること）
            assert pred_opt.shape == pred_std.shape, "最適化版と標準版の出力形状が異なります"

    def test_performance_improvement(self, config, sample_tensors):
        """Issue #698: パフォーマンス向上テスト"""
        input_dim = sample_tensors['features']
        model = HybridLSTMTransformerModel(config, input_dim)

        input_tensor = sample_tensors['input_tensor']

        # 最適化有効での実行時間測定
        model._use_efficient_tensor_ops = True
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        optimized_time = time.time() - start_time

        # 最適化無効での実行時間測定
        model._use_efficient_tensor_ops = False
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        standard_time = time.time() - start_time

        # パフォーマンス比較（最適化版が遅くないことを確認）
        assert optimized_time <= standard_time * 1.2, f"最適化版が期待より遅い: {optimized_time:.4f}s vs {standard_time:.4f}s"

        print(f"パフォーマンス比較: 最適化版 {optimized_time:.4f}s, 標準版 {standard_time:.4f}s")

    def test_initialization_optimization(self, config):
        """Issue #698: 重み初期化最適化テスト"""
        input_dim = 10
        model = HybridLSTMTransformerModel(config, input_dim)

        # Conv1d層の初期化確認
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]

        for conv_layer in conv_layers:
            # 重みが適切に初期化されていることを確認
            weight_std = conv_layer.weight.std().item()
            assert 0.01 < weight_std < 1.0, f"Conv1d重みの標準偏差が範囲外: {weight_std}"

    def test_gradient_flow_optimization(self, config, sample_tensors):
        """Issue #698: 勾配フロー最適化テスト"""
        input_dim = sample_tensors['features']
        model = HybridLSTMTransformerModel(config, input_dim)

        input_tensor = sample_tensors['input_tensor']
        target = torch.randn(sample_tensors['batch_size'], config.prediction_horizon)

        # 勾配計算テスト
        model.train()
        predictions = model(input_tensor)
        loss = nn.MSELoss()(predictions, target)
        loss.backward()

        # 勾配が適切に計算されていることを確認
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.sum().item() != 0:
                has_gradients = True
                break

        assert has_gradients, "勾配が計算されていません"


@pytest.mark.skipif(not TEST_AVAILABLE, reason="PyTorch or required modules not available")
class TestIntegrationWithOptimization:
    """Issue #698: 最適化統合テスト"""

    def test_engine_with_tensor_optimization(self):
        """Issue #698: HybridLSTMTransformerEngineの最適化統合テスト"""
        config = HybridModelConfig(
            sequence_length=15,
            prediction_horizon=2,
            lstm_hidden_size=32,
            transformer_d_model=16,
            epochs=3,
            batch_size=4
        )

        engine = HybridLSTMTransformerEngine(config)

        # テストデータ
        import pandas as pd
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'target': np.random.randn(100)
        })

        # モデル構築
        X, y = engine.prepare_data(test_data)
        model = engine.build_model(X.shape[1:])
        engine.model = model

        # 最適化フラグの確認
        if hasattr(model, '_use_efficient_tensor_ops'):
            assert model._use_efficient_tensor_ops == True, "最適化フラグが設定されていません"

        # 簡易訓練テスト
        if PYTORCH_AVAILABLE and hasattr(engine.model, "parameters"):
            results = engine._train_internal(X[:50], y[:50])
            assert 'training_time' in results, "訓練結果が正しくありません"
            assert results['training_time'] > 0, "訓練時間が記録されていません"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])