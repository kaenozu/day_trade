#!/usr/bin/env python3
"""
HybridLSTMTransformer Data Loading Optimization Tests

Issue #697対応: HybridLSTMTransformerデータ読み込み最適化テスト
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('C:/gemini-desktop/day_trade/src')

# テストに必要なモジュールをインポート
try:
    from day_trade.ml.hybrid_lstm_transformer import (
        HybridLSTMTransformerEngine,
        HybridModelConfig,
        TimeSeriesDataset,
        PYTORCH_AVAILABLE
    )
    TEST_AVAILABLE = PYTORCH_AVAILABLE
except ImportError as e:
    TEST_AVAILABLE = False
    print(f"テストモジュールインポートエラー: {e}")


@pytest.mark.skipif(not TEST_AVAILABLE, reason="PyTorch or required modules not available")
class TestDataLoadingOptimization:
    """Issue #697: データ読み込み最適化テスト"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return HybridModelConfig(
            sequence_length=15,
            prediction_horizon=2,
            lstm_hidden_size=32,
            transformer_d_model=16,
            epochs=3,
            batch_size=4
        )

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        batch_size, seq_len, features = 8, 15, 5
        np.random.seed(42)

        X = np.random.randn(batch_size * 5, seq_len, features).astype(np.float32)
        y = np.random.randn(batch_size * 5, 2).astype(np.float32)

        return X, y

    def test_timeseries_dataset_tensor_input(self, sample_data):
        """Issue #697: TimeSeriesDatasetのTensor入力対応テスト"""
        X, y = sample_data

        # NumPy入力テスト
        dataset_numpy = TimeSeriesDataset(X[:8], y[:8])
        assert len(dataset_numpy) == 8
        sample_X, sample_y = dataset_numpy[0]
        assert isinstance(sample_X, torch.Tensor)
        assert isinstance(sample_y, torch.Tensor)

        # Tensor入力テスト
        X_tensor = torch.from_numpy(X[:8]).float()
        y_tensor = torch.from_numpy(y[:8]).float()
        dataset_tensor = TimeSeriesDataset(X_tensor, y_tensor)

        assert len(dataset_tensor) == 8
        sample_X_t, sample_y_t = dataset_tensor[0]
        assert isinstance(sample_X_t, torch.Tensor)
        assert isinstance(sample_y_t, torch.Tensor)

        # データ一致性確認
        assert torch.allclose(sample_X, sample_X_t, atol=1e-6)
        assert torch.allclose(sample_y, sample_y_t, atol=1e-6)

    def test_efficient_dataset_creation(self, sample_data):
        """Issue #697: 効率的データセット作成テスト"""
        X, y = sample_data

        # CPU デバイス
        cpu_device = torch.device('cpu')
        dataset_cpu = TimeSeriesDataset.create_efficient_dataset(
            X[:8], y[:8], device=cpu_device, use_pinned_memory=False
        )

        assert len(dataset_cpu) == 8
        assert dataset_cpu.device == cpu_device

        # GPU デバイス（利用可能な場合）
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda')
            dataset_gpu = TimeSeriesDataset.create_efficient_dataset(
                X[:8], y[:8], device=cuda_device, use_pinned_memory=True
            )

            assert len(dataset_gpu) == 8
            # pin_memory使用時はCPUデバイスになる
            assert dataset_gpu.device.type == 'cpu'

    def test_pinned_memory_functionality(self, sample_data):
        """Issue #697: ピンメモリ機能テスト"""
        X, y = sample_data

        # ピンメモリ対応データセット作成
        dataset = TimeSeriesDataset(X[:8], y[:8], device=torch.device('cpu'))

        # データセットからサンプル取得
        sample_X, sample_y = dataset[0]

        # テンソルのpinned状態は環境に依存するため、型チェックのみ
        assert isinstance(sample_X, torch.Tensor)
        assert isinstance(sample_y, torch.Tensor)
        assert sample_X.dtype == torch.float32
        assert sample_y.dtype == torch.float32

    def test_dataloader_optimization_config(self, config):
        """Issue #697: DataLoader最適化設定テスト"""
        engine = HybridLSTMTransformerEngine(config)

        # 最適化設定の確認
        assert hasattr(engine, '_dataloader_config')
        dl_config = engine._dataloader_config

        # 必須項目の存在確認
        required_keys = ['num_workers', 'pin_memory', 'persistent_workers', 'drop_last', 'prefetch_factor']
        for key in required_keys:
            assert key in dl_config, f"{key}が設定に含まれていません"

        # 値の型確認
        assert isinstance(dl_config['num_workers'], int)
        assert isinstance(dl_config['pin_memory'], bool)
        assert isinstance(dl_config['persistent_workers'], bool)
        assert isinstance(dl_config['drop_last'], bool)
        assert isinstance(dl_config['prefetch_factor'], int)

        # 値の合理性確認
        assert dl_config['num_workers'] >= 0
        assert dl_config['prefetch_factor'] > 0

    @patch('torch.cuda.is_available')
    def test_dataloader_config_cuda_vs_cpu(self, mock_cuda_available, config):
        """Issue #697: CUDA/CPU環境でのDataLoader設定テスト"""
        # CUDA利用可能環境
        mock_cuda_available.return_value = True
        engine_cuda = HybridLSTMTransformerEngine(config)

        cuda_config = engine_cuda._dataloader_config
        assert cuda_config['pin_memory'] == True
        assert cuda_config['persistent_workers'] == True
        assert cuda_config['num_workers'] > 0

        # CPU環境
        mock_cuda_available.return_value = False
        engine_cpu = HybridLSTMTransformerEngine(config)

        cpu_config = engine_cpu._dataloader_config
        assert cpu_config['pin_memory'] == False
        # persistent_workersはnum_workersに依存
        assert cpu_config['num_workers'] >= 0

    def test_non_blocking_transfer(self, sample_data):
        """Issue #697: non_blocking転送のテスト"""
        X, y = sample_data

        # テストDataLoader作成
        dataset = TimeSeriesDataset(X[:8], y[:8])
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for batch_X, batch_y in dataloader:
            # non_blocking転送テスト
            batch_X_gpu = batch_X.to(device, non_blocking=True)
            batch_y_gpu = batch_y.to(device, non_blocking=True)

            # 転送成功確認
            assert batch_X_gpu.device == device
            assert batch_y_gpu.device == device
            assert batch_X_gpu.shape == batch_X.shape
            assert batch_y_gpu.shape == batch_y.shape
            break  # 最初のバッチのみテスト

    def test_dataloader_performance_optimization(self, sample_data):
        """Issue #697: DataLoaderパフォーマンス最適化テスト"""
        X, y = sample_data

        # 基本DataLoader
        dataset_basic = TimeSeriesDataset(X, y)
        dataloader_basic = DataLoader(dataset_basic, batch_size=8, shuffle=False)

        # 最適化DataLoader
        dataset_opt = TimeSeriesDataset.create_efficient_dataset(
            X, y, device=torch.device('cpu'), use_pinned_memory=False
        )
        dataloader_opt = DataLoader(
            dataset_opt,
            batch_size=8,
            shuffle=False,
            num_workers=0,  # テスト環境では0に設定
            pin_memory=False
        )

        # 基本実行時間測定
        start_time = time.time()
        for batch_X, batch_y in dataloader_basic:
            pass
        basic_time = time.time() - start_time

        # 最適化実行時間測定
        start_time = time.time()
        for batch_X, batch_y in dataloader_opt:
            pass
        opt_time = time.time() - start_time

        # パフォーマンス確認（最適化版が大幅に遅くないことを確認）
        assert opt_time <= basic_time * 2.0, f"最適化版が期待より遅い: {opt_time:.4f}s vs {basic_time:.4f}s"

        print(f"DataLoader パフォーマンス: 基本版 {basic_time:.4f}s, 最適化版 {opt_time:.4f}s")

    def test_memory_efficiency(self, sample_data):
        """Issue #697: メモリ効率性テスト"""
        X, y = sample_data

        # GPU-CPU変換なしのデータセット
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        dataset_efficient = TimeSeriesDataset(X_tensor, y_tensor)

        # 従来のNumPy経由データセット
        dataset_traditional = TimeSeriesDataset(X, y)

        # データの一致性確認
        sample_eff = dataset_efficient[0]
        sample_trad = dataset_traditional[0]

        assert torch.allclose(sample_eff[0], sample_trad[0], atol=1e-6)
        assert torch.allclose(sample_eff[1], sample_trad[1], atol=1e-6)

        # メモリレイアウト確認
        assert sample_eff[0].is_contiguous()
        assert sample_eff[1].is_contiguous()

    def test_batch_processing_optimization(self, sample_data):
        """Issue #697: バッチ処理最適化テスト"""
        X, y = sample_data

        # 効率的データセット
        dataset = TimeSeriesDataset.create_efficient_dataset(X, y)

        # drop_last=Trueでの動作確認
        dataloader = DataLoader(
            dataset,
            batch_size=7,  # 40 / 7 = 5余り5 → drop_lastでバッチ数は5
            shuffle=False,
            drop_last=True
        )

        batch_count = 0
        for batch_X, batch_y in dataloader:
            batch_count += 1
            assert batch_X.size(0) == 7  # 全バッチが指定サイズ
            assert batch_y.size(0) == 7

        expected_batches = len(X) // 7  # drop_lastで不完全バッチを除外
        assert batch_count == expected_batches

    def test_data_type_consistency(self, sample_data):
        """Issue #697: データ型一貫性テスト"""
        X, y = sample_data

        # float32データでの一貫性
        X_f32 = X.astype(np.float32)
        y_f32 = y.astype(np.float32)

        dataset = TimeSeriesDataset(X_f32, y_f32)
        sample_X, sample_y = dataset[0]

        assert sample_X.dtype == torch.float32
        assert sample_y.dtype == torch.float32

        # DataLoaderを通した後の型確認
        dataloader = DataLoader(dataset, batch_size=2)
        for batch_X, batch_y in dataloader:
            assert batch_X.dtype == torch.float32
            assert batch_y.dtype == torch.float32
            break


@pytest.mark.skipif(not TEST_AVAILABLE, reason="PyTorch or required modules not available")
class TestIntegrationDataLoading:
    """Issue #697: データ読み込み統合テスト"""

    def test_training_with_optimized_dataloader(self):
        """Issue #697: 最適化DataLoaderでの学習統合テスト"""
        config = HybridModelConfig(
            sequence_length=10,
            prediction_horizon=1,
            lstm_hidden_size=16,
            transformer_d_model=8,
            epochs=2,
            batch_size=4
        )

        engine = HybridLSTMTransformerEngine(config)

        # テストデータ
        import pandas as pd
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'target': np.random.randn(50)
        })

        # データ準備
        X, y = engine.prepare_data(test_data)

        # モデル構築
        model = engine.build_model(X.shape[1:])
        engine.model = model

        # DataLoader最適化設定の確認
        assert hasattr(engine, '_dataloader_config')

        # 学習実行（最適化されたDataLoaderが使用される）
        if PYTORCH_AVAILABLE and hasattr(engine.model, "parameters"):
            results = engine._train_internal(X[:30], y[:30])
            assert 'training_time' in results
            assert results['training_time'] > 0

            # 学習が正常に完了することを確認
            assert engine.is_trained


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])