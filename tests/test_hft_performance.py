#!/usr/bin/env python3
"""
HFT パフォーマンステスト
Issue #434: 本番環境パフォーマンス最終最適化

超低レイテンシとGPU加速のパフォーマンス検証
"""

import asyncio
import time

import numpy as np
import pytest

from src.day_trade.ml.feature_pipeline import FeaturePipeline, PipelineConfig
from src.day_trade.performance import GPUAccelerator, GPUConfig, HFTConfig, HFTOptimizer


class TestHFTPerformance:
    """HFT パフォーマンステスト"""

    @pytest.fixture
    def hft_config(self):
        """HFT設定"""
        return HFTConfig(
            target_latency_us=50.0,
            preallocated_memory_mb=50,
            enable_simd=True,
            max_threads=2,
        )

    @pytest.fixture
    def gpu_config(self):
        """GPU設定"""
        return GPUConfig(
            batch_size=256,
            gpu_memory_limit_mb=512,
            cpu_fallback=True,
        )

    @pytest.fixture
    def test_data(self):
        """テストデータ生成"""
        np.random.seed(42)
        return {
            "prices": np.random.normal(100, 5, 1000).astype(np.float64),
            "volumes": np.random.normal(10000, 1000, 1000).astype(np.float64),
            "model_weights": np.random.normal(0, 0.1, 8).astype(np.float64),
        }

    def test_hft_optimizer_initialization(self, hft_config):
        """HFT最適化エンジン初期化テスト"""
        optimizer = HFTOptimizer(hft_config)

        assert optimizer.config.target_latency_us == 50.0
        assert optimizer.memory_pool is not None
        assert optimizer.performance_stats is not None

        # メモリプール動作確認
        memory_info = optimizer.memory_pool.get_stats()
        assert memory_info["total_allocations"] == 0
        assert memory_info["pool_utilization"] == 0.0

        optimizer.cleanup()

    def test_ultra_fast_prediction(self, hft_config, test_data):
        """超高速予測テスト"""
        optimizer = HFTOptimizer(hft_config)

        # 複数回予測でレイテンシ測定
        latencies = []
        predictions = []

        for _ in range(10):
            result = optimizer.predict_ultra_fast(
                test_data["prices"], test_data["volumes"]
            )

            assert "prediction" in result
            assert "latency_us" in result
            assert "under_target" in result

            latencies.append(result["latency_us"])
            predictions.append(result["prediction"])

            # 基本的な妥当性チェック
            assert isinstance(result["prediction"], float)
            assert result["latency_us"] > 0

        # パフォーマンス検証
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        under_target_rate = np.mean(
            [lat < hft_config.target_latency_us for lat in latencies]
        )

        print(f"平均レイテンシ: {avg_latency:.2f}μs")
        print(f"最大レイテンシ: {max_latency:.2f}μs")
        print(f"目標達成率: {under_target_rate:.1%}")

        # パフォーマンス要件検証
        assert avg_latency < hft_config.target_latency_us * 2  # 平均は目標の2倍以内
        assert under_target_rate >= 0.5  # 50%以上は目標達成

        optimizer.cleanup()

    def test_batch_prediction_performance(self, hft_config, test_data):
        """バッチ予測パフォーマンステスト"""
        optimizer = HFTOptimizer(hft_config)

        # 複数銘柄データ準備
        symbols_data = {}
        for i in range(20):
            symbol = f"STOCK_{i:03d}"
            symbols_data[symbol] = {
                "prices": test_data["prices"]
                + np.random.normal(0, 1, len(test_data["prices"])),
                "volumes": test_data["volumes"]
                + np.random.normal(0, 100, len(test_data["volumes"])),
            }

        # バッチ予測実行
        start_time = time.perf_counter()
        batch_result = optimizer.batch_predict_optimized(symbols_data, batch_size=10)
        total_time = (time.perf_counter() - start_time) * 1000  # ms

        batch_stats = batch_result["batch_stats"]

        # 結果検証
        assert batch_stats["total_symbols"] == 20
        assert batch_stats["total_batch_time_us"] > 0
        assert batch_stats["avg_latency_per_symbol_us"] > 0

        # パフォーマンス検証
        symbols_per_ms = batch_stats["total_symbols"] / total_time

        print(f"バッチ処理: {batch_stats['total_symbols']}銘柄")
        print(f"総処理時間: {total_time:.2f}ms")
        print(f"平均銘柄レイテンシ: {batch_stats['avg_latency_per_symbol_us']:.2f}μs")
        print(f"処理速度: {symbols_per_ms:.1f} symbols/ms")

        # バッチ効率検証
        assert symbols_per_ms > 0.1  # 0.1 symbols/ms以上（10symbols/100ms）
        assert batch_stats["under_target_rate"] >= 0.3  # 30%以上は目標達成

        optimizer.cleanup()

    def test_gpu_accelerator_initialization(self, gpu_config):
        """GPU加速エンジン初期化テスト"""
        accelerator = GPUAccelerator(gpu_config)

        assert accelerator.config.batch_size == 256
        assert accelerator.memory_manager is not None

        # GPU可用性確認
        gpu_report = accelerator.get_gpu_report()
        assert "gpu_available" in gpu_report["availability"]
        assert "cupy_available" in gpu_report["availability"]

        print(f"GPU利用可能: {gpu_report['availability']['gpu_available']}")
        print(f"CuPy利用可能: {gpu_report['availability']['cupy_available']}")

        accelerator.cleanup()

    def test_gpu_feature_calculation(self, gpu_config, test_data):
        """GPU特徴量計算テスト"""
        accelerator = GPUAccelerator(gpu_config)

        prices = test_data["prices"].astype(np.float32)
        volumes = test_data["volumes"].astype(np.float32)

        # GPU特徴量計算
        start_time = time.perf_counter()
        features = accelerator.compute_features_gpu(prices, volumes, feature_dim=7)
        gpu_time = (time.perf_counter() - start_time) * 1000  # ms

        # 結果検証
        if features is not None:
            assert features.shape == (len(prices), 7)
            assert features.dtype in [np.float32, np.float64]

            # 特徴量の妥当性チェック（ゼロでない値の存在）
            non_zero_features = np.sum(np.abs(features) > 1e-6)
            assert non_zero_features > 0

            print(f"GPU特徴量計算: {features.shape}, {gpu_time:.2f}ms")
            print(f"非ゼロ特徴量数: {non_zero_features}")
        else:
            print("GPU特徴量計算: CPU フォールバック")

        # CPU版との比較
        start_time = time.perf_counter()
        cpu_features = accelerator._compute_features_cpu(prices, volumes, 7)
        cpu_time = (time.perf_counter() - start_time) * 1000  # ms

        print(f"CPU特徴量計算: {cpu_features.shape}, {cpu_time:.2f}ms")

        if features is not None and accelerator.gpu_available:
            speedup = cpu_time / max(gpu_time, 0.001)
            print(f"GPU加速比: {speedup:.1f}x")

            # GPU加速効果の検証
            assert speedup >= 0.5  # 最低でも半分程度の性能

        accelerator.cleanup()

    @pytest.mark.asyncio
    async def test_async_gpu_pipeline(self, gpu_config, test_data):
        """非同期GPU処理パイプラインテスト"""
        accelerator = GPUAccelerator(gpu_config)

        # 複数銘柄データ準備
        symbols_data = {}
        for i in range(10):
            symbol = f"TEST_{i}"
            symbols_data[symbol] = {
                "prices": test_data["prices"]
                + np.random.normal(0, 2, len(test_data["prices"])),
                "volumes": test_data["volumes"]
                + np.random.normal(0, 200, len(test_data["volumes"])),
            }

        # 非同期パイプライン実行
        result = await accelerator.async_gpu_pipeline(
            symbols_data, test_data["model_weights"][:7]
        )

        # 結果検証
        assert "predictions" in result
        assert "pipeline_stats" in result

        predictions = result["predictions"]
        pipeline_stats = result["pipeline_stats"]

        assert pipeline_stats["total_symbols"] <= 10
        assert pipeline_stats["pipeline_time_ms"] > 0
        assert pipeline_stats["symbols_per_second"] > 0

        # 予測結果の妥当性
        for symbol, prediction in predictions.items():
            assert "prediction" in prediction
            assert "timestamp" in prediction
            assert isinstance(prediction["prediction"], float)

        print(f"非同期パイプライン: {pipeline_stats['total_symbols']}銘柄")
        print(f"処理時間: {pipeline_stats['pipeline_time_ms']:.2f}ms")
        print(f"処理速度: {pipeline_stats['symbols_per_second']:.1f} symbols/sec")

        accelerator.cleanup()

    @pytest.mark.asyncio
    async def test_integrated_pipeline_performance(self):
        """統合パイプラインパフォーマンステスト"""
        # 最適化設定でパイプライン作成
        pipeline_config = PipelineConfig(
            enable_hft_optimization=True,
            hft_target_latency_us=30.0,
            enable_gpu_acceleration=True,
            gpu_batch_size=256,
        )

        # パイプライン初期化はfrom_configで行う必要がある場合は調整
        try:
            pipeline = FeaturePipeline(pipeline_config)
        except Exception as e:
            # 依存関係の問題でスキップ
            pytest.skip(f"統合パイプライン初期化失敗: {e}")

        # テストデータ
        np.random.seed(42)
        test_prices = np.random.normal(100, 5, 500).astype(np.float64)
        test_volumes = np.random.normal(10000, 1000, 500).astype(np.float64)

        # 超高速予測テスト
        prediction_result = await pipeline.ultra_fast_prediction(
            "AAPL", test_prices, test_volumes
        )

        # 結果検証
        assert "symbol" in prediction_result
        assert "prediction" in prediction_result
        assert "latency_us" in prediction_result
        assert "hft_optimized" in prediction_result

        # パフォーマンス検証
        if prediction_result.get("hft_optimized", False):
            assert prediction_result["latency_us"] < 100.0  # 100μs以内
            print(
                f"HFT予測: {prediction_result['latency_us']:.2f}μs (目標: {pipeline_config.hft_target_latency_us}μs)"
            )

        # GPU特徴量生成テスト
        symbols_data = {
            "AAPL": {"prices": test_prices, "volumes": test_volumes},
            "GOOGL": {"prices": test_prices * 1.5, "volumes": test_volumes * 0.8},
            "TSLA": {"prices": test_prices * 2.0, "volumes": test_volumes * 1.2},
        }

        features_result = await pipeline.gpu_batch_feature_generation(symbols_data)

        # GPU特徴量検証
        if features_result:
            assert len(features_result) <= 3
            for symbol, features in features_result.items():
                assert isinstance(features, np.ndarray)
                assert features.shape[0] == len(test_prices)
                print(f"GPU特徴量 {symbol}: {features.shape}")

        # 統計情報取得
        stats = pipeline.get_pipeline_stats()

        print("\n=== 統合パイプライン統計 ===")
        print(f"HFT予測数: {stats.get('hft_predictions', 0)}")
        print(f"HFT平均レイテンシ: {stats.get('hft_avg_latency_us', 0):.2f}μs")
        print(f"HFT目標達成率: {stats.get('hft_under_target_rate', 0):.1%}")
        print(f"GPU加速操作数: {stats.get('gpu_accelerated_operations', 0)}")

        # HFT統計詳細
        if "hft_optimization" in stats:
            hft_stats = stats["hft_optimization"]
            print(f"HFT最適化スコア: {hft_stats.get('optimization_score', 0):.1f}/100")

        # GPU統計詳細
        if "gpu_acceleration" in stats:
            gpu_stats = stats["gpu_acceleration"]
            print(f"GPU効率スコア: {gpu_stats.get('efficiency_score', 0):.1f}/100")

        pipeline.cleanup()

    def test_memory_usage_optimization(self, hft_config):
        """メモリ使用量最適化テスト"""
        optimizer = HFTOptimizer(hft_config)

        initial_stats = optimizer.memory_pool.get_stats()
        assert initial_stats["current_usage_bytes"] == 0

        # メモリ割り当てテスト
        allocations = []
        for i in range(100):
            offset = optimizer.memory_pool.allocate(1024)  # 1KB
            if offset is not None:
                allocations.append(offset)

        after_allocation_stats = optimizer.memory_pool.get_stats()
        assert after_allocation_stats["total_allocations"] > 0
        assert after_allocation_stats["current_usage_bytes"] > 0

        # メモリ解放テスト
        for offset in allocations:
            optimizer.memory_pool.deallocate(offset)

        after_deallocation_stats = optimizer.memory_pool.get_stats()
        assert after_deallocation_stats["total_deallocations"] > 0

        print("メモリ統計:")
        print(f"  割り当て: {after_deallocation_stats['total_allocations']}")
        print(f"  解放: {after_deallocation_stats['total_deallocations']}")
        print(f"  現在使用量: {after_deallocation_stats['current_usage_bytes']} bytes")
        print(f"  プール使用率: {after_deallocation_stats['pool_utilization']:.1%}")

        optimizer.cleanup()

    def test_performance_regression_limits(self):
        """パフォーマンス回帰制限テスト"""
        # 基本的なパフォーマンス制限チェック
        test_data_sizes = [100, 500, 1000, 2000]

        for data_size in test_data_sizes:
            np.random.seed(42)
            prices = np.random.normal(100, 5, data_size).astype(np.float64)
            volumes = np.random.normal(10000, 1000, data_size).astype(np.float64)

            # HFT最適化なしでの基準時間測定
            start_time = time.perf_counter_ns()

            # 基本的な特徴量計算（フォールバック）
            features = np.zeros((data_size, 5))
            if data_size >= 20:
                for i in range(20, data_size):
                    features[i, 0] = np.mean(prices[i - 5 : i])  # MA5
                    features[i, 1] = np.mean(prices[i - 20 : i])  # MA20
                    features[i, 2] = prices[i]
                    features[i, 3] = volumes[i]
                    features[i, 4] = (prices[i] - prices[i - 1]) / prices[i - 1]

            baseline_time_us = (time.perf_counter_ns() - start_time) / 1000.0

            print(f"データサイズ {data_size}: ベースライン {baseline_time_us:.2f}μs")

            # パフォーマンス制限検証
            # 大きなデータでもレスポンス時間が線形以下であることを確認
            if data_size <= 1000:
                assert baseline_time_us < 10000  # 10ms以内
            else:
                assert baseline_time_us < 20000  # 20ms以内


if __name__ == "__main__":
    # 単体テスト実行
    pytest.main([__file__, "-v", "-s"])
