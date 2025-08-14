#!/usr/bin/env python3
"""
推論最適化システム包括テストスイート
Comprehensive Test Suite for Inference Optimization System

Issue #761: MLモデル推論パイプラインの高速化と最適化
"""

import pytest
import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from src.day_trade.inference import (
    ModelOptimizationEngine, ModelOptimizationConfig,
    MemoryOptimizer, MemoryConfig,
    ParallelInferenceEngine, ParallelConfig, InferenceTask,
    AdvancedOptimizer, OptimizationConfig,
    OptimizedInferenceSystem, InferenceSystemConfig,
    create_optimized_inference_system
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelOptimization:
    """モデル最適化テスト"""

    @pytest.fixture
    def optimization_config(self):
        return ModelOptimizationConfig(
            enable_onnx=True,
            enable_quantization=True,
            enable_batch_inference=True,
            max_batch_size=8,
            num_threads=2
        )

    @pytest.fixture
    def test_data(self):
        return np.random.randn(10, 20).astype(np.float32)

    def test_optimization_config_creation(self, optimization_config):
        """最適化設定作成テスト"""
        assert optimization_config.enable_onnx is True
        assert optimization_config.max_batch_size == 8
        assert optimization_config.num_threads == 2

    def test_model_optimization_engine_initialization(self, optimization_config):
        """モデル最適化エンジン初期化テスト"""
        engine = ModelOptimizationEngine(optimization_config)
        assert engine.config == optimization_config
        assert engine.optimizers is not None

    @pytest.mark.asyncio
    async def test_batch_inference_engine(self, optimization_config, test_data):
        """バッチ推論エンジンテスト"""
        from src.day_trade.inference.model_optimizer import BatchInferenceEngine

        batch_engine = BatchInferenceEngine(optimization_config)
        assert batch_engine.config.max_batch_size == 8

        # バッチ処理のシミュレーション
        # 実際のモデルが必要なため、ここでは基本的なテストのみ


class TestMemoryOptimization:
    """メモリ最適化テスト"""

    @pytest.fixture
    def memory_config(self):
        return MemoryConfig(
            max_models_in_memory=3,
            feature_cache_size_mb=100,
            memory_warning_threshold=0.8
        )

    @pytest.fixture
    def memory_optimizer(self, memory_config):
        return MemoryOptimizer(memory_config)

    def test_memory_config_creation(self, memory_config):
        """メモリ設定作成テスト"""
        assert memory_config.max_models_in_memory == 3
        assert memory_config.feature_cache_size_mb == 100
        assert memory_config.memory_warning_threshold == 0.8

    def test_memory_optimizer_initialization(self, memory_optimizer):
        """メモリ最適化システム初期化テスト"""
        assert memory_optimizer.config is not None
        assert memory_optimizer.model_pool is not None
        assert memory_optimizer.feature_cache is not None

    def test_model_pool_functionality(self, memory_optimizer):
        """モデルプール機能テスト"""
        model_pool = memory_optimizer.model_pool

        # ダミーモデルローダー
        def dummy_loader():
            return f"DummyModel_{time.time()}"

        # モデル読み込みテスト
        model1 = model_pool.load_model("test_model_1", dummy_loader)
        assert model1 is not None
        assert "test_model_1" in model_pool.models

        # 同じモデル再読み込み（キャッシュヒット）
        model2 = model_pool.load_model("test_model_1", dummy_loader)
        assert model1 == model2

        # モデル解放
        model_pool.release_model("test_model_1")

    def test_feature_cache_functionality(self, memory_optimizer):
        """特徴量キャッシュ機能テスト"""
        feature_cache = memory_optimizer.feature_cache

        # テスト特徴量
        test_features = np.random.randn(100, 10).astype(np.float32)
        cache_key = "test_features_key"

        # キャッシュ保存
        success = feature_cache.store_features(cache_key, test_features)
        assert success is True

        # キャッシュ取得
        cached_features = feature_cache.get_features(cache_key)
        assert cached_features is not None
        assert np.array_equal(cached_features, test_features)

        # 存在しないキーの取得
        non_existent = feature_cache.get_features("non_existent_key")
        assert non_existent is None

    def test_memory_stats_collection(self, memory_optimizer):
        """メモリ統計収集テスト"""
        stats = memory_optimizer.get_comprehensive_stats()

        assert "memory_stats" in stats
        assert "model_pool_stats" in stats
        assert "feature_cache_stats" in stats

        memory_stats = stats["memory_stats"]
        assert "memory_usage_percent" in memory_stats
        assert isinstance(memory_stats["memory_usage_percent"], float)


class TestParallelProcessing:
    """並列処理テスト"""

    @pytest.fixture
    def parallel_config(self):
        return ParallelConfig(
            cpu_workers=2,
            enable_gpu_parallel=False,  # テスト環境ではGPUを無効化
            async_workers=3,
            cpu_batch_size=4
        )

    @pytest.fixture
    def parallel_engine(self, parallel_config):
        return ParallelInferenceEngine(parallel_config)

    def test_parallel_config_creation(self, parallel_config):
        """並列処理設定作成テスト"""
        assert parallel_config.cpu_workers == 2
        assert parallel_config.enable_gpu_parallel is False
        assert parallel_config.async_workers == 3

    @pytest.mark.asyncio
    async def test_parallel_engine_initialization(self, parallel_engine):
        """並列エンジン初期化テスト"""
        assert parallel_engine.config is not None
        assert parallel_engine.cpu_pool is not None
        assert parallel_engine.async_pool is not None

    @pytest.mark.asyncio
    async def test_parallel_engine_lifecycle(self, parallel_engine):
        """並列エンジンライフサイクルテスト"""
        # エンジン開始
        await parallel_engine.start()

        # 統計確認
        stats = parallel_engine.get_comprehensive_stats()
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 0

        # エンジン停止
        await parallel_engine.stop()

    def test_inference_task_creation(self):
        """推論タスク作成テスト"""
        test_data = np.random.randn(1, 10).astype(np.float32)

        task = InferenceTask(
            task_id="test_task_1",
            model_id="test_model",
            input_data=test_data,
            priority=1
        )

        assert task.task_id == "test_task_1"
        assert task.model_id == "test_model"
        assert np.array_equal(task.input_data, test_data)
        assert task.priority == 1


class TestAdvancedOptimization:
    """高度最適化テスト"""

    @pytest.fixture
    def optimization_config(self):
        return OptimizationConfig(
            enable_dynamic_model_selection=True,
            enable_inference_caching=True,
            enable_profiling=True,
            enable_ab_testing=True
        )

    @pytest.fixture
    def advanced_optimizer(self, optimization_config):
        return AdvancedOptimizer(optimization_config)

    def test_optimization_config_creation(self, optimization_config):
        """高度最適化設定作成テスト"""
        assert optimization_config.enable_dynamic_model_selection is True
        assert optimization_config.enable_inference_caching is True
        assert optimization_config.enable_profiling is True

    def test_advanced_optimizer_initialization(self, advanced_optimizer):
        """高度最適化システム初期化テスト"""
        assert advanced_optimizer.config is not None
        assert advanced_optimizer.model_selector is not None
        assert advanced_optimizer.inference_cache is not None
        assert advanced_optimizer.profiler is not None
        assert advanced_optimizer.ab_tester is not None

    def test_dynamic_model_selection(self, advanced_optimizer):
        """動的モデル選択テスト"""
        model_selector = advanced_optimizer.model_selector

        # モデル登録
        models = ["model_fast", "model_accurate", "model_balanced"]
        for model_id in models:
            model_selector.register_model(model_id)

        # モデル選択
        selected_model = model_selector.select_optimal_model(models)
        assert selected_model in models

        # メトリクス更新
        model_selector.update_model_metrics(selected_model, latency_ms=10.0, accuracy=0.95, throughput=100.0)

        # ランキング確認
        rankings = model_selector.get_model_rankings()
        assert len(rankings) == len(models)
        assert all(isinstance(rank, tuple) and len(rank) == 2 for rank in rankings)

    def test_inference_caching(self, advanced_optimizer):
        """推論キャッシュテスト"""
        cache = advanced_optimizer.inference_cache

        # テストデータ
        input_data = np.random.randn(1, 10).astype(np.float32)
        output_data = np.random.randn(1, 5).astype(np.float32)
        model_id = "test_model"
        confidence = 0.95

        # キャッシュミス確認
        cached_result = cache.get_cached_result(input_data, model_id)
        assert cached_result is None

        # 結果保存
        cache.store_result(input_data, output_data, model_id, confidence)

        # キャッシュヒット確認
        cached_result = cache.get_cached_result(input_data, model_id)
        assert cached_result is not None

        cached_output, cached_confidence = cached_result
        assert np.array_equal(cached_output, output_data)
        assert cached_confidence == confidence

    def test_ab_testing(self, advanced_optimizer):
        """A/Bテストテスト"""
        ab_tester = advanced_optimizer.ab_tester

        # A/Bテスト開始
        success = ab_tester.start_ab_test(
            "test_ab_001",
            "model_a",
            "model_b",
            ["latency_ms", "accuracy"]
        )
        assert success is True

        # グループ割り当て
        user_id = "test_user_123"
        assigned_model = ab_tester.assign_test_group("test_ab_001", user_id)
        assert assigned_model in ["model_a", "model_b"]

        # 結果記録
        metrics = {"latency_ms": 15.0, "accuracy": 0.92}
        ab_tester.record_test_result("test_ab_001", assigned_model, metrics)

        # テスト停止
        final_analysis = ab_tester.stop_test("test_ab_001")
        assert isinstance(final_analysis, dict)


class TestIntegratedSystem:
    """統合システムテスト"""

    @pytest.fixture
    def system_config(self):
        return InferenceSystemConfig(
            system_name="TestOptimizedSystem",
            target_latency_ms=5.0,
            target_throughput_per_second=200.0,
            supported_models=["test_model_1", "test_model_2"]
        )

    @pytest.mark.asyncio
    async def test_system_creation_helper(self):
        """システム作成ヘルパーテスト"""
        config_overrides = {
            "system_name": "HelperTestSystem",
            "target_latency_ms": 3.0
        }

        system = await create_optimized_inference_system(config_overrides)

        assert system.config.system_name == "HelperTestSystem"
        assert system.config.target_latency_ms == 3.0
        assert system.is_initialized is True

        await system.stop()

    @pytest.mark.asyncio
    async def test_system_lifecycle(self, system_config):
        """システムライフサイクルテスト"""
        system = OptimizedInferenceSystem(system_config)

        # 初期化
        await system.initialize()
        assert system.is_initialized is True
        assert system.health_status == "healthy"

        # システム状態確認
        status = system.get_system_status()
        assert status["system_name"] == "TestOptimizedSystem"
        assert status["is_running"] is False  # まだ開始していない

        # 停止
        await system.stop()
        assert system.health_status == "stopped"

    @pytest.mark.asyncio
    async def test_prediction_functionality(self, system_config):
        """予測機能テスト"""
        system = OptimizedInferenceSystem(system_config)
        await system.initialize()

        # システム状態を手動で実行中に設定（テスト用）
        system.is_running = True

        try:
            # 単一予測テスト
            input_data = np.random.randn(1, 10).astype(np.float32)
            result = await system.predict(input_data, model_id="test_model_1", user_id="test_user")

            assert "prediction" in result
            assert "processing_time_ms" in result
            assert "model_used" in result
            assert result["model_used"] == "test_model_1"
            assert isinstance(result["processing_time_ms"], float)

            # バッチ予測テスト
            batch_data = [np.random.randn(1, 10).astype(np.float32) for _ in range(3)]
            batch_results = await system.predict_batch(batch_data, model_id="test_model_1")

            assert len(batch_results) == 3
            for result in batch_results:
                assert "prediction" in result or "error" in result

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, system_config):
        """パフォーマンス監視テスト"""
        system = OptimizedInferenceSystem(system_config)
        await system.initialize()
        system.is_running = True

        try:
            # いくつかの予測を実行して統計を蓄積
            for i in range(5):
                input_data = np.random.randn(1, 10).astype(np.float32)
                await system.predict(input_data, user_id=f"perf_test_user_{i}")

            # パフォーマンスメトリクス確認
            metrics = system.get_performance_metrics()

            assert "current_performance" in metrics
            assert "target_achievement" in metrics
            assert "optimization_impact" in metrics

            current_perf = metrics["current_performance"]
            assert "avg_latency_ms" in current_perf
            assert "current_throughput" in current_perf
            assert "success_rate" in current_perf

            # 成功率確認
            assert 0.0 <= current_perf["success_rate"] <= 1.0

        finally:
            await system.stop()


class TestPerformanceBenchmarks:
    """パフォーマンスベンチマークテスト"""

    @pytest.mark.asyncio
    async def test_latency_benchmark(self):
        """レイテンシベンチマーク"""
        config_overrides = {
            "system_name": "LatencyBenchmarkSystem",
            "target_latency_ms": 2.0
        }

        system = await create_optimized_inference_system(config_overrides)
        system.is_running = True

        try:
            latencies = []

            # 100回の予測実行
            for i in range(100):
                input_data = np.random.randn(1, 20).astype(np.float32)
                start_time = time.perf_counter()

                result = await system.predict(input_data, user_id=f"benchmark_user_{i}")

                end_time = time.perf_counter()
                actual_latency = (end_time - start_time) * 1000  # ms

                latencies.append(actual_latency)

                # 結果が成功していることを確認
                assert "prediction" in result

            # 統計計算
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            logger.info(f"Latency Benchmark Results:")
            logger.info(f"  Average: {avg_latency:.2f}ms")
            logger.info(f"  P95: {p95_latency:.2f}ms")
            logger.info(f"  P99: {p99_latency:.2f}ms")

            # 基本的な性能確認（環境に依存するため緩い条件）
            assert avg_latency < 100.0  # 100ms未満
            assert p95_latency < 200.0   # P95が200ms未満

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """スループットベンチマーク"""
        config_overrides = {
            "system_name": "ThroughputBenchmarkSystem",
            "target_throughput_per_second": 100.0
        }

        system = await create_optimized_inference_system(config_overrides)
        system.is_running = True

        try:
            # 並列リクエスト生成
            num_requests = 50
            tasks = []

            start_time = time.perf_counter()

            for i in range(num_requests):
                input_data = np.random.randn(1, 15).astype(np.float32)
                task = system.predict(input_data, user_id=f"throughput_user_{i}")
                tasks.append(task)

            # 全リクエスト完了待機
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # 成功したリクエスト数計算
            successful_requests = sum(
                1 for result in results
                if not isinstance(result, Exception) and "prediction" in result
            )

            throughput = successful_requests / total_time

            logger.info(f"Throughput Benchmark Results:")
            logger.info(f"  Total requests: {num_requests}")
            logger.info(f"  Successful requests: {successful_requests}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Throughput: {throughput:.2f} requests/sec")

            # 基本的な性能確認
            assert successful_requests >= num_requests * 0.9  # 90%以上成功
            assert throughput > 10.0  # 最低10 req/sec

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """メモリ効率テスト"""
        import psutil
        import gc

        # ベースラインメモリ使用量
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        config_overrides = {
            "system_name": "MemoryEfficiencySystem"
        }

        system = await create_optimized_inference_system(config_overrides)
        system.is_running = True

        try:
            # システム開始後のメモリ使用量
            gc.collect()
            after_start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            # 大量予測実行
            for i in range(200):
                input_data = np.random.randn(1, 50).astype(np.float32)
                await system.predict(input_data, user_id=f"memory_user_{i}")

                # 定期的なガベージコレクション
                if i % 50 == 0:
                    gc.collect()

            # 最終メモリ使用量
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            logger.info(f"Memory Efficiency Results:")
            logger.info(f"  Baseline memory: {baseline_memory:.1f}MB")
            logger.info(f"  After start: {after_start_memory:.1f}MB")
            logger.info(f"  Final memory: {final_memory:.1f}MB")
            logger.info(f"  Memory increase: {final_memory - baseline_memory:.1f}MB")

            # メモリ効率確認（大幅な増加がないこと）
            memory_increase = final_memory - baseline_memory
            assert memory_increase < 500.0  # 500MB未満の増加

        finally:
            await system.stop()
            gc.collect()


# テスト実行用のメイン関数
if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v", "--tb=short"])