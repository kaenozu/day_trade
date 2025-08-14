#!/usr/bin/env python3
"""
パフォーマンスベンチマークテストスイート
Performance Benchmark Test Suite

Issue #760: 包括的テスト自動化と検証フレームワークの構築
Issue #761対応パフォーマンス目標検証
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock
import json
import os
from pathlib import Path

# テスト対象システム
from src.day_trade.inference import OptimizedInferenceSystem, create_optimized_inference_system
from src.day_trade.analysis.ml_models import UnifiedMLEngine

# テストフレームワーク
from src.day_trade.testing import (
    TestFramework,
    TestConfig,
    PerformanceAssertions,
    TestDataManager,
    BaseTestCase,
    TestResult
)

# ログ設定
logger = logging.getLogger(__name__)


class PerformanceBenchmarkConfig:
    """パフォーマンスベンチマーク設定"""

    # Issue #761 目標値
    TARGET_INFERENCE_LATENCY_MS = 5.0
    TARGET_THROUGHPUT_PER_SEC = 10000.0
    TARGET_MEMORY_EFFICIENCY_RATIO = 0.5
    TARGET_ACCURACY_RETENTION = 0.97

    # テスト設定
    WARMUP_ITERATIONS = 10
    BENCHMARK_ITERATIONS = 100
    CONCURRENCY_LEVELS = [1, 5, 10, 20, 50]
    BATCH_SIZES = [1, 8, 16, 32, 64]

    # リソース制限
    MAX_MEMORY_MB = 1024
    MAX_CPU_PERCENT = 80.0
    TIMEOUT_SECONDS = 300


class InferenceLatencyBenchmark(BaseTestCase):
    """推論レイテンシベンチマーク"""

    async def setup(self) -> None:
        """セットアップ"""
        self.data_manager = TestDataManager()
        self.test_data = self.data_manager.get_feature_data(
            samples=1000, features=20, target_type="regression"
        )

        # モック推論システム
        self.inference_system = await self._create_mock_system()

    async def teardown(self) -> None:
        """クリーンアップ"""
        if hasattr(self, 'inference_system'):
            await self.inference_system.shutdown()

    async def _create_mock_system(self):
        """モック推論システム作成"""
        def mock_model():
            model = Mock()
            # リアルな推論時間をシミュレート
            def predict(X):
                time.sleep(0.001)  # 1ms sleep
                return np.random.randn(X.shape[0], 1)
            model.predict = predict
            return model

        system = create_optimized_inference_system(
            model_loader=mock_model,
            model_id="benchmark_model"
        )
        await system.initialize()
        return system

    async def execute(self) -> TestResult:
        """レイテンシベンチマーク実行"""
        try:
            X_test, _ = self.test_data
            latencies = []

            # ウォームアップ
            for _ in range(PerformanceBenchmarkConfig.WARMUP_ITERATIONS):
                await self.inference_system.predict(X_test[:1])

            # ベンチマーク実行
            for _ in range(PerformanceBenchmarkConfig.BENCHMARK_ITERATIONS):
                start_time = time.perf_counter()
                await self.inference_system.predict(X_test[:1])
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)

            # 統計計算
            metrics = {
                "avg_latency_ms": np.mean(latencies),
                "median_latency_ms": np.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "std_latency_ms": np.std(latencies)
            }

            # 目標値アサーション
            assert metrics["avg_latency_ms"] < PerformanceBenchmarkConfig.TARGET_INFERENCE_LATENCY_MS, (
                f"Average latency {metrics['avg_latency_ms']:.2f}ms exceeds target "
                f"{PerformanceBenchmarkConfig.TARGET_INFERENCE_LATENCY_MS}ms"
            )

            assert metrics["p95_latency_ms"] < PerformanceBenchmarkConfig.TARGET_INFERENCE_LATENCY_MS * 2, (
                f"P95 latency {metrics['p95_latency_ms']:.2f}ms exceeds 2x target"
            )

            return TestResult(
                test_name="InferenceLatencyBenchmark",
                status="passed",
                duration_seconds=0,
                performance_metrics=metrics
            )

        except Exception as e:
            return TestResult(
                test_name="InferenceLatencyBenchmark",
                status="failed",
                duration_seconds=0,
                error_message=str(e)
            )


class ThroughputBenchmark(BaseTestCase):
    """スループットベンチマーク"""

    async def setup(self) -> None:
        """セットアップ"""
        self.data_manager = TestDataManager()
        self.test_data = self.data_manager.get_feature_data(
            samples=10000, features=20, target_type="regression"
        )

        self.inference_system = await self._create_mock_system()

    async def teardown(self) -> None:
        """クリーンアップ"""
        if hasattr(self, 'inference_system'):
            await self.inference_system.shutdown()

    async def _create_mock_system(self):
        """モック推論システム作成"""
        def mock_model():
            model = Mock()
            def predict(X):
                # バッチサイズに応じた処理時間
                time.sleep(0.0001 * X.shape[0])  # 0.1ms per sample
                return np.random.randn(X.shape[0], 1)
            model.predict = predict
            return model

        system = create_optimized_inference_system(
            model_loader=mock_model,
            model_id="throughput_model"
        )
        await system.initialize()
        return system

    async def execute(self) -> TestResult:
        """スループットベンチマーク実行"""
        try:
            X_test, _ = self.test_data
            batch_results = {}

            # 各バッチサイズでテスト
            for batch_size in PerformanceBenchmarkConfig.BATCH_SIZES:
                throughputs = []

                # 複数回測定
                for run in range(10):
                    num_batches = 50
                    total_samples = 0

                    start_time = time.perf_counter()

                    for i in range(num_batches):
                        start_idx = (i * batch_size) % len(X_test)
                        end_idx = start_idx + batch_size
                        batch_data = X_test[start_idx:end_idx]

                        await self.inference_system.predict(batch_data)
                        total_samples += batch_data.shape[0]

                    total_time = time.perf_counter() - start_time
                    throughput = total_samples / total_time
                    throughputs.append(throughput)

                batch_results[f"batch_size_{batch_size}"] = {
                    "avg_throughput": np.mean(throughputs),
                    "max_throughput": np.max(throughputs),
                    "std_throughput": np.std(throughputs)
                }

            # 最高スループット確認
            max_throughput = max(
                result["max_throughput"]
                for result in batch_results.values()
            )

            # 目標値アサーション
            # 実際の目標値は低めに設定（モックシステムのため）
            target_throughput = 1000.0  # 実システムでは10,000/sec

            assert max_throughput > target_throughput, (
                f"Max throughput {max_throughput:.1f}/sec below target {target_throughput}/sec"
            )

            return TestResult(
                test_name="ThroughputBenchmark",
                status="passed",
                duration_seconds=0,
                performance_metrics={
                    "max_throughput_per_sec": max_throughput,
                    "batch_results": batch_results
                }
            )

        except Exception as e:
            return TestResult(
                test_name="ThroughputBenchmark",
                status="failed",
                duration_seconds=0,
                error_message=str(e)
            )


class ConcurrencyBenchmark(BaseTestCase):
    """並行処理ベンチマーク"""

    async def setup(self) -> None:
        """セットアップ"""
        self.data_manager = TestDataManager()
        self.test_data = self.data_manager.get_feature_data(
            samples=1000, features=15, target_type="regression"
        )

        self.inference_system = await self._create_mock_system()

    async def teardown(self) -> None:
        """クリーンアップ"""
        if hasattr(self, 'inference_system'):
            await self.inference_system.shutdown()

    async def _create_mock_system(self):
        """モック推論システム作成"""
        def mock_model():
            model = Mock()
            def predict(X):
                # 非同期処理をシミュレート
                time.sleep(0.002)  # 2ms
                return np.random.randn(X.shape[0], 1)
            model.predict = predict
            return model

        system = create_optimized_inference_system(
            model_loader=mock_model,
            model_id="concurrency_model"
        )
        await system.initialize()
        return system

    async def execute(self) -> TestResult:
        """並行処理ベンチマーク実行"""
        try:
            X_test, _ = self.test_data
            concurrency_results = {}

            # 各並行レベルでテスト
            for concurrency in PerformanceBenchmarkConfig.CONCURRENCY_LEVELS:

                # 並行タスク作成
                tasks = []
                for i in range(concurrency):
                    task_data = X_test[i:i+1]
                    task = asyncio.create_task(
                        self.inference_system.predict(task_data)
                    )
                    tasks.append(task)

                # 実行時間測定
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks)
                concurrent_time = time.perf_counter() - start_time

                # スループット計算
                concurrent_throughput = concurrency / concurrent_time

                # 逐次実行時間推定
                sequential_time_estimate = concurrency * 0.002  # 2ms per task
                speedup_ratio = sequential_time_estimate / concurrent_time

                concurrency_results[f"concurrency_{concurrency}"] = {
                    "execution_time_sec": concurrent_time,
                    "throughput_per_sec": concurrent_throughput,
                    "speedup_ratio": speedup_ratio,
                    "successful_tasks": len([r for r in results if r is not None])
                }

            # 最高並行性能確認
            best_speedup = max(
                result["speedup_ratio"]
                for result in concurrency_results.values()
            )

            # 並行処理効率アサーション
            assert best_speedup > 2.0, (
                f"Best speedup ratio {best_speedup:.1f}x below minimum 2.0x"
            )

            return TestResult(
                test_name="ConcurrencyBenchmark",
                status="passed",
                duration_seconds=0,
                performance_metrics={
                    "best_speedup_ratio": best_speedup,
                    "concurrency_results": concurrency_results
                }
            )

        except Exception as e:
            return TestResult(
                test_name="ConcurrencyBenchmark",
                status="failed",
                duration_seconds=0,
                error_message=str(e)
            )


class MemoryEfficiencyBenchmark(BaseTestCase):
    """メモリ効率ベンチマーク"""

    async def setup(self) -> None:
        """セットアップ"""
        self.data_manager = TestDataManager()
        self.large_test_data = self.data_manager.get_feature_data(
            samples=5000, features=50, target_type="regression"
        )

        self.inference_system = await self._create_mock_system()

    async def teardown(self) -> None:
        """クリーンアップ"""
        if hasattr(self, 'inference_system'):
            await self.inference_system.shutdown()

    async def _create_mock_system(self):
        """モック推論システム作成"""
        def mock_model():
            model = Mock()
            def predict(X):
                # メモリ使用をシミュレート
                temp_data = np.random.randn(1000, 100)  # 一時的なメモリ使用
                time.sleep(0.001)
                return np.random.randn(X.shape[0], 1)
            model.predict = predict
            return model

        system = create_optimized_inference_system(
            model_loader=mock_model,
            model_id="memory_model"
        )
        await system.initialize()
        return system

    async def execute(self) -> TestResult:
        """メモリ効率ベンチマーク実行"""
        try:
            X_test, _ = self.large_test_data

            import gc
            process = psutil.Process()

            # 初期メモリ測定
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 大量推論実行
            memory_samples = []
            num_iterations = 500

            for i in range(num_iterations):
                # 推論実行
                batch_data = X_test[i:i+10]
                await self.inference_system.predict(batch_data)

                # 定期的にメモリ測定
                if i % 50 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)

            # 最終メモリ測定
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024

            # メモリ統計
            memory_increase = final_memory - initial_memory
            peak_memory = max(memory_samples) if memory_samples else final_memory
            memory_efficiency = 1.0 - (memory_increase / initial_memory)

            # メモリ効率アサーション
            assert memory_increase < PerformanceBenchmarkConfig.MAX_MEMORY_MB, (
                f"Memory increase {memory_increase:.1f}MB exceeds limit "
                f"{PerformanceBenchmarkConfig.MAX_MEMORY_MB}MB"
            )

            assert memory_efficiency > 0.7, (
                f"Memory efficiency {memory_efficiency:.2f} below 0.7"
            )

            return TestResult(
                test_name="MemoryEfficiencyBenchmark",
                status="passed",
                duration_seconds=0,
                performance_metrics={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_efficiency": memory_efficiency
                }
            )

        except Exception as e:
            return TestResult(
                test_name="MemoryEfficiencyBenchmark",
                status="failed",
                duration_seconds=0,
                error_message=str(e)
            )


class SystemResourcesBenchmark(BaseTestCase):
    """システムリソースベンチマーク"""

    async def setup(self) -> None:
        """セットアップ"""
        self.data_manager = TestDataManager()
        self.test_data = self.data_manager.get_feature_data(
            samples=2000, features=30, target_type="regression"
        )

        self.inference_system = await self._create_mock_system()
        self.resource_monitor = SystemResourceMonitor()

    async def teardown(self) -> None:
        """クリーンアップ"""
        if hasattr(self, 'inference_system'):
            await self.inference_system.shutdown()
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()

    async def _create_mock_system(self):
        """モック推論システム作成"""
        def mock_model():
            model = Mock()
            def predict(X):
                # CPU使用をシミュレート
                dummy_computation = np.sum(np.random.randn(100, 100))
                time.sleep(0.002)
                return np.random.randn(X.shape[0], 1)
            model.predict = predict
            return model

        system = create_optimized_inference_system(
            model_loader=mock_model,
            model_id="resource_model"
        )
        await system.initialize()
        return system

    async def execute(self) -> TestResult:
        """システムリソースベンチマーク実行"""
        try:
            X_test, _ = self.test_data

            # リソース監視開始
            self.resource_monitor.start()

            # 高負荷推論実行
            num_concurrent = 20
            num_iterations = 50

            for iteration in range(num_iterations):
                # 並行タスク作成
                tasks = []
                for i in range(num_concurrent):
                    batch_data = X_test[i:i+5]
                    task = asyncio.create_task(
                        self.inference_system.predict(batch_data)
                    )
                    tasks.append(task)

                # 実行
                await asyncio.gather(*tasks)

                # 短い休憩
                await asyncio.sleep(0.01)

            # 監視停止とメトリクス取得
            self.resource_monitor.stop()
            resource_metrics = self.resource_monitor.get_metrics()

            # リソース制限アサーション
            assert resource_metrics["max_cpu_percent"] < PerformanceBenchmarkConfig.MAX_CPU_PERCENT, (
                f"Peak CPU usage {resource_metrics['max_cpu_percent']:.1f}% exceeds limit "
                f"{PerformanceBenchmarkConfig.MAX_CPU_PERCENT}%"
            )

            return TestResult(
                test_name="SystemResourcesBenchmark",
                status="passed",
                duration_seconds=0,
                performance_metrics=resource_metrics
            )

        except Exception as e:
            return TestResult(
                test_name="SystemResourcesBenchmark",
                status="failed",
                duration_seconds=0,
                error_message=str(e)
            )


class SystemResourceMonitor:
    """システムリソース監視"""

    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None

    def start(self):
        """監視開始"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)

                time.sleep(0.1)
            except Exception:
                break

    def get_metrics(self) -> Dict[str, float]:
        """メトリクス取得"""
        if not self.cpu_samples or not self.memory_samples:
            return {}

        return {
            "avg_cpu_percent": np.mean(self.cpu_samples),
            "max_cpu_percent": np.max(self.cpu_samples),
            "avg_memory_percent": np.mean(self.memory_samples),
            "max_memory_percent": np.max(self.memory_samples)
        }


# pytest マーカー付きテスト関数
@pytest.mark.performance
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_inference_latency_benchmark():
    """推論レイテンシベンチマーク pytest"""
    config = TestConfig()
    test = InferenceLatencyBenchmark(config)
    result = await test.run()

    assert result.status == "passed"
    assert "avg_latency_ms" in result.performance_metrics


@pytest.mark.performance
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_throughput_benchmark():
    """スループットベンチマーク pytest"""
    config = TestConfig()
    test = ThroughputBenchmark(config)
    result = await test.run()

    assert result.status == "passed"
    assert "max_throughput_per_sec" in result.performance_metrics


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_efficiency_benchmark():
    """メモリ効率ベンチマーク pytest"""
    config = TestConfig()
    test = MemoryEfficiencyBenchmark(config)
    result = await test.run()

    assert result.status == "passed"
    assert "memory_efficiency" in result.performance_metrics


# ベンチマークスイート実行
async def run_performance_benchmarks():
    """パフォーマンスベンチマークスイート実行"""
    config = TestConfig(
        test_timeout_seconds=600,
        parallel_execution=False,  # リソース測定のため順次実行
        verbose_logging=True
    )

    framework = TestFramework(config)
    suite = framework.create_suite("PerformanceBenchmarks")

    # ベンチマークテスト追加
    suite.add_test(InferenceLatencyBenchmark(config))
    suite.add_test(ThroughputBenchmark(config))
    suite.add_test(ConcurrencyBenchmark(config))
    suite.add_test(MemoryEfficiencyBenchmark(config))
    suite.add_test(SystemResourcesBenchmark(config))

    # 実行
    results = await framework.run_all_tests()

    # ベンチマーク結果保存
    _save_benchmark_results(results)

    return results


def _save_benchmark_results(results: Dict[str, Any]):
    """ベンチマーク結果保存"""
    try:
        output_dir = Path("test_reports/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"benchmark_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Benchmark results saved: {results_file}")

    except Exception as e:
        logger.error(f"Failed to save benchmark results: {e}")


if __name__ == "__main__":
    # ベンチマーク実行
    async def main():
        results = await run_performance_benchmarks()

        print("\n=== Performance Benchmark Results ===")
        summary = results["summary"]
        print(f"Total benchmarks: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total duration: {summary['total_duration']:.2f}s")

        # 個別ベンチマーク結果
        for suite_name, suite_results in results["suite_results"].items():
            print(f"\n{suite_name}:")
            for result in suite_results:
                if hasattr(result, 'performance_metrics') and result.performance_metrics:
                    print(f"  {result.test_name}:")
                    for metric, value in result.performance_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric}: {value:.3f}")
                        else:
                            print(f"    {metric}: {value}")

        return results

    asyncio.run(main())