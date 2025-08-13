#!/usr/bin/env python3
"""
分散コンピューティング統合テスト
Issue #384: 並列処理のさらなる強化 - 統合テスト

DaskとRayの統合動作、パフォーマンス検証、フォールバック機能テスト
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

# プロジェクトモジュール
try:
    from ...distributed.dask_data_processor import (
        DaskBatchProcessor,
        DaskDataProcessor,
        DaskStockAnalyzer,
        create_dask_data_processor,
    )
    from ...distributed.distributed_computing_manager import (
        ComputingBackend,
        DistributedComputingManager,
        DistributedResult,
        DistributedTask,
        TaskDistributionStrategy,
        create_distributed_computing_manager,
    )
    from ...utils.logging_config import get_context_logger
    from ...utils.parallel_executor_manager import TaskType
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # モッククラス（テスト時フォールバック）
    pytest.skip("分散処理モジュールがインポートできません", allow_module_level=True)

logger = get_context_logger(__name__)


class TestDistributedComputingIntegration:
    """分散コンピューティング統合テストクラス"""

    @pytest.fixture
    async def distributed_manager(self):
        """分散コンピューティングマネージャーフィクスチャ"""
        manager = create_distributed_computing_manager(
            preferred_backend=ComputingBackend.AUTO,
            fallback_strategy=True,
            enable_performance_tracking=True,
        )

        # 初期化
        await manager.initialize(
            {
                "dask": {
                    "enable_distributed": True,
                    "n_workers": 2,
                    "memory_limit": "1GB",
                },
                "ray": {"num_cpus": 2, "log_to_driver": False},
            }
        )

        yield manager

        # クリーンアップ
        await manager.cleanup()

    @pytest.fixture
    async def dask_processor(self):
        """Daskデータプロセッサーフィクスチャ"""
        processor = create_dask_data_processor(
            enable_distributed=True, n_workers=2, memory_limit="1GB"
        )

        yield processor

        # クリーンアップ
        processor.cleanup()

    @pytest.mark.asyncio
    async def test_backend_initialization(self, distributed_manager):
        """バックエンド初期化テスト"""
        health_status = distributed_manager.get_health_status()

        assert health_status["status"] in ["healthy", "degraded"]
        assert health_status["available_backends_count"] >= 1
        assert isinstance(health_status["distributed_processing_enabled"], bool)

        # 統計確認
        stats = distributed_manager.get_comprehensive_stats()
        assert "manager_stats" in stats
        assert "available_backends" in stats
        assert len(stats["available_backends"]) >= 1

    @pytest.mark.asyncio
    async def test_single_task_execution(self, distributed_manager):
        """単一タスク実行テスト"""

        def simple_computation(n: int) -> int:
            return sum(range(n))

        task = DistributedTask(
            task_id="test_single_task",
            task_function=simple_computation,
            args=(100,),
            task_type=TaskType.CPU_BOUND,
        )

        result = await distributed_manager.execute_distributed_task(task)

        assert result.success is True
        assert result.result == sum(range(100))
        assert result.task_id == "test_single_task"
        assert result.execution_time_seconds > 0
        assert result.backend_used in ComputingBackend

    @pytest.mark.asyncio
    async def test_batch_task_execution(self, distributed_manager):
        """バッチタスク実行テスト"""

        def multiply_task(x: int, y: int) -> int:
            return x * y

        # バッチタスク生成
        tasks = []
        for i in range(5):
            task = DistributedTask(
                task_id=f"batch_task_{i}",
                task_function=multiply_task,
                args=(i, 2),
                task_type=TaskType.CPU_BOUND,
            )
            tasks.append(task)

        results = await distributed_manager.execute_distributed_batch(
            tasks, strategy=TaskDistributionStrategy.DYNAMIC
        )

        assert len(results) == 5

        for i, result in enumerate(results):
            assert result.success is True
            assert result.result == i * 2
            assert result.task_id == f"batch_task_{i}"

    @pytest.mark.asyncio
    async def test_task_distribution_strategies(self, distributed_manager):
        """タスク分散戦略テスト"""

        def simple_task(value: int) -> int:
            return value**2

        tasks = [
            DistributedTask(
                task_id=f"strategy_task_{i}",
                task_function=simple_task,
                args=(i,),
                task_type=TaskType.CPU_BOUND if i % 2 == 0 else TaskType.IO_BOUND,
            )
            for i in range(6)
        ]

        # 異なる戦略でテスト
        strategies = [
            TaskDistributionStrategy.DYNAMIC,
            TaskDistributionStrategy.ROUND_ROBIN,
            TaskDistributionStrategy.AFFINITY,
        ]

        for strategy in strategies:
            results = await distributed_manager.execute_distributed_batch(
                tasks, strategy=strategy
            )

            assert len(results) == 6
            success_count = sum(1 for r in results if r.success)
            assert success_count >= 4  # 少なくとも4個は成功すること

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, distributed_manager):
        """エラーハンドリングとフォールバックテスト"""

        def error_task(should_error: bool) -> str:
            if should_error:
                raise ValueError("Intentional test error")
            return "success"

        # エラータスクと正常タスクの混在
        tasks = [
            DistributedTask(
                task_id=f"error_test_{i}",
                task_function=error_task,
                args=(i % 2 == 0,),  # 偶数番目でエラー
                task_type=TaskType.CPU_BOUND,
            )
            for i in range(4)
        ]

        results = await distributed_manager.execute_distributed_batch(tasks)

        assert len(results) == 4

        # エラー結果確認
        error_results = [r for r in results if not r.success]
        success_results = [r for r in results if r.success]

        assert len(error_results) == 2  # 偶数番目（0, 2）
        assert len(success_results) == 2  # 奇数番目（1, 3）

        for error_result in error_results:
            assert error_result.error is not None
            assert "Intentional test error" in str(error_result.error)

    @pytest.mark.asyncio
    async def test_dask_data_processing(self, dask_processor):
        """Daskデータ処理テスト"""

        # テストデータ
        test_symbols = ["TEST_A", "TEST_B", "TEST_C"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # 並列データ処理
        result_df = await dask_processor.process_multiple_symbols_parallel(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            include_technical=False,
            store_intermediate=False,
        )

        # 結果検証（モックデータでも構造確認）
        assert isinstance(result_df, type(pd.DataFrame()))

        # 統計確認
        stats = dask_processor.get_stats()
        assert "processed_symbols" in stats
        assert "total_processing_time_ms" in stats

    @pytest.mark.asyncio
    async def test_dask_correlation_analysis(self, dask_processor):
        """Dask相関分析テスト"""

        test_symbols = ["TEST_X", "TEST_Y", "TEST_Z"]

        analyzer = DaskStockAnalyzer(dask_processor)

        # 相関分析実行
        correlation_df = dask_processor.analyze_market_correlation_distributed(
            symbols=test_symbols, analysis_period_days=30, correlation_window=10
        )

        # 結果検証
        assert isinstance(correlation_df, type(pd.DataFrame()))

        # 相関マトリックスの基本プロパティ
        if not correlation_df.empty:
            assert correlation_df.shape[0] == correlation_df.shape[1]  # 正方行列
            # 対角成分が1に近い（自己相関）
            for i in range(min(3, len(correlation_df))):
                if i < len(correlation_df) and i < len(correlation_df.columns):
                    diagonal_value = (
                        correlation_df.iloc[i, i]
                        if not pd.isna(correlation_df.iloc[i, i])
                        else 0
                    )
                    assert abs(diagonal_value - 1.0) < 0.1 or diagonal_value == 0

    @pytest.mark.asyncio
    async def test_batch_pipeline_processing(self, dask_processor):
        """バッチパイプライン処理テスト"""

        batch_processor = DaskBatchProcessor(dask_processor)

        test_symbols = ["PIPE_A", "PIPE_B"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        pipeline_steps = ["fetch_data", "technical_analysis", "data_cleaning"]

        # パイプライン実行
        result = await batch_processor.process_market_data_pipeline(
            symbols=test_symbols,
            pipeline_steps=pipeline_steps,
            start_date=start_date,
            end_date=end_date,
            output_format="parquet",
        )

        # 結果検証
        assert "pipeline_timestamp" in result
        assert "symbols_processed" in result
        assert "steps_executed" in result
        assert result["steps_executed"] == pipeline_steps
        assert isinstance(result["processing_successful"], bool)

    @pytest.mark.asyncio
    async def test_performance_comparison(self, distributed_manager, dask_processor):
        """パフォーマンス比較テスト"""

        def cpu_intensive_computation(n: int) -> float:
            """CPU集約的計算"""
            result = 0.0
            for i in range(n):
                result += (i**0.5) * (i**0.5)
            return result

        # テストケース
        computation_sizes = [1000, 5000, 10000]

        performance_results = {}

        for size in computation_sizes:
            tasks = [
                DistributedTask(
                    task_id=f"perf_test_{size}_{i}",
                    task_function=cpu_intensive_computation,
                    args=(size,),
                    task_type=TaskType.CPU_BOUND,
                )
                for i in range(3)
            ]

            # 実行時間測定
            start_time = time.time()
            results = await distributed_manager.execute_distributed_batch(tasks)
            execution_time = time.time() - start_time

            # 結果記録
            success_count = sum(1 for r in results if r.success)
            performance_results[size] = {
                "execution_time": execution_time,
                "success_count": success_count,
                "total_tasks": len(tasks),
            }

        # パフォーマンス検証
        for size, metrics in performance_results.items():
            assert (
                metrics["success_count"] >= metrics["total_tasks"] * 0.8
            )  # 80%以上成功
            assert metrics["execution_time"] > 0
            logger.info(
                f"Size {size}: {metrics['execution_time']:.3f}s, "
                f"Success: {metrics['success_count']}/{metrics['total_tasks']}"
            )

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, distributed_manager):
        """リソースクリーンアップテスト"""

        # テンポラリディレクトリ状態確認
        temp_dir = distributed_manager.temp_dir
        initial_exists = temp_dir.exists()

        # 簡単なタスク実行
        task = DistributedTask(
            task_id="cleanup_test",
            task_function=lambda x: x * 2,
            args=(42,),
            task_type=TaskType.CPU_BOUND,
        )

        result = await distributed_manager.execute_distributed_task(task)
        assert result.success is True

        # 統計確認
        stats_before_cleanup = distributed_manager.get_comprehensive_stats()
        assert stats_before_cleanup["manager_stats"]["tasks_executed"] > 0

        # クリーンアップ実行（フィクスチャで自動実行されるが、明示的テスト）
        await distributed_manager.cleanup()

        # クリーンアップ後の状態確認
        health_after_cleanup = distributed_manager.get_health_status()

        # 基本的なクリーンアップ確認（完全な検証は実装依存）
        assert isinstance(health_after_cleanup, dict)

    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, distributed_manager):
        """並行タスク実行テスト"""

        def concurrent_task(task_id: str, duration: float) -> Dict[str, Any]:
            import time

            start = time.time()
            time.sleep(duration)
            end = time.time()
            return {
                "task_id": task_id,
                "start_time": start,
                "end_time": end,
                "actual_duration": end - start,
            }

        # 異なる実行時間のタスクを準備
        tasks = [
            DistributedTask(
                task_id=f"concurrent_{i}",
                task_function=concurrent_task,
                args=(f"concurrent_{i}", 0.1 * i),
                task_type=TaskType.IO_BOUND,
            )
            for i in range(1, 4)  # 0.1s, 0.2s, 0.3s
        ]

        # バッチ実行
        start_time = time.time()
        results = await distributed_manager.execute_distributed_batch(tasks)
        total_time = time.time() - start_time

        # 並行実行効果の確認
        assert len(results) == 3
        success_results = [r for r in results if r.success]
        assert len(success_results) >= 2

        # 並行実行により、個別実行時間の合計より短いことを確認
        expected_sequential_time = 0.1 + 0.2 + 0.3  # 0.6秒
        assert total_time < expected_sequential_time * 1.5  # 並行実行効果を考慮した閾値


# パフォーマンスベンチマーク
class TestDistributedComputingPerformance:
    """分散コンピューティングパフォーマンステスト"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_scaling_performance(self):
        """スケーリングパフォーマンステスト"""

        # 異なるワーカー数での性能テスト
        worker_configs = [1, 2, 4]
        task_counts = [10, 20, 50]

        results = {}

        for workers in worker_configs:
            manager = create_distributed_computing_manager()

            try:
                await manager.initialize(
                    {"dask": {"n_workers": workers, "enable_distributed": True}}
                )

                for task_count in task_counts:
                    # CPU集約的タスクセット
                    tasks = [
                        DistributedTask(
                            task_id=f"scale_test_{workers}w_{task_count}t_{i}",
                            task_function=lambda n: sum(i**2 for i in range(n)),
                            args=(1000,),
                            task_type=TaskType.CPU_BOUND,
                        )
                        for i in range(task_count)
                    ]

                    # 実行時間測定
                    start_time = time.time()
                    batch_results = await manager.execute_distributed_batch(tasks)
                    execution_time = time.time() - start_time

                    # 結果記録
                    success_count = sum(1 for r in batch_results if r.success)

                    results[f"{workers}w_{task_count}t"] = {
                        "workers": workers,
                        "tasks": task_count,
                        "execution_time": execution_time,
                        "success_rate": success_count / task_count,
                        "throughput": task_count / execution_time,
                    }

                    logger.info(
                        f"Workers: {workers}, Tasks: {task_count}, "
                        f"Time: {execution_time:.3f}s, "
                        f"Throughput: {task_count / execution_time:.2f} tasks/s"
                    )

            finally:
                await manager.cleanup()

        # スケーリング効果の基本検証
        assert len(results) > 0

        # 結果出力
        for config, metrics in results.items():
            print(f"{config}: {metrics}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_efficiency(self):
        """メモリ効率テスト"""

        import gc

        import psutil

        # メモリ使用量ベースライン
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        manager = create_distributed_computing_manager()

        try:
            await manager.initialize(
                {"dask": {"n_workers": 2, "memory_limit": "500MB"}}
            )

            # メモリ集約的タスク
            def memory_intensive_task(size: int) -> List[int]:
                return list(range(size))

            tasks = [
                DistributedTask(
                    task_id=f"memory_test_{i}",
                    task_function=memory_intensive_task,
                    args=(10000,),
                    task_type=TaskType.CPU_BOUND,
                )
                for i in range(10)
            ]

            # 実行前メモリ
            before_execution = psutil.Process().memory_info().rss / 1024 / 1024

            # タスク実行
            results = await manager.execute_distributed_batch(tasks)

            # 実行後メモリ
            after_execution = psutil.Process().memory_info().rss / 1024 / 1024

            # 結果検証
            success_count = sum(1 for r in results if r.success)
            assert success_count >= len(tasks) * 0.8

            # メモリ使用量ログ
            memory_increase = after_execution - initial_memory
            logger.info(
                f"Initial: {initial_memory:.1f}MB, "
                f"Before: {before_execution:.1f}MB, "
                f"After: {after_execution:.1f}MB, "
                f"Increase: {memory_increase:.1f}MB"
            )

            # メモリリークの簡易チェック（大幅な増加がないこと）
            assert memory_increase < 200  # 200MB未満の増加

        finally:
            await manager.cleanup()

            # クリーンアップ後メモリ
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Final memory: {final_memory:.1f}MB")


if __name__ == "__main__":
    # 統合テスト直接実行
    async def run_integration_tests():
        print("=== Issue #384 分散コンピューティング統合テスト ===")

        # 基本的な統合テスト実行
        manager = create_distributed_computing_manager()

        try:
            # 初期化
            init_result = await manager.initialize(
                {
                    "dask": {"enable_distributed": True, "n_workers": 2},
                    "ray": {"num_cpus": 2},
                }
            )
            print(f"初期化結果: {init_result}")

            # 簡単なテスト実行
            task = DistributedTask(
                task_id="integration_test",
                task_function=lambda x: x**2,
                args=(10,),
                task_type=TaskType.CPU_BOUND,
            )

            result = await manager.execute_distributed_task(task)
            print(
                f"テスト結果: success={result.success}, result={result.result}, backend={result.backend_used.value}"
            )

            # 統計情報
            stats = manager.get_comprehensive_stats()
            print(f"統計: {stats['manager_stats']}")

        except Exception as e:
            print(f"統合テストエラー: {e}")

        finally:
            await manager.cleanup()

    asyncio.run(run_integration_tests())
    print("=== 分散コンピューティング統合テスト完了 ===")


# pytest設定用
import pandas as pd
