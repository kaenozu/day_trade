"""
統一パフォーマンス最適化エンジンの包括的テストスイート

Issue #10: テストカバレッジ向上とエラーハンドリング強化
優先度: #1 (Priority Score: 215)
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from day_trade.core.performance.optimization_engine import (
    PerformanceMetricType, OptimizationStrategy, PerformanceMetric,
    PerformanceBenchmark, OptimizationResult, PerformanceProfiler,
    ResourceMonitor, CacheOptimizer, ParallelProcessingOptimizer,
    MemoryOptimizer, OptimizationEngine, performance_monitor,
    global_optimization_engine
)


class TestPerformanceMetric:
    """PerformanceMetricのテスト"""

    def test_metric_creation(self):
        """メトリクス作成テスト"""
        metric = PerformanceMetric(
            name="test_metric",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=123.45,
            component="test_component",
            operation="test_operation"
        )

        assert metric.name == "test_metric"
        assert metric.metric_type == PerformanceMetricType.EXECUTION_TIME
        assert metric.value == 123.45
        assert metric.component == "test_component"
        assert metric.operation == "test_operation"
        assert isinstance(metric.timestamp, datetime)

    def test_metric_with_metadata(self):
        """メタデータ付きメトリクステスト"""
        metadata = {"key1": "value1", "key2": 42}
        metric = PerformanceMetric(
            name="test_metric",
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            value=100.0,
            metadata=metadata
        )

        assert metric.metadata == metadata


class TestPerformanceProfiler:
    """PerformanceProfilerのテスト"""

    @pytest.fixture
    def profiler(self):
        """プロファイラーのインスタンスを作成"""
        return PerformanceProfiler()

    def test_start_stop_profiling(self, profiler):
        """プロファイリング開始・停止テスト"""
        operation_name = "test_operation"

        # プロファイリング開始
        profiler.start_profiling(operation_name)
        assert operation_name in profiler._profiles

        # 何かの処理をシミュレート
        time.sleep(0.01)

        # プロファイリング停止
        result = profiler.stop_profiling(operation_name)

        assert result["operation"] == operation_name
        assert "total_calls" in result
        assert "total_time" in result
        assert operation_name not in profiler._profiles

    def test_stop_profiling_nonexistent(self, profiler):
        """存在しないプロファイルの停止テスト"""
        result = profiler.stop_profiling("nonexistent")
        assert result == {}

    def test_add_metric(self, profiler):
        """メトリクス追加テスト"""
        metric = PerformanceMetric(
            name="test_metric",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=100.0
        )

        profiler.add_metric(metric)
        assert len(profiler._metrics) == 1
        assert profiler._metrics[0] == metric

    def test_get_metrics_filtering(self, profiler):
        """メトリクスフィルタリングテスト"""
        # テストデータ作成
        metrics = [
            PerformanceMetric("metric1", PerformanceMetricType.EXECUTION_TIME, 100, component="comp1"),
            PerformanceMetric("metric2", PerformanceMetricType.MEMORY_USAGE, 200, component="comp2"),
            PerformanceMetric("metric3", PerformanceMetricType.EXECUTION_TIME, 300, operation="op1")
        ]

        for metric in metrics:
            profiler.add_metric(metric)

        # コンポーネントでフィルタリング
        comp1_metrics = profiler.get_metrics(component="comp1")
        assert len(comp1_metrics) == 1
        assert comp1_metrics[0].component == "comp1"

        # メトリクスタイプでフィルタリング
        exec_metrics = profiler.get_metrics(metric_type=PerformanceMetricType.EXECUTION_TIME)
        assert len(exec_metrics) == 2

        # オペレーションでフィルタリング
        op1_metrics = profiler.get_metrics(operation="op1")
        assert len(op1_metrics) == 1

    def test_set_benchmark(self, profiler):
        """ベンチマーク設定テスト"""
        benchmark = PerformanceBenchmark(
            operation_name="test_op",
            target_response_time_ms=100.0,
            target_throughput_ops=1000.0,
            max_memory_mb=512.0,
            max_cpu_percent=80.0
        )

        profiler.set_benchmark(benchmark)
        assert "test_op" in profiler._benchmarks
        assert profiler._benchmarks["test_op"] == benchmark

    def test_check_benchmarks(self, profiler):
        """ベンチマークチェックテスト"""
        # ベンチマーク設定
        benchmark = PerformanceBenchmark(
            operation_name="test_op",
            target_response_time_ms=100.0,
            target_throughput_ops=1000.0,
            max_memory_mb=512.0,
            max_cpu_percent=80.0
        )
        profiler.set_benchmark(benchmark)

        # テストメトリクス追加
        metric = PerformanceMetric(
            name="test_op_execution_time",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=80.0,  # ベンチマークより良い
            operation="test_op"
        )
        profiler.add_metric(metric)

        # ベンチマークチェック
        results = profiler.check_benchmarks()

        assert "test_op" in results
        result = results["test_op"]
        assert result["benchmark_response_time_ms"] == 100.0
        assert result["actual_response_time_ms"] == 80.0
        assert result["meets_benchmark"] is True
        assert result["variance_percent"] < 0  # 良い結果


class TestResourceMonitor:
    """ResourceMonitorのテスト"""

    @pytest.fixture
    def monitor(self):
        """リソースモニターのインスタンスを作成"""
        return ResourceMonitor(interval_seconds=0.1)

    def test_monitor_lifecycle(self, monitor):
        """モニターのライフサイクルテスト"""
        assert not monitor._monitoring

        # 監視開始
        monitor.start_monitoring()
        assert monitor._monitoring
        assert monitor._monitor_thread is not None

        # 少し待ってメトリクスが収集されることを確認
        time.sleep(0.2)
        assert len(monitor._metrics) > 0

        # 監視停止
        monitor.stop_monitoring()
        assert not monitor._monitoring

    def test_get_current_usage(self, monitor):
        """現在の使用率取得テスト"""
        usage = monitor.get_current_usage()

        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "memory_available_gb" in usage
        assert "process_memory_mb" in usage

        # 値の妥当性チェック
        assert 0 <= usage["cpu_percent"] <= 100
        assert 0 <= usage["memory_percent"] <= 100
        assert usage["memory_available_gb"] >= 0
        assert usage["process_memory_mb"] >= 0

    def test_duplicate_start_monitoring(self, monitor):
        """重複監視開始テスト"""
        monitor.start_monitoring()
        thread1 = monitor._monitor_thread

        # 重複して開始しても問題ないことを確認
        monitor.start_monitoring()
        assert monitor._monitor_thread == thread1

        monitor.stop_monitoring()


class TestCacheOptimizer:
    """CacheOptimizerのテスト"""

    @pytest.fixture
    def optimizer(self):
        """キャッシュオプティマイザーのインスタンスを作成"""
        return CacheOptimizer()

    def test_record_cache_access(self, optimizer):
        """キャッシュアクセス記録テスト"""
        cache_name = "test_cache"

        # ヒット・ミスを記録
        optimizer.record_cache_access(cache_name, True)  # ヒット
        optimizer.record_cache_access(cache_name, False)  # ミス
        optimizer.record_cache_access(cache_name, True)  # ヒット

        stats = optimizer.get_cache_stats()

        assert cache_name in stats
        cache_stats = stats[cache_name]
        assert cache_stats["hits"] == 2
        assert cache_stats["misses"] == 1
        assert cache_stats["total_requests"] == 3
        assert cache_stats["hit_rate"] == 2/3
        assert cache_stats["efficiency"] == "medium"

    def test_cache_efficiency_classification(self, optimizer):
        """キャッシュ効率分類テスト"""
        # 高効率キャッシュ
        for _ in range(9):
            optimizer.record_cache_access("high_cache", True)
        optimizer.record_cache_access("high_cache", False)

        # 低効率キャッシュ
        for _ in range(7):
            optimizer.record_cache_access("low_cache", False)
        for _ in range(3):
            optimizer.record_cache_access("low_cache", True)

        stats = optimizer.get_cache_stats()

        assert stats["high_cache"]["efficiency"] == "high"
        assert stats["low_cache"]["efficiency"] == "low"

    def test_optimize_cache_size(self, optimizer):
        """キャッシュサイズ最適化テスト"""
        cache_name = "test_cache"
        current_size = 1000

        # ヒット率が低い場合（サイズ増加）
        for _ in range(3):
            optimizer.record_cache_access(cache_name, False)
        optimizer.record_cache_access(cache_name, True)

        new_size = optimizer.optimize_cache_size(cache_name, current_size)
        assert new_size > current_size

        # 新しいオプティマイザーでヒット率が高い場合をテスト
        optimizer2 = CacheOptimizer()
        for _ in range(19):
            optimizer2.record_cache_access(cache_name, True)
        optimizer2.record_cache_access(cache_name, False)

        new_size2 = optimizer2.optimize_cache_size(cache_name, current_size)
        assert new_size2 < current_size

    def test_unknown_cache_optimization(self, optimizer):
        """未知のキャッシュ最適化テスト"""
        current_size = 1000
        new_size = optimizer.optimize_cache_size("unknown_cache", current_size)
        assert new_size == current_size


class TestParallelProcessingOptimizer:
    """ParallelProcessingOptimizerのテスト"""

    def test_context_manager(self):
        """コンテキストマネージャーテスト"""
        with ParallelProcessingOptimizer() as optimizer:
            assert optimizer is not None
        # コンテキスト終了時にcleanupが呼ばれることを確認

    def test_io_bound_optimization(self):
        """IO集約的タスク最適化テスト"""
        def dummy_io_task():
            time.sleep(0.01)
            return "completed"

        tasks = [dummy_io_task for _ in range(3)]

        with ParallelProcessingOptimizer() as optimizer:
            results = optimizer.optimize_for_io_bound(tasks)

        assert len(results) == 3
        assert all(result == "completed" for result in results)

    def test_cpu_bound_optimization(self):
        """CPU集約的タスク最適化テスト"""
        def dummy_cpu_task():
            # CPU集約的な計算をシミュレート
            return sum(i * i for i in range(1000))

        tasks = [dummy_cpu_task for _ in range(2)]

        with ParallelProcessingOptimizer() as optimizer:
            results = optimizer.optimize_for_cpu_bound(tasks)

        assert len(results) == 2
        assert all(isinstance(result, int) for result in results)

    def test_cleanup(self):
        """クリーンアップテスト"""
        optimizer = ParallelProcessingOptimizer()

        # タスクを実行してエグゼキューターを作成
        tasks = [lambda: "test"]
        optimizer.optimize_for_io_bound(tasks)

        assert optimizer._thread_pool is not None

        # クリーンアップ
        optimizer.cleanup()
        assert optimizer._thread_pool is None


class TestMemoryOptimizer:
    """MemoryOptimizerのテスト"""

    def test_optimize_garbage_collection(self):
        """ガベージコレクション最適化テスト"""
        result = MemoryOptimizer.optimize_garbage_collection()

        assert "objects_collected" in result
        assert "memory_freed_mb" in result
        assert "before_memory_mb" in result
        assert "after_memory_mb" in result

        assert isinstance(result["objects_collected"], int)
        assert isinstance(result["memory_freed_mb"], float)
        assert result["objects_collected"] >= 0

    def test_get_memory_profile(self):
        """メモリプロファイル取得テスト"""
        profile = MemoryOptimizer.get_memory_profile()

        assert "rss_mb" in profile
        assert "vms_mb" in profile
        assert "gc_stats" in profile
        assert "gc_thresholds" in profile

        gc_stats = profile["gc_stats"]
        assert "generation_0" in gc_stats
        assert "generation_1" in gc_stats
        assert "generation_2" in gc_stats

        assert isinstance(profile["rss_mb"], float)
        assert profile["rss_mb"] > 0


class TestOptimizationEngine:
    """OptimizationEngineのテスト"""

    @pytest.fixture
    def engine(self):
        """最適化エンジンのインスタンスを作成"""
        return OptimizationEngine()

    def test_engine_initialization(self, engine):
        """エンジン初期化テスト"""
        assert engine.profiler is not None
        assert engine.resource_monitor is not None
        assert engine.cache_optimizer is not None
        assert engine.memory_optimizer is not None
        assert isinstance(engine._optimization_results, list)

    def test_monitoring_lifecycle(self, engine):
        """監視ライフサイクルテスト"""
        engine.start_monitoring()
        assert engine.resource_monitor._monitoring

        engine.stop_monitoring()
        assert not engine.resource_monitor._monitoring

    def test_analyze_performance(self, engine):
        """パフォーマンス分析テスト"""
        # テストデータを追加
        metric1 = PerformanceMetric(
            name="test_execution",
            metric_type=PerformanceMetricType.EXECUTION_TIME,
            value=150.0,
            component="test_component"
        )
        metric2 = PerformanceMetric(
            name="test_memory",
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            value=512.0,
            component="test_component"
        )

        engine.profiler.add_metric(metric1)
        engine.profiler.add_metric(metric2)

        # キャッシュ統計を追加
        engine.cache_optimizer.record_cache_access("test_cache", True)
        engine.cache_optimizer.record_cache_access("test_cache", False)

        analysis = engine.analyze_performance("test_component")

        assert "component" in analysis
        assert "average_execution_time_ms" in analysis
        assert "max_memory_usage_mb" in analysis
        assert "benchmark_results" in analysis
        assert "cache_stats" in analysis
        assert "memory_profile" in analysis
        assert "current_resource_usage" in analysis

        assert analysis["component"] == "test_component"
        assert analysis["average_execution_time_ms"] == 150.0
        assert analysis["max_memory_usage_mb"] == 512.0

    def test_suggest_optimizations(self, engine):
        """最適化提案テスト"""
        # 高負荷な分析データをシミュレート
        analysis = {
            "average_execution_time_ms": 1500,  # 高い実行時間
            "max_memory_usage_mb": 2000,  # 高いメモリ使用量
            "cache_stats": {
                "bad_cache": {"hit_rate": 0.3}  # 低いヒット率
            },
            "current_resource_usage": {
                "cpu_percent": 85,  # 高いCPU使用率
                "memory_percent": 85  # 高いメモリ使用率
            }
        }

        suggestions = engine.suggest_optimizations(analysis)

        assert len(suggestions) > 0
        assert any("並列処理" in s or "キャッシュ" in s for s in suggestions)
        assert any("メモリ" in s for s in suggestions)
        assert any("CPU" in s for s in suggestions)

    def test_auto_optimize(self, engine):
        """自動最適化テスト"""
        # テストデータ準備
        engine.cache_optimizer.record_cache_access("test_cache", False)
        engine.cache_optimizer.record_cache_access("test_cache", False)
        engine.cache_optimizer.record_cache_access("test_cache", True)

        result = engine.auto_optimize("test_component")

        assert "analysis" in result
        assert "optimizations_applied" in result
        assert "suggestions" in result
        assert "gc_result" in result

        assert isinstance(result["optimizations_applied"], list)
        assert isinstance(result["suggestions"], list)


class TestPerformanceMonitorDecorator:
    """performance_monitorデコレーターのテスト"""

    @pytest.mark.asyncio
    async def test_async_function_monitoring(self):
        """非同期関数監視テスト"""
        @performance_monitor(component="test", operation="async_test")
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await async_test_function()
        assert result == "async_result"

        # メトリクスが記録されているかチェック
        metrics = global_optimization_engine.profiler.get_metrics(
            component="test",
            operation="async_test"
        )
        assert len(metrics) > 0

    def test_sync_function_monitoring(self):
        """同期関数監視テスト"""
        @performance_monitor(component="test", operation="sync_test")
        def sync_test_function():
            time.sleep(0.01)
            return "sync_result"

        result = sync_test_function()
        assert result == "sync_result"

        # メトリクスが記録されているかチェック
        metrics = global_optimization_engine.profiler.get_metrics(
            component="test",
            operation="sync_test"
        )
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_async_function_with_exception(self):
        """非同期関数例外テスト"""
        @performance_monitor(component="test", operation="async_error")
        async def async_error_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await async_error_function()

        # エラーが発生してもメトリクスが記録されるかチェック
        metrics = global_optimization_engine.profiler.get_metrics(
            component="test",
            operation="async_error"
        )
        assert len(metrics) > 0

    def test_sync_function_with_exception(self):
        """同期関数例外テスト"""
        @performance_monitor(component="test", operation="sync_error")
        def sync_error_function():
            time.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            sync_error_function()

        # エラーが発生してもメトリクスが記録されるかチェック
        metrics = global_optimization_engine.profiler.get_metrics(
            component="test",
            operation="sync_error"
        )
        assert len(metrics) > 0


class TestErrorScenarios:
    """エラーシナリオのテスト"""

    def test_profiler_with_invalid_operation(self):
        """無効なオペレーションでのプロファイラーテスト"""
        profiler = PerformanceProfiler()

        # 存在しないプロファイルを停止
        result = profiler.stop_profiling("nonexistent")
        assert result == {}

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_monitor_error_handling(self, mock_memory, mock_cpu):
        """リソースモニターエラーハンドリングテスト"""
        mock_cpu.side_effect = Exception("CPU error")
        mock_memory.side_effect = Exception("Memory error")

        monitor = ResourceMonitor(interval_seconds=0.01)

        # エラーが発生しても監視が継続することを確認
        monitor.start_monitoring()
        time.sleep(0.05)
        monitor.stop_monitoring()

        # get_current_usageでエラーが処理されることを確認
        usage = monitor.get_current_usage()
        assert usage == {}

    def test_memory_optimizer_error_handling(self):
        """メモリオプティマイザーエラーハンドリングテスト"""
        with patch('psutil.Process') as mock_process:
            mock_process.side_effect = Exception("Process error")

            # エラーが発生しても空の辞書が返されることを確認
            profile = MemoryOptimizer.get_memory_profile()
            assert profile == {}


class TestPerformanceIntegration:
    """パフォーマンス統合テスト"""

    def test_full_optimization_workflow(self):
        """完全な最適化ワークフローテスト"""
        engine = OptimizationEngine()

        # 1. 監視開始
        engine.start_monitoring()

        # 2. 模擬作業負荷
        @performance_monitor(component="integration", operation="test_workflow")
        def simulate_workload():
            time.sleep(0.02)
            return "workload_complete"

        # 複数回実行
        for _ in range(3):
            simulate_workload()

        # 3. キャッシュアクセスをシミュレート
        for _ in range(10):
            engine.cache_optimizer.record_cache_access("integration_cache", True)
        for _ in range(5):
            engine.cache_optimizer.record_cache_access("integration_cache", False)

        # 4. 分析実行
        analysis = engine.analyze_performance("integration")

        # 5. 自動最適化実行
        optimization_result = engine.auto_optimize("integration")

        # 6. 監視停止
        engine.stop_monitoring()

        # 結果検証
        assert analysis["component"] == "integration"
        assert analysis["average_execution_time_ms"] > 0
        assert "integration_cache" in analysis["cache_stats"]

        assert "analysis" in optimization_result
        assert "optimizations_applied" in optimization_result
        assert "suggestions" in optimization_result

    def test_concurrent_optimization(self):
        """並行最適化テスト"""
        engine = OptimizationEngine()

        def worker_task(worker_id):
            @performance_monitor(component="concurrent", operation=f"worker_{worker_id}")
            def worker_function():
                time.sleep(0.01)
                return f"worker_{worker_id}_complete"

            return worker_function()

        # 並行実行
        with ParallelProcessingOptimizer() as optimizer:
            tasks = [lambda i=i: worker_task(i) for i in range(5)]
            results = optimizer.optimize_for_io_bound(tasks)

        assert len(results) == 5
        assert all("complete" in result for result in results)

        # メトリクス確認
        metrics = engine.profiler.get_metrics(component="concurrent")
        assert len(metrics) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])