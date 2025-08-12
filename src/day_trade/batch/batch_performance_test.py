#!/usr/bin/env python3
"""
バッチ処理システム包括的パフォーマンステスト
Issue #376 で実装したすべてのバッチ最適化システムの性能検証

テスト対象:
- APIリクエスト統合システム
- 統合データフェッチャー
- データベースバルクオペレーション最適化
- 並列バッチ処理エンジン
- 統合バッチデータフロー最適化
"""

import asyncio
import gc
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import psutil

from ..utils.logging_config import get_context_logger
from .api_request_consolidator import RequestPriority as APIRequestPriority

# テスト対象システムのインポート
from .api_request_consolidator import create_consolidator
from .database_bulk_optimizer import OptimizationStrategy, create_bulk_optimizer
from .integrated_data_fetcher import RequestPriority as DataRequestPriority
from .integrated_data_fetcher import create_integrated_fetcher
from .parallel_batch_engine import ProcessingMode, TaskPriority, create_batch_engine
from .unified_batch_dataflow import (
    DataFlowStage,
    FlowPriority,
    OptimizationTarget,
    create_standard_flow_stages,
    create_unified_dataflow,
)

logger = get_context_logger(__name__)


class PerformanceTestResult:
    """パフォーマンステスト結果"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.success = False
        self.metrics = {}
        self.errors = []
        self.system_info = self._get_system_info()

    def complete(self, success: bool = True):
        """テスト完了"""
        self.end_time = time.time()
        self.success = success
        self.metrics["total_time_seconds"] = self.end_time - self.start_time

    def add_metric(self, key: str, value: Any):
        """メトリクス追加"""
        self.metrics[key] = value

    def add_error(self, error: str):
        """エラー追加"""
        self.errors.append(error)
        logger.error(f"{self.test_name}: {error}")

    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": os.name,
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"error": str(e)}

    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "errors": self.errors,
            "system_info": self.system_info,
        }


class BatchPerformanceTester:
    """バッチ処理システム包括的パフォーマンステスター"""

    def __init__(self, test_config: Dict[str, Any] = None):
        self.test_config = test_config or self._get_default_config()
        self.test_results = []
        self.temp_files = []  # クリーンアップ用

        logger.info("バッチ処理システム包括的パフォーマンステスト開始")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルトテスト設定"""
        return {
            "test_symbols": [
                "7203",
                "8306",
                "9984",
                "6758",
                "4689",
                "2914",
                "6861",
                "8035",
            ],
            "batch_sizes": [10, 50, 100, 200],
            "worker_counts": [2, 4, 8],
            "data_sizes": [100, 1000, 5000, 10000],
            "concurrent_requests": [1, 5, 10, 20],
            "timeout_seconds": 300,
            "enable_memory_monitoring": True,
            "enable_detailed_metrics": True,
        }

    def run_all_tests(self) -> List[PerformanceTestResult]:
        """全テスト実行"""
        test_methods = [
            self._test_api_request_consolidator,
            self._test_integrated_data_fetcher,
            self._test_database_bulk_optimizer,
            self._test_parallel_batch_engine,
            self._test_unified_dataflow,
            self._test_system_integration,
            self._test_scalability,
            self._test_stress_test,
        ]

        for test_method in test_methods:
            try:
                logger.info(f"テスト実行開始: {test_method.__name__}")
                result = test_method()
                self.test_results.append(result)

                if result.success:
                    logger.info(
                        f"テスト成功: {test_method.__name__} - {result.metrics.get('total_time_seconds', 0):.2f}秒"
                    )
                else:
                    logger.warning(
                        f"テスト失敗: {test_method.__name__} - {result.errors}"
                    )

            except Exception as e:
                error_result = PerformanceTestResult(test_method.__name__)
                error_result.add_error(str(e))
                error_result.complete(False)
                self.test_results.append(error_result)
                logger.error(f"テスト例外: {test_method.__name__} - {e}")

        return self.test_results

    def _test_api_request_consolidator(self) -> PerformanceTestResult:
        """APIリクエスト統合システムテスト"""
        result = PerformanceTestResult("api_request_consolidator")

        try:
            # 複数バッチサイズでテスト
            for batch_size in self.test_config["batch_sizes"]:
                consolidator = create_consolidator(
                    batch_size=batch_size, max_workers=4, enable_caching=True
                )
                consolidator.start()

                try:
                    # パフォーマンステスト
                    symbols = self.test_config["test_symbols"]
                    request_count = 0
                    successful_requests = 0
                    total_response_time = 0.0

                    # 複数リクエスト並行実行
                    start_time = time.time()

                    def response_callback(response):
                        nonlocal successful_requests, total_response_time
                        if response.success:
                            successful_requests += 1
                        total_response_time += response.response_time

                    # リクエスト投入
                    request_ids = []
                    for i in range(10):  # 10回のリクエスト
                        request_id = consolidator.submit_request(
                            endpoint="stock_data",
                            symbols=symbols,
                            priority=(
                                APIRequestPriority.HIGH
                                if i % 2 == 0
                                else APIRequestPriority.NORMAL
                            ),
                            callback=response_callback,
                        )
                        request_ids.append(request_id)
                        request_count += 1

                    # 完了待機
                    time.sleep(3.0)

                    test_time = time.time() - start_time

                    # 統計取得
                    stats = consolidator.get_stats()
                    status = consolidator.get_status()

                    # メトリクス記録
                    result.add_metric(
                        f"batch_size_{batch_size}_success_rate",
                        successful_requests / request_count,
                    )
                    result.add_metric(
                        f"batch_size_{batch_size}_avg_response_time",
                        total_response_time / max(request_count, 1),
                    )
                    result.add_metric(
                        f"batch_size_{batch_size}_consolidation_ratio",
                        stats.consolidation_ratio,
                    )
                    result.add_metric(
                        f"batch_size_{batch_size}_throughput", stats.throughput_rps
                    )
                    result.add_metric(f"batch_size_{batch_size}_test_time", test_time)

                finally:
                    consolidator.stop()

            # 全体メトリクス計算
            success_rates = [
                v for k, v in result.metrics.items() if "success_rate" in k
            ]
            result.add_metric(
                "overall_avg_success_rate", sum(success_rates) / len(success_rates)
            )

            result.complete(True)

        except Exception as e:
            result.add_error(f"APIリクエスト統合テストエラー: {e}")
            result.complete(False)

        return result

    def _test_integrated_data_fetcher(self) -> PerformanceTestResult:
        """統合データフェッチャーテスト"""
        result = PerformanceTestResult("integrated_data_fetcher")

        try:
            fetcher = create_integrated_fetcher(
                batch_size=50, max_workers=4, enable_caching=True
            )
            fetcher.start()

            try:
                symbols = self.test_config["test_symbols"]

                # 複数の優先度でテスト
                priorities = [
                    DataRequestPriority.LOW,
                    DataRequestPriority.NORMAL,
                    DataRequestPriority.HIGH,
                ]

                for priority in priorities:
                    start_time = time.time()

                    response = fetcher.fetch_data(
                        symbols=symbols,
                        period="5d",
                        priority=priority,
                        enable_fallback=True,
                        timeout=30.0,
                    )

                    fetch_time = time.time() - start_time

                    # メトリクス記録
                    priority_name = priority.name.lower()
                    result.add_metric(
                        f"{priority_name}_success_count", response.success_count
                    )
                    result.add_metric(f"{priority_name}_total_symbols", len(symbols))
                    result.add_metric(
                        f"{priority_name}_success_rate",
                        response.success_count / len(symbols),
                    )
                    result.add_metric(
                        f"{priority_name}_processing_time",
                        response.total_processing_time,
                    )
                    result.add_metric(
                        f"{priority_name}_cache_hit_rate", response.cache_hit_rate
                    )
                    result.add_metric(f"{priority_name}_fetch_time", fetch_time)

                # パフォーマンス統計
                stats = fetcher.get_performance_stats()
                result.add_metric("total_requests", stats["total_requests"])
                result.add_metric("overall_success_rate", stats["success_rate"])
                result.add_metric("overall_cache_hit_rate", stats["cache_hit_rate"])
                result.add_metric("fallback_rate", stats["fallback_rate"])
                result.add_metric(
                    "average_response_time", stats["average_response_time"]
                )

                result.complete(True)

            finally:
                fetcher.stop()

        except Exception as e:
            result.add_error(f"統合データフェッチャーテストエラー: {e}")
            result.complete(False)

        return result

    def _test_database_bulk_optimizer(self) -> PerformanceTestResult:
        """データベースバルクオペレーション最適化テスト"""
        result = PerformanceTestResult("database_bulk_optimizer")

        # テストDBファイル作成
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            test_db_path = tmp_file.name

        self.temp_files.append(test_db_path)

        try:
            optimizer = create_bulk_optimizer(
                db_path=test_db_path,
                batch_size=1000,
                max_workers=4,
                optimization_strategy=OptimizationStrategy.SPEED_OPTIMIZED,
            )

            # テストテーブル作成
            with optimizer.connection_pool.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS test_data (
                        id INTEGER PRIMARY KEY,
                        symbol TEXT,
                        price REAL,
                        volume INTEGER,
                        date TEXT
                    )
                """
                )
                conn.commit()

            # 複数データサイズでテスト
            for data_size in self.test_config["data_sizes"]:
                # テストデータ生成
                test_data = []
                symbols = self.test_config["test_symbols"]

                for i in range(data_size):
                    symbol = symbols[i % len(symbols)]
                    test_data.append(
                        {
                            "id": i + 1,
                            "symbol": symbol,
                            "price": 100.0 + (i % 1000),
                            "volume": 1000000 + (i % 500000),
                            "date": f"2024-01-{(i % 30) + 1:02d}",
                        }
                    )

                # バルク挿入テスト
                insert_result = optimizer.bulk_insert(
                    table_name="test_data",
                    data=test_data,
                    conflict_resolution="REPLACE",
                )

                # バルクUPSERTテスト（データの一部を更新）
                update_data = test_data[: data_size // 2].copy()
                for record in update_data:
                    record["price"] *= 1.1

                upsert_result = optimizer.bulk_upsert(
                    table_name="test_data", data=update_data, key_columns=["id"]
                )

                # メトリクス記録
                size_key = f"data_size_{data_size}"
                result.add_metric(
                    f"{size_key}_insert_throughput", insert_result.throughput_rps
                )
                result.add_metric(
                    f"{size_key}_insert_success_rate", insert_result.success_rate
                )
                result.add_metric(
                    f"{size_key}_upsert_throughput", upsert_result.throughput_rps
                )
                result.add_metric(
                    f"{size_key}_upsert_success_rate", upsert_result.success_rate
                )

            # 全体パフォーマンス統計
            stats = optimizer.get_performance_stats()
            result.add_metric("total_operations", stats["total_operations"])
            result.add_metric("overall_success_rate", stats["success_rate"])
            result.add_metric("average_throughput", stats["average_throughput_rps"])
            result.add_metric("database_size_mb", stats["database_size_mb"])

            optimizer.close()
            result.complete(True)

        except Exception as e:
            result.add_error(f"データベースバルク最適化テストエラー: {e}")
            result.complete(False)

        return result

    def _test_parallel_batch_engine(self) -> PerformanceTestResult:
        """並列バッチ処理エンジンテスト"""
        result = PerformanceTestResult("parallel_batch_engine")

        try:
            # テスト用関数
            def test_task(n: int, delay: float = 0.01) -> int:
                time.sleep(delay)
                return sum(i for i in range(n))

            def failing_task(should_fail: bool = True) -> str:
                if should_fail:
                    raise ValueError("Intentional test failure")
                return "Success"

            # 複数ワーカー数でテスト
            for worker_count in self.test_config["worker_counts"]:
                engine = create_batch_engine(
                    workers=worker_count,
                    processing_mode=ProcessingMode.THREAD_BASED,
                    enable_adaptive_scaling=True,
                )
                engine.start()

                try:
                    task_ids = []

                    # 成功タスク投入
                    for i in range(20):
                        task_id = engine.submit_task(
                            test_task,
                            100 + i * 10,
                            delay=0.005,
                            priority=(
                                TaskPriority.HIGH if i % 3 == 0 else TaskPriority.NORMAL
                            ),
                            task_id=f"success_task_{i}",
                        )
                        task_ids.append(task_id)

                    # 失敗タスク投入（リトライテスト）
                    for i in range(5):
                        task_id = engine.submit_task(
                            failing_task,
                            should_fail=(i < 3),  # 最初の3つは失敗
                            retry_count=2,
                            task_id=f"fail_task_{i}",
                        )
                        task_ids.append(task_id)

                    # 結果待機
                    time.sleep(2.0)

                    # バッチ結果取得
                    results = engine.get_results_batch(task_ids, timeout=10.0)

                    # 統計計算
                    successful_results = [r for r in results.values() if r.success]
                    failed_results = [r for r in results.values() if not r.success]

                    # エンジン統計
                    stats = engine.get_stats()

                    # メトリクス記録
                    worker_key = f"workers_{worker_count}"
                    result.add_metric(f"{worker_key}_completed_tasks", len(results))
                    result.add_metric(
                        f"{worker_key}_successful_tasks", len(successful_results)
                    )
                    result.add_metric(f"{worker_key}_failed_tasks", len(failed_results))
                    result.add_metric(f"{worker_key}_success_rate", stats.success_rate)
                    result.add_metric(
                        f"{worker_key}_throughput_tps", stats.throughput_tps
                    )
                    result.add_metric(
                        f"{worker_key}_avg_execution_time", stats.average_execution_time
                    )

                finally:
                    engine.stop(timeout=10.0)

            result.complete(True)

        except Exception as e:
            result.add_error(f"並列バッチエンジンテストエラー: {e}")
            result.complete(False)

        return result

    def _test_unified_dataflow(self) -> PerformanceTestResult:
        """統合バッチデータフロー最適化テスト"""
        result = PerformanceTestResult("unified_dataflow")

        # テストDBファイル作成
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            test_db_path = tmp_file.name

        self.temp_files.append(test_db_path)

        try:
            dataflow = create_unified_dataflow(
                db_path=test_db_path, enable_all=True, monitoring_interval=2.0
            )
            dataflow.start()

            try:
                symbols = self.test_config["test_symbols"][:5]  # テスト用に少なめ

                # 複数最適化目標でテスト
                optimization_targets = [
                    OptimizationTarget.LATENCY,
                    OptimizationTarget.THROUGHPUT,
                    OptimizationTarget.BALANCED,
                ]

                for target in optimization_targets:
                    # テストフロー段階作成
                    stages = [
                        DataFlowStage(
                            stage_id="data_fetch",
                            name="データ取得",
                            handler="data_fetcher",
                            config={"period": "5d", "interval": "1d"},
                        )
                        # DB段階はテストでは省略（複雑すぎるため）
                    ]

                    # フロー実行
                    flow_result = dataflow.execute_flow(
                        symbols=symbols,
                        stages=stages,
                        flow_id=f"test_flow_{target.value}",
                        priority=FlowPriority.HIGH,
                        optimization_target=target,
                    )

                    # メトリクス記録
                    target_key = target.value
                    result.add_metric(f"{target_key}_flow_success", flow_result.success)
                    result.add_metric(
                        f"{target_key}_execution_time", flow_result.total_execution_time
                    )
                    result.add_metric(
                        f"{target_key}_records_processed",
                        flow_result.total_records_processed,
                    )
                    result.add_metric(
                        f"{target_key}_cache_hit_rate",
                        flow_result.overall_cache_hit_rate,
                    )
                    result.add_metric(
                        f"{target_key}_stage_count", len(flow_result.stage_results)
                    )

                # システム全体の性能メトリクス
                metrics = dataflow.get_performance_metrics()
                result.add_metric("total_flows", metrics.total_flows)
                result.add_metric("overall_success_rate", metrics.success_rate)
                result.add_metric("avg_execution_time", metrics.average_execution_time)
                result.add_metric("avg_throughput", metrics.average_throughput)
                result.add_metric("avg_cache_hit_rate", metrics.average_cache_hit_rate)

                # システムステータス
                status = dataflow.get_system_status()
                result.add_metric("systems_count", len(status["systems"]))
                result.add_metric("completed_flows", status["completed_flows"])

                result.complete(True)

            finally:
                dataflow.stop(timeout=15.0)

        except Exception as e:
            result.add_error(f"統合データフローテストエラー: {e}")
            result.complete(False)

        return result

    def _test_system_integration(self) -> PerformanceTestResult:
        """システム統合テスト"""
        result = PerformanceTestResult("system_integration")

        try:
            # 複数システムの同時実行テスト
            systems_running = []

            # APIリクエスト統合システム
            consolidator = create_consolidator(batch_size=30, max_workers=2)
            consolidator.start()
            systems_running.append(("consolidator", consolidator))

            # 統合データフェッチャー
            fetcher = create_integrated_fetcher(batch_size=30, max_workers=2)
            fetcher.start()
            systems_running.append(("fetcher", fetcher))

            # 並列バッチエンジン
            engine = create_batch_engine(workers=2, enable_adaptive_scaling=False)
            engine.start()
            systems_running.append(("engine", engine))

            try:
                symbols = self.test_config["test_symbols"][:4]  # システム負荷軽減

                # 並行実行テスト
                start_time = time.time()

                # APIコンソリデーター
                consolidator_responses = {}

                def consolidator_callback(response):
                    consolidator_responses[response.request_id] = response

                consolidator.submit_request(
                    endpoint="test_endpoint",
                    symbols=symbols,
                    callback=consolidator_callback,
                )

                # データフェッチャー
                fetcher_response = fetcher.fetch_data(
                    symbols=symbols, period="5d", timeout=10.0
                )

                # バッチエンジン
                def simple_task(symbol: str) -> str:
                    time.sleep(0.01)
                    return f"processed_{symbol}"

                engine_task_ids = []
                for symbol in symbols:
                    task_id = engine.submit_task(simple_task, symbol=symbol)
                    engine_task_ids.append(task_id)

                # 結果待機
                time.sleep(3.0)
                engine_results = engine.get_results_batch(engine_task_ids, timeout=5.0)

                integration_time = time.time() - start_time

                # 結果検証
                consolidator_success = len(consolidator_responses) > 0
                fetcher_success = fetcher_response.success_count > 0
                engine_success = (
                    len([r for r in engine_results.values() if r.success]) > 0
                )

                # メトリクス記録
                result.add_metric("integration_time", integration_time)
                result.add_metric("consolidator_success", consolidator_success)
                result.add_metric("fetcher_success", fetcher_success)
                result.add_metric("engine_success", engine_success)
                result.add_metric(
                    "overall_integration_success",
                    all([consolidator_success, fetcher_success, engine_success]),
                )
                result.add_metric("systems_tested", len(systems_running))

                result.complete(True)

            finally:
                # 全システム停止
                for name, system in systems_running:
                    try:
                        if hasattr(system, "stop"):
                            system.stop()
                        elif hasattr(system, "close"):
                            system.close()
                    except Exception as e:
                        result.add_error(f"{name} stop error: {e}")

        except Exception as e:
            result.add_error(f"システム統合テストエラー: {e}")
            result.complete(False)

        return result

    def _test_scalability(self) -> PerformanceTestResult:
        """スケーラビリティテスト"""
        result = PerformanceTestResult("scalability")

        try:
            # 異なる負荷レベルでテスト
            load_levels = [1, 5, 10, 20]

            for load_level in load_levels:
                consolidator = create_consolidator(
                    batch_size=20, max_workers=min(load_level, 8), enable_caching=True
                )
                consolidator.start()

                try:
                    symbols = self.test_config["test_symbols"]

                    # 負荷生成
                    responses_received = []
                    start_time = time.time()

                    def response_callback(response):
                        responses_received.append(response)

                    # 並行リクエスト投入
                    request_ids = []
                    for i in range(load_level):
                        request_id = consolidator.submit_request(
                            endpoint="scalability_test",
                            symbols=symbols,
                            callback=response_callback,
                        )
                        request_ids.append(request_id)

                    # 完了待機
                    timeout = 30.0
                    wait_start = time.time()
                    while (
                        len(responses_received) < load_level
                        and (time.time() - wait_start) < timeout
                    ):
                        time.sleep(0.1)

                    test_duration = time.time() - start_time

                    # パフォーマンスメトリクス
                    stats = consolidator.get_stats()
                    status = consolidator.get_status()

                    # メトリクス記録
                    load_key = f"load_{load_level}"
                    result.add_metric(
                        f"{load_key}_responses_received", len(responses_received)
                    )
                    result.add_metric(f"{load_key}_test_duration", test_duration)
                    result.add_metric(
                        f"{load_key}_requests_per_second",
                        len(responses_received) / test_duration,
                    )
                    result.add_metric(
                        f"{load_key}_consolidation_ratio", stats.consolidation_ratio
                    )
                    result.add_metric(f"{load_key}_success_rate", stats.success_rate)
                    result.add_metric(
                        f"{load_key}_avg_response_time", stats.avg_response_time
                    )

                finally:
                    consolidator.stop()

            result.complete(True)

        except Exception as e:
            result.add_error(f"スケーラビリティテストエラー: {e}")
            result.complete(False)

        return result

    def _test_stress_test(self) -> PerformanceTestResult:
        """ストレステスト"""
        result = PerformanceTestResult("stress_test")

        try:
            # 高負荷テスト設定
            high_worker_count = min(psutil.cpu_count(), 8)
            high_batch_size = 100
            stress_duration = 10.0  # 10秒間

            engine = create_batch_engine(
                workers=high_worker_count,
                processing_mode=ProcessingMode.THREAD_BASED,
                enable_adaptive_scaling=True,
            )
            engine.start()

            try:

                def stress_task(
                    task_id: int, processing_time: float = 0.001
                ) -> Dict[str, Any]:
                    """ストレステスト用タスク"""
                    start_time = time.time()
                    time.sleep(processing_time)

                    # メモリ使用量測定
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    return {
                        "task_id": task_id,
                        "processing_time": processing_time,
                        "actual_time": time.time() - start_time,
                        "memory_mb": memory_mb,
                    }

                # ストレステスト実行
                task_ids = []
                start_time = time.time()
                task_count = 0

                # 指定時間内にタスクを継続投入
                while (time.time() - start_time) < stress_duration:
                    task_id = engine.submit_task(
                        stress_task,
                        task_id=task_count,
                        processing_time=0.001,
                        priority=TaskPriority.NORMAL,
                    )
                    task_ids.append(task_id)
                    task_count += 1

                    # CPUを休ませる
                    time.sleep(0.001)

                stress_end_time = time.time()

                # 全タスクの完了待機
                time.sleep(5.0)

                # 結果収集
                results = engine.get_results_batch(task_ids, timeout=10.0)

                # 統計計算
                successful_tasks = [r for r in results.values() if r.success]
                failed_tasks = [r for r in results.values() if not r.success]

                if successful_tasks:
                    avg_execution_time = sum(
                        r.execution_time for r in successful_tasks
                    ) / len(successful_tasks)
                    peak_memory = max(r.memory_peak_mb for r in successful_tasks)
                else:
                    avg_execution_time = 0.0
                    peak_memory = 0.0

                # エンジン統計
                stats = engine.get_stats()

                # メトリクス記録
                result.add_metric("stress_duration_seconds", stress_duration)
                result.add_metric("tasks_submitted", task_count)
                result.add_metric("tasks_completed", len(results))
                result.add_metric("successful_tasks", len(successful_tasks))
                result.add_metric("failed_tasks", len(failed_tasks))
                result.add_metric(
                    "success_rate",
                    len(successful_tasks) / task_count if task_count > 0 else 0.0,
                )
                result.add_metric("tasks_per_second", task_count / stress_duration)
                result.add_metric("throughput_tps", stats.throughput_tps)
                result.add_metric("avg_execution_time", avg_execution_time)
                result.add_metric("peak_memory_mb", peak_memory)
                result.add_metric("engine_success_rate", stats.success_rate)
                result.add_metric(
                    "engine_avg_execution_time", stats.average_execution_time
                )

                # システムリソース
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent
                result.add_metric("system_cpu_percent", cpu_percent)
                result.add_metric("system_memory_percent", memory_percent)

                result.complete(True)

            finally:
                engine.stop(timeout=15.0)

        except Exception as e:
            result.add_error(f"ストレステストエラー: {e}")
            result.complete(False)

        return result

    def generate_report(self) -> Dict[str, Any]:
        """テストレポート生成"""
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.success])

        report = {
            "timestamp": time.time(),
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0.0
                ),
            },
            "test_config": self.test_config,
            "test_results": [result.to_dict() for result in self.test_results],
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": os.name,
            },
        }

        return report

    def save_report(self, filepath: str):
        """レポート保存"""
        report = self.generate_report()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"テストレポート保存: {filepath}")

    def cleanup(self):
        """クリーンアップ"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"一時ファイル削除エラー {temp_file}: {e}")

        # メモリクリーンアップ
        gc.collect()


def run_comprehensive_performance_test() -> Dict[str, Any]:
    """包括的パフォーマンステスト実行"""
    print("=== バッチ処理システム包括的パフォーマンステスト開始 ===")

    tester = BatchPerformanceTester()

    try:
        # 全テスト実行
        results = tester.run_all_tests()

        # レポート生成
        report = tester.generate_report()

        # 結果表示
        print("\n=== テスト結果サマリー ===")
        print(f"総テスト数: {report['test_summary']['total_tests']}")
        print(f"成功: {report['test_summary']['successful_tests']}")
        print(f"失敗: {report['test_summary']['failed_tests']}")
        print(f"成功率: {report['test_summary']['success_rate']:.1%}")

        print("\n=== 個別テスト結果 ===")
        for result in results:
            status = "✓" if result.success else "✗"
            time_taken = result.metrics.get("total_time_seconds", 0)
            print(f"{status} {result.test_name}: {time_taken:.2f}秒")

            if result.errors:
                for error in result.errors[:2]:  # 最初の2つのエラーのみ表示
                    print(f"    エラー: {error}")

        # 重要メトリクス表示
        print("\n=== 重要パフォーマンスメトリクス ===")

        for result in results:
            if result.success and result.metrics:
                print(f"\n{result.test_name}:")

                # 重要メトリクスのみ表示
                important_metrics = [
                    "overall_avg_success_rate",
                    "overall_success_rate",
                    "success_rate",
                    "average_throughput",
                    "throughput_tps",
                    "tasks_per_second",
                    "avg_execution_time",
                    "average_response_time",
                    "overall_cache_hit_rate",
                    "cache_hit_rate",
                ]

                displayed_count = 0
                for metric, value in result.metrics.items():
                    if (
                        any(important in metric for important in important_metrics)
                        and displayed_count < 3
                    ):
                        if isinstance(value, float):
                            if "rate" in metric or "ratio" in metric:
                                print(f"  {metric}: {value:.1%}")
                            else:
                                print(f"  {metric}: {value:.3f}")
                        else:
                            print(f"  {metric}: {value}")
                        displayed_count += 1

        # レポートファイル保存
        timestamp = int(time.time())
        report_path = f"batch_performance_test_results_{timestamp}.json"
        tester.save_report(report_path)
        print(f"\n詳細レポート保存: {report_path}")

        return report

    finally:
        tester.cleanup()
        print("\n=== パフォーマンステスト完了 ===")


if __name__ == "__main__":
    report = run_comprehensive_performance_test()
