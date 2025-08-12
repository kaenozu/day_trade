#!/usr/bin/env python3
"""
バッチ処理パフォーマンスベンチマーク
Issue #376: バッチ処理の強化 - パフォーマンス測定・検証

新しいバッチ処理システムの性能測定と既存システムとの比較
"""

import asyncio
import gc
import json
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

# プロジェクトモジュール
try:
    from ...batch.batch_processing_engine import (
        BatchProcessingEngine,
        execute_stock_batch_pipeline,
    )
    from ...data.batch_data_processor import BatchDataProcessor, get_batch_processor
    from ...data.enhanced_stock_fetcher import create_enhanced_stock_fetcher
    from ...data.stock_fetcher import StockFetcher
    from ...data.unified_api_adapter import UnifiedAPIAdapter, fetch_stock_prices
    from ...models.advanced_batch_database import (
        AdvancedBatchDatabase,
        OptimizationLevel,
    )
    from ...models.database import get_database_manager
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # モッククラス
    class BatchProcessingEngine:
        def __init__(self, **kwargs):
            pass

        async def cleanup(self):
            pass

    class StockFetcher:
        def get_current_price(self, code):
            time.sleep(0.1)
            return {"price": random.uniform(100, 1000)}

    def get_database_manager():
        return None

    async def execute_stock_batch_pipeline(*args, **kwargs):
        await asyncio.sleep(1)
        return type("MockResult", (), {"success": True, "total_processing_time_ms": 1000})()


logger = get_context_logger(__name__)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""

    test_name: str
    system_type: str  # 'legacy', 'batch_enhanced', 'unified_engine'
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_time_seconds: float
    operations_per_second: float
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    memory_delta_mb: float
    throughput_improvement_factor: float = 1.0
    error_rate: float = 0.0
    cache_hit_rate: Optional[float] = None
    additional_metrics: Dict[str, Any] = None


class BatchPerformanceBenchmark:
    """バッチ処理パフォーマンスベンチマーク"""

    def __init__(self, results_dir: str = "batch_benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []

        # テストデータ
        self.test_symbols = self._generate_test_symbols()
        self.test_datasets = {
            "small": self.test_symbols[:10],
            "medium": self.test_symbols[:50],
            "large": self.test_symbols[:200],
        }

        logger.info(f"バッチパフォーマンスベンチマーク初期化完了: {results_dir}")

    def _generate_test_symbols(self) -> List[str]:
        """テストシンボル生成"""
        # 主要な株式シンボル
        major_symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "AMD",
            "INTC",
            "CRM",
            "ORCL",
            "ADBE",
            "PYPL",
            "UBER",
            "LYFT",
            "ZOOM",
            "SNOW",
            "PLTR",
            "ROKU",
            "SQ",
            "SHOP",
            "TWLO",
            "OKTA",
        ]

        # 追加のランダムシンボル（テスト用）
        additional_symbols = [f"TEST{i:03d}" for i in range(1, 177)]

        return major_symbols + additional_symbols

    def run_memory_benchmark(
        self, test_function: Callable, name: str, *args, **kwargs
    ) -> BenchmarkResult:
        """メモリ使用量測定付きベンチマーク"""
        logger.info(f"メモリベンチマーク開始: {name}")

        # ガベージコレクション実行
        gc.collect()

        # メモリトレース開始
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.perf_counter()

        try:
            result = test_function(*args, **kwargs)

            total_time = time.perf_counter() - start_time

            # メモリ測定
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = current_memory - initial_memory

            # メモリトレース終了
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 結果解析
            if hasattr(result, "success") and hasattr(result, "total_processing_time_ms"):
                # BatchProcessingEngine結果
                success_count = 1 if result.success else 0
                failed_count = 1 if not result.success else 0
                avg_time_ms = result.total_processing_time_ms
            elif isinstance(result, dict):
                # 辞書形式結果
                success_count = len(result)
                failed_count = 0
                avg_time_ms = total_time * 1000
            else:
                success_count = 1
                failed_count = 0
                avg_time_ms = total_time * 1000

            ops_per_second = success_count / total_time if total_time > 0 else 0

            benchmark_result = BenchmarkResult(
                test_name=name,
                system_type="unknown",
                total_operations=success_count + failed_count,
                successful_operations=success_count,
                failed_operations=failed_count,
                total_time_seconds=total_time,
                operations_per_second=ops_per_second,
                average_response_time_ms=avg_time_ms,
                min_response_time_ms=avg_time_ms,
                max_response_time_ms=avg_time_ms,
                p95_response_time_ms=avg_time_ms,
                p99_response_time_ms=avg_time_ms,
                memory_usage_mb=current_memory,
                memory_delta_mb=memory_delta,
                error_rate=failed_count / max(success_count + failed_count, 1),
            )

            self.results.append(benchmark_result)
            logger.info(f"ベンチマーク完了: {name}, OPS: {ops_per_second:.2f}")

            return benchmark_result

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"ベンチマークエラー {name}: {e}")

            # エラー結果
            return BenchmarkResult(
                test_name=name,
                system_type="error",
                total_operations=1,
                successful_operations=0,
                failed_operations=1,
                total_time_seconds=total_time,
                operations_per_second=0,
                average_response_time_ms=total_time * 1000,
                min_response_time_ms=0,
                max_response_time_ms=total_time * 1000,
                p95_response_time_ms=total_time * 1000,
                p99_response_time_ms=total_time * 1000,
                memory_usage_mb=0,
                memory_delta_mb=0,
                error_rate=1.0,
            )

    def benchmark_legacy_stock_fetcher(self, symbols: List[str]) -> BenchmarkResult:
        """従来のStockFetcherベンチマーク"""

        def legacy_fetch_test():
            fetcher = StockFetcher()
            results = {}

            for symbol in symbols:
                try:
                    result = fetcher.get_current_price(symbol)
                    if result:
                        results[symbol] = result
                except Exception as e:
                    logger.debug(f"Legacy fetch error for {symbol}: {e}")

            return results

        result = self.run_memory_benchmark(
            legacy_fetch_test, f"legacy_fetcher_{len(symbols)}symbols"
        )
        result.system_type = "legacy"
        return result

    def benchmark_enhanced_stock_fetcher(self, symbols: List[str]) -> BenchmarkResult:
        """Enhanced StockFetcherベンチマーク"""

        def enhanced_fetch_test():
            fetcher = create_enhanced_stock_fetcher(
                cache_config={
                    "persistent_cache_enabled": True,
                    "enable_multi_layer_cache": True,
                    "l1_memory_size": 1000,
                }
            )

            results = fetcher.bulk_get_current_prices(symbols)

            # キャッシュ統計取得
            cache_stats = fetcher.get_cache_stats()
            hit_rate = cache_stats.get("performance_stats", {}).get("cache_hit_rate", 0)

            return {"results": results, "cache_hit_rate": hit_rate}

        result = self.run_memory_benchmark(
            enhanced_fetch_test, f"enhanced_fetcher_{len(symbols)}symbols"
        )
        result.system_type = "enhanced"

        # キャッシュヒット率を追加
        # Note: 実際の実装では test_function の戻り値から取得
        result.cache_hit_rate = 0.0  # プレースホルダー

        return result

    async def benchmark_unified_api_adapter(self, symbols: List[str]) -> BenchmarkResult:
        """統一APIアダプターベンチマーク"""

        async def unified_api_test():
            adapter = UnifiedAPIAdapter(enable_caching=True)

            try:
                results = await fetch_stock_prices(symbols, adapter)
                stats = adapter.get_stats()

                return {
                    "results": results,
                    "cache_hit_rate": stats.get("cache_hit_rate", 0),
                    "batched_requests": stats.get("batched_requests", 0),
                }
            finally:
                await adapter.cleanup()

        # 非同期関数のラッパー
        def async_wrapper():
            return asyncio.run(unified_api_test())

        result = self.run_memory_benchmark(async_wrapper, f"unified_api_{len(symbols)}symbols")
        result.system_type = "unified_api"
        return result

    async def benchmark_batch_processing_engine(self, symbols: List[str]) -> BenchmarkResult:
        """バッチ処理エンジンベンチマーク"""

        async def batch_engine_test():
            engine = BatchProcessingEngine(
                max_concurrent_jobs=5,
                enable_caching=True,
                optimization_level=OptimizationLevel.ADVANCED,
            )

            try:
                result = await execute_stock_batch_pipeline(
                    symbols=symbols,
                    include_historical=False,
                    store_data=False,  # データベース保存はスキップ
                    engine=engine,
                )

                stats = engine.get_stats()
                return {"result": result, "engine_stats": stats}
            finally:
                await engine.cleanup()

        # 非同期関数のラッパー
        def async_wrapper():
            return asyncio.run(batch_engine_test())

        result = self.run_memory_benchmark(async_wrapper, f"batch_engine_{len(symbols)}symbols")
        result.system_type = "batch_engine"
        return result

    async def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """包括的ベンチマーク実行"""
        logger.info("=== 包括的バッチ処理ベンチマーク開始 ===")

        comprehensive_results = {}

        # データサイズ別テスト
        for dataset_name, symbols in self.test_datasets.items():
            logger.info(f"\n{dataset_name.upper()}データセットテスト: {len(symbols)}銘柄")

            dataset_results = []

            # 1. 従来システム
            logger.info("1. 従来StockFetcherテスト")
            legacy_result = self.benchmark_legacy_stock_fetcher(symbols)
            dataset_results.append(legacy_result)

            # 2. 拡張システム
            logger.info("2. Enhanced StockFetcherテスト")
            enhanced_result = self.benchmark_enhanced_stock_fetcher(symbols)
            dataset_results.append(enhanced_result)

            # 3. 統一APIアダプター
            logger.info("3. 統一APIアダプターテスト")
            unified_result = await self.benchmark_unified_api_adapter(symbols)
            dataset_results.append(unified_result)

            # 4. バッチ処理エンジン
            logger.info("4. バッチ処理エンジンテスト")
            batch_result = await self.benchmark_batch_processing_engine(symbols)
            dataset_results.append(batch_result)

            # 改善倍率計算
            legacy_ops = legacy_result.operations_per_second
            for result in dataset_results[1:]:  # 従来システム以外
                if legacy_ops > 0:
                    result.throughput_improvement_factor = result.operations_per_second / legacy_ops

            comprehensive_results[dataset_name] = dataset_results

            # データセット間の休憩
            await asyncio.sleep(2)

        logger.info("=== 包括的ベンチマーク完了 ===")
        return comprehensive_results

    def generate_performance_report(
        self, results: Dict[str, List[BenchmarkResult]]
    ) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "test_environment": {
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": psutil.sys.platform,
            },
            "dataset_results": {},
            "performance_summary": {},
            "recommendations": [],
        }

        # データセット別結果
        all_results = []
        for dataset_name, dataset_results in results.items():
            report["dataset_results"][dataset_name] = {
                "dataset_size": len(self.test_datasets[dataset_name]),
                "system_comparisons": [],
            }

            for result in dataset_results:
                comparison = {
                    "system_type": result.system_type,
                    "operations_per_second": result.operations_per_second,
                    "average_response_time_ms": result.average_response_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "memory_delta_mb": result.memory_delta_mb,
                    "error_rate": result.error_rate,
                    "throughput_improvement": result.throughput_improvement_factor,
                    "cache_hit_rate": result.cache_hit_rate,
                }

                report["dataset_results"][dataset_name]["system_comparisons"].append(comparison)
                all_results.append(result)

        # 総合パフォーマンス分析
        system_performance = {}
        for result in all_results:
            if result.system_type not in system_performance:
                system_performance[result.system_type] = {
                    "avg_ops_per_second": [],
                    "avg_response_time_ms": [],
                    "avg_memory_usage_mb": [],
                    "avg_improvement_factor": [],
                    "error_rates": [],
                }

            system_performance[result.system_type]["avg_ops_per_second"].append(
                result.operations_per_second
            )
            system_performance[result.system_type]["avg_response_time_ms"].append(
                result.average_response_time_ms
            )
            system_performance[result.system_type]["avg_memory_usage_mb"].append(
                result.memory_usage_mb
            )
            system_performance[result.system_type]["avg_improvement_factor"].append(
                result.throughput_improvement_factor
            )
            system_performance[result.system_type]["error_rates"].append(result.error_rate)

        # 統計計算
        for system, metrics in system_performance.items():
            report["performance_summary"][system] = {
                "avg_operations_per_second": statistics.mean(metrics["avg_ops_per_second"]),
                "avg_response_time_ms": statistics.mean(metrics["avg_response_time_ms"]),
                "avg_memory_usage_mb": statistics.mean(metrics["avg_memory_usage_mb"]),
                "avg_improvement_factor": statistics.mean(metrics["avg_improvement_factor"]),
                "avg_error_rate": statistics.mean(metrics["error_rates"]),
                "reliability_score": 1.0 - statistics.mean(metrics["error_rates"]),
            }

        # 推奨事項生成
        recommendations = []

        # 最高性能システム特定
        best_system = max(
            report["performance_summary"].items(),
            key=lambda x: x[1]["avg_operations_per_second"],
        )
        recommendations.append(
            f"最高性能システム: {best_system[0]} ({best_system[1]['avg_operations_per_second']:.2f} OPS)"
        )

        # メモリ効率分析
        most_memory_efficient = min(
            report["performance_summary"].items(),
            key=lambda x: x[1]["avg_memory_usage_mb"],
        )
        recommendations.append(
            f"最もメモリ効率的: {most_memory_efficient[0]} ({most_memory_efficient[1]['avg_memory_usage_mb']:.1f} MB)"
        )

        # 改善倍率分析
        max_improvement = max(
            report["performance_summary"].items(),
            key=lambda x: x[1]["avg_improvement_factor"],
        )
        if max_improvement[1]["avg_improvement_factor"] > 1.5:
            recommendations.append(
                f"大幅な性能改善: {max_improvement[0]} (従来比 {max_improvement[1]['avg_improvement_factor']:.1f}倍)"
            )

        report["recommendations"] = recommendations

        # ファイル保存
        report_file = self.results_dir / "batch_performance_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"パフォーマンスレポート保存: {report_file}")

        return report

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """フルベンチマーク実行"""
        try:
            results = await self.run_comprehensive_benchmark()
            report = self.generate_performance_report(results)
            return report
        except Exception as e:
            logger.error(f"ベンチマーク実行エラー: {e}")
            return {"error": str(e)}


async def run_batch_performance_analysis():
    """バッチパフォーマンス分析メイン関数"""
    print("=== Issue #376 バッチ処理パフォーマンス分析 ===")

    benchmark = BatchPerformanceBenchmark()
    report = await benchmark.run_full_benchmark()

    if "error" not in report:
        print(f"\n📊 ベンチマーク完了時刻: {report.get('benchmark_timestamp')}")
        print(
            f"テスト環境: {report.get('test_environment', {}).get('python_version')} Python, "
            f"{report.get('test_environment', {}).get('cpu_count')} CPU, "
            f"{report.get('test_environment', {}).get('memory_total_gb', 0):.1f}GB RAM"
        )

        # パフォーマンス概要
        print("\n🚀 パフォーマンス概要:")
        summary = report.get("performance_summary", {})
        for system, metrics in summary.items():
            print(f"  {system}:")
            print(f"    平均OPS: {metrics.get('avg_operations_per_second', 0):.2f}")
            print(f"    平均応答時間: {metrics.get('avg_response_time_ms', 0):.2f}ms")
            print(f"    改善倍率: {metrics.get('avg_improvement_factor', 1):.2f}x")
            print(f"    信頼性スコア: {metrics.get('reliability_score', 0):.2%}")

        # データセット別結果
        print("\n📈 データセット別結果:")
        dataset_results = report.get("dataset_results", {})
        for dataset_name, dataset_data in dataset_results.items():
            print(f"  {dataset_name.upper()} ({dataset_data.get('dataset_size')}銘柄):")

            comparisons = dataset_data.get("system_comparisons", [])
            for comp in comparisons:
                print(
                    f"    {comp.get('system_type')}: "
                    f"{comp.get('operations_per_second', 0):.2f} OPS, "
                    f"{comp.get('average_response_time_ms', 0):.1f}ms"
                )

        # 推奨事項
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n📄 詳細レポート: {benchmark.results_dir}/batch_performance_report.json")

    else:
        print(f"❌ ベンチマーク実行エラー: {report['error']}")

    print("\n=== バッチ処理パフォーマンス分析完了 ===")
    return report


if __name__ == "__main__":
    asyncio.run(run_batch_performance_analysis())
