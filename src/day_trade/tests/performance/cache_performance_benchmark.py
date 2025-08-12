#!/usr/bin/env python3
"""
キャッシュパフォーマンスベンチマーク
Issue #377: 高度なキャッシング戦略の検証

様々なキャッシングシステムの性能測定・比較テスト
"""

import gc
import json
import random
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

# プロジェクトモジュール
try:
    from ...cache.distributed_cache_system import get_distributed_cache
    from ...cache.persistent_cache_system import get_persistent_cache
    from ...cache.smart_invalidation_strategies import get_invalidation_manager
    from ...data.enhanced_stock_fetcher import (
        EnhancedStockFetcher,
        create_enhanced_stock_fetcher,
    )
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # 簡易モックフォールバック
    class EnhancedStockFetcher:
        def __init__(self, **kwargs):
            self.cache_config = kwargs

        def get_current_price(self, code):
            time.sleep(0.1)  # API遅延シミュレート
            return {"price": random.uniform(100, 1000), "timestamp": time.time()}

        def get_cache_stats(self):
            return {"hits": 50, "misses": 50, "hit_rate": 0.5}

    def create_enhanced_stock_fetcher(**kwargs):
        return EnhancedStockFetcher(**kwargs)

    def get_persistent_cache(**kwargs):
        return None

    def get_distributed_cache(**kwargs):
        return None


logger = get_context_logger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""

    operation_name: str
    total_operations: int
    success_operations: int
    failed_operations: int
    total_time_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    operations_per_second: float
    memory_usage_mb: float
    memory_delta_mb: float
    cache_hit_rate: Optional[float] = None
    cache_stats: Optional[Dict[str, Any]] = None


class BenchmarkRunner:
    """ベンチマーク実行クラス"""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        self.test_data = self._generate_test_data()

        logger.info(f"ベンチマーク実行準備完了: {results_dir}")

    def _generate_test_data(self) -> List[str]:
        """テストデータ生成"""
        # 日本の主要株式銘柄コード
        major_stocks = [
            "7203",
            "6758",
            "9984",
            "9983",
            "6861",
            "8306",
            "9432",
            "4502",
            "8316",
            "7267",
            "6367",
            "9434",
            "4568",
            "6902",
            "6954",
            "8035",
            "9020",
            "4063",
            "6098",
            "3382",
            "6501",
            "7974",
            "4901",
            "9022",
        ]

        # 追加のランダム銘柄コード生成
        additional_stocks = [f"{random.randint(1000, 9999)}" for _ in range(176)]

        return major_stocks + additional_stocks

    def run_memory_benchmark(
        self, test_function: Callable, name: str, iterations: int = 100
    ) -> PerformanceMetrics:
        """メモリ使用量測定付きベンチマーク実行"""
        logger.info(f"メモリベンチマーク開始: {name} ({iterations}回)")

        # ガベージコレクション実行
        gc.collect()

        # メモリトレース開始
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        execution_times = []
        success_count = 0
        failed_count = 0

        start_time = time.perf_counter()

        for i in range(iterations):
            try:
                operation_start = time.perf_counter()
                result = test_function()
                operation_end = time.perf_counter()

                execution_times.append((operation_end - operation_start) * 1000)  # ms
                success_count += 1

                if result is None:
                    failed_count += 1
                    success_count -= 1

            except Exception as e:
                logger.error(f"ベンチマーク実行エラー ({i}): {e}")
                failed_count += 1

        total_time = time.perf_counter() - start_time

        # メモリ測定
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - initial_memory

        # メモリトレース終了
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 統計計算
        if execution_times:
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            median_time = statistics.median(execution_times)
            p95_time = np.percentile(execution_times, 95)
            p99_time = np.percentile(execution_times, 99)
        else:
            avg_time = min_time = max_time = median_time = p95_time = p99_time = 0

        ops_per_second = success_count / total_time if total_time > 0 else 0

        metrics = PerformanceMetrics(
            operation_name=name,
            total_operations=iterations,
            success_operations=success_count,
            failed_operations=failed_count,
            total_time_seconds=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            median_time_ms=median_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            operations_per_second=ops_per_second,
            memory_usage_mb=current_memory,
            memory_delta_mb=memory_delta,
        )

        self.results.append(metrics)
        logger.info(
            f"ベンチマーク完了: {name}, 平均時間: {avg_time:.2f}ms, OPS: {ops_per_second:.1f}"
        )

        return metrics

    def benchmark_enhanced_stock_fetcher(self) -> List[PerformanceMetrics]:
        """Enhanced Stock Fetcherベンチマーク"""
        logger.info("Enhanced Stock Fetcher ベンチマーク開始")
        results = []

        # 様々なキャッシュ設定でテスト
        cache_configs = [
            {"persistent_cache_enabled": False, "distributed_cache_enabled": False},
            {"persistent_cache_enabled": True, "distributed_cache_enabled": False},
            {
                "persistent_cache_enabled": True,
                "distributed_cache_enabled": False,
                "smart_invalidation_enabled": True,
            },
            {
                "persistent_cache_enabled": True,
                "distributed_cache_enabled": False,
                "enable_multi_layer_cache": True,
                "l1_memory_size": 2000,
            },
        ]

        config_names = [
            "キャッシュなし",
            "永続キャッシュのみ",
            "スマート無効化",
            "マルチレイヤー",
        ]

        for config, name in zip(cache_configs, config_names):
            try:
                fetcher = create_enhanced_stock_fetcher(cache_config=config)

                # ウォームアップ (キャッシュ効果測定用)
                def warmup():
                    for code in self.test_data[:20]:
                        fetcher.get_current_price(code)

                warmup()

                # 実際のベンチマーク
                def benchmark_operation():
                    code = random.choice(self.test_data)
                    return fetcher.get_current_price(code)

                metrics = self.run_memory_benchmark(
                    benchmark_operation, f"stock_fetcher_{name}", iterations=200
                )

                # キャッシュ統計追加
                try:
                    cache_stats = fetcher.get_cache_stats()
                    metrics.cache_hit_rate = cache_stats.get("performance_stats", {}).get(
                        "cache_hit_rate", 0
                    )
                    metrics.cache_stats = cache_stats
                except Exception as e:
                    logger.warning(f"キャッシュ統計取得失敗: {e}")

                results.append(metrics)

            except Exception as e:
                logger.error(f"Stock Fetcherベンチマークエラー ({name}): {e}")

        return results

    def benchmark_concurrent_access(self) -> PerformanceMetrics:
        """並行アクセスベンチマーク"""
        logger.info("並行アクセスベンチマーク開始")

        fetcher = create_enhanced_stock_fetcher(
            cache_config={
                "persistent_cache_enabled": True,
                "enable_multi_layer_cache": True,
                "l1_memory_size": 1000,
            }
        )

        # ウォームアップ
        for code in self.test_data[:10]:
            fetcher.get_current_price(code)

        def worker_function(worker_id: int) -> List[float]:
            """ワーカー関数"""
            times = []
            for i in range(50):  # 各ワーカーが50回実行
                code = random.choice(self.test_data)
                start_time = time.perf_counter()
                result = fetcher.get_current_price(code)
                end_time = time.perf_counter()

                if result is not None:
                    times.append((end_time - start_time) * 1000)

            return times

        # 並行実行
        num_workers = 10
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_workers)]
            all_times = []

            for future in as_completed(futures):
                try:
                    worker_times = future.result()
                    all_times.extend(worker_times)
                except Exception as e:
                    logger.error(f"ワーカーエラー: {e}")

        total_time = time.perf_counter() - start_time

        if all_times:
            metrics = PerformanceMetrics(
                operation_name="並行アクセス",
                total_operations=len(all_times),
                success_operations=len(all_times),
                failed_operations=0,
                total_time_seconds=total_time,
                avg_time_ms=statistics.mean(all_times),
                min_time_ms=min(all_times),
                max_time_ms=max(all_times),
                median_time_ms=statistics.median(all_times),
                p95_time_ms=np.percentile(all_times, 95),
                p99_time_ms=np.percentile(all_times, 99),
                operations_per_second=len(all_times) / total_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                memory_delta_mb=0,
            )

            # キャッシュ統計
            try:
                cache_stats = fetcher.get_cache_stats()
                metrics.cache_hit_rate = cache_stats.get("performance_stats", {}).get(
                    "cache_hit_rate", 0
                )
                metrics.cache_stats = cache_stats
            except Exception:
                pass

            self.results.append(metrics)
            return metrics

        return None

    def benchmark_cache_eviction(self) -> PerformanceMetrics:
        """キャッシュ退避性能ベンチマーク"""
        logger.info("キャッシュ退避性能ベンチマーク開始")

        # 小さなキャッシュサイズで強制的に退避を発生させる
        fetcher = create_enhanced_stock_fetcher(
            cache_config={
                "persistent_cache_enabled": True,
                "l1_memory_size": 50,  # 小さなL1キャッシュ
                "enable_multi_layer_cache": True,
            }
        )

        def eviction_test():
            # 大量のユニークなデータでキャッシュを溢れさせる
            code = f"{random.randint(10000, 99999)}"
            return fetcher.get_current_price(code)

        return self.run_memory_benchmark(eviction_test, "キャッシュ退避", iterations=300)

    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """ベンチマーク結果レポート生成"""
        if not self.results:
            logger.warning("ベンチマーク結果がありません")
            return {}

        # 結果をDataFrameに変換
        df = pd.DataFrame([asdict(r) for r in self.results])

        # 統計サマリー作成
        summary = {
            "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "performance_rankings": {},
            "memory_analysis": {},
            "cache_effectiveness": {},
            "recommendations": [],
        }

        # パフォーマンスランキング
        if "operations_per_second" in df.columns:
            perf_ranking = df.nlargest(5, "operations_per_second")[
                ["operation_name", "operations_per_second", "avg_time_ms"]
            ]
            summary["performance_rankings"] = perf_ranking.to_dict("records")

        # メモリ分析
        if "memory_delta_mb" in df.columns:
            memory_stats = {
                "max_memory_usage": df["memory_usage_mb"].max(),
                "avg_memory_delta": df["memory_delta_mb"].mean(),
                "high_memory_operations": df[df["memory_delta_mb"] > 10]["operation_name"].tolist(),
            }
            summary["memory_analysis"] = memory_stats

        # キャッシュ効果分析
        cache_results = df[df["cache_hit_rate"].notna()]
        if not cache_results.empty:
            cache_analysis = {
                "avg_hit_rate": cache_results["cache_hit_rate"].mean(),
                "best_cache_config": cache_results.loc[
                    cache_results["cache_hit_rate"].idxmax(), "operation_name"
                ],
                "cache_performance_impact": cache_results["operations_per_second"].max()
                / cache_results["operations_per_second"].min(),
            }
            summary["cache_effectiveness"] = cache_analysis

        # 推奨事項
        recommendations = []

        # キャッシュヒット率が低い場合
        if cache_results.empty or cache_results["cache_hit_rate"].mean() < 0.5:
            recommendations.append(
                "キャッシュヒット率が低いです。TTL設定の見直しまたはキャッシュサイズ拡張を検討してください。"
            )

        # メモリ使用量が多い場合
        if df["memory_delta_mb"].max() > 50:
            recommendations.append(
                "メモリ使用量が多いです。データ圧縮またはキャッシュサイズ制限を検討してください。"
            )

        # 応答時間が遅い場合
        if df["avg_time_ms"].max() > 1000:
            recommendations.append(
                "応答時間が遅いです。キャッシュレイヤーの追加を検討してください。"
            )

        summary["recommendations"] = recommendations

        # 詳細データも含める
        summary["detailed_results"] = df.to_dict("records")

        # ファイル出力
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"ベンチマーク結果保存: {output_path}")

        return summary

    def run_full_benchmark(self) -> Dict[str, Any]:
        """フルベンチマーク実行"""
        logger.info("=== フルキャッシュパフォーマンスベンチマーク開始 ===")

        try:
            # 各種ベンチマーク実行
            logger.info("1. Enhanced Stock Fetcher ベンチマーク")
            self.benchmark_enhanced_stock_fetcher()

            logger.info("2. 並行アクセスベンチマーク")
            self.benchmark_concurrent_access()

            logger.info("3. キャッシュ退避ベンチマーク")
            self.benchmark_cache_eviction()

            # レポート生成
            logger.info("4. レポート生成")
            report = self.generate_report("cache_performance_report.json")

            logger.info("=== フルベンチマーク完了 ===")
            return report

        except Exception as e:
            logger.error(f"ベンチマーク実行エラー: {e}")
            return {"error": str(e)}


def run_cache_performance_analysis():
    """キャッシュ性能分析メイン関数"""
    print("=== Issue #377 キャッシュパフォーマンス分析 ===")

    # ベンチマーク実行
    runner = BenchmarkRunner()
    report = runner.run_full_benchmark()

    # 結果表示
    if "error" not in report:
        print("\n📊 ベンチマーク結果サマリー:")
        print(f"実行時刻: {report.get('benchmark_timestamp')}")
        print(f"総テスト数: {report.get('total_benchmarks')}")

        # パフォーマンスランキング
        rankings = report.get("performance_rankings", [])
        if rankings:
            print("\n🏆 パフォーマンスランキング (OPS):")
            for i, result in enumerate(rankings, 1):
                print(
                    f"  {i}. {result['operation_name']}: {result['operations_per_second']:.1f} OPS "
                    f"(平均 {result['avg_time_ms']:.2f}ms)"
                )

        # キャッシュ効果
        cache_effectiveness = report.get("cache_effectiveness", {})
        if cache_effectiveness:
            print("\n💾 キャッシュ効果:")
            print(f"  平均ヒット率: {cache_effectiveness.get('avg_hit_rate', 0):.2%}")
            print(f"  最高性能設定: {cache_effectiveness.get('best_cache_config')}")
            print(f"  性能向上倍率: {cache_effectiveness.get('cache_performance_impact', 1):.1f}x")

        # メモリ分析
        memory_analysis = report.get("memory_analysis", {})
        if memory_analysis:
            print("\n🧠 メモリ分析:")
            print(f"  最大メモリ使用量: {memory_analysis.get('max_memory_usage', 0):.1f} MB")
            print(f"  平均メモリ増加: {memory_analysis.get('avg_memory_delta', 0):.1f} MB")

        # 推奨事項
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n📄 詳細レポート: {runner.results_dir}/cache_performance_report.json")
    else:
        print(f"❌ ベンチマーク実行エラー: {report['error']}")

    print("\n=== キャッシュパフォーマンス分析完了 ===")
    return report


if __name__ == "__main__":
    run_cache_performance_analysis()
