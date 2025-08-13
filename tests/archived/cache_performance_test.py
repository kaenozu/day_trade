#!/usr/bin/env python3
"""
Cache Performance Test Suite
Issue #324: Cache Strategy Optimization

統合キャッシュマネージャーの性能テストと既存キャッシュシステムとの比較
"""

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.day_trade.utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )

    UNIFIED_CACHE_AVAILABLE = True
    print("Unified cache manager imported successfully")
except ImportError as e:
    print(f"Unified cache import error: {e}")
    UNIFIED_CACHE_AVAILABLE = False

try:
    from src.day_trade.data.stock_fetcher import DataCache

    OLD_CACHE_AVAILABLE = True
    print("Legacy cache system imported successfully")
except ImportError as e:
    print(f"Legacy cache import error: {e}")
    OLD_CACHE_AVAILABLE = False


class CachePerformanceBenchmark:
    """キャッシュパフォーマンステストクラス"""

    def __init__(self):
        self.results = {}
        print("Cache Performance Benchmark initialized")

    def generate_test_data(self, num_items: int = 1000) -> List[Dict[str, Any]]:
        """テストデータ生成"""
        test_data = []

        for i in range(num_items):
            size_category = random.choice(["small", "medium", "large"])

            if size_category == "small":
                data = f"data_{i}" * random.randint(1, 10)  # ~50-500 bytes
                priority = random.uniform(5.0, 10.0)
            elif size_category == "medium":
                data = f"data_{i}" * random.randint(100, 500)  # ~5-25KB
                priority = random.uniform(2.0, 7.0)
            else:  # large
                data = f"data_{i}" * random.randint(1000, 2000)  # ~50-100KB
                priority = random.uniform(0.5, 3.0)

            test_data.append(
                {
                    "key": f"test_key_{i}",
                    "value": data,
                    "priority": priority,
                    "category": size_category,
                }
            )

        return test_data

    def test_unified_cache_performance(
        self, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """統合キャッシュマネージャーパフォーマンステスト"""
        if not UNIFIED_CACHE_AVAILABLE:
            return {"error": "Unified cache not available"}

        print("\n=== Testing Unified Cache Manager ===")

        # キャッシュマネージャー初期化
        cache_manager = UnifiedCacheManager(
            l1_memory_mb=32, l2_memory_mb=128, l3_disk_mb=512
        )

        results = {
            "put_times": [],
            "get_times": [],
            "hit_rates": {},
            "memory_usage": [],
            "errors": 0,
        }

        # データ保存テスト
        print("  Testing PUT operations...")
        put_start_time = time.time()

        for item in test_data:
            start = time.time()
            try:
                success = cache_manager.put(
                    item["key"], item["value"], priority=item["priority"]
                )
                if not success:
                    results["errors"] += 1
            except Exception as e:
                print(f"    PUT error: {e}")
                results["errors"] += 1

            results["put_times"].append((time.time() - start) * 1000)  # ms

        put_total_time = time.time() - put_start_time

        # データ取得テスト (全データ)
        print("  Testing GET operations (full dataset)...")
        get_start_time = time.time()
        hits = 0

        for item in test_data:
            start = time.time()
            try:
                value = cache_manager.get(item["key"])
                if value is not None:
                    hits += 1
            except Exception as e:
                print(f"    GET error: {e}")
                results["errors"] += 1

            results["get_times"].append((time.time() - start) * 1000)  # ms

        get_total_time = time.time() - get_start_time

        # ランダムアクセステスト
        print("  Testing random access pattern...")
        random_access_hits = 0
        random_keys = random.choices([item["key"] for item in test_data], k=500)

        for key in random_keys:
            value = cache_manager.get(key)
            if value is not None:
                random_access_hits += 1

        # 統計取得
        stats = cache_manager.get_comprehensive_stats()

        return {
            "total_items": len(test_data),
            "put_total_time": put_total_time,
            "get_total_time": get_total_time,
            "avg_put_time_ms": np.mean(results["put_times"]),
            "avg_get_time_ms": np.mean(results["get_times"]),
            "full_hit_rate": hits / len(test_data),
            "random_hit_rate": random_access_hits / len(random_keys),
            "overall_hit_rate": stats["overall"]["hit_rate"],
            "l1_hit_ratio": stats["overall"]["l1_hit_ratio"],
            "l2_hit_ratio": stats["overall"]["l2_hit_ratio"],
            "l3_hit_ratio": stats["overall"]["l3_hit_ratio"],
            "memory_usage_mb": stats["memory_usage_total_mb"],
            "disk_usage_mb": stats["disk_usage_mb"],
            "errors": results["errors"],
            "layer_stats": stats["layers"],
        }

    def test_legacy_cache_performance(
        self, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """既存キャッシュシステムパフォーマンステスト"""
        if not OLD_CACHE_AVAILABLE:
            return {"error": "Legacy cache not available"}

        print("\n=== Testing Legacy Cache System ===")

        # 既存キャッシュ初期化
        cache = DataCache(ttl_seconds=300, max_size=1000, stale_while_revalidate=60)

        results = {"put_times": [], "get_times": [], "errors": 0}

        # データ保存テスト
        print("  Testing PUT operations...")
        put_start_time = time.time()

        for item in test_data:
            start = time.time()
            try:
                cache.put(item["key"], item["value"])
            except Exception as e:
                print(f"    PUT error: {e}")
                results["errors"] += 1

            results["put_times"].append((time.time() - start) * 1000)  # ms

        put_total_time = time.time() - put_start_time

        # データ取得テスト
        print("  Testing GET operations...")
        get_start_time = time.time()
        hits = 0

        for item in test_data:
            start = time.time()
            try:
                value = cache.get(item["key"])
                if value is not None:
                    hits += 1
            except Exception as e:
                print(f"    GET error: {e}")
                results["errors"] += 1

            results["get_times"].append((time.time() - start) * 1000)  # ms

        get_total_time = time.time() - get_start_time

        return {
            "total_items": len(test_data),
            "put_total_time": put_total_time,
            "get_total_time": get_total_time,
            "avg_put_time_ms": np.mean(results["put_times"]),
            "avg_get_time_ms": np.mean(results["get_times"]),
            "hit_rate": hits / len(test_data),
            "errors": results["errors"],
        }

    def test_concurrent_access(
        self, cache_manager, test_data: List[Dict[str, Any]], num_threads: int = 4
    ) -> Dict[str, Any]:
        """並行アクセステスト"""
        print(f"\n=== Testing Concurrent Access ({num_threads} threads) ===")

        results = {
            "total_operations": 0,
            "total_time": 0,
            "errors": 0,
            "thread_results": [],
        }

        def worker_thread(thread_id: int, data_slice: List[Dict[str, Any]]):
            """ワーカースレッド"""
            thread_start = time.time()
            operations = 0
            errors = 0

            for item in data_slice:
                try:
                    # 書き込み
                    cache_manager.put(item["key"], item["value"], item["priority"])
                    operations += 1

                    # 読み込み
                    value = cache_manager.get(item["key"])
                    operations += 1

                except Exception:
                    errors += 1

            thread_time = time.time() - thread_start

            return {
                "thread_id": thread_id,
                "operations": operations,
                "time": thread_time,
                "errors": errors,
                "ops_per_sec": operations / thread_time,
            }

        # データを分割
        chunk_size = len(test_data) // num_threads
        data_chunks = [
            test_data[i : i + chunk_size] for i in range(0, len(test_data), chunk_size)
        ]

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, i, chunk)
                for i, chunk in enumerate(data_chunks)
            ]

            for future in futures:
                thread_result = future.result()
                results["thread_results"].append(thread_result)
                results["total_operations"] += thread_result["operations"]
                results["errors"] += thread_result["errors"]

        results["total_time"] = time.time() - start_time
        results["overall_ops_per_sec"] = (
            results["total_operations"] / results["total_time"]
        )

        return results

    def test_memory_pressure(self, cache_manager) -> Dict[str, Any]:
        """メモリプレッシャーテスト"""
        print("\n=== Testing Memory Pressure ===")

        # 大きなデータを段階的に追加
        memory_usage = []
        pressure_results = {"stages": [], "final_stats": {}}

        for stage in range(1, 6):  # 5段階
            print(f"  Stage {stage}: Adding large data...")

            # 大きなデータを追加
            large_data = "X" * (100 * 1024 * stage)  # 100KB * stage
            key = f"memory_pressure_{stage}"

            success = cache_manager.put(key, large_data, priority=5.0)

            # 統計取得
            stats = cache_manager.get_comprehensive_stats()

            stage_result = {
                "stage": stage,
                "data_size_kb": len(large_data) / 1024,
                "put_success": success,
                "memory_usage_mb": stats["memory_usage_total_mb"],
                "disk_usage_mb": stats["disk_usage_mb"],
                "l1_entries": stats["layers"]["L1"]["entries"],
                "l2_entries": stats["layers"]["L2"]["entries"],
                "l3_entries": stats["layers"]["L3"]["entries"],
            }

            pressure_results["stages"].append(stage_result)

            # メモリ最適化実行
            cache_manager.optimize_memory()

        pressure_results["final_stats"] = cache_manager.get_comprehensive_stats()

        return pressure_results

    def generate_performance_report(
        self,
        unified_results: Dict,
        legacy_results: Dict = None,
        concurrent_results: Dict = None,
        memory_results: Dict = None,
    ) -> str:
        """性能レポート生成"""

        report = f"""
Cache Performance Benchmark Report
{'='*60}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

UNIFIED CACHE MANAGER RESULTS:
"""
        if "error" not in unified_results:
            report += f"""
Dataset Size: {unified_results['total_items']} items
PUT Performance:
  Total time: {unified_results['put_total_time']:.2f}s
  Average time: {unified_results['avg_put_time_ms']:.2f}ms per operation
  Throughput: {unified_results['total_items']/unified_results['put_total_time']:.0f} ops/sec

GET Performance:
  Total time: {unified_results['get_total_time']:.2f}s
  Average time: {unified_results['avg_get_time_ms']:.2f}ms per operation
  Throughput: {unified_results['total_items']/unified_results['get_total_time']:.0f} ops/sec

Cache Hit Rates:
  Full dataset: {unified_results['full_hit_rate']:.1%}
  Random access: {unified_results['random_hit_rate']:.1%}
  Overall: {unified_results['overall_hit_rate']:.1%}

Layer Performance:
  L1 hits: {unified_results['l1_hit_ratio']:.1%}
  L2 hits: {unified_results['l2_hit_ratio']:.1%}
  L3 hits: {unified_results['l3_hit_ratio']:.1%}

Resource Usage:
  Memory: {unified_results['memory_usage_mb']:.1f}MB
  Disk: {unified_results['disk_usage_mb']:.1f}MB
  Errors: {unified_results['errors']}
"""
        else:
            report += f"  ERROR: {unified_results['error']}\n"

        if legacy_results and "error" not in legacy_results:
            report += f"""
LEGACY CACHE SYSTEM RESULTS:
PUT Performance:
  Total time: {legacy_results['put_total_time']:.2f}s
  Average time: {legacy_results['avg_put_time_ms']:.2f}ms per operation
  Throughput: {legacy_results['total_items']/legacy_results['put_total_time']:.0f} ops/sec

GET Performance:
  Total time: {legacy_results['get_total_time']:.2f}s
  Average time: {legacy_results['avg_get_time_ms']:.2f}ms per operation
  Throughput: {legacy_results['total_items']/legacy_results['get_total_time']:.0f} ops/sec

Hit Rate: {legacy_results['hit_rate']:.1%}
Errors: {legacy_results['errors']}
"""

            # パフォーマンス比較
            if "error" not in unified_results:
                put_improvement = (
                    (
                        legacy_results["avg_put_time_ms"]
                        - unified_results["avg_put_time_ms"]
                    )
                    / legacy_results["avg_put_time_ms"]
                    * 100
                )
                get_improvement = (
                    (
                        legacy_results["avg_get_time_ms"]
                        - unified_results["avg_get_time_ms"]
                    )
                    / legacy_results["avg_get_time_ms"]
                    * 100
                )

                report += f"""
PERFORMANCE COMPARISON:
PUT operation improvement: {put_improvement:.1f}%
GET operation improvement: {get_improvement:.1f}%
Hit rate improvement: {(unified_results['overall_hit_rate'] - legacy_results['hit_rate'])*100:.1f} percentage points
"""

        if concurrent_results:
            report += f"""
CONCURRENT ACCESS TEST:
Total operations: {concurrent_results['total_operations']}
Total time: {concurrent_results['total_time']:.2f}s
Overall throughput: {concurrent_results['overall_ops_per_sec']:.0f} ops/sec
Errors: {concurrent_results['errors']}

Per-thread results:
"""
            for thread_result in concurrent_results["thread_results"]:
                report += f"  Thread {thread_result['thread_id']}: {thread_result['ops_per_sec']:.0f} ops/sec\n"

        if memory_results:
            report += """
MEMORY PRESSURE TEST:
"""
            for stage in memory_results["stages"]:
                report += f"  Stage {stage['stage']}: {stage['data_size_kb']:.0f}KB data, Memory: {stage['memory_usage_mb']:.1f}MB\n"

        report += f"\n{'='*60}\n"
        return report


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("Cache Performance Benchmark Suite")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not UNIFIED_CACHE_AVAILABLE:
        print("[ERROR] Unified cache manager not available")
        return

    # ベンチマーククラス初期化
    benchmark = CachePerformanceBenchmark()

    # テストデータ生成
    print("\n1. Generating test data...")
    test_data = benchmark.generate_test_data(num_items=1000)
    print(f"   Generated {len(test_data)} test items")

    # 統合キャッシュテスト
    print("\n2. Testing Unified Cache Manager...")
    unified_results = benchmark.test_unified_cache_performance(test_data)

    # レガシーキャッシュテスト
    legacy_results = None
    if OLD_CACHE_AVAILABLE:
        print("\n3. Testing Legacy Cache System...")
        legacy_results = benchmark.test_legacy_cache_performance(test_data)
    else:
        print("\n3. Legacy cache system not available - skipping")

    # 並行アクセステスト
    if UNIFIED_CACHE_AVAILABLE:
        print("\n4. Testing concurrent access...")
        cache_manager = UnifiedCacheManager()
        concurrent_results = benchmark.test_concurrent_access(
            cache_manager, test_data[:200], num_threads=4
        )

        # メモリプレッシャーテスト
        print("\n5. Testing memory pressure...")
        memory_results = benchmark.test_memory_pressure(cache_manager)
    else:
        concurrent_results = None
        memory_results = None

    # レポート生成
    print("\n6. Generating performance report...")
    report = benchmark.generate_performance_report(
        unified_results, legacy_results, concurrent_results, memory_results
    )

    # レポート保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"cache_performance_report_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved: {report_file}")

    # サマリー表示
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    if "error" not in unified_results:
        print("Unified Cache Performance:")
        print(f"  PUT: {unified_results['avg_put_time_ms']:.2f}ms avg")
        print(f"  GET: {unified_results['avg_get_time_ms']:.2f}ms avg")
        print(f"  Hit Rate: {unified_results['overall_hit_rate']:.1%}")
        print(f"  Memory Usage: {unified_results['memory_usage_mb']:.1f}MB")

        if legacy_results and "error" not in legacy_results:
            put_improvement = (
                (legacy_results["avg_put_time_ms"] - unified_results["avg_put_time_ms"])
                / legacy_results["avg_put_time_ms"]
                * 100
            )
            get_improvement = (
                (legacy_results["avg_get_time_ms"] - unified_results["avg_get_time_ms"])
                / legacy_results["avg_get_time_ms"]
                * 100
            )

            print("\nImprovement over Legacy:")
            print(f"  PUT: {put_improvement:.1f}% faster")
            print(f"  GET: {get_improvement:.1f}% faster")

    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Cache performance benchmark completed!")
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
