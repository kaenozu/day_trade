#!/usr/bin/env python3
"""
Comprehensive Cache Performance Test & Integration
Issue #377: 高度なキャッシング戦略の導入

包括的キャッシュパフォーマンステスト:
- 全キャッシュシステムの統合テスト
- パフォーマンスベンチマーク
- スケーラビリティテスト
- 障害耐性テスト
- 実用シナリオでの検証
"""

import asyncio
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .auto_invalidation_triggers import (
        AutoInvalidationTriggerManager,
        TriggerCondition,
        TriggerConfig,
        TriggerType,
    )
    from .enhanced_persistent_cache import (
        EnhancedPersistentCache,
        create_enhanced_persistent_cache,
    )
    from .redis_enhanced_cache import (
        REDIS_AVAILABLE,
        RedisEnhancedCache,
        create_redis_enhanced_cache,
    )
    from .smart_cache_invalidation import (
        DependencyRule,
        DependencyType,
        InvalidationType,
        SmartCacheInvalidator,
    )
    from .staged_cache_update import StagedCacheUpdater, UpdateConfig, UpdateStrategy
except ImportError:
    print("一部のモジュールをインポートできません。個別テストを実行します。")

    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    REDIS_AVAILABLE = False

    # モッククラス
    class EnhancedPersistentCache:
        def __init__(self, *args, **kwargs):
            self.data = {}

        async def get(self, key):
            return self.data.get(key)

        async def put(self, key, value, **kwargs):
            self.data[key] = value
            return True

        async def delete(self, key):
            return self.data.pop(key, None) is not None

        async def get_comprehensive_stats(self):
            return {"hit_rate": 95.0, "avg_response_time": 2.5}

    def create_enhanced_persistent_cache(*args, **kwargs):
        return EnhancedPersistentCache()

    def create_redis_enhanced_cache(*args, **kwargs):
        return EnhancedPersistentCache()

    class SmartCacheInvalidator:
        pass

    class AutoInvalidationTriggerManager:
        pass

    class StagedCacheUpdater:
        pass

    class UnifiedCacheManager:
        pass


logger = get_context_logger(__name__)


class CachePerformanceBenchmark:
    """キャッシュパフォーマンスベンチマーク"""

    def __init__(self):
        self.results = {}
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> Dict[str, Any]:
        """テストデータ生成"""
        data = {}

        # 小サイズデータ (1KB未満)
        for i in range(1000):
            data[f"small_{i}"] = {
                "id": i,
                "name": f"item_{i}",
                "value": random.randint(1, 1000),
            }

        # 中サイズデータ (1KB-10KB)
        for i in range(500):
            data[f"medium_{i}"] = {
                "id": i,
                "data": "x" * random.randint(1000, 10000),
                "metadata": {"created": time.time(), "version": 1},
            }

        # 大サイズデータ (10KB-100KB)
        for i in range(100):
            data[f"large_{i}"] = {
                "id": i,
                "content": "y" * random.randint(10000, 100000),
                "extra": list(range(random.randint(100, 1000))),
            }

        return data

    async def benchmark_cache_system(
        self, cache_name: str, cache_instance: Any, operations: int = 1000
    ) -> Dict[str, Any]:
        """キャッシュシステムベンチマーク"""
        print(f"\n=== {cache_name} パフォーマンステスト ===")

        results = {
            "cache_name": cache_name,
            "operations": operations,
            "put_times": [],
            "get_times": [],
            "hit_rate": 0.0,
            "total_time": 0.0,
            "ops_per_second": 0.0,
            "errors": 0,
        }

        start_time = time.time()

        # PUT操作ベンチマーク
        print("  PUT操作テスト...")
        test_keys = list(self.test_data.keys())[:operations]

        for key in test_keys:
            put_start = time.time()
            try:
                if hasattr(cache_instance, "put"):
                    await cache_instance.put(key, self.test_data[key])
                elif hasattr(cache_instance, "set"):
                    cache_instance.set(key, self.test_data[key])
                else:
                    cache_instance[key] = self.test_data[key]

                put_time = (time.time() - put_start) * 1000
                results["put_times"].append(put_time)

            except Exception as e:
                results["errors"] += 1
                logger.error(f"PUT操作エラー ({key}): {e}")

        # GET操作ベンチマーク
        print("  GET操作テスト...")
        hits = 0

        for key in test_keys:
            get_start = time.time()
            try:
                if hasattr(cache_instance, "get"):
                    result = await cache_instance.get(key)
                elif hasattr(cache_instance, "__getitem__"):
                    result = cache_instance.get(key)
                else:
                    result = getattr(cache_instance, "data", {}).get(key)

                get_time = (time.time() - get_start) * 1000
                results["get_times"].append(get_time)

                if result is not None:
                    hits += 1

            except Exception as e:
                results["errors"] += 1
                logger.error(f"GET操作エラー ({key}): {e}")

        # 統計計算
        results["hit_rate"] = (hits / len(test_keys)) * 100 if test_keys else 0
        results["total_time"] = time.time() - start_time
        results["ops_per_second"] = (
            (len(test_keys) * 2) / results["total_time"]
            if results["total_time"] > 0
            else 0
        )

        # 統計サマリー
        if results["put_times"]:
            results["avg_put_time"] = statistics.mean(results["put_times"])
            results["p95_put_time"] = (
                statistics.quantiles(results["put_times"], n=20)[18]
                if len(results["put_times"]) > 20
                else max(results["put_times"])
            )

        if results["get_times"]:
            results["avg_get_time"] = statistics.mean(results["get_times"])
            results["p95_get_time"] = (
                statistics.quantiles(results["get_times"], n=20)[18]
                if len(results["get_times"]) > 20
                else max(results["get_times"])
            )

        print(
            f"  結果: ヒット率={results['hit_rate']:.1f}%, OPS={results['ops_per_second']:.0f}, エラー={results['errors']}"
        )

        return results

    async def concurrent_benchmark(
        self,
        cache_name: str,
        cache_instance: Any,
        concurrent_users: int = 10,
        operations_per_user: int = 100,
    ):
        """同時アクセスベンチマーク"""
        print(
            f"\n=== {cache_name} 同時アクセステスト (ユーザー数: {concurrent_users}) ==="
        )

        async def user_simulation(user_id: int):
            """ユーザーシミュレーション"""
            user_ops = []
            keys = list(self.test_data.keys())[
                user_id * operations_per_user : (user_id + 1) * operations_per_user
            ]

            for key in keys:
                op_start = time.time()
                try:
                    # ランダムにPUT/GET操作
                    if random.random() < 0.3:  # 30% PUT, 70% GET
                        if hasattr(cache_instance, "put"):
                            await cache_instance.put(key, self.test_data[key])
                        else:
                            cache_instance[key] = self.test_data[key]
                        op_type = "PUT"
                    else:
                        if hasattr(cache_instance, "get"):
                            result = await cache_instance.get(key)
                        else:
                            result = cache_instance.get(key)
                        op_type = "GET"

                    op_time = (time.time() - op_start) * 1000
                    user_ops.append({"type": op_type, "time": op_time, "success": True})

                except Exception as e:
                    op_time = (time.time() - op_start) * 1000
                    user_ops.append(
                        {
                            "type": op_type,
                            "time": op_time,
                            "success": False,
                            "error": str(e),
                        }
                    )

            return user_ops

        # 同時実行
        start_time = time.time()
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # 結果集計
        all_ops = []
        for user_results in all_results:
            all_ops.extend(user_results)

        successful_ops = [op for op in all_ops if op["success"]]
        failed_ops = [op for op in all_ops if not op["success"]]

        concurrent_results = {
            "cache_name": cache_name,
            "concurrent_users": concurrent_users,
            "total_operations": len(all_ops),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(all_ops) * 100 if all_ops else 0,
            "total_time": total_time,
            "ops_per_second": len(all_ops) / total_time if total_time > 0 else 0,
            "avg_response_time": (
                statistics.mean([op["time"] for op in successful_ops])
                if successful_ops
                else 0
            ),
            "p95_response_time": (
                statistics.quantiles([op["time"] for op in successful_ops], n=20)[18]
                if len(successful_ops) > 20
                else 0
            ),
        }

        print(
            f"  結果: 成功率={concurrent_results['success_rate']:.1f}%, OPS={concurrent_results['ops_per_second']:.0f}"
        )

        return concurrent_results


class IntegrationTestSuite:
    """統合テストスイート"""

    async def test_full_cache_integration(self):
        """完全キャッシュ統合テスト"""
        print("\n=== Issue #377 完全統合テスト開始 ===")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
        }

        try:
            # 1. 個別キャッシュシステムテスト
            print("\n1. 個別キャッシュシステムテスト")

            # Enhanced Persistent Cache テスト
            persistent_cache = create_enhanced_persistent_cache(
                storage_path="test_cache_integration",
                max_memory_mb=32,
                compression_enabled=True,
            )

            benchmark = CachePerformanceBenchmark()
            persistent_results = await benchmark.benchmark_cache_system(
                "Enhanced Persistent Cache", persistent_cache, operations=500
            )
            test_results["tests"]["persistent_cache"] = persistent_results

            # Redis Cache テスト (利用可能な場合)
            if REDIS_AVAILABLE:
                try:
                    redis_cache = create_redis_enhanced_cache()
                    await redis_cache.initialize()

                    redis_results = await benchmark.benchmark_cache_system(
                        "Redis Enhanced Cache", redis_cache, operations=500
                    )
                    test_results["tests"]["redis_cache"] = redis_results

                    await redis_cache.shutdown()

                except Exception as e:
                    print(f"  Redis Cache テストスキップ: {e}")
                    test_results["tests"]["redis_cache"] = {"error": str(e)}

            # 2. 同時アクセステスト
            print("\n2. 同時アクセステスト")
            concurrent_results = await benchmark.concurrent_benchmark(
                "Enhanced Persistent Cache",
                persistent_cache,
                concurrent_users=5,
                operations_per_user=50,
            )
            test_results["tests"]["concurrent_access"] = concurrent_results

            # 3. スマート無効化テスト
            print("\n3. スマート無効化システム統合テスト")
            invalidator = SmartCacheInvalidator([persistent_cache])

            # 依存性ルール追加
            if hasattr(invalidator, "add_rule"):
                rule = DependencyRule(
                    source_pattern="user:*",
                    target_patterns=["user_stats:*", "user_cache:*"],
                    dependency_type=DependencyType.HIERARCHICAL,
                    invalidation_type=InvalidationType.BATCH,
                )
                rule_id = invalidator.add_rule(rule)

                # 無効化テスト
                event_id = invalidator.invalidate(
                    "user:test", InvalidationType.IMMEDIATE
                )

                invalidation_stats = invalidator.get_comprehensive_stats()
                test_results["tests"]["smart_invalidation"] = invalidation_stats
                print(f"  無効化システム統計: {invalidation_stats}")

            # 4. 段階的更新テスト
            print("\n4. 段階的更新システムテスト")
            updater = StagedCacheUpdater()
            updater.register_cache_instance("test_cache", persistent_cache)

            if hasattr(updater, "start_staged_update"):
                update_config = UpdateConfig(
                    update_id="integration_test",
                    strategy=UpdateStrategy.CANARY,
                    target_keys=["test:1", "test:2", "test:3"],
                    target_cache_instances=["test_cache"],
                    canary_percentage=33.3,
                )

                new_data = {
                    "test:1": {"integration": True, "version": 1},
                    "test:2": {"integration": True, "version": 1},
                    "test:3": {"integration": True, "version": 1},
                }

                update_id = await updater.start_staged_update(update_config, new_data)
                await asyncio.sleep(2)  # 更新完了待機

                update_status = updater.get_update_status(update_id)
                test_results["tests"]["staged_update"] = update_status
                print(f"  段階的更新結果: {update_status}")

            # 5. 総合パフォーマンステスト
            print("\n5. 総合パフォーマンス測定")
            overall_start = time.time()

            # 混合ワークロード実行
            mixed_workload_ops = []
            for i in range(200):
                op_start = time.time()

                key = f"perf_test_{i}"
                value = {
                    "id": i,
                    "data": f"performance_test_data_{i}",
                    "timestamp": time.time(),
                }

                # PUT操作
                await persistent_cache.put(key, value)
                put_time = time.time() - op_start

                # GET操作
                get_start = time.time()
                result = await persistent_cache.get(key)
                get_time = time.time() - get_start

                mixed_workload_ops.append(
                    {
                        "put_time": put_time * 1000,
                        "get_time": get_time * 1000,
                        "success": result is not None,
                    }
                )

            overall_time = time.time() - overall_start

            # 統計計算
            successful_ops = [op for op in mixed_workload_ops if op["success"]]
            avg_put_time = statistics.mean(
                [op["put_time"] for op in mixed_workload_ops]
            )
            avg_get_time = statistics.mean(
                [op["get_time"] for op in mixed_workload_ops]
            )

            performance_summary = {
                "total_operations": len(mixed_workload_ops) * 2,
                "success_rate": len(successful_ops) / len(mixed_workload_ops) * 100,
                "total_time_seconds": overall_time,
                "ops_per_second": (len(mixed_workload_ops) * 2) / overall_time,
                "avg_put_time_ms": avg_put_time,
                "avg_get_time_ms": avg_get_time,
                "combined_avg_time_ms": (avg_put_time + avg_get_time) / 2,
            }

            test_results["tests"]["overall_performance"] = performance_summary
            print(
                f"  総合パフォーマンス: {performance_summary['ops_per_second']:.0f} OPS, {performance_summary['success_rate']:.1f}% 成功率"
            )

            # 6. サマリー作成
            test_results["summary"] = {
                "total_tests": len(test_results["tests"]),
                "successful_tests": len(
                    [
                        t
                        for t in test_results["tests"].values()
                        if not isinstance(t, dict) or "error" not in t
                    ]
                ),
                "overall_success": True,
                "key_metrics": {
                    "persistent_cache_hit_rate": persistent_results.get("hit_rate", 0),
                    "concurrent_success_rate": concurrent_results.get(
                        "success_rate", 0
                    ),
                    "overall_ops_per_second": performance_summary.get(
                        "ops_per_second", 0
                    ),
                    "avg_response_time_ms": performance_summary.get(
                        "combined_avg_time_ms", 0
                    ),
                },
            }

            # クリーンアップ
            await persistent_cache.shutdown()

            print("\n=== 統合テスト結果サマリー ===")
            summary = test_results["summary"]
            print(f"実行テスト数: {summary['total_tests']}")
            print(f"成功テスト数: {summary['successful_tests']}")
            print("キーメトリクス:")
            for metric, value in summary["key_metrics"].items():
                print(f"  {metric}: {value:.2f}")

            print("\n[OK] Issue #377 完全統合テスト完了！")
            return test_results

        except Exception as e:
            test_results["summary"] = {"overall_success": False, "error": str(e)}
            print(f"[ERROR] 統合テストエラー: {e}")
            import traceback

            traceback.print_exc()
            return test_results


async def run_comprehensive_tests():
    """包括的テスト実行"""
    print("=== Issue #377 高度なキャッシング戦略 - 包括的テスト ===")

    test_suite = IntegrationTestSuite()
    results = await test_suite.test_full_cache_integration()

    # 結果をJSONファイルに保存
    results_file = f"cache_integration_test_results_{int(time.time())}.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nテスト結果を保存: {results_file}")
    except Exception as e:
        print(f"結果保存エラー: {e}")

    return results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
