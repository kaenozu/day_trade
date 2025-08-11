#!/usr/bin/env python3
"""
キャッシュストレステスト
Issue #377: 高度なキャッシング戦略の負荷テスト

高負荷状況でのキャッシュシステム安定性・性能テスト
"""

import gc
import json
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

# プロジェクトモジュール
try:
    from ...cache.persistent_cache_system import get_persistent_cache
    from ...data.enhanced_stock_fetcher import create_enhanced_stock_fetcher
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # 簡易モックフォールバック
    def create_enhanced_stock_fetcher(**kwargs):
        class MockFetcher:
            def get_current_price(self, code):
                time.sleep(0.05)  # API遅延シミュレート
                return {"price": random.uniform(100, 1000), "timestamp": time.time()}

            def get_cache_stats(self):
                return {"hits": 100, "misses": 50, "hit_rate": 0.67}

        return MockFetcher()

    def get_persistent_cache(**kwargs):
        return None


logger = get_context_logger(__name__)


@dataclass
class StressTestResult:
    """ストレステスト結果"""

    test_name: str
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    avg_response_time_ms: float
    max_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    concurrent_users: int
    cache_hit_rate: Optional[float] = None
    errors: List[str] = None


class StressTestRunner:
    """ストレステスト実行クラス"""

    def __init__(self, results_dir: str = "stress_test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        self.stop_event = threading.Event()

        # テスト用株式コード
        self.test_codes = [f"{random.randint(1000, 9999)}" for _ in range(500)]

        logger.info(f"ストレステスト実行準備完了: {results_dir}")

    def monitor_system_resources(self, duration_seconds: int) -> Dict[str, float]:
        """システムリソース監視"""
        process = psutil.Process()
        cpu_samples = []
        memory_samples = []

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            try:
                cpu_samples.append(process.cpu_percent())
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"リソース監視エラー: {e}")

        return {
            "avg_cpu_percent": statistics.mean(cpu_samples) if cpu_samples else 0,
            "max_cpu_percent": max(cpu_samples) if cpu_samples else 0,
            "avg_memory_mb": statistics.mean(memory_samples) if memory_samples else 0,
            "max_memory_mb": max(memory_samples) if memory_samples else 0,
        }

    def stress_test_single_thread(
        self, duration_seconds: int = 60, operations_per_second: int = 10
    ) -> StressTestResult:
        """シングルスレッドストレステスト"""
        logger.info(
            f"シングルスレッドストレステスト開始: {duration_seconds}秒間, {operations_per_second}OPS"
        )

        # Enhanced Stock Fetcher準備
        fetcher = create_enhanced_stock_fetcher(
            cache_config={
                "persistent_cache_enabled": True,
                "enable_multi_layer_cache": True,
                "l1_memory_size": 1000,
            }
        )

        # カウンター初期化
        total_ops = 0
        successful_ops = 0
        failed_ops = 0
        response_times = []
        errors = []

        start_time = time.time()
        interval = 1.0 / operations_per_second

        # リソース監視開始
        resource_monitor = threading.Thread(
            target=lambda: self.monitor_system_resources(duration_seconds), daemon=True
        )
        resource_stats = {}

        try:
            while time.time() - start_time < duration_seconds:
                operation_start = time.perf_counter()

                try:
                    code = random.choice(self.test_codes)
                    result = fetcher.get_current_price(code)

                    operation_end = time.perf_counter()
                    response_times.append((operation_end - operation_start) * 1000)

                    if result:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                        errors.append("No result returned")

                except Exception as e:
                    failed_ops += 1
                    errors.append(str(e))
                    logger.debug(f"操作エラー: {e}")

                total_ops += 1

                # レート制御
                elapsed = time.perf_counter() - operation_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            logger.info("ストレステスト中断")

        total_duration = time.time() - start_time

        # キャッシュ統計取得
        cache_hit_rate = None
        try:
            cache_stats = fetcher.get_cache_stats()
            cache_hit_rate = cache_stats.get("performance_stats", {}).get(
                "cache_hit_rate", 0
            )
        except Exception:
            pass

        # システムリソース取得
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()

        result = StressTestResult(
            test_name="シングルスレッド",
            duration_seconds=total_duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            operations_per_second=total_ops / total_duration,
            avg_response_time_ms=statistics.mean(response_times)
            if response_times
            else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            error_rate=failed_ops / total_ops if total_ops > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            concurrent_users=1,
            cache_hit_rate=cache_hit_rate,
            errors=list(set(errors)),  # 重複除去
        )

        self.results.append(result)
        logger.info(
            f"シングルスレッドテスト完了: {successful_ops}/{total_ops} 成功, OPS={result.operations_per_second:.1f}"
        )

        return result

    def stress_test_multi_thread(
        self,
        duration_seconds: int = 60,
        concurrent_threads: int = 10,
        operations_per_thread: int = 5,
    ) -> StressTestResult:
        """マルチスレッドストレステスト"""
        logger.info(
            f"マルチスレッドストレステスト開始: {concurrent_threads}スレッド, {duration_seconds}秒間"
        )

        # 共有結果収集
        results_queue = Queue()
        stop_event = threading.Event()

        def worker_thread(thread_id: int):
            """ワーカースレッド"""
            fetcher = create_enhanced_stock_fetcher(
                cache_config={
                    "persistent_cache_enabled": True,
                    "enable_multi_layer_cache": True,
                    "l1_memory_size": 500,
                }
            )

            local_stats = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "response_times": [],
                "errors": [],
            }

            interval = 1.0 / operations_per_thread
            thread_start = time.time()

            while (
                not stop_event.is_set()
                and (time.time() - thread_start) < duration_seconds
            ):
                operation_start = time.perf_counter()

                try:
                    code = random.choice(self.test_codes)
                    result = fetcher.get_current_price(code)

                    operation_end = time.perf_counter()
                    local_stats["response_times"].append(
                        (operation_end - operation_start) * 1000
                    )

                    if result:
                        local_stats["success"] += 1
                    else:
                        local_stats["failed"] += 1
                        local_stats["errors"].append("No result")

                except Exception as e:
                    local_stats["failed"] += 1
                    local_stats["errors"].append(str(e))

                local_stats["total"] += 1

                # レート制御
                elapsed = time.perf_counter() - operation_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            # 結果をキューに送信
            results_queue.put((thread_id, local_stats))

        # スレッド開始
        start_time = time.time()
        threads = []

        for i in range(concurrent_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # 指定時間待機
        time.sleep(duration_seconds)
        stop_event.set()

        # スレッド終了待機
        for thread in threads:
            thread.join(timeout=5)

        total_duration = time.time() - start_time

        # 結果集約
        total_ops = 0
        successful_ops = 0
        failed_ops = 0
        all_response_times = []
        all_errors = []

        while not results_queue.empty():
            try:
                thread_id, stats = results_queue.get_nowait()
                total_ops += stats["total"]
                successful_ops += stats["success"]
                failed_ops += stats["failed"]
                all_response_times.extend(stats["response_times"])
                all_errors.extend(stats["errors"])
            except Empty:
                break

        # システムリソース
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()

        result = StressTestResult(
            test_name=f"マルチスレッド_{concurrent_threads}",
            duration_seconds=total_duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            operations_per_second=total_ops / total_duration,
            avg_response_time_ms=statistics.mean(all_response_times)
            if all_response_times
            else 0,
            max_response_time_ms=max(all_response_times) if all_response_times else 0,
            error_rate=failed_ops / total_ops if total_ops > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            concurrent_users=concurrent_threads,
            errors=list(set(all_errors)),
        )

        self.results.append(result)
        logger.info(
            f"マルチスレッドテスト完了: {successful_ops}/{total_ops} 成功, OPS={result.operations_per_second:.1f}"
        )

        return result

    def stress_test_memory_pressure(
        self, duration_seconds: int = 30, memory_pressure_mb: int = 100
    ) -> StressTestResult:
        """メモリプレッシャーストレステスト"""
        logger.info(
            f"メモリプレッシャーテスト開始: {memory_pressure_mb}MB圧迫, {duration_seconds}秒間"
        )

        # メモリプレッシャー生成（大量のダミーデータ）
        memory_hogs = []

        try:
            # 指定されたメモリ量のダミーデータ作成
            chunk_size = 10  # 10MBずつ
            chunks = memory_pressure_mb // chunk_size

            for i in range(chunks):
                # 10MBのランダムデータ
                dummy_data = np.random.bytes(chunk_size * 1024 * 1024)
                memory_hogs.append(dummy_data)

            logger.info(
                f"メモリプレッシャー生成完了: {len(memory_hogs) * chunk_size}MB"
            )

            # Enhanced Stock Fetcher（小さなキャッシュサイズ）
            fetcher = create_enhanced_stock_fetcher(
                cache_config={
                    "persistent_cache_enabled": True,
                    "l1_memory_size": 100,  # 小さなメモリキャッシュ
                    "enable_multi_layer_cache": True,
                }
            )

            # ストレステスト実行
            total_ops = 0
            successful_ops = 0
            failed_ops = 0
            response_times = []
            errors = []

            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                operation_start = time.perf_counter()

                try:
                    code = random.choice(self.test_codes)
                    result = fetcher.get_current_price(code)

                    operation_end = time.perf_counter()
                    response_times.append((operation_end - operation_start) * 1000)

                    if result:
                        successful_ops += 1
                    else:
                        failed_ops += 1
                        errors.append("No result")

                except Exception as e:
                    failed_ops += 1
                    errors.append(str(e))

                total_ops += 1
                time.sleep(0.1)  # 10OPS制限

            total_duration = time.time() - start_time

            # システムリソース
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()

            result = StressTestResult(
                test_name=f"メモリプレッシャー_{memory_pressure_mb}MB",
                duration_seconds=total_duration,
                total_operations=total_ops,
                successful_operations=successful_ops,
                failed_operations=failed_ops,
                operations_per_second=total_ops / total_duration,
                avg_response_time_ms=statistics.mean(response_times)
                if response_times
                else 0,
                max_response_time_ms=max(response_times) if response_times else 0,
                error_rate=failed_ops / total_ops if total_ops > 0 else 0,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                concurrent_users=1,
                errors=list(set(errors)),
            )

            self.results.append(result)
            logger.info(
                f"メモリプレッシャーテスト完了: {successful_ops}/{total_ops} 成功"
            )

        finally:
            # メモリ解放
            memory_hogs.clear()
            gc.collect()

        return result

    def stress_test_cache_invalidation(
        self, duration_seconds: int = 60
    ) -> StressTestResult:
        """キャッシュ無効化ストレステスト"""
        logger.info(f"キャッシュ無効化ストレステスト開始: {duration_seconds}秒間")

        fetcher = create_enhanced_stock_fetcher(
            cache_config={
                "persistent_cache_enabled": True,
                "smart_invalidation_enabled": True,
                "enable_multi_layer_cache": True,
                "l1_memory_size": 1000,
            }
        )

        # 初期データでキャッシュをウォームアップ
        for code in self.test_codes[:100]:
            fetcher.get_current_price(code)

        total_ops = 0
        successful_ops = 0
        failed_ops = 0
        response_times = []
        errors = []

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            operation_start = time.perf_counter()

            try:
                # 50%の確率でキャッシュ無効化
                if random.random() < 0.5:
                    # 既存のコードでキャッシュヒット狙い
                    code = random.choice(self.test_codes[:100])
                else:
                    # 新しいコードでキャッシュミス発生
                    code = f"{random.randint(10000, 99999)}"

                result = fetcher.get_current_price(code)

                # 10%の確率で特定銘柄のキャッシュ無効化
                if random.random() < 0.1:
                    invalidate_code = random.choice(self.test_codes[:50])
                    fetcher.invalidate_symbol_cache(invalidate_code)

                operation_end = time.perf_counter()
                response_times.append((operation_end - operation_start) * 1000)

                if result:
                    successful_ops += 1
                else:
                    failed_ops += 1
                    errors.append("No result")

            except Exception as e:
                failed_ops += 1
                errors.append(str(e))

            total_ops += 1
            time.sleep(0.05)  # 20OPS

        total_duration = time.time() - start_time

        # キャッシュ統計
        cache_hit_rate = None
        try:
            cache_stats = fetcher.get_cache_stats()
            cache_hit_rate = cache_stats.get("performance_stats", {}).get(
                "cache_hit_rate", 0
            )
        except Exception:
            pass

        # システムリソース
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()

        result = StressTestResult(
            test_name="キャッシュ無効化",
            duration_seconds=total_duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            operations_per_second=total_ops / total_duration,
            avg_response_time_ms=statistics.mean(response_times)
            if response_times
            else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            error_rate=failed_ops / total_ops if total_ops > 0 else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            concurrent_users=1,
            cache_hit_rate=cache_hit_rate,
            errors=list(set(errors)),
        )

        self.results.append(result)
        logger.info(f"キャッシュ無効化テスト完了: ヒット率={cache_hit_rate:.2%}")

        return result

    def generate_stress_report(self, output_file: str = None) -> Dict[str, Any]:
        """ストレステスト結果レポート生成"""
        if not self.results:
            logger.warning("ストレステスト結果がありません")
            return {}

        # 結果をDataFrameに変換
        df = pd.DataFrame([asdict(r) for r in self.results])

        report = {
            "stress_test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.results),
            "test_summary": {},
            "stability_analysis": {},
            "resource_usage": {},
            "error_analysis": {},
            "recommendations": [],
        }

        # テストサマリー
        report["test_summary"] = {
            "best_performance_ops": df["operations_per_second"].max(),
            "worst_performance_ops": df["operations_per_second"].min(),
            "avg_error_rate": df["error_rate"].mean(),
            "max_error_rate": df["error_rate"].max(),
            "avg_response_time": df["avg_response_time_ms"].mean(),
            "max_response_time": df["max_response_time_ms"].max(),
        }

        # 安定性分析
        stable_tests = df[df["error_rate"] < 0.01]  # エラー率1%未満
        report["stability_analysis"] = {
            "stable_tests_count": len(stable_tests),
            "stability_rate": len(stable_tests) / len(df) * 100,
            "most_stable_test": stable_tests.loc[
                stable_tests["error_rate"].idxmin(), "test_name"
            ]
            if not stable_tests.empty
            else None,
            "least_stable_test": df.loc[df["error_rate"].idxmax(), "test_name"],
        }

        # リソース使用量
        report["resource_usage"] = {
            "max_memory_usage": df["memory_usage_mb"].max(),
            "avg_memory_usage": df["memory_usage_mb"].mean(),
            "max_cpu_usage": df["cpu_usage_percent"].max(),
            "avg_cpu_usage": df["cpu_usage_percent"].mean(),
            "high_resource_tests": df[df["memory_usage_mb"] > 100][
                "test_name"
            ].tolist(),
        }

        # エラー分析
        all_errors = []
        for result in self.results:
            if result.errors:
                all_errors.extend(result.errors)

        if all_errors:
            error_counts = {}
            for error in all_errors:
                error_counts[error] = error_counts.get(error, 0) + 1

            report["error_analysis"] = {
                "total_unique_errors": len(error_counts),
                "most_common_errors": sorted(
                    error_counts.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }

        # 推奨事項
        recommendations = []

        if df["error_rate"].mean() > 0.05:
            recommendations.append(
                "エラー率が高いです。キャッシュサイズの拡張またはタイムアウト設定の見直しを検討してください。"
            )

        if df["max_response_time_ms"].max() > 5000:
            recommendations.append(
                "応答時間が長すぎます。キャッシュヒット率の改善またはAPIタイムアウトの調整を検討してください。"
            )

        if df["memory_usage_mb"].max() > 200:
            recommendations.append(
                "メモリ使用量が多いです。キャッシュサイズ制限またはデータ圧縮を検討してください。"
            )

        multithread_results = df[df["test_name"].str.contains("マルチスレッド")]
        if not multithread_results.empty:
            single_ops = df[df["concurrent_users"] == 1]["operations_per_second"].max()
            multi_ops = multithread_results["operations_per_second"].max()

            if multi_ops < single_ops * 2:  # スケーラビリティが悪い
                recommendations.append(
                    "並行処理のスケーラビリティが低いです。ロック競合の軽減を検討してください。"
                )

        report["recommendations"] = recommendations

        # 詳細結果
        report["detailed_results"] = df.to_dict("records")

        # ファイル出力
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"ストレステスト結果保存: {output_path}")

        return report

    def run_full_stress_test(self) -> Dict[str, Any]:
        """フルストレステスト実行"""
        logger.info("=== フルキャッシュストレステスト開始 ===")

        try:
            # 各種ストレステスト実行
            logger.info("1. シングルスレッドストレステスト")
            self.stress_test_single_thread(
                duration_seconds=30, operations_per_second=20
            )

            logger.info("2. マルチスレッドストレステスト")
            self.stress_test_multi_thread(duration_seconds=30, concurrent_threads=5)
            self.stress_test_multi_thread(duration_seconds=30, concurrent_threads=10)

            logger.info("3. メモリプレッシャーテスト")
            self.stress_test_memory_pressure(duration_seconds=20, memory_pressure_mb=50)

            logger.info("4. キャッシュ無効化テスト")
            self.stress_test_cache_invalidation(duration_seconds=30)

            # レポート生成
            logger.info("5. ストレステストレポート生成")
            report = self.generate_stress_report("cache_stress_test_report.json")

            logger.info("=== フルストレステスト完了 ===")
            return report

        except Exception as e:
            logger.error(f"ストレステスト実行エラー: {e}")
            return {"error": str(e)}


def run_cache_stress_test():
    """キャッシュストレステストメイン関数"""
    print("=== Issue #377 キャッシュストレステスト ===")

    runner = StressTestRunner()
    report = runner.run_full_stress_test()

    # 結果表示
    if "error" not in report:
        print("\n🔥 ストレステスト結果サマリー:")
        print(f"実行時刻: {report.get('stress_test_timestamp')}")
        print(f"総テスト数: {report.get('total_tests')}")

        # パフォーマンス
        summary = report.get("test_summary", {})
        print("\n⚡ パフォーマンス:")
        print(f"  最高性能: {summary.get('best_performance_ops', 0):.1f} OPS")
        print(f"  最低性能: {summary.get('worst_performance_ops', 0):.1f} OPS")
        print(f"  平均応答時間: {summary.get('avg_response_time', 0):.2f}ms")
        print(f"  最大応答時間: {summary.get('max_response_time', 0):.2f}ms")

        # 安定性
        stability = report.get("stability_analysis", {})
        print("\n🛡️ 安定性:")
        print(f"  安定テスト率: {stability.get('stability_rate', 0):.1f}%")
        print(f"  平均エラー率: {summary.get('avg_error_rate', 0):.2%}")
        print(f"  最大エラー率: {summary.get('max_error_rate', 0):.2%}")

        # リソース使用量
        resources = report.get("resource_usage", {})
        print("\n💾 リソース使用量:")
        print(f"  最大メモリ: {resources.get('max_memory_usage', 0):.1f}MB")
        print(f"  平均CPU: {resources.get('avg_cpu_usage', 0):.1f}%")

        # 推奨事項
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("\n✅ すべてのテストで良好な結果です")

        print(f"\n📄 詳細レポート: {runner.results_dir}/cache_stress_test_report.json")
    else:
        print(f"❌ ストレステスト実行エラー: {report['error']}")

    print("\n=== キャッシュストレステスト完了 ===")
    return report


if __name__ == "__main__":
    run_cache_stress_test()
