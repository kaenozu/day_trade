#!/usr/bin/env python3
"""
最適化されたパフォーマンステストスイート
現在システムの詳細改善・完成度向上フェーズ

システム全体のパフォーマンス分析・ボトルネック特定・最適化検証
"""

import asyncio
import gc
import json
import statistics
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import psutil

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput: float  # operations per second
    latency_p50: float
    latency_p95: float
    latency_p99: float
    success_rate: float
    error_count: int


@dataclass
class ResourceUsage:
    """リソース使用量"""

    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    network_io_mb: float
    file_descriptors: int
    thread_count: int


@dataclass
class PerformanceTestResult:
    """パフォーマンステスト結果"""

    timestamp: datetime
    test_suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_execution_time: float
    test_metrics: List[PerformanceMetrics]
    system_resources: ResourceUsage
    bottlenecks: List[str]
    optimization_recommendations: List[str]


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self.start_time = None
        self.memory_tracker = None
        self.cpu_samples = []
        self.latency_samples = []

    def start_profiling(self):
        """プロファイリング開始"""
        self.start_time = time.perf_counter()
        tracemalloc.start()
        gc.collect()

    def stop_profiling(self) -> Tuple[float, float, float]:
        """プロファイリング終了"""
        execution_time = time.perf_counter() - self.start_time if self.start_time else 0

        # メモリ使用量
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024

        # CPU使用量
        cpu_percent = psutil.cpu_percent(interval=0.1) if psutil else 0

        return execution_time, memory_mb, cpu_percent

    def add_latency_sample(self, latency: float):
        """レイテンシサンプル追加"""
        self.latency_samples.append(latency)

    def calculate_latency_percentiles(self) -> Tuple[float, float, float]:
        """レイテンシパーセンタイル計算"""
        if not self.latency_samples:
            return 0, 0, 0

        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)

        p50 = sorted_samples[int(n * 0.5)]
        p95 = sorted_samples[int(n * 0.95)]
        p99 = sorted_samples[int(n * 0.99)]

        return p50, p95, p99


class OptimizedPerformanceTestSuite:
    """最適化パフォーマンステストスイート"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.test_results = []
        self.system_baseline = None

        print("=" * 80)
        print("[PERF] 最適化パフォーマンステストスイート")
        print("現在システムの詳細改善・完成度向上フェーズ")
        print("=" * 80)

        # システムベースライン測定
        self._measure_system_baseline()

    def _measure_system_baseline(self):
        """システムベースライン測定"""
        print("[BASELINE] システムベースライン測定中...")

        if psutil:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_io_counters()
            network = psutil.net_io_counters()

            self.system_baseline = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory.used / 1024 / 1024,
                disk_io_mb=disk.read_bytes / 1024 / 1024 if disk else 0,
                network_io_mb=network.bytes_recv / 1024 / 1024 if network else 0,
                file_descriptors=(
                    len(psutil.Process().open_files())
                    if hasattr(psutil.Process(), "open_files")
                    else 0
                ),
                thread_count=threading.active_count(),
            )
        else:
            self.system_baseline = ResourceUsage(0, 0, 0, 0, 0, 0)

        print(
            f"[BASELINE] CPU: {self.system_baseline.cpu_percent:.1f}%, "
            f"Memory: {self.system_baseline.memory_mb:.0f}MB"
        )

    def run_performance_test(
        self, test_name: str, test_function: Callable, iterations: int = 100, **kwargs
    ) -> PerformanceMetrics:
        """パフォーマンステスト実行"""
        print(f"\n[TEST] {test_name} ({iterations}回実行)")

        # テスト準備
        gc.collect()
        success_count = 0
        error_count = 0
        execution_times = []

        # プロファイリング開始
        self.profiler.start_profiling()

        # テスト実行
        for i in range(iterations):
            iteration_start = time.perf_counter()

            try:
                result = test_function(**kwargs)
                success_count += 1

                iteration_time = time.perf_counter() - iteration_start
                execution_times.append(iteration_time)
                self.profiler.add_latency_sample(iteration_time)

            except Exception as e:
                error_count += 1
                print(f"[ERROR] イテレーション {i+1}: {e}")

        # プロファイリング終了
        total_time, memory_mb, cpu_percent = self.profiler.stop_profiling()

        # メトリクス計算
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        throughput = success_count / total_time if total_time > 0 else 0
        success_rate = (success_count / iterations) * 100

        p50, p95, p99 = self.profiler.calculate_latency_percentiles()

        metrics = PerformanceMetrics(
            test_name=test_name,
            execution_time=avg_execution_time,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            throughput=throughput,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            success_rate=success_rate,
            error_count=error_count,
        )

        print(f"[RESULT] {test_name}:")
        print(f"  平均実行時間: {avg_execution_time*1000:.2f}ms")
        print(f"  スループット: {throughput:.1f} ops/sec")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  P95レイテンシ: {p95*1000:.2f}ms")

        return metrics

    def test_data_loading_performance(self) -> PerformanceMetrics:
        """データ読み込み性能テスト"""

        def load_test_data():
            # 疑似データ生成
            import numpy as np
            import pandas as pd

            # 10,000行のテストデータ
            data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=10000, freq="1min"),
                    "price": np.random.uniform(100, 200, 10000),
                    "volume": np.random.randint(1000, 100000, 10000),
                }
            )

            # データ処理（ソート、フィルタリング、集計）
            processed = data.sort_values("timestamp")
            filtered = processed[processed["volume"] > 5000]
            aggregated = filtered.groupby(filtered["timestamp"].dt.hour).agg(
                {"price": ["mean", "max", "min"], "volume": "sum"}
            )

            return len(aggregated)

        return self.run_performance_test("データ読み込み・処理", load_test_data, iterations=50)

    def test_computation_performance(self) -> PerformanceMetrics:
        """計算性能テスト"""

        def computation_test():
            import numpy as np

            # 重い計算処理
            size = 1000
            matrix_a = np.random.rand(size, size)
            matrix_b = np.random.rand(size, size)

            # 行列乗算
            result = np.dot(matrix_a, matrix_b)

            # 統計計算
            stats = {
                "mean": np.mean(result),
                "std": np.std(result),
                "max": np.max(result),
                "min": np.min(result),
            }

            return stats["mean"]

        return self.run_performance_test("数値計算処理", computation_test, iterations=20)

    def test_concurrent_processing(self) -> PerformanceMetrics:
        """並行処理性能テスト"""

        def concurrent_test():
            def worker_task(x):
                # CPU集約的タスク
                total = 0
                for i in range(10000):
                    total += i * x
                return total % 1000000

            # スレッドプール実行
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker_task, i) for i in range(20)]
                results = [f.result() for f in futures]

            return sum(results)

        return self.run_performance_test("並行処理", concurrent_test, iterations=10)

    def test_memory_efficiency(self) -> PerformanceMetrics:
        """メモリ効率テスト"""

        def memory_test():
            # メモリ集約的処理
            data_sets = []

            # 大量データ生成・保持
            for i in range(100):
                data = list(range(10000))
                processed = [x * 2 + 1 for x in data if x % 2 == 0]
                data_sets.append(processed)

            # データ集約
            total = sum(sum(ds) for ds in data_sets)

            # メモリクリーンアップ
            del data_sets
            gc.collect()

            return total % 1000000

        return self.run_performance_test("メモリ効率", memory_test, iterations=20)

    def test_io_performance(self) -> PerformanceMetrics:
        """I/O性能テスト"""

        def io_test():
            import json
            import tempfile

            # 一時ファイル作成
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
                # JSON データ書き込み
                test_data = {
                    "records": [
                        {"id": i, "value": i * 1.5, "flag": i % 2 == 0} for i in range(1000)
                    ]
                }
                json.dump(test_data, f)
                temp_path = f.name

            # ファイル読み込み
            with open(temp_path) as f:
                loaded_data = json.load(f)

            # ファイル削除
            Path(temp_path).unlink()

            return len(loaded_data["records"])

        return self.run_performance_test("I/O処理", io_test, iterations=30)

    def test_async_performance(self) -> PerformanceMetrics:
        """非同期処理性能テスト"""

        async def async_test():
            async def async_task(delay, value):
                await asyncio.sleep(delay)
                return value * 2

            # 非同期タスク実行
            tasks = [async_task(0.001, i) for i in range(100)]
            results = await asyncio.gather(*tasks)

            return sum(results)

        def run_async_test():
            return asyncio.run(async_test())

        return self.run_performance_test("非同期処理", run_async_test, iterations=10)

    def analyze_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """ボトルネック分析"""
        bottlenecks = []

        # 実行時間分析
        slow_tests = [m for m in metrics if m.execution_time > 0.1]  # 100ms以上
        if slow_tests:
            bottlenecks.append(f"実行時間: {len(slow_tests)}個のテストが100ms以上")

        # メモリ使用量分析
        memory_heavy = [m for m in metrics if m.memory_usage_mb > 100]  # 100MB以上
        if memory_heavy:
            bottlenecks.append(f"メモリ使用量: {len(memory_heavy)}個のテストが100MB以上")

        # CPU使用率分析
        cpu_intensive = [m for m in metrics if m.cpu_usage_percent > 80]
        if cpu_intensive:
            bottlenecks.append(f"CPU使用率: {len(cpu_intensive)}個のテストが80%以上")

        # スループット分析
        low_throughput = [m for m in metrics if m.throughput < 10]  # 10 ops/sec未満
        if low_throughput:
            bottlenecks.append(f"スループット: {len(low_throughput)}個のテストが10ops/sec未満")

        # レイテンシ分析
        high_latency = [m for m in metrics if m.latency_p95 > 0.1]  # P95が100ms以上
        if high_latency:
            bottlenecks.append(f"レイテンシ: {len(high_latency)}個のテストでP95が100ms以上")

        return bottlenecks

    def generate_optimization_recommendations(
        self, metrics: List[PerformanceMetrics], bottlenecks: List[str]
    ) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []

        # ボトルネックベース推奨事項
        if any("実行時間" in b for b in bottlenecks):
            recommendations.extend(
                [
                    "アルゴリズムの最適化（O(n²)からO(nlogn)への改善等）",
                    "キャッシュ戦略の導入",
                    "処理の並列化検討",
                ]
            )

        if any("メモリ使用量" in b for b in bottlenecks):
            recommendations.extend(
                [
                    "メモリ効率的なデータ構造の使用",
                    "ストリーミング処理への移行",
                    "オブジェクトプールの導入",
                ]
            )

        if any("CPU使用率" in b for b in bottlenecks):
            recommendations.extend(
                ["CPU集約的処理の最適化", "マルチプロセシング活用", "非同期処理の導入"]
            )

        if any("スループット" in b for b in bottlenecks):
            recommendations.extend(["バッチ処理の導入", "接続プールの最適化", "I/O待機の削減"])

        if any("レイテンシ" in b for b in bottlenecks):
            recommendations.extend(
                [
                    "レスポンス時間の最適化",
                    "キャッシュ戦略の見直し",
                    "データベースクエリ最適化",
                ]
            )

        # 一般的推奨事項
        recommendations.extend(
            [
                "プロファイリングツールによる詳細分析",
                "継続的パフォーマンス監視の導入",
                "負荷テストの定期実行",
                "コードレビューでのパフォーマンス観点追加",
            ]
        )

        return recommendations[:10]  # 上位10項目

    def run_comprehensive_performance_test(self) -> PerformanceTestResult:
        """包括的パフォーマンステスト実行"""
        start_time = time.time()

        print("\n[START] 包括的パフォーマンステスト開始")

        # テスト実行
        test_methods = [
            self.test_data_loading_performance,
            self.test_computation_performance,
            self.test_concurrent_processing,
            self.test_memory_efficiency,
            self.test_io_performance,
            self.test_async_performance,
        ]

        test_metrics = []
        passed_tests = 0
        failed_tests = 0

        for test_method in test_methods:
            try:
                metrics = test_method()
                test_metrics.append(metrics)

                if metrics.success_rate >= 95:  # 95%以上成功率
                    passed_tests += 1
                else:
                    failed_tests += 1

            except Exception as e:
                failed_tests += 1
                print(f"[ERROR] テスト失敗: {test_method.__name__}: {e}")

        # システムリソース測定
        if psutil:
            current_process = psutil.Process()
            system_resources = ResourceUsage(
                cpu_percent=psutil.cpu_percent(),
                memory_mb=current_process.memory_info().rss / 1024 / 1024,
                disk_io_mb=0,  # 簡略化
                network_io_mb=0,  # 簡略化
                file_descriptors=(
                    len(current_process.open_files())
                    if hasattr(current_process, "open_files")
                    else 0
                ),
                thread_count=current_process.num_threads(),
            )
        else:
            system_resources = ResourceUsage(0, 0, 0, 0, 0, 0)

        # 分析
        bottlenecks = self.analyze_bottlenecks(test_metrics)
        recommendations = self.generate_optimization_recommendations(test_metrics, bottlenecks)

        total_time = time.time() - start_time

        result = PerformanceTestResult(
            timestamp=datetime.now(),
            test_suite_name="包括的パフォーマンステスト",
            total_tests=len(test_methods),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_execution_time=total_time,
            test_metrics=test_metrics,
            system_resources=system_resources,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
        )

        print(f"\n[COMPLETE] テスト完了 ({total_time:.2f}秒)")
        print(f"成功: {passed_tests}/{len(test_methods)}")
        print(f"失敗: {failed_tests}/{len(test_methods)}")

        return result

    def generate_performance_report(self, result: PerformanceTestResult) -> str:
        """パフォーマンスレポート生成"""
        report = f"""
=== パフォーマンステストレポート ===
実行時刻: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
テストスイート: {result.test_suite_name}
総実行時間: {result.total_execution_time:.2f}秒

=== テスト結果サマリー ===
総テスト数: {result.total_tests}
成功: {result.passed_tests}
失敗: {result.failed_tests}
成功率: {(result.passed_tests/result.total_tests)*100:.1f}%

=== パフォーマンスメトリクス ==="""

        for metrics in result.test_metrics:
            report += f"""
{metrics.test_name}:
  実行時間: {metrics.execution_time*1000:.2f}ms
  メモリ使用量: {metrics.memory_usage_mb:.1f}MB
  スループット: {metrics.throughput:.1f} ops/sec
  P95レイテンシ: {metrics.latency_p95*1000:.2f}ms
  成功率: {metrics.success_rate:.1f}%"""

        report += f"""

=== システムリソース ===
CPU使用率: {result.system_resources.cpu_percent:.1f}%
メモリ使用量: {result.system_resources.memory_mb:.1f}MB
スレッド数: {result.system_resources.thread_count}

=== 検出されたボトルネック ==="""

        for bottleneck in result.bottlenecks:
            report += f"\n- {bottleneck}"

        report += "\n\n=== 最適化推奨事項（上位5項目）==="
        for rec in result.optimization_recommendations[:5]:
            report += f"\n- {rec}"

        return report

    def save_performance_report(self, result: PerformanceTestResult, filename: str = None) -> str:
        """パフォーマンスレポート保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_report_{timestamp}.json"

        # JSON用データ変換
        report_data = asdict(result)
        report_data["timestamp"] = result.timestamp.isoformat()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return f"パフォーマンスレポート保存完了: {filename}"


def main():
    """メイン実行"""
    test_suite = OptimizedPerformanceTestSuite()

    try:
        # 包括的パフォーマンステスト実行
        result = test_suite.run_comprehensive_performance_test()

        # レポート表示（エンコーディング対応）
        try:
            print("\n" + "=" * 80)
            report = test_suite.generate_performance_report(result)
            print(report.encode("ascii", "replace").decode("ascii"))
            print("=" * 80)
        except UnicodeEncodeError:
            print("\n[SUMMARY] パフォーマンステスト完了")
            print(f"総テスト数: {result.total_tests}")
            print(f"成功: {result.passed_tests}/{result.total_tests}")
            print(f"実行時間: {result.total_execution_time:.2f}秒")
            print("詳細は保存されたJSONレポートをご確認ください")
            print("=" * 80)

        # レポート保存
        saved_file = test_suite.save_performance_report(result)
        print(f"\n[REPORT] {saved_file}")

    except KeyboardInterrupt:
        print("\n[STOP] パフォーマンステスト中断")
    except Exception as e:
        print(f"\n[ERROR] テストエラー: {e}")


if __name__ == "__main__":
    main()
