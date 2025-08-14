#!/usr/bin/env python3
"""
パフォーマンスベンチマーク
Day Trade ML System 性能基準測定・最適化

統合対象:
- Issue #487: EnsembleSystem (93%精度) パフォーマンス
- Issue #755: テスト体制 パフォーマンス検証
- Issue #800: 本番環境 スケーラビリティ
"""

import os
import sys
import time
import json
import logging
import asyncio
import psutil
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    test_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    system_state: Dict[str, float]
    details: Optional[Dict] = None

@dataclass
class SystemResourceMetrics:
    """システムリソースメトリクス"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: datetime

class PerformanceBenchmark:
    """パフォーマンスベンチマーク"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.resource_metrics: List[SystemResourceMetrics] = []
        self.monitoring_active = False

        # ベンチマーク設定
        self.benchmark_config = {
            'ml_prediction_samples': [100, 500, 1000, 2000, 5000],
            'concurrent_users': [1, 5, 10, 25, 50, 100],
            'data_sizes': [1000, 5000, 10000, 50000, 100000],  # records
            'test_duration_seconds': 300,
            'resource_monitoring_interval': 1.0
        }

        # パフォーマンス目標
        self.performance_targets = {
            'ml_prediction_latency_ms': 500,
            'ml_prediction_throughput_rps': 100,
            'data_processing_latency_ms': 1000,
            'memory_usage_percent': 80,
            'cpu_usage_percent': 70,
            'disk_io_threshold_mbps': 100
        }

    async def run_comprehensive_benchmark(self) -> Dict:
        """包括的ベンチマーク実行"""
        logger.info("Starting comprehensive performance benchmark...")

        benchmark_start = datetime.utcnow()

        # リソース監視開始
        self._start_resource_monitoring()

        try:
            # 1. ML予測性能ベンチマーク
            ml_results = await self._benchmark_ml_performance()

            # 2. データ処理性能ベンチマーク
            data_results = await self._benchmark_data_processing()

            # 3. スケーラビリティベンチマーク
            scalability_results = await self._benchmark_scalability()

            # 4. メモリ使用量ベンチマーク
            memory_results = await self._benchmark_memory_usage()

            # 5. I/O性能ベンチマーク
            io_results = await self._benchmark_io_performance()

            # 6. 並行処理性能ベンチマーク
            concurrency_results = await self._benchmark_concurrency()

        finally:
            # リソース監視停止
            self._stop_resource_monitoring()

        benchmark_duration = (datetime.utcnow() - benchmark_start).total_seconds()

        # 結果統合
        all_results = (ml_results + data_results + scalability_results +
                      memory_results + io_results + concurrency_results)

        # 統計サマリー
        summary = self._generate_performance_summary(all_results)

        benchmark_report = {
            'benchmark_start': benchmark_start.isoformat(),
            'benchmark_duration_seconds': benchmark_duration,
            'total_tests': len(all_results),
            'results': [asdict(result) for result in all_results],
            'resource_metrics': [asdict(metric) for metric in self.resource_metrics],
            'summary': summary,
            'recommendations': self._generate_optimization_recommendations(summary),
            'performance_grade': self._calculate_overall_grade(summary)
        }

        return benchmark_report

    async def _benchmark_ml_performance(self) -> List[BenchmarkResult]:
        """ML性能ベンチマーク"""
        logger.info("Benchmarking ML prediction performance...")

        results = []

        for sample_size in self.benchmark_config['ml_prediction_samples']:
            # レイテンシ測定
            latencies = []

            for _ in range(sample_size):
                start_time = time.time()

                # ML予測実行（模擬）
                await self._simulate_ml_prediction()

                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)

            # 統計計算
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = sample_size / (sum(latencies) / 1000)  # RPS

            # 結果記録
            results.extend([
                BenchmarkResult(
                    test_name="ml_prediction_performance",
                    metric_name="avg_latency",
                    value=avg_latency,
                    unit="ms",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'sample_size': sample_size}
                ),
                BenchmarkResult(
                    test_name="ml_prediction_performance",
                    metric_name="p95_latency",
                    value=p95_latency,
                    unit="ms",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'sample_size': sample_size}
                ),
                BenchmarkResult(
                    test_name="ml_prediction_performance",
                    metric_name="throughput",
                    value=throughput,
                    unit="rps",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'sample_size': sample_size}
                )
            ])

            logger.info(f"ML Performance - Sample Size: {sample_size}, "
                       f"Avg Latency: {avg_latency:.2f}ms, "
                       f"Throughput: {throughput:.2f} RPS")

        return results

    async def _benchmark_data_processing(self) -> List[BenchmarkResult]:
        """データ処理性能ベンチマーク"""
        logger.info("Benchmarking data processing performance...")

        results = []

        for data_size in self.benchmark_config['data_sizes']:
            # データ生成
            test_data = self._generate_test_data(data_size)

            # データ処理時間測定
            start_time = time.time()

            # データ前処理実行（模擬）
            processed_data = await self._simulate_data_processing(test_data)

            processing_time = (time.time() - start_time) * 1000  # ms
            throughput = data_size / (processing_time / 1000)  # records/sec

            results.extend([
                BenchmarkResult(
                    test_name="data_processing_performance",
                    metric_name="processing_latency",
                    value=processing_time,
                    unit="ms",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'data_size': data_size}
                ),
                BenchmarkResult(
                    test_name="data_processing_performance",
                    metric_name="throughput",
                    value=throughput,
                    unit="records/sec",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'data_size': data_size}
                )
            ])

            logger.info(f"Data Processing - Size: {data_size}, "
                       f"Latency: {processing_time:.2f}ms, "
                       f"Throughput: {throughput:.2f} records/sec")

        return results

    async def _benchmark_scalability(self) -> List[BenchmarkResult]:
        """スケーラビリティベンチマーク"""
        logger.info("Benchmarking system scalability...")

        results = []

        for concurrent_users in self.benchmark_config['concurrent_users']:
            # 並行実行
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                # 並行タスク実行
                futures = [
                    executor.submit(self._execute_concurrent_task)
                    for _ in range(concurrent_users)
                ]

                # 完了待機
                completed_tasks = 0
                total_response_time = 0

                for future in as_completed(futures):
                    try:
                        task_duration = future.result()
                        completed_tasks += 1
                        total_response_time += task_duration
                    except Exception as e:
                        logger.warning(f"Concurrent task failed: {str(e)}")

            total_time = time.time() - start_time
            avg_response_time = total_response_time / completed_tasks if completed_tasks > 0 else float('inf')
            throughput = completed_tasks / total_time

            results.extend([
                BenchmarkResult(
                    test_name="scalability_test",
                    metric_name="avg_response_time",
                    value=avg_response_time * 1000,  # ms
                    unit="ms",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'concurrent_users': concurrent_users}
                ),
                BenchmarkResult(
                    test_name="scalability_test",
                    metric_name="throughput",
                    value=throughput,
                    unit="tasks/sec",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'concurrent_users': concurrent_users}
                )
            ])

            logger.info(f"Scalability - Users: {concurrent_users}, "
                       f"Avg Response: {avg_response_time*1000:.2f}ms, "
                       f"Throughput: {throughput:.2f} tasks/sec")

        return results

    async def _benchmark_memory_usage(self) -> List[BenchmarkResult]:
        """メモリ使用量ベンチマーク"""
        logger.info("Benchmarking memory usage...")

        results = []

        # ベースラインメモリ使用量
        baseline_memory = psutil.virtual_memory().percent

        # 負荷段階的増加
        load_levels = [1, 5, 10, 20, 50]

        for load_level in load_levels:
            # メモリ負荷生成
            memory_intensive_data = []

            for _ in range(load_level):
                # 大きなデータ構造作成
                large_array = np.random.rand(100000, 10)
                memory_intensive_data.append(large_array)

            # メモリ使用量測定
            current_memory = psutil.virtual_memory().percent
            memory_increase = current_memory - baseline_memory

            results.append(
                BenchmarkResult(
                    test_name="memory_usage_test",
                    metric_name="memory_usage",
                    value=current_memory,
                    unit="percent",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'load_level': load_level, 'memory_increase': memory_increase}
                )
            )

            logger.info(f"Memory Usage - Load Level: {load_level}, "
                       f"Usage: {current_memory:.1f}%, "
                       f"Increase: {memory_increase:.1f}%")

            # メモリクリーンアップ
            del memory_intensive_data
            time.sleep(1)

        return results

    async def _benchmark_io_performance(self) -> List[BenchmarkResult]:
        """I/O性能ベンチマーク"""
        logger.info("Benchmarking I/O performance...")

        results = []

        # ディスクI/O測定
        test_file_sizes = [1, 10, 100, 500, 1000]  # MB

        for size_mb in test_file_sizes:
            # 書き込み性能測定
            test_data = os.urandom(size_mb * 1024 * 1024)
            test_file = f"benchmark_test_{size_mb}mb.tmp"

            # 書き込み時間測定
            start_time = time.time()
            with open(test_file, 'wb') as f:
                f.write(test_data)
            write_time = time.time() - start_time
            write_throughput = size_mb / write_time  # MB/s

            # 読み込み時間測定
            start_time = time.time()
            with open(test_file, 'rb') as f:
                _ = f.read()
            read_time = time.time() - start_time
            read_throughput = size_mb / read_time  # MB/s

            # ファイル削除
            try:
                os.remove(test_file)
            except:
                pass

            results.extend([
                BenchmarkResult(
                    test_name="io_performance_test",
                    metric_name="write_throughput",
                    value=write_throughput,
                    unit="MB/s",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'file_size_mb': size_mb}
                ),
                BenchmarkResult(
                    test_name="io_performance_test",
                    metric_name="read_throughput",
                    value=read_throughput,
                    unit="MB/s",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'file_size_mb': size_mb}
                )
            ])

            logger.info(f"I/O Performance - Size: {size_mb}MB, "
                       f"Write: {write_throughput:.2f} MB/s, "
                       f"Read: {read_throughput:.2f} MB/s")

        return results

    async def _benchmark_concurrency(self) -> List[BenchmarkResult]:
        """並行処理性能ベンチマーク"""
        logger.info("Benchmarking concurrency performance...")

        results = []

        # CPU集約的タスクのベンチマーク
        cpu_task_sizes = [1000, 5000, 10000, 50000]

        for task_size in cpu_task_sizes:
            # シーケンシャル実行
            start_time = time.time()
            await self._cpu_intensive_task(task_size)
            sequential_time = time.time() - start_time

            # 並行実行（CPUコア数に応じて）
            num_cores = psutil.cpu_count()
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = [
                    executor.submit(self._cpu_intensive_task_sync, task_size // num_cores)
                    for _ in range(num_cores)
                ]

                for future in as_completed(futures):
                    future.result()

            parallel_time = time.time() - start_time
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / num_cores

            results.extend([
                BenchmarkResult(
                    test_name="concurrency_test",
                    metric_name="sequential_time",
                    value=sequential_time,
                    unit="seconds",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'task_size': task_size}
                ),
                BenchmarkResult(
                    test_name="concurrency_test",
                    metric_name="parallel_time",
                    value=parallel_time,
                    unit="seconds",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'task_size': task_size}
                ),
                BenchmarkResult(
                    test_name="concurrency_test",
                    metric_name="speedup",
                    value=speedup,
                    unit="ratio",
                    timestamp=datetime.utcnow(),
                    system_state=self._get_current_system_state(),
                    details={'task_size': task_size, 'num_cores': num_cores}
                )
            ])

            logger.info(f"Concurrency - Task Size: {task_size}, "
                       f"Speedup: {speedup:.2f}x, "
                       f"Efficiency: {efficiency:.2f}")

        return results

    def _start_resource_monitoring(self):
        """リソース監視開始"""
        self.monitoring_active = True

        def monitor_resources():
            while self.monitoring_active:
                try:
                    # システムリソース取得
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk_io = psutil.disk_io_counters()
                    network_io = psutil.net_io_counters()

                    metrics = SystemResourceMetrics(
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_used_gb=memory.used / (1024**3),
                        disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
                        disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
                        network_sent_mb=network_io.bytes_sent / (1024**2) if network_io else 0,
                        network_recv_mb=network_io.bytes_recv / (1024**2) if network_io else 0,
                        timestamp=datetime.utcnow()
                    )

                    self.resource_metrics.append(metrics)

                except Exception as e:
                    logger.warning(f"Resource monitoring error: {str(e)}")

                time.sleep(self.benchmark_config['resource_monitoring_interval'])

        monitoring_thread = threading.Thread(target=monitor_resources)
        monitoring_thread.daemon = True
        monitoring_thread.start()

    def _stop_resource_monitoring(self):
        """リソース監視停止"""
        self.monitoring_active = False

    def _get_current_system_state(self) -> Dict[str, float]:
        """現在のシステム状態取得"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
        except:
            return {}

    async def _simulate_ml_prediction(self):
        """ML予測シミュレーション"""
        # CPU集約的な計算をシミュレート
        await asyncio.sleep(0.01)  # 10ms
        np.random.rand(1000, 100).dot(np.random.rand(100, 50))

    async def _simulate_data_processing(self, data):
        """データ処理シミュレーション"""
        # データ変換処理をシミュレート
        await asyncio.sleep(0.001 * len(data) / 1000)  # データサイズに比例
        return [x * 2 for x in data[:1000]]  # サンプル変換

    def _generate_test_data(self, size: int) -> List[float]:
        """テストデータ生成"""
        return np.random.rand(size).tolist()

    def _execute_concurrent_task(self) -> float:
        """並行タスク実行"""
        start_time = time.time()

        # 簡単な計算タスク
        result = sum(i * i for i in range(10000))

        return time.time() - start_time

    async def _cpu_intensive_task(self, size: int):
        """CPU集約的タスク（非同期）"""
        await asyncio.get_event_loop().run_in_executor(
            None, self._cpu_intensive_task_sync, size
        )

    def _cpu_intensive_task_sync(self, size: int):
        """CPU集約的タスク（同期）"""
        result = 0
        for i in range(size):
            result += i * i * np.sin(i)
        return result

    def _generate_performance_summary(self, results: List[BenchmarkResult]) -> Dict:
        """パフォーマンスサマリー生成"""
        summary = {}

        # テスト別結果集約
        test_groups = {}
        for result in results:
            test_name = result.test_name
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(result)

        # 各テストの統計
        for test_name, test_results in test_groups.items():
            test_summary = {}

            # メトリクス別統計
            metrics = {}
            for result in test_results:
                metric_name = result.metric_name
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(result.value)

            for metric_name, values in metrics.items():
                test_summary[metric_name] = {
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }

            summary[test_name] = test_summary

        # 全体統計
        summary['overall'] = {
            'total_tests': len(results),
            'test_types': len(test_groups),
            'benchmark_duration': max(r.timestamp for r in results) - min(r.timestamp for r in results)
        }

        return summary

    def _generate_optimization_recommendations(self, summary: Dict) -> List[str]:
        """最適化提案生成"""
        recommendations = []

        # ML性能チェック
        ml_summary = summary.get('ml_prediction_performance', {})
        if 'avg_latency' in ml_summary:
            avg_latency = ml_summary['avg_latency']['avg']
            if avg_latency > self.performance_targets['ml_prediction_latency_ms']:
                recommendations.append(
                    f"ML予測レイテンシ改善: {avg_latency:.1f}ms > 目標{self.performance_targets['ml_prediction_latency_ms']}ms"
                )

        # スケーラビリティチェック
        scalability_summary = summary.get('scalability_test', {})
        if 'throughput' in scalability_summary:
            max_throughput = scalability_summary['throughput']['max']
            if max_throughput < self.performance_targets['ml_prediction_throughput_rps']:
                recommendations.append(
                    f"スループット向上: {max_throughput:.1f} RPS < 目標{self.performance_targets['ml_prediction_throughput_rps']} RPS"
                )

        # メモリ使用量チェック
        memory_summary = summary.get('memory_usage_test', {})
        if 'memory_usage' in memory_summary:
            max_memory = memory_summary['memory_usage']['max']
            if max_memory > self.performance_targets['memory_usage_percent']:
                recommendations.append(
                    f"メモリ使用量最適化: {max_memory:.1f}% > 目標{self.performance_targets['memory_usage_percent']}%"
                )

        # I/O性能チェック
        io_summary = summary.get('io_performance_test', {})
        if 'write_throughput' in io_summary:
            min_write = io_summary['write_throughput']['min']
            if min_write < self.performance_targets['disk_io_threshold_mbps']:
                recommendations.append(
                    f"ディスクI/O改善: {min_write:.1f} MB/s < 推奨{self.performance_targets['disk_io_threshold_mbps']} MB/s"
                )

        if not recommendations:
            recommendations.append("パフォーマンス目標を達成しています。現状維持を推奨します。")

        return recommendations

    def _calculate_overall_grade(self, summary: Dict) -> str:
        """総合評価算出"""
        score = 0
        max_score = 0

        # ML性能評価
        ml_summary = summary.get('ml_prediction_performance', {})
        if 'avg_latency' in ml_summary:
            latency = ml_summary['avg_latency']['avg']
            if latency <= self.performance_targets['ml_prediction_latency_ms']:
                score += 25
            max_score += 25

        # スケーラビリティ評価
        scalability_summary = summary.get('scalability_test', {})
        if 'throughput' in scalability_summary:
            throughput = scalability_summary['throughput']['max']
            if throughput >= self.performance_targets['ml_prediction_throughput_rps']:
                score += 25
            max_score += 25

        # メモリ効率評価
        memory_summary = summary.get('memory_usage_test', {})
        if 'memory_usage' in memory_summary:
            memory_usage = memory_summary['memory_usage']['max']
            if memory_usage <= self.performance_targets['memory_usage_percent']:
                score += 25
            max_score += 25

        # I/O性能評価
        io_summary = summary.get('io_performance_test', {})
        if 'write_throughput' in io_summary:
            write_throughput = io_summary['write_throughput']['min']
            if write_throughput >= self.performance_targets['disk_io_threshold_mbps']:
                score += 25
            max_score += 25

        # グレード算出
        if max_score == 0:
            return "N/A"

        percentage = (score / max_score) * 100

        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def save_benchmark_report(self, benchmark_report: Dict, filename: str = None):
        """ベンチマークレポート保存"""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_benchmark_{timestamp}.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Benchmark report saved to: {filename}")

    def generate_performance_charts(self, benchmark_report: Dict, output_dir: str = "charts"):
        """パフォーマンスチャート生成"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. レイテンシ分布チャート
        self._create_latency_chart(benchmark_report, output_dir)

        # 2. スループットチャート
        self._create_throughput_chart(benchmark_report, output_dir)

        # 3. リソース使用量チャート
        self._create_resource_usage_chart(benchmark_report, output_dir)

        # 4. スケーラビリティチャート
        self._create_scalability_chart(benchmark_report, output_dir)

    def _create_latency_chart(self, report: Dict, output_dir: str):
        """レイテンシチャート作成"""
        # 実装省略（matplotlib/seaborn使用）
        pass

    def _create_throughput_chart(self, report: Dict, output_dir: str):
        """スループットチャート作成"""
        # 実装省略
        pass

    def _create_resource_usage_chart(self, report: Dict, output_dir: str):
        """リソース使用量チャート作成"""
        # 実装省略
        pass

    def _create_scalability_chart(self, report: Dict, output_dir: str):
        """スケーラビリティチャート作成"""
        # 実装省略
        pass

async def main():
    """パフォーマンスベンチマーク実行"""
    benchmark = PerformanceBenchmark()

    print("Day Trade ML System - パフォーマンスベンチマーク開始")
    print("=" * 60)

    # ベンチマーク実行
    report = await benchmark.run_comprehensive_benchmark()

    # 結果出力
    print(f"\nパフォーマンスベンチマーク結果")
    print(f"総合評価: {report['performance_grade']}")
    print(f"実行時間: {report['benchmark_duration_seconds']:.1f}秒")
    print(f"総テスト数: {report['total_tests']}")

    print(f"\nパフォーマンスサマリー:")
    for test_name, test_summary in report['summary'].items():
        if test_name != 'overall':
            print(f"\n{test_name}:")
            for metric_name, stats in test_summary.items():
                if isinstance(stats, dict) and 'avg' in stats:
                    print(f"  {metric_name}: 平均 {stats['avg']:.2f}, 最大 {stats['max']:.2f}")

    print(f"\n最適化提案:")
    for rec in report['recommendations']:
        print(f"• {rec}")

    # レポート保存
    benchmark.save_benchmark_report(report)

    # チャート生成
    # benchmark.generate_performance_charts(report)

    print(f"\n詳細レポートが保存されました")

if __name__ == '__main__':
    asyncio.run(main())