#!/usr/bin/env python3
"""
超低レイテンシHFTシステム統合テスト
Issue #443: HFT超低レイテンシ最適化 - <10μs実現戦略

統合テスト: Rustコア + システム最適化 + パフォーマンス測定
"""

import asyncio
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from day_trade.performance.system_optimization import (
        SystemOptimizationConfig,
        SystemOptimizer,
        setup_ultra_low_latency_system,
    )
    from day_trade.performance.ultra_low_latency_core import (
        UltraLowLatencyConfig,
        UltraLowLatencyCore,
        create_ultra_low_latency_core,
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"モジュール不足: {e}")
    MODULES_AVAILABLE = False


def test_basic_ultra_low_latency():
    """基本的な超低レイテンシテスト"""
    print("=== 基本超低レイテンシテスト ===")

    # 超低レイテンシコア作成
    core = create_ultra_low_latency_core(
        target_latency_us=10.0,
        cpu_cores=[2, 3],
        memory_mb=256
    )

    print("Ultra Low Latency Core initialized")

    # シンプルなトレード実行テスト
    results = []
    for i in range(100):
        result = core.execute_trade_ultra_fast(
            "USDJPY", "buy", 10000, 150.0 + (i * 0.001)
        )
        results.append(result)

        if i % 20 == 0:
            print(f"Trade {i+1}: {result['latency_us']:.2f}μs (target: <10μs)")

    # 統計計算
    latencies = [r['latency_us'] for r in results]
    under_target = sum(1 for lat in latencies if lat < 10.0)

    print("\nResults:")
    print(f"Total trades: {len(results)}")
    print(f"Average latency: {mean(latencies):.2f}μs")
    print(f"Min latency: {min(latencies):.2f}μs")
    print(f"Max latency: {max(latencies):.2f}μs")
    if len(latencies) > 1:
        print(f"Std deviation: {stdev(latencies):.2f}μs")
    print(f"Under 10μs target: {under_target}/{len(results)} ({under_target/len(results)*100:.1f}%)")

    # パフォーマンスレポート
    report = core.get_performance_report()
    print(f"Success rate: {report['success_rate']:.1f}%")
    print(f"Target achievement: {report['performance']['target_achievement_rate']:.1f}%")

    core.cleanup()

    return {
        'avg_latency': mean(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'under_target_rate': under_target / len(results),
        'report': report
    }


def test_system_optimization():
    """システム最適化テスト"""
    print("\n=== システム最適化テスト ===")

    # システム最適化適用
    optimizer = setup_ultra_low_latency_system([2, 3])

    # システム状況確認
    status = optimizer.get_system_status()

    print(f"Platform: {status['platform']}")
    print(f"Applied optimizations: {len(status['applied_optimizations'])}")

    for optimization in status['applied_optimizations']:
        print(f"  - {optimization}")

    # CPU情報
    cpu_info = status.get('cpu_info', {})
    if cpu_info:
        print(f"CPU cores configured: {cpu_info.get('cores')}")
        print(f"CPU affinity set: {cpu_info.get('affinity_set')}")

        if 'current_affinity' in cpu_info:
            print(f"Current CPU affinity: {cpu_info['current_affinity']}")

    # メモリ情報
    memory_info = status.get('memory_info', {})
    if memory_info and 'total' in memory_info:
        total_gb = memory_info['total'] / (1024**3)
        available_gb = memory_info['available'] / (1024**3)
        print(f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")

    return status


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")

    # 通常設定での測定
    print("Normal configuration testing...")
    normal_core = create_ultra_low_latency_core(target_latency_us=50.0, memory_mb=128)

    normal_results = []
    for i in range(50):
        result = normal_core.execute_trade_ultra_fast("USDJPY", "buy", 10000, 150.0)
        normal_results.append(result['latency_us'])

    normal_avg = mean(normal_results)
    normal_core.cleanup()

    # 最適化設定での測定
    print("Optimized configuration testing...")
    optimized_core = create_ultra_low_latency_core(
        target_latency_us=10.0,
        cpu_cores=[2, 3],
        memory_mb=512
    )

    optimized_results = []
    for i in range(50):
        result = optimized_core.execute_trade_ultra_fast("USDJPY", "buy", 10000, 150.0)
        optimized_results.append(result['latency_us'])

    optimized_avg = mean(optimized_results)
    optimized_core.cleanup()

    # 比較結果
    improvement = ((normal_avg - optimized_avg) / normal_avg) * 100

    print("\nPerformance Comparison:")
    print(f"Normal config average: {normal_avg:.2f}μs")
    print(f"Optimized config average: {optimized_avg:.2f}μs")
    print(f"Improvement: {improvement:.1f}%")

    return {
        'normal_avg': normal_avg,
        'optimized_avg': optimized_avg,
        'improvement_percent': improvement
    }


def test_latency_distribution():
    """レイテンシ分布テスト"""
    print("\n=== レイテンシ分布テスト ===")

    core = create_ultra_low_latency_core(target_latency_us=10.0)

    # 大量実行でのレイテンシ分布測定
    latencies = []
    for i in range(1000):
        result = core.execute_trade_ultra_fast("USDJPY", "buy", 10000, 150.0)
        latencies.append(result['latency_us'])

        if i % 200 == 0:
            print(f"Progress: {i+1}/1000")

    # パーセンタイル計算
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

    # 目標内分析
    under_5us = sum(1 for lat in latencies if lat < 5.0)
    under_10us = sum(1 for lat in latencies if lat < 10.0)
    under_15us = sum(1 for lat in latencies if lat < 15.0)

    print("\nLatency Distribution (N=1000):")
    print(f"Mean: {mean(latencies):.2f}μs")
    print(f"P50 (median): {p50:.2f}μs")
    print(f"P90: {p90:.2f}μs")
    print(f"P95: {p95:.2f}μs")
    print(f"P99: {p99:.2f}μs")
    print(f"Min: {min(latencies):.2f}μs")
    print(f"Max: {max(latencies):.2f}μs")

    print("\nTarget Achievement:")
    print(f"< 5μs:  {under_5us}/1000 ({under_5us/10:.1f}%)")
    print(f"< 10μs: {under_10us}/1000 ({under_10us/10:.1f}%)")
    print(f"< 15μs: {under_15us}/1000 ({under_15us/10:.1f}%)")

    core.cleanup()

    return {
        'latencies': latencies,
        'percentiles': {'p50': p50, 'p90': p90, 'p95': p95, 'p99': p99},
        'under_targets': {
            '5us': under_5us/10,
            '10us': under_10us/10,
            '15us': under_15us/10
        }
    }


def test_concurrent_performance():
    """同時実行パフォーマンステスト"""
    print("\n=== 同時実行パフォーマンステスト ===")

    import threading

    # 2つのコアで同時実行
    cores = [
        create_ultra_low_latency_core(target_latency_us=10.0, memory_mb=256),
        create_ultra_low_latency_core(target_latency_us=10.0, memory_mb=256)
    ]

    results = [[] for _ in cores]

    def worker(core_index: int, num_trades: int):
        for i in range(num_trades):
            result = cores[core_index].execute_trade_ultra_fast(
                "USDJPY", "buy", 10000, 150.0 + (i * 0.001)
            )
            results[core_index].append(result['latency_us'])

    # 同時実行開始
    print("Starting concurrent execution...")
    threads = []
    for i in range(len(cores)):
        thread = threading.Thread(target=worker, args=(i, 100))
        threads.append(thread)
        thread.start()

    # 完了待機
    for thread in threads:
        thread.join()

    # 結果分析
    all_latencies = []
    for i, core_results in enumerate(results):
        all_latencies.extend(core_results)
        avg_latency = mean(core_results)
        print(f"Core {i+1}: {len(core_results)} trades, avg {avg_latency:.2f}μs")

    overall_avg = mean(all_latencies)
    under_10us = sum(1 for lat in all_latencies if lat < 10.0)

    print("\nConcurrent Performance Summary:")
    print(f"Total trades: {len(all_latencies)}")
    print(f"Overall average: {overall_avg:.2f}μs")
    print(f"Under 10μs target: {under_10us}/{len(all_latencies)} ({under_10us/len(all_latencies)*100:.1f}%)")

    # クリーンアップ
    for core in cores:
        core.cleanup()

    return {
        'total_trades': len(all_latencies),
        'overall_avg': overall_avg,
        'target_rate': under_10us / len(all_latencies)
    }


def main():
    """メインテスト実行"""
    if not MODULES_AVAILABLE:
        print("Required modules not available. Exiting.")
        return False

    print("Ultra Low Latency HFT System Test")
    print("=" * 50)

    try:
        # 基本テスト
        basic_results = test_basic_ultra_low_latency()

        # システム最適化テスト
        system_status = test_system_optimization()

        # パフォーマンス比較
        comparison_results = test_performance_comparison()

        # レイテンシ分布テスト
        distribution_results = test_latency_distribution()

        # 同時実行テスト
        concurrent_results = test_concurrent_performance()

        # 総合評価
        print("\n" + "=" * 50)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 50)

        print("Basic Performance:")
        print(f"  Average latency: {basic_results['avg_latency']:.2f}μs")
        print(f"  Target achievement: {basic_results['under_target_rate']*100:.1f}%")

        print("System Optimization:")
        print(f"  Optimizations applied: {len(system_status['applied_optimizations'])}")
        print(f"  Platform: {system_status['platform']}")

        print("Performance Improvement:")
        print(f"  Improvement vs normal: {comparison_results['improvement_percent']:.1f}%")
        print(f"  Optimized average: {comparison_results['optimized_avg']:.2f}μs")

        print("Latency Distribution:")
        print(f"  P95 latency: {distribution_results['percentiles']['p95']:.2f}μs")
        print(f"  P99 latency: {distribution_results['percentiles']['p99']:.2f}μs")
        print(f"  <10μs achievement: {distribution_results['under_targets']['10us']:.1f}%")

        print("Concurrent Performance:")
        print(f"  Concurrent average: {concurrent_results['overall_avg']:.2f}μs")
        print(f"  Concurrent target rate: {concurrent_results['target_rate']*100:.1f}%")

        # 目標達成判定
        target_achieved = (
            basic_results['avg_latency'] < 15.0 and  # 平均15μs以下
            distribution_results['under_targets']['10us'] > 70 and  # 70%以上が10μs以下
            comparison_results['improvement_percent'] > 20  # 20%以上の改善
        )

        print(f"\nOverall Assessment: {'SUCCESS' if target_achieved else 'NEEDS IMPROVEMENT'}")

        if target_achieved:
            print("Ultra-low latency targets achieved!")
        else:
            print("Further optimization required for target achievement.")

        return target_achieved

    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
