#!/usr/bin/env python3
"""
HFT最適化エンジン簡易テスト
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from day_trade.performance import HFTOptimizer, HFTConfig

def main():
    print("=== HFT Performance Test ===")

    # Config
    config = HFTConfig(
        target_latency_us=50.0,
        preallocated_memory_mb=10,
        enable_simd=True,
    )

    # Initialize
    optimizer = HFTOptimizer(config)
    print("HFT Optimizer initialized")

    # Test data
    np.random.seed(42)
    prices = np.random.normal(100, 5, 100).astype(np.float64)
    volumes = np.random.normal(10000, 1000, 100).astype(np.float64)

    # Performance test
    latencies = []
    predictions = []

    for i in range(10):
        result = optimizer.predict_ultra_fast(prices, volumes)
        latencies.append(result['latency_us'])
        predictions.append(result['prediction'])

        if i == 0:
            print(f"First prediction: {result['prediction']:.4f}")
            print(f"First latency: {result['latency_us']:.2f}μs")
            print(f"Under target: {result['under_target']}")

    # Statistics
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    under_target_rate = np.mean([lat < config.target_latency_us for lat in latencies])

    print(f"\nPerformance Results:")
    print(f"Average latency: {avg_latency:.2f}μs")
    print(f"Max latency: {max_latency:.2f}μs")
    print(f"Under target rate: {under_target_rate:.1%}")

    # Optimization report
    report = optimizer.get_optimization_report()
    print(f"Optimization score: {report['optimization_score']:.1f}/100")

    # Memory statistics
    memory_stats = optimizer.memory_pool.get_stats()
    print(f"Memory allocations: {memory_stats['total_allocations']}")
    print(f"Pool utilization: {memory_stats['pool_utilization']:.1%}")

    # Cleanup
    optimizer.cleanup()
    print("Test completed successfully")

    return avg_latency < config.target_latency_us * 2  # Success if avg < 2x target

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)