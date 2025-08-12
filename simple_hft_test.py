#!/usr/bin/env python3
"""
高頻度取引エンジン 簡易パフォーマンステスト
Issue #366: High-Frequency Trading Optimization Engine Simple Test
"""

import asyncio
import sys
from pathlib import Path

# パス調整
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.day_trade.core.optimization_strategy import OptimizationConfig
from src.day_trade.trading.high_frequency_engine import (
    create_high_frequency_trading_engine,
)


async def simple_performance_test():
    """簡易パフォーマンステスト"""
    print("=" * 60)
    print("高頻度取引エンジン 簡易テスト開始")
    print("=" * 60)

    # 設定
    config = OptimizationConfig()
    symbols = ["AAPL", "MSFT", "GOOGL"]

    try:
        print("\n設定:")
        print(f"  対象銘柄: {symbols}")
        print("  テスト時間: 10秒")

        # エンジン作成・初期化
        print("\nエンジン初期化中...")
        engine = await create_high_frequency_trading_engine(config, symbols)
        print("初期化完了")

        # ベンチマーク実行
        print("\nベンチマーク実行中...")
        benchmark_results = await engine.run_performance_benchmark(duration_seconds=10)

        # 結果表示
        print("\n結果:")
        print(f"  処理注文数: {benchmark_results['total_orders_processed']:,}")
        print(f"  平均遅延: {benchmark_results['average_latency_microseconds']:.1f} マイクロ秒")
        print(
            f"  スループット: {benchmark_results['peak_throughput_orders_per_second']:.0f} 注文/秒"
        )
        print(f"  エラー率: {benchmark_results['error_rate_percent']:.3f}%")

        # パフォーマンス評価
        latency = benchmark_results["average_latency_microseconds"]
        throughput = benchmark_results["peak_throughput_orders_per_second"]
        error_rate = benchmark_results["error_rate_percent"]

        print("\n評価:")

        # 遅延評価
        if latency < 100:
            print(f"  遅延: 優秀 ({latency:.1f}μs)")
        elif latency < 500:
            print(f"  遅延: 良好 ({latency:.1f}μs)")
        else:
            print(f"  遅延: 改善余地 ({latency:.1f}μs)")

        # スループット評価
        if throughput > 5000:
            print(f"  スループット: 優秀 ({throughput:.0f} 注文/秒)")
        elif throughput > 1000:
            print(f"  スループット: 良好 ({throughput:.0f} 注文/秒)")
        else:
            print(f"  スループット: 改善余地 ({throughput:.0f} 注文/秒)")

        # 安定性評価
        if error_rate < 1.0:
            print(f"  安定性: 優秀 ({error_rate:.3f}% エラー)")
        elif error_rate < 5.0:
            print(f"  安定性: 良好 ({error_rate:.3f}% エラー)")
        else:
            print(f"  安定性: 改善余地 ({error_rate:.3f}% エラー)")

        print("\n高頻度取引エンジン テスト完了!")
        return True

    except Exception as e:
        print(f"テストエラー: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(simple_performance_test())
    sys.exit(0 if success else 1)
