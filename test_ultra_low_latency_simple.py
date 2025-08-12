#!/usr/bin/env python3
"""
超低レイテンシHFTシステム簡単テスト
Issue #443: HFT超低レイテンシ最適化
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_ultra_low_latency_basic():
    """基本的な超低レイテンシテスト"""
    print("=== Ultra Low Latency Basic Test ===")

    try:
        from day_trade.performance.ultra_low_latency_core import (
            create_ultra_low_latency_core,
        )

        # コア作成
        core = create_ultra_low_latency_core(target_latency_us=10.0)
        print("Ultra Low Latency Core created successfully")

        # シンプルなトレードテスト
        results = []
        print("Running 50 trade executions...")

        for i in range(50):
            start_time = time.time()

            result = core.execute_trade_ultra_fast("USDJPY", "buy", 10000, 150.0)

            if result and result.get("success", False):
                latency_us = result["latency_us"]
                results.append(latency_us)

                if i % 10 == 0:
                    print(f"Trade {i+1}: {latency_us:.2f}μs")
            else:
                print(f"Trade {i+1}: Failed")

        # 結果統計
        if results:
            avg_latency = sum(results) / len(results)
            min_latency = min(results)
            max_latency = max(results)
            under_target = sum(1 for r in results if r < 10.0)

            print("\nResults Summary:")
            print(f"Successful trades: {len(results)}/50")
            print(f"Average latency: {avg_latency:.2f}μs")
            print(f"Min latency: {min_latency:.2f}μs")
            print(f"Max latency: {max_latency:.2f}μs")
            print(
                f"Under 10μs target: {under_target}/{len(results)} ({under_target/len(results)*100:.1f}%)"
            )

            # パフォーマンスレポート
            try:
                report = core.get_performance_report()
                print(
                    f"Performance report available: {report.get('total_trades', 0)} trades tracked"
                )
            except Exception as e:
                print(f"Performance report error: {e}")
        else:
            print("No successful trades recorded")

        # クリーンアップ
        core.cleanup()

        return len(results) > 0 and avg_latency < 50.0  # 成功基準

    except ImportError as e:
        print(f"Module import error: {e}")
        return False
    except Exception as e:
        print(f"Test execution error: {e}")
        return False


def test_system_optimization():
    """システム最適化テスト"""
    print("\n=== System Optimization Test ===")

    try:
        from day_trade.performance.system_optimization import (
            setup_ultra_low_latency_system,
        )

        # システム最適化適用
        optimizer = setup_ultra_low_latency_system([2, 3])
        print("System optimization setup completed")

        # システム状況確認
        status = optimizer.get_system_status()

        print(f"Platform: {status.get('platform', 'unknown')}")
        optimizations = status.get("applied_optimizations", [])
        print(f"Applied optimizations: {len(optimizations)}")

        for opt in optimizations[:5]:  # 最初の5つのみ表示
            print(f"  - {opt}")

        if len(optimizations) > 5:
            print(f"  ... and {len(optimizations) - 5} more")

        return len(optimizations) > 0

    except ImportError as e:
        print(f"System optimization module not available: {e}")
        return False
    except Exception as e:
        print(f"System optimization error: {e}")
        return False


def test_performance_capabilities():
    """パフォーマンス能力テスト"""
    print("\n=== Performance Capabilities Test ===")

    try:
        from day_trade.performance import (
            get_performance_info,
            verify_system_capabilities,
        )

        # パフォーマンス情報取得
        info = get_performance_info()
        print(f"Performance system version: {info['version']}")

        systems = info.get("systems", {})
        for name, desc in systems.items():
            print(f"  {name}: {desc}")

        # システム能力検証
        capabilities = verify_system_capabilities()
        print("\nSystem Capabilities:")
        for capability, available in capabilities.items():
            status = "Available" if available else "Not Available"
            print(f"  {capability}: {status}")

        return capabilities.get("ultra_low_latency", False)

    except ImportError as e:
        print(f"Performance module not available: {e}")
        return False
    except Exception as e:
        print(f"Performance capabilities error: {e}")
        return False


def main():
    """メインテスト実行"""
    print("Ultra Low Latency HFT System Simple Test")
    print("=" * 50)

    results = {}

    # 基本的な超低レイテンシテスト
    results["basic_test"] = test_ultra_low_latency_basic()

    # システム最適化テスト
    results["system_test"] = test_system_optimization()

    # パフォーマンス能力テスト
    results["capabilities_test"] = test_performance_capabilities()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    passed_tests = sum(results.values())
    total_tests = len(results)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests >= 2:  # 3つのうち2つ以上成功
        print("Ultra Low Latency system is functional!")
        return True
    else:
        print("Ultra Low Latency system needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
