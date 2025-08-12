#!/usr/bin/env python3
"""
簡単なパフォーマンステスト - ASCII Safe Version

85銘柄対応での処理速度を測定
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.data.batch_data_fetcher import BatchDataFetcher  # noqa: E402


def test_data_fetching():
    """データ取得テスト"""
    print("=== Data Fetching Performance Test ===")

    # テスト用銘柄（主要10銘柄）
    test_symbols = [
        "7203",
        "8306",
        "9984",
        "6758",
        "4689",
        "9434",
        "8001",
        "7267",
        "6861",
        "2914",
    ]

    fetcher = BatchDataFetcher(max_workers=3)

    # シーケンシャル取得
    print("1. Sequential fetch test (10 stocks)")
    start_time = time.time()
    sequential_data = fetcher.fetch_multiple_symbols(
        test_symbols, period="5d", use_parallel=False
    )
    sequential_time = time.time() - start_time

    print(f"  - Time: {sequential_time:.2f}s")
    print(f"  - Per stock: {sequential_time/len(test_symbols):.2f}s")
    print(f"  - Success: {len(sequential_data)}/{len(test_symbols)}")

    # 並列取得
    print("2. Parallel fetch test (10 stocks)")
    start_time = time.time()
    parallel_data = fetcher.fetch_multiple_symbols(
        test_symbols, period="5d", use_parallel=True
    )
    parallel_time = time.time() - start_time

    print(f"  - Time: {parallel_time:.2f}s")
    print(f"  - Per stock: {parallel_time/len(test_symbols):.2f}s")
    print(f"  - Success: {len(parallel_data)}/{len(test_symbols)}")

    if sequential_time > 0 and parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"  - Speedup: {speedup:.1f}x")

    # 85銘柄予測
    print("3. Estimate for 85 stocks")
    if parallel_time > 0:
        estimated_85 = (parallel_time / len(test_symbols)) * 85
        print(f"  - Estimated time: {estimated_85:.1f}s")
        print(f"  - 30s target: {'OK' if estimated_85 <= 30 else 'EXCEED'}")

    return len(sequential_data) > 0 or len(parallel_data) > 0


def test_ml_performance():
    """ML処理性能テスト"""
    print("\n=== ML Performance Test ===")

    try:
        from day_trade.data.advanced_ml_engine import AdvancedMLEngine

        # テスト用データ取得
        fetcher = BatchDataFetcher()
        test_symbols = ["7203", "4563"]  # トヨタとアンジェス

        stock_data = fetcher.fetch_multiple_symbols(test_symbols, period="30d")

        if not stock_data:
            print("No data available for ML test")
            return False

        ml_engine = AdvancedMLEngine()
        total_time = 0
        successful_analyses = 0

        for symbol, data in stock_data.items():
            if data.empty:
                continue

            print(f"ML analysis for {symbol}:")
            start_time = time.time()

            try:
                # 特徴量準備
                features = ml_engine.prepare_ml_features(data)

                # ML予測実行
                (
                    trend_score,
                    vol_score,
                    pattern_score,
                ) = ml_engine.predict_advanced_scores(symbol, data, features)

                # 投資助言生成
                advice = ml_engine.generate_investment_advice(symbol, data, features)

                processing_time = time.time() - start_time
                total_time += processing_time
                successful_analyses += 1

                print(f"  - Time: {processing_time:.3f}s")
                print(
                    f"  - Advice: {advice['advice']} (Confidence: {advice['confidence']:.1f}%)"
                )

            except Exception as e:
                processing_time = time.time() - start_time
                print(f"  - Error: {e}")
                print(f"  - Failed in: {processing_time:.3f}s")

        # 全銘柄予測
        if successful_analyses > 0:
            avg_time = total_time / successful_analyses
            estimated_85 = avg_time * 85

            print("ML Summary:")
            print(f"  - Avg time per stock: {avg_time:.3f}s")
            print(f"  - 85 stocks estimated: {estimated_85:.1f}s")
            print(f"  - 10s target: {'OK' if estimated_85 <= 10 else 'EXCEED'}")
            print(f"  - Success rate: {successful_analyses}/{len(stock_data)}")

        return successful_analyses > 0

    except ImportError as e:
        print(f"ML engine import error: {e}")
        return False


def test_memory_usage():
    """メモリ使用量テスト"""
    print("\n=== Memory Usage Test ===")

    try:
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_memory:.1f}MB")

        # データ取得後メモリ
        fetcher = BatchDataFetcher()
        fetcher.fetch_multiple_symbols(["7203", "8306", "9984"], period="30d")

        after_data_memory = process.memory_info().rss / 1024 / 1024
        print(
            f"After data fetch: {after_data_memory:.1f}MB (+{after_data_memory-initial_memory:.1f}MB)"
        )

        # 85銘柄予想
        data_per_stock = (after_data_memory - initial_memory) / 3
        estimated_85_memory = initial_memory + data_per_stock * 85

        print(f"Estimated 85 stocks: {estimated_85_memory:.1f}MB")
        print(f"2GB target: {'OK' if estimated_85_memory <= 2048 else 'EXCEED'}")

        return True

    except ImportError:
        print("psutil not available - memory test skipped")
        return False


def main():
    """メインテスト実行"""
    print("Performance Benchmark for 85 Stocks")
    print("=" * 50)

    try:
        # データ取得テスト
        data_ok = test_data_fetching()

        # ML処理テスト
        ml_ok = test_ml_performance()

        # メモリテスト
        mem_ok = test_memory_usage()

        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"  - Data fetching: {'PASS' if data_ok else 'FAIL'}")
        print(f"  - ML processing: {'PASS' if ml_ok else 'FAIL'}")
        print(f"  - Memory usage: {'PASS' if mem_ok else 'SKIP'}")

        print("\nRecommendations:")
        print("1. Increase max_workers for data fetching")
        print("2. Implement ML feature calculation parallelization")
        print("3. Add data caching for recent results")
        print("4. Consider batch processing for memory efficiency")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test error: {e}")


if __name__ == "__main__":
    main()
