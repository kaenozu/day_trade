#!/usr/bin/env python3
"""
並列ML処理パフォーマンステスト
Issue #323: ML Processing Parallelization for Throughput Improvement

4-8倍のスループット改善効果を測定・検証
"""

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.day_trade.data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
    from src.day_trade.data.optimized_ml_engine import OptimizedMLEngine

    ENGINES_AVAILABLE = True
    print("並列MLエンジン正常読込")
except ImportError as e:
    print(f"MLエンジン読込エラー: {e}")
    ENGINES_AVAILABLE = False


# テスト用データ生成
def generate_test_stock_data(
    symbols: List[str], days: int = 100
) -> Dict[str, pd.DataFrame]:
    """テスト用株式データ生成"""
    stock_data = {}

    for symbol in symbols:
        dates = pd.date_range(start="2024-01-01", periods=days)

        # リアルな価格変動を模擬
        base_price = np.random.uniform(1000, 5000)
        price_series = []
        current_price = base_price

        for i in range(days):
            # ランダムウォーク + トレンド
            daily_change = np.random.normal(0, 0.02)  # 2%標準偏差
            trend_factor = 0.0001 * (i - days / 2)  # 微弱トレンド

            current_price *= 1 + daily_change + trend_factor
            price_series.append(current_price)

        prices = np.array(price_series)

        # OHLC生成
        highs = prices * np.random.uniform(1.001, 1.05, days)  # 高値
        lows = prices * np.random.uniform(0.95, 0.999, days)  # 安値
        opens = np.roll(prices, 1)  # 前日終値ベース開始価格
        opens[0] = prices[0]

        volumes = np.random.randint(100000, 10000000, days)

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )

        stock_data[symbol] = df

    return stock_data


def benchmark_sequential_processing(
    stock_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """シーケンシャル処理ベンチマーク"""
    print("\n=== シーケンシャル処理ベンチマーク ===")

    if not ENGINES_AVAILABLE:
        print("エンジン利用不可 - ベンチマークスキップ")
        return {}

    try:
        engine = OptimizedMLEngine()

        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        results = {}
        processing_times = []

        for i, (symbol, data) in enumerate(stock_data.items(), 1):
            symbol_start = time.time()

            try:
                # ML特徴量計算
                features = engine.prepare_lightweight_features(data)

                # 予測生成
                prediction = (
                    engine.predict_investment_score(features)
                    if hasattr(engine, "predict_investment_score")
                    else 0.5
                )

                results[symbol] = {
                    "features_count": len(features),
                    "prediction": prediction,
                    "success": True,
                }

                symbol_time = time.time() - symbol_start
                processing_times.append(symbol_time)

                print(f"  [{i}/{len(stock_data)}] {symbol}: {symbol_time:.3f}秒")

            except Exception as e:
                print(f"  [{i}/{len(stock_data)}] {symbol}: エラー - {e}")
                results[symbol] = {"error": str(e), "success": False}

        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        successful_count = sum(1 for r in results.values() if r.get("success", False))

        return {
            "total_time": total_time,
            "avg_time_per_symbol": total_time / len(stock_data),
            "successful_symbols": successful_count,
            "total_symbols": len(stock_data),
            "success_rate": successful_count / len(stock_data),
            "memory_used_mb": memory_used,
            "processing_times": processing_times,
            "results": results,
        }

    except Exception as e:
        print(f"シーケンシャルベンチマークエラー: {e}")
        return {}


def benchmark_parallel_processing(
    stock_data: Dict[str, pd.DataFrame], cpu_workers: int = 4, enable_cache: bool = True
) -> Dict[str, Any]:
    """並列処理ベンチマーク"""
    print(
        f"\n=== 並列処理ベンチマーク (workers={cpu_workers}, cache={enable_cache}) ==="
    )

    if not ENGINES_AVAILABLE:
        print("エンジン利用不可 - ベンチマークスキップ")
        return {}

    try:
        with AdvancedParallelMLEngine(
            cpu_workers=cpu_workers,
            io_workers=cpu_workers * 2,
            memory_limit_gb=1.5,
            cache_enabled=enable_cache,
            enable_monitoring=True,
        ) as engine:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # バッチ処理実行
            results, total_time = engine.batch_process_symbols(
                stock_data, use_cache=enable_cache, timeout_per_symbol=30
            )

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory

            # 性能統計取得
            performance_stats = engine.get_performance_stats()

            successful_count = sum(1 for r in results.values() if not r.get("error"))

            print(f"  処理時間: {total_time:.3f}秒")
            print(f"  成功銘柄: {successful_count}/{len(stock_data)}")
            print(f"  銘柄あたり: {total_time/len(stock_data):.3f}秒")
            print(f"  メモリ使用: {memory_used:.1f}MB")

            return {
                "total_time": total_time,
                "avg_time_per_symbol": total_time / len(stock_data),
                "successful_symbols": successful_count,
                "total_symbols": len(stock_data),
                "success_rate": successful_count / len(stock_data),
                "memory_used_mb": memory_used,
                "performance_stats": performance_stats,
                "results": results,
                "cpu_workers": cpu_workers,
                "cache_enabled": enable_cache,
            }

    except Exception as e:
        print(f"並列ベンチマークエラー: {e}")
        import traceback

        traceback.print_exc()
        return {}


def analyze_scalability(base_symbols: List[str], max_symbols: int = 100):
    """スケーラビリティ分析"""
    print(f"\n=== スケーラビリティ分析 (最大{max_symbols}銘柄) ===")

    # テスト銘柄数の設定
    test_sizes = [5, 10, 20, 50]
    if max_symbols >= 100:
        test_sizes.append(100)

    scalability_results = []

    for size in test_sizes:
        print(f"\n--- {size}銘柄テスト ---")

        # テストデータ生成
        test_symbols = [f"SYM{i:04d}" for i in range(size)]
        test_data = generate_test_stock_data(test_symbols, days=50)  # 短縮データ

        # シーケンシャル測定
        sequential_result = benchmark_sequential_processing(test_data)

        # 並列測定（4 workers）
        parallel_result = benchmark_parallel_processing(test_data, cpu_workers=4)

        # 並列測定（8 workers）
        parallel_8_result = benchmark_parallel_processing(test_data, cpu_workers=8)

        # 結果分析
        if sequential_result and parallel_result:
            speedup_4 = (
                sequential_result["total_time"] / parallel_result["total_time"]
                if parallel_result["total_time"] > 0
                else 0
            )

            speedup_8 = (
                sequential_result["total_time"] / parallel_8_result["total_time"]
                if parallel_8_result.get("total_time", 0) > 0
                else 0
            )

            efficiency_4 = speedup_4 / 4 * 100  # 4workers効率
            efficiency_8 = speedup_8 / 8 * 100  # 8workers効率

            scalability_results.append(
                {
                    "symbols": size,
                    "sequential_time": sequential_result["total_time"],
                    "parallel_4_time": parallel_result["total_time"],
                    "parallel_8_time": parallel_8_result.get("total_time", 0),
                    "speedup_4x": speedup_4,
                    "speedup_8x": speedup_8,
                    "efficiency_4_percent": efficiency_4,
                    "efficiency_8_percent": efficiency_8,
                    "memory_sequential_mb": sequential_result.get("memory_used_mb", 0),
                    "memory_parallel_mb": parallel_result.get("memory_used_mb", 0),
                }
            )

            print(f"  4 workers: {speedup_4:.2f}x高速化 ({efficiency_4:.1f}%効率)")
            print(f"  8 workers: {speedup_8:.2f}x高速化 ({efficiency_8:.1f}%効率)")

        # メモリクリーンアップ
        del test_data
        gc.collect()

    return scalability_results


def estimate_production_performance(scalability_results: List[Dict]):
    """本番環境性能予測"""
    print("\n=== 本番環境性能予測 ===")

    if not scalability_results:
        print("スケーラビリティデータ不足 - 予測不可")
        return

    # 線形回帰による予測
    sizes = [r["symbols"] for r in scalability_results]
    sequential_times = [r["sequential_time"] for r in scalability_results]
    parallel_times = [r["parallel_4_time"] for r in scalability_results]

    if len(sizes) >= 2:
        # 単純な線形予測
        sequential_slope = (sequential_times[-1] - sequential_times[0]) / (
            sizes[-1] - sizes[0]
        )
        parallel_slope = (parallel_times[-1] - parallel_times[0]) / (
            sizes[-1] - sizes[0]
        )

        # 予測計算
        target_sizes = [85, 500]  # TOPIX100, TOPIX500

        print("予測結果:")
        for target in target_sizes:
            seq_pred = sequential_times[0] + sequential_slope * (target - sizes[0])
            par_pred = parallel_times[0] + parallel_slope * (target - sizes[0])

            speedup = seq_pred / par_pred if par_pred > 0 else 0

            print(f"  {target}銘柄:")
            print(f"    シーケンシャル: {seq_pred:.1f}秒 ({seq_pred/60:.1f}分)")
            print(f"    並列処理: {par_pred:.1f}秒 ({par_pred/60:.1f}分)")
            print(f"    高速化: {speedup:.1f}倍")
            print(f"    改善効果: {(seq_pred-par_pred)/60:.1f}分短縮")


def generate_performance_report(
    sequential_results: Dict,
    parallel_results: List[Dict],
    scalability_results: List[Dict],
):
    """性能レポート生成"""
    print("\n" + "=" * 70)
    print("並列ML処理パフォーマンス最終レポート")
    print("=" * 70)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"測定日時: {timestamp}")

    print("\n【基本性能比較】")
    if sequential_results and parallel_results:
        best_parallel = min(
            parallel_results, key=lambda x: x.get("total_time", float("inf"))
        )

        if best_parallel and sequential_results.get("total_time", 0) > 0:
            speedup = sequential_results["total_time"] / best_parallel["total_time"]

            print(f"シーケンシャル処理: {sequential_results['total_time']:.2f}秒")
            print(
                f"最適並列処理: {best_parallel['total_time']:.2f}秒 ({best_parallel['cpu_workers']} workers)"
            )
            print(f"高速化効果: {speedup:.1f}倍")
            print(f"成功率: {best_parallel['success_rate']:.1%}")

            if "performance_stats" in best_parallel:
                cache_hit_rate = best_parallel["performance_stats"].get(
                    "cache_hit_rate", 0
                )
                print(f"キャッシュヒット率: {cache_hit_rate:.1%}")

    print("\n【スケーラビリティ分析】")
    if scalability_results:
        print("銘柄数 | シーケンシャル | 並列4x | 並列8x | 高速化4x | 高速化8x")
        print("-" * 65)

        for result in scalability_results:
            print(
                f"{result['symbols']:6d} | "
                f"{result['sequential_time']:11.2f}s | "
                f"{result['parallel_4_time']:6.2f}s | "
                f"{result['parallel_8_time']:6.2f}s | "
                f"{result['speedup_4x']:7.1f}x | "
                f"{result['speedup_8x']:7.1f}x"
            )

    print("\n【リソース効率性】")
    if parallel_results:
        for result in parallel_results:
            workers = result.get("cpu_workers", 0)
            memory = result.get("memory_used_mb", 0)
            success_rate = result.get("success_rate", 0)

            print(f"{workers} workers: {memory:.1f}MB使用, 成功率{success_rate:.1%}")

    print("\n【推奨設定】")
    if scalability_results:
        # 最適な設定を推奨
        best_efficiency = max(
            scalability_results, key=lambda x: x.get("efficiency_4_percent", 0)
        )
        print(
            f"推奨並列度: 4 workers (効率 {best_efficiency.get('efficiency_4_percent', 0):.1f}%)"
        )
        print(f"期待高速化: {best_efficiency.get('speedup_4x', 0):.1f}倍")
        print("キャッシュ利用: 推奨 (ヒット率向上による更なる高速化)")

    print("=" * 70)


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("並列ML処理パフォーマンステスト")
    print("Issue #323: ML Processing Parallelization")
    print("=" * 70)

    # システム情報表示
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"システム情報: CPU {cpu_count}コア, メモリ {memory_gb:.1f}GB")

    if not ENGINES_AVAILABLE:
        print("必要なエンジンが利用できません")
        return

    # テストデータ生成
    print("\n1. テストデータ生成...")
    base_symbols = [
        "7203",
        "8306",
        "9984",
        "6758",
        "4689",
        "9434",
        "8001",
        "7267",
        "6861",
        "4563",
    ]
    test_stock_data = generate_test_stock_data(base_symbols, days=60)
    print(f"   生成完了: {len(test_stock_data)}銘柄")

    # 基本ベンチマーク
    print("\n2. 基本性能ベンチマーク...")
    sequential_results = benchmark_sequential_processing(test_stock_data)

    parallel_results = []
    for workers in [4, 8]:
        result = benchmark_parallel_processing(test_stock_data, cpu_workers=workers)
        if result:
            parallel_results.append(result)

    # キャッシュ効果測定
    print("\n3. キャッシュ効果測定...")
    cache_result = benchmark_parallel_processing(
        test_stock_data, cpu_workers=4, enable_cache=True
    )
    if cache_result:
        # 2回目実行でキャッシュ効果確認
        cache_result_2 = benchmark_parallel_processing(
            test_stock_data, cpu_workers=4, enable_cache=True
        )

        if cache_result_2 and cache_result["total_time"] > 0:
            cache_speedup = cache_result["total_time"] / cache_result_2["total_time"]
            print(f"   キャッシュ効果: {cache_speedup:.1f}倍高速化")

    # スケーラビリティ分析
    print("\n4. スケーラビリティ分析...")
    scalability_results = analyze_scalability(base_symbols, max_symbols=50)

    # 本番性能予測
    estimate_production_performance(scalability_results)

    # 最終レポート
    generate_performance_report(
        sequential_results, parallel_results, scalability_results
    )

    print("\n✅ パフォーマンステスト完了")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました")
    except Exception as e:
        print(f"\n\nテストエラー: {e}")
        import traceback

        traceback.print_exc()
