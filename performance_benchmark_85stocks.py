#!/usr/bin/env python3
"""
85銘柄対応パフォーマンスベンチマーク

現在の処理速度を測定し、ボトルネックを特定する
"""

import asyncio
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.data.advanced_ml_engine import AdvancedMLEngine  # noqa: E402
from day_trade.data.batch_data_fetcher import BatchDataFetcher  # noqa: E402
from day_trade.utils.performance_monitor import PerformanceMonitor  # noqa: E402


def load_stock_symbols():
    """主要85銘柄のリストを返す"""
    return [
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
        "4755",
        "3659",
        "9613",
        "2432",
        "4385",
        "9437",
        "4704",
        "4751",
        "8058",
        "8411",
        "8766",
        "8316",
        "8031",
        "8053",
        "7751",
        "6981",
        "5401",
        "7011",
        "6503",
        "6954",
        "7974",
        "6367",
        "4502",
        "3382",
        "2801",
        "2502",
        "4523",
        "9983",
        "9101",
        "9201",
        "9202",
        "5020",
        "9501",
        "9502",
        "8802",
        "1801",
        "1803",
        "8604",
        "7182",
        "4005",
        "4061",
        "8795",
        "9432",
        "4777",
        "3776",
        "4478",
        "4485",
        "4490",
        "3900",
        "3774",
        "4382",
        "4386",
        "4475",
        "4421",
        "3655",
        "3844",
        "4833",
        "4563",
        "4592",
        "4564",
        "4588",
        "4596",
        "4591",
        "4565",
        "7707",
        "3692",
        "3656",
        "3760",
        "9449",
        "4726",
        "7779",
        "6178",
        "4847",
        "4598",
        "4880",
    ]


async def benchmark_data_fetching():
    """データ取得性能ベンチマーク"""
    print("=== データ取得性能ベンチマーク ===")

    symbols = load_stock_symbols()
    print(f"対象銘柄数: {len(symbols)}")

    fetcher = BatchDataFetcher(max_workers=5)
    monitor = PerformanceMonitor()

    # シーケンシャル取得テスト
    print("\n1. シーケンシャル取得テスト (10銘柄サンプル)")
    sample_symbols = symbols[:10]

    with monitor.measure_operation("sequential_fetch_10"):
        start_time = time.time()
        sequential_data = fetcher.fetch_multiple_symbols(
            sample_symbols, period="5d", use_parallel=False
        )
        sequential_time = time.time() - start_time

    print(f"  - 取得時間: {sequential_time:.2f}秒")
    print(f"  - 銘柄あたり: {sequential_time/len(sample_symbols):.2f}秒")
    print(
        f"  - 成功率: {len(sequential_data)}/{len(sample_symbols)} = {len(sequential_data)/len(sample_symbols)*100:.1f}%"
    )

    # 並列取得テスト (10銘柄)
    print("\n2. 並列取得テスト (10銘柄サンプル)")

    with monitor.measure_operation("parallel_fetch_10"):
        start_time = time.time()
        parallel_data = fetcher.fetch_multiple_symbols(
            sample_symbols, period="5d", use_parallel=True
        )
        parallel_time = time.time() - start_time

    print(f"  - 取得時間: {parallel_time:.2f}秒")
    print(f"  - 銘柄あたり: {parallel_time/len(sample_symbols):.2f}秒")
    print(
        f"  - 成功率: {len(parallel_data)}/{len(sample_symbols)} = {len(parallel_data)/len(sample_symbols)*100:.1f}%"
    )
    print(f"  - 速度改善: {sequential_time/parallel_time:.1f}x倍高速")

    # 全銘柄取得予測
    print("\n3. 全85銘柄取得時間予測")
    estimated_time = (parallel_time / len(sample_symbols)) * len(symbols)
    print(f"  - 予想時間: {estimated_time:.1f}秒")
    print(f"  - 30秒目標: {'✅ OK' if estimated_time <= 30 else '❌ 超過'}")

    return sequential_data if sequential_data else parallel_data


async def benchmark_ml_processing():
    """ML処理性能ベンチマーク"""
    print("\n=== ML処理性能ベンチマーク ===")

    # サンプルデータでMLエンジンテスト
    symbols = load_stock_symbols()[:5]  # 5銘柄でテスト
    fetcher = BatchDataFetcher(max_workers=3)
    monitor = PerformanceMonitor()

    print(f"ML処理テスト対象: {len(symbols)}銘柄")

    # データ取得
    stock_data = fetcher.fetch_multiple_symbols(symbols, period="30d", use_parallel=True)

    if not stock_data:
        print("❌ データ取得に失敗しました")
        return

    # MLエンジン初期化
    ml_engine = AdvancedMLEngine()

    total_processing_time = 0
    successful_analyses = 0

    for symbol, data in stock_data.items():
        if data.empty:
            continue

        print(f"\n銘柄 {symbol} ML分析:")

        with monitor.measure_operation(f"ml_analysis_{symbol}"):
            start_time = time.time()
            try:
                # 特徴量準備
                features = ml_engine.prepare_ml_features(data)

                # ML予測
                (
                    trend_score,
                    vol_score,
                    pattern_score,
                ) = ml_engine.predict_advanced_scores(symbol, data, features)

                # 投資助言生成
                advice = ml_engine.generate_investment_advice(symbol, data, features)

                processing_time = time.time() - start_time
                total_processing_time += processing_time
                successful_analyses += 1

                print(f"  - 処理時間: {processing_time:.3f}秒")
                print(f"  - 推奨: {advice['advice']} (信頼度: {advice['confidence']:.1f}%)")

            except Exception as e:
                processing_time = time.time() - start_time
                print(f"  - エラー: {e}")
                print(f"  - 失敗時間: {processing_time:.3f}秒")

    # 全銘柄処理時間予測
    if successful_analyses > 0:
        avg_time_per_stock = total_processing_time / successful_analyses
        total_symbols = len(load_stock_symbols())
        estimated_total_time = avg_time_per_stock * total_symbols

        print("\n4. ML処理時間サマリー")
        print(f"  - 平均処理時間/銘柄: {avg_time_per_stock:.3f}秒")
        print(f"  - 85銘柄予想時間: {estimated_total_time:.1f}秒")
        print(f"  - 10秒目標: {'✅ OK' if estimated_total_time <= 10 else '❌ 超過'}")
        print(
            f"  - 成功率: {successful_analyses}/{len(stock_data)} = {successful_analyses/len(stock_data)*100:.1f}%"
        )


async def benchmark_memory_usage():
    """メモリ使用量ベンチマーク"""
    print("\n=== メモリ使用量ベンチマーク ===")

    import psutil

    process = psutil.Process()

    # 初期メモリ使用量
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初期メモリ使用量: {initial_memory:.1f}MB")

    # データ読み込み後
    symbols = load_stock_symbols()
    fetcher = BatchDataFetcher()

    sample_data = fetcher.fetch_multiple_symbols(symbols[:20], period="30d")
    after_data_memory = process.memory_info().rss / 1024 / 1024

    print(
        f"データ読み込み後: {after_data_memory:.1f}MB (+{after_data_memory-initial_memory:.1f}MB)"
    )

    # ML処理後
    if sample_data:
        ml_engine = AdvancedMLEngine()
        for symbol, data in list(sample_data.items())[:3]:
            if not data.empty:
                features = ml_engine.prepare_ml_features(data)
                ml_engine.predict_advanced_scores(symbol, data, features)

    after_ml_memory = process.memory_info().rss / 1024 / 1024
    print(f"ML処理後: {after_ml_memory:.1f}MB (+{after_ml_memory-after_data_memory:.1f}MB)")

    # 85銘柄予想メモリ使用量
    data_memory_per_stock = (after_data_memory - initial_memory) / min(20, len(sample_data))
    ml_memory_per_stock = (after_ml_memory - after_data_memory) / 3

    estimated_total_memory = initial_memory + data_memory_per_stock * 85 + ml_memory_per_stock * 85

    print(f"85銘柄予想メモリ: {estimated_total_memory:.1f}MB")
    print(f"2GB目標: {'✅ OK' if estimated_total_memory <= 2048 else '❌ 超過'}")


async def main():
    """メインベンチマーク実行"""
    print("🔍 85銘柄対応パフォーマンスベンチマーク開始")
    print("=" * 60)

    try:
        # データ取得ベンチマーク
        await benchmark_data_fetching()

        # ML処理ベンチマーク
        await benchmark_ml_processing()

        # メモリ使用量ベンチマーク
        await benchmark_memory_usage()

        print("\n" + "=" * 60)
        print("✅ ベンチマーク完了")

        print("\n📋 改善提案:")
        print("1. データ取得: max_workers増加 (3→8)")
        print("2. ML処理: 特徴量計算の並列化")
        print("3. メモリ: バッチサイズによる分割処理")
        print("4. キャッシュ: 直近データの保持戦略")

    except KeyboardInterrupt:
        print("\n⏹️ ベンチマーク中断")
    except Exception as e:
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
