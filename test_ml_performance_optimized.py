#!/usr/bin/env python3
"""
ML処理高速化のパフォーマンステスト

高速化前後の比較と85銘柄での処理時間測定
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.config.config_manager import ConfigManager
from day_trade.data.advanced_ml_engine import AdvancedMLEngine
from day_trade.data.batch_data_fetcher import BatchDataFetcher


def test_ml_performance_comparison():
    """ML処理速度比較テスト"""
    print("ML処理高速化比較テスト")
    print("=" * 50)

    # 設定読み込み
    try:
        ConfigManager()
        test_symbols = [
            "7203",
            "8306",
            "9984",
            "6758",
            "4689",  # 主要5銘柄
            "4563",
            "4592",
            "3655",
            "4382",  # 新興4銘柄
        ]

        # データ取得
        print("テスト用データ取得中...")
        data_fetcher = BatchDataFetcher(max_workers=5)
        stock_data = data_fetcher.fetch_multiple_symbols(
            test_symbols, period="60d", use_parallel=True
        )

        successful_symbols = [s for s, data in stock_data.items() if not data.empty]
        print(f"データ取得成功: {len(successful_symbols)}銘柄")

        if len(successful_symbols) < 3:
            print("ERROR: テスト用データが不足しています")
            return

        # テスト1: 通常モード
        print("\n=== テスト1: 通常モード ===")
        ml_engine_normal = AdvancedMLEngine(fast_mode=False)

        start_time = time.time()
        normal_results = {}

        for symbol in successful_symbols:
            data = stock_data[symbol]
            features = ml_engine_normal.prepare_ml_features(data)
            advice = ml_engine_normal.generate_investment_advice(symbol, data, features)
            normal_results[symbol] = advice

            elapsed = time.time() - start_time
            print(
                f"  {symbol}: {advice['advice']} ({advice['confidence']:.1f}%) - {elapsed:.2f}s累計"
            )

        normal_total_time = time.time() - start_time
        normal_avg_time = normal_total_time / len(successful_symbols)

        print("\n通常モード結果:")
        print(f"  - 総処理時間: {normal_total_time:.2f}秒")
        print(f"  - 平均時間/銘柄: {normal_avg_time:.3f}秒")
        print(f"  - 85銘柄推定時間: {normal_avg_time * 85:.1f}秒")

        # テスト2: 高速モード
        print("\n=== テスト2: 高速モード ===")
        ml_engine_fast = AdvancedMLEngine(fast_mode=True)

        start_time = time.time()
        fast_results = {}

        for symbol in successful_symbols:
            data = stock_data[symbol]
            advice = ml_engine_fast.generate_fast_investment_advice(symbol, data)
            fast_results[symbol] = advice

            elapsed = time.time() - start_time
            print(
                f"  {symbol}: {advice['advice']} ({advice['confidence']:.1f}%) - {elapsed:.2f}s累計"
            )

        fast_total_time = time.time() - start_time
        fast_avg_time = fast_total_time / len(successful_symbols)

        print("\n高速モード結果:")
        print(f"  - 総処理時間: {fast_total_time:.2f}秒")
        print(f"  - 平均時間/銘柄: {fast_avg_time:.3f}秒")
        print(f"  - 85銘柄推定時間: {fast_avg_time * 85:.1f}秒")

        # 比較結果
        speedup = normal_avg_time / fast_avg_time if fast_avg_time > 0 else 1
        estimated_85_fast = fast_avg_time * 85

        print("\n=== 比較結果 ===")
        print(f"高速化倍率: {speedup:.1f}倍")
        print(f"85銘柄推定時間: {estimated_85_fast:.1f}秒")
        print(f"目標10秒: {'OK' if estimated_85_fast <= 10 else 'EXCEED'}")

        # 精度比較（簡易）
        print("\n=== 精度比較 ===")
        agreement_count = 0
        for symbol in successful_symbols:
            if normal_results[symbol]["advice"] == fast_results[symbol]["advice"]:
                agreement_count += 1

        accuracy_rate = agreement_count / len(successful_symbols) * 100
        print(
            f"助言一致率: {accuracy_rate:.1f}% ({agreement_count}/{len(successful_symbols)})"
        )

        # 詳細結果表示
        print("\n=== 詳細比較 ===")
        print(f"{'Symbol':>6} | {'Normal':>12} | {'Fast':>12} | {'Match':>5}")
        print("-" * 50)
        for symbol in successful_symbols:
            normal_advice = normal_results[symbol]["advice"]
            fast_advice = fast_results[symbol]["advice"]
            match = "✓" if normal_advice == fast_advice else "✗"
            print(f"{symbol:>6} | {normal_advice:>12} | {fast_advice:>12} | {match:>5}")

        return {
            "normal_time": normal_total_time,
            "fast_time": fast_total_time,
            "speedup": speedup,
            "estimated_85_time": estimated_85_fast,
            "meets_target": estimated_85_fast <= 10,
            "accuracy_rate": accuracy_rate,
        }

    except Exception as e:
        print(f"テストエラー: {e}")
        return None


def test_85_stocks_simulation():
    """85銘柄シミュレーション"""
    print("\n" + "=" * 50)
    print("85銘柄高速処理シミュレーション")
    print("=" * 50)

    try:
        config = ConfigManager()
        symbols_85 = config.get_all_symbols()[:20]  # 実際は20銘柄でテスト

        print(f"シミュレーション対象: {len(symbols_85)}銘柄")

        # データ取得
        print("データ取得中...")
        data_fetcher = BatchDataFetcher(max_workers=10)
        stock_data = data_fetcher.fetch_multiple_symbols(
            symbols_85, period="30d", use_parallel=True
        )

        successful_data = {s: data for s, data in stock_data.items() if not data.empty}
        print(f"データ取得成功: {len(successful_data)}銘柄")

        # 高速モードで一括処理
        ml_engine = AdvancedMLEngine(fast_mode=True)

        print("高速ML分析開始...")
        start_time = time.time()

        results = {}
        for i, (symbol, data) in enumerate(successful_data.items(), 1):
            advice = ml_engine.generate_fast_investment_advice(symbol, data)
            results[symbol] = advice

            if i % 5 == 0:  # 5銘柄ごとに進捗表示
                elapsed = time.time() - start_time
                print(f"  進捗: {i}/{len(successful_data)} ({elapsed:.1f}s)")

        total_time = time.time() - start_time
        avg_time = total_time / len(successful_data)
        estimated_85_time = avg_time * 85

        print("\n=== シミュレーション結果 ===")
        print(f"処理銘柄数: {len(successful_data)}")
        print(f"総処理時間: {total_time:.2f}秒")
        print(f"平均処理時間: {avg_time:.3f}秒/銘柄")
        print(f"85銘柄推定時間: {estimated_85_time:.1f}秒")
        print(f"目標達成: {'YES' if estimated_85_time <= 10 else 'NO'}")

        # 助言サマリー
        advice_summary = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for result in results.values():
            advice_summary[result["advice"]] += 1

        print("\n=== 助言サマリー ===")
        for advice, count in advice_summary.items():
            pct = count / len(results) * 100
            print(f"{advice}: {count}銘柄 ({pct:.1f}%)")

        return estimated_85_time

    except Exception as e:
        print(f"シミュレーションエラー: {e}")
        return None


def main():
    """メインテスト実行"""
    print("ML高速化パフォーマンステスト")
    print("=" * 60)

    # 比較テスト
    comparison_result = test_ml_performance_comparison()

    # 85銘柄シミュレーション
    simulation_time = test_85_stocks_simulation()

    # 最終結果
    print("\n" + "=" * 60)
    print("最終結果サマリー")
    print("=" * 60)

    if comparison_result:
        print(f"高速化倍率: {comparison_result['speedup']:.1f}倍")
        print(f"助言精度維持: {comparison_result['accuracy_rate']:.1f}%")
        print(f"85銘柄推定時間: {comparison_result['estimated_85_time']:.1f}秒")
        print(
            f"目標10秒達成: {'SUCCESS' if comparison_result['meets_target'] else 'FAILED'}"
        )

    if simulation_time:
        print(f"実際シミュレーション: {simulation_time:.1f}秒")

    print("\n高速化実装完了！")


if __name__ == "__main__":
    main()
