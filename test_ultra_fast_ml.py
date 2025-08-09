#!/usr/bin/env python3
"""
超高速ML処理のベンチマークテスト

85銘柄を10秒以下で処理する目標のテスト
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.config.config_manager import ConfigManager
from day_trade.data.batch_data_fetcher import BatchDataFetcher
from day_trade.data.ultra_fast_ml_engine import UltraFastMLEngine


def test_ultra_fast_performance():
    """超高速ML処理テスト"""
    print("超高速ML処理ベンチマーク")
    print("=" * 50)

    try:
        # 設定読み込み
        ConfigManager()

        # テスト銘柄（多めに設定）
        test_symbols = [
            "7203", "8306", "9984", "6758", "4689",  # 主要5銘柄
            "4563", "4592", "3655", "4382", "4475",  # 新興5銘柄
            "7267", "6861", "2914", "9434", "8001",  # 追加5銘柄
            "7779", "3692", "4592", "4564", "4588",  # さらに5銘柄
        ]

        print(f"テスト対象: {len(test_symbols)}銘柄")

        # データ取得
        print("データ取得中...")
        data_fetcher = BatchDataFetcher(max_workers=10)
        stock_data = data_fetcher.fetch_multiple_symbols(
            test_symbols, period="30d", use_parallel=True
        )

        successful_data = {s: data for s, data in stock_data.items() if not data.empty}
        print(f"データ取得成功: {len(successful_data)}銘柄")

        if len(successful_data) < 10:
            print(f"WARNING: データ不足（{len(successful_data)}銘柄）")

        # 超高速エンジン初期化
        ultra_engine = UltraFastMLEngine()

        # ウォームアップ（初回キャッシュ構築）
        print("ウォームアップ実行...")
        warmup_symbols = list(successful_data.keys())[:3]
        for symbol in warmup_symbols:
            ultra_engine.ultra_fast_advice(symbol, successful_data[symbol])

        # 本測定
        print("本測定開始...")
        start_time = time.time()

        results = {}
        for i, (symbol, data) in enumerate(successful_data.items(), 1):
            advice = ultra_engine.ultra_fast_advice(symbol, data)
            results[symbol] = advice

            # 進捗表示
            if i % 5 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  進捗: {i}/{len(successful_data)} ({elapsed:.2f}s, {rate:.1f}銘柄/s)")

        total_time = time.time() - start_time

        # 結果分析
        avg_time = total_time / len(successful_data)
        estimated_85_time = avg_time * 85
        throughput = len(successful_data) / total_time

        print("\n=== 超高速処理結果 ===")
        print(f"処理銘柄数: {len(successful_data)}")
        print(f"総処理時間: {total_time:.2f}秒")
        print(f"平均処理時間: {avg_time:.4f}秒/銘柄")
        print(f"スループット: {throughput:.1f}銘柄/秒")
        print(f"85銘柄推定時間: {estimated_85_time:.1f}秒")
        print(f"目標10秒達成: {'SUCCESS' if estimated_85_time <= 10 else 'FAILED'}")

        # キャッシュ効果分析
        cache_info = ultra_engine.get_cache_info()
        print("\n=== キャッシュ効果 ===")
        print(f"訓練済みモデル: {cache_info['trained_models']}個")
        print(f"特徴量キャッシュ: {cache_info['feature_cache']}個")
        print(f"メモリ使用量: {cache_info['memory_usage_kb']:.1f}KB")

        # 助言分布
        advice_dist = {"BUY": 0, "SELL": 0, "HOLD": 0}
        confidence_sum = 0

        for result in results.values():
            advice_dist[result["advice"]] += 1
            confidence_sum += result["confidence"]

        avg_confidence = confidence_sum / len(results) if results else 0

        print("\n=== 助言分析 ===")
        for advice, count in advice_dist.items():
            pct = count / len(results) * 100 if results else 0
            print(f"{advice}: {count}銘柄 ({pct:.1f}%)")
        print(f"平均信頼度: {avg_confidence:.1f}%")

        # 詳細結果（上位10銘柄）
        print("\n=== 詳細結果（上位10銘柄） ===")
        print(f"{'Symbol':>6} | {'Advice':>5} | {'Conf':>4} | {'Reason':>15}")
        print("-" * 40)

        for i, (symbol, result) in enumerate(results.items()):
            if i >= 10:
                break
            print(f"{symbol:>6} | {result['advice']:>5} | {result['confidence']:>4.0f}% | {result['reason']:>15}")

        return {
            "total_time": total_time,
            "avg_time": avg_time,
            "estimated_85_time": estimated_85_time,
            "throughput": throughput,
            "meets_target": estimated_85_time <= 10,
            "processed_count": len(successful_data)
        }

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def stress_test_85_symbols():
    """85銘柄ストレステスト（推定）"""
    print("\n" + "=" * 50)
    print("85銘柄ストレステスト")
    print("=" * 50)

    try:
        # 実際の85銘柄のサブセットでテスト
        ConfigManager()

        # 実際の85銘柄リストを作成
        all_symbols = [
            # Technology (10銘柄)
            "9984", "6758", "4689", "4755", "3659", "9613", "2432", "4385", "4704", "4751",
            # Financial (7銘柄)
            "8306", "8411", "8766", "8316", "8604", "7182", "8795",
            # Transportation (5銘柄)
            "7203", "7267", "9101", "9201", "9202",
            # Industrial (7銘柄)
            "6861", "7011", "6503", "6954", "6367", "7751", "6981",
            # Healthcare (2銘柄)
            "4502", "4523",
            # Consumer (6銘柄)
            "2914", "7974", "3382", "2801", "2502", "9983",
            # DayTrading (12銘柄)
            "4478", "4485", "4490", "3900", "3774", "4382", "4386", "4475", "4421", "3655", "3844", "4833",
            # BioTech (8銘柄)
            "4563", "4592", "4564", "4588", "4596", "4591", "4565", "7707",
            # Gaming (5銘柄)
            "3692", "3656", "3760", "9449", "4726",
            # FutureTech (5銘柄)
            "7779", "6178", "4847", "4598", "4880",
            # その他 (18銘柄)
            "8001", "8058", "8031", "8053", "5401", "4005", "4061", "5020", "9501", "9502",
            "9434", "9437", "9432", "8802", "1801", "1803", "4777", "3776"
        ]

        # ランダムに40銘柄選択（より大規模テスト）
        import random
        random.seed(42)
        test_symbols = random.sample(all_symbols, min(40, len(all_symbols)))

        print(f"ストレステスト対象: {len(test_symbols)}銘柄（85銘柄の代表）")

        # データ取得
        data_fetcher = BatchDataFetcher(max_workers=15)
        stock_data = data_fetcher.fetch_multiple_symbols(
            test_symbols, period="20d", use_parallel=True
        )

        successful_data = {s: data for s, data in stock_data.items() if not data.empty}
        print(f"データ取得成功: {len(successful_data)}銘柄")

        # 超高速バッチ処理
        ultra_engine = UltraFastMLEngine()

        print("バッチ処理開始...")
        start_time = time.time()

        # バッチ処理実行
        ultra_engine.batch_ultra_fast_analysis(successful_data)

        batch_time = time.time() - start_time

        # 85銘柄推定
        scale_factor = 85 / len(successful_data) if successful_data else 1
        estimated_85_time = batch_time * scale_factor

        print("\n=== ストレステスト結果 ===")
        print(f"実測銘柄数: {len(successful_data)}")
        print(f"実測処理時間: {batch_time:.2f}秒")
        print(f"スケール倍率: {scale_factor:.1f}倍")
        print(f"85銘柄推定時間: {estimated_85_time:.1f}秒")
        print(f"目標達成: {'SUCCESS' if estimated_85_time <= 10 else 'FAILED'}")

        if estimated_85_time <= 10:
            print(f"🎉 目標達成！85銘柄を{estimated_85_time:.1f}秒で処理可能")
        else:
            shortage = estimated_85_time - 10
            print(f"⚠️  目標まであと{shortage:.1f}秒の短縮が必要")

        return estimated_85_time

    except Exception as e:
        print(f"ストレステストエラー: {e}")
        return None


def main():
    """メインテスト実行"""
    print("超高速ML処理 最終ベンチマーク")
    print("=" * 60)

    # パフォーマンステスト
    result = test_ultra_fast_performance()

    # ストレステスト
    stress_result = stress_test_85_symbols()

    # 最終判定
    print("\n" + "=" * 60)
    print("最終判定")
    print("=" * 60)

    if result and stress_result:
        min_estimated_time = min(result["estimated_85_time"], stress_result)
        max_estimated_time = max(result["estimated_85_time"], stress_result)

        print(f"85銘柄処理推定時間: {min_estimated_time:.1f}秒 〜 {max_estimated_time:.1f}秒")

        if max_estimated_time <= 10:
            print("🎉 目標達成！85銘柄を10秒以内で処理可能")
            print("✅ パフォーマンス最適化完了")
        else:
            print(f"⚠️  目標未達成（{max_estimated_time:.1f}秒 > 10秒）")
            print("🔄 さらなる最適化が必要")
    else:
        print("❌ テスト実行に失敗")

    print("\n超高速ML処理実装テスト完了")


if __name__ == "__main__":
    main()
