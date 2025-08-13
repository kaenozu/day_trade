#!/usr/bin/env python3
"""
実市場データ検証テスト

Issue #321: 実データでの最終動作確認テスト
実際の市場データの取得・品質検証の実行テスト
"""

import sys
import time
from pathlib import Path

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from day_trade.data.real_data_validator import DataQuality, RealDataValidator
    from day_trade.utils.performance_monitor import get_performance_monitor
    from day_trade.utils.structured_logging import get_structured_logger
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("一部機能をスキップして実行します")

print("REAL MARKET DATA VALIDATION TEST")
print("Issue #321: 実データでの最終動作確認テスト")
print("=" * 60)


def test_real_data_validation():
    """実市場データ検証テスト"""
    print("\n=== 実市場データ検証テスト ===")

    try:
        # 実データ検証システム初期化
        validator = RealDataValidator()
        print(f"対象銘柄数: {len(validator.test_symbols)}銘柄")
        print("検証開始...")

        # 実市場データ検証実行（最近30日）
        start_time = time.time()
        validation_results = validator.validate_market_data(days=30)
        validation_time = time.time() - start_time

        print(f"検証時間: {validation_time:.2f}秒")
        print(f"検証完了: {len(validation_results)}銘柄")

        # 結果分析
        quality_counts = {}
        usable_count = 0

        for metrics in validation_results.values():
            quality = metrics.overall_quality
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

            if quality in [DataQuality.EXCELLENT, DataQuality.GOOD]:
                usable_count += 1

        print("\n品質分布:")
        for quality, count in quality_counts.items():
            percentage = count / len(validation_results) * 100
            print(f"  {quality.value.upper()}: {count}銘柄 ({percentage:.1f}%)")

        print(f"\nML分析利用可能銘柄: {usable_count}銘柄")

        # 成功条件確認
        success_conditions = [
            len(validation_results) >= 80,  # 80銘柄以上検証完了
            usable_count >= 60,  # 60銘柄以上利用可能
            validation_time <= 300,  # 5分以内で完了
        ]

        if all(success_conditions):
            print("✅ 実市場データ検証テスト: 成功")
            print(f"  - 検証銘柄数: {len(validation_results)}")
            print(f"  - 利用可能銘柄: {usable_count}")
            print(f"  - 検証時間: {validation_time:.1f}秒")
            return True
        else:
            print("❌ 実市場データ検証テスト: 失敗")
            print(
                f"  - 検証銘柄数不足: {len(validation_results)} < 80"
                if len(validation_results) < 80
                else ""
            )
            print(
                f"  - 利用可能銘柄不足: {usable_count} < 60"
                if usable_count < 60
                else ""
            )
            print(
                f"  - 検証時間超過: {validation_time:.1f}s > 300s"
                if validation_time > 300
                else ""
            )
            return False

    except Exception as e:
        print(f"実市場データ検証テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_quality_analysis():
    """データ品質分析テスト"""
    print("\n=== データ品質分析テスト ===")

    try:
        validator = RealDataValidator()

        # 少数銘柄での詳細分析（時間短縮のため）
        test_symbols = validator.topix_core30[:10]  # 上位10銘柄
        validator.test_symbols = test_symbols

        print(f"詳細分析対象: {len(test_symbols)}銘柄")

        # データ検証実行
        validator.validate_market_data(days=7)  # 1週間データ

        # 詳細レポート生成
        report = validator.generate_validation_report()
        print("\n" + report)

        # 高品質銘柄抽出
        excellent = validator.get_high_quality_symbols(DataQuality.EXCELLENT)
        good = validator.get_high_quality_symbols(DataQuality.GOOD)

        print(f"\n優良品質銘柄: {excellent}")
        print(f"良品質銘柄: {good}")

        # 分析結果確認
        total_high_quality = len(excellent) + len(good)
        success = total_high_quality >= len(test_symbols) * 0.7  # 70%以上が高品質

        if success:
            print("✅ データ品質分析テスト: 成功")
            print(
                f"  - 高品質率: {total_high_quality}/{len(test_symbols)} ({total_high_quality/len(test_symbols)*100:.1f}%)"
            )
        else:
            print("❌ データ品質分析テスト: 失敗")
            print(
                f"  - 高品質率不足: {total_high_quality}/{len(test_symbols)} ({total_high_quality/len(test_symbols)*100:.1f}%)"
            )

        return success

    except Exception as e:
        print(f"データ品質分析テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_network_connectivity():
    """ネットワーク接続テスト"""
    print("\n=== ネットワーク接続テスト ===")

    try:
        import urllib.request

        import yfinance as yf

        # 基本ネットワーク接続テスト
        try:
            urllib.request.urlopen("https://www.google.com", timeout=5)
            print("✅ 基本ネットワーク接続: 成功")
            network_ok = True
        except:
            print("❌ 基本ネットワーク接続: 失敗")
            network_ok = False

        # Yahoo Finance API接続テスト
        try:
            ticker = yf.Ticker("7203.T")  # トヨタ
            print("✅ Yahoo Finance API: 成功")
            api_ok = True
        except:
            print("❌ Yahoo Finance API: 失敗")
            api_ok = False

        # データ取得テスト
        try:
            hist = ticker.history(period="5d")
            if not hist.empty:
                print(f"✅ データ取得テスト: 成功 ({len(hist)}レコード)")
                data_ok = True
            else:
                print("❌ データ取得テスト: 失敗（データなし）")
                data_ok = False
        except:
            print("❌ データ取得テスト: 失敗")
            data_ok = False

        success = network_ok and api_ok and data_ok

        if success:
            print("✅ ネットワーク接続テスト: 全て成功")
        else:
            print("❌ ネットワーク接続テスト: 一部失敗")
            print("  - インターネット接続を確認してください")
            print("  - APIアクセス制限がないか確認してください")

        return success

    except Exception as e:
        print(f"ネットワーク接続テストエラー: {e}")
        return False


def main():
    """メイン実行"""
    print("実市場データ検証テスト開始...")

    test_results = []

    # テスト実行
    tests = [
        ("ネットワーク接続テスト", test_network_connectivity),
        ("データ品質分析テスト", test_data_quality_analysis),
        ("実市場データ検証テスト", test_real_data_validation),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} で例外発生: {e}")
            test_results.append((test_name, False))

    # 最終結果
    print(f"\n{'='*60}")
    print("最終テスト結果")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    for test_name, success in test_results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {status} {test_name}")

    success_rate = passed / total
    print(f"\n成功率: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("\n🎉 実市場データ検証システム: 準備完了")
        print("実データでのML分析テストに進むことができます")
        return True
    else:
        print("\n⚠️ 一部テストに失敗しました")
        print("問題を解決してから次のステップに進んでください")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n{'='*60}")
        print("次の推奨アクション:")
        print("  1. 実データML分析テスト実行")
        print("  2. 実市場データポートフォリオ最適化")
        print("  3. 実データバックテスト実行")
        print(f"{'='*60}")

    exit(0 if success else 1)
