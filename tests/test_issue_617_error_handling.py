#!/usr/bin/env python3
"""
Issue #617: RealMarketDataManager Error Handling Improvements Test
RealMarketDataManagerエラーハンドリング改善テスト
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_technical_error_analysis():
    """テクニカル指標エラー分析テスト"""
    print("=== Issue #617 Technical Error Analysis Test ===\n")

    try:
        from src.day_trade.data.real_market_data import RealMarketDataManager

        manager = RealMarketDataManager()

        # テストケース：各エラータイプの分析
        test_cases = [
            {
                "error": KeyError("Close"),
                "indicator": "RSI",
                "expected_message": "RSIデータ列不足エラー",
                "name": "KeyError（Close列不足）"
            },
            {
                "error": ValueError("empty DataFrame"),
                "indicator": "MACD",
                "expected_message": "MACDデータ不足エラー",
                "name": "ValueError（データ不足）"
            },
            {
                "error": IndexError("list index out of range"),
                "indicator": "VOLUME_RATIO",
                "expected_message": "VOLUME_RATIOデータ配列アクセスエラー",
                "name": "IndexError（配列範囲外）"
            },
            {
                "error": ZeroDivisionError("division by zero"),
                "indicator": "PRICE_CHANGE",
                "expected_message": "PRICE_CHANGEゼロ除算エラー",
                "name": "ZeroDivisionError（ゼロ除算）"
            },
            {
                "error": AttributeError("'NoneType' object has no attribute 'mean'"),
                "indicator": "RSI",
                "expected_message": "RSI属性エラー",
                "name": "AttributeError（属性不足）"
            }
        ]

        print("テクニカル指標エラー分析結果:")
        print("-" * 60)

        results = []

        for test_case in test_cases:
            # テストデータ作成
            test_data = pd.DataFrame({
                'Close': [100, 105, 103, 108, 102],
                'Volume': [1000, 1200, 900, 1500, 800]
            })

            # エラー分析実行
            error_info = manager._analyze_technical_error(
                test_case["error"],
                test_case["indicator"],
                test_data
            )

            # 結果検証
            message_valid = test_case["expected_message"] in error_info['message']
            value_valid = 'value' in error_info and error_info['value'] is not None

            test_passed = message_valid and value_valid
            results.append(test_passed)

            status = "OK PASS" if test_passed else "NG FAIL"
            print(f"{test_case['name']:30s}: {status}")
            print(f"  値: {error_info['value']}")
            print(f"  メッセージ: {error_info['message']}")
            print()

        all_passed = all(results)
        print(f"テクニカル指標エラー分析: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"テクニカル指標エラー分析テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_error_analysis():
    """MLスコアエラー分析テスト"""
    print("\n=== Issue #617 ML Error Analysis Test ===\n")

    try:
        from src.day_trade.data.real_market_data import RealMarketDataManager

        manager = RealMarketDataManager()

        # テストケース：ML特有のエラータイプ分析
        test_cases = [
            {
                "error": KeyError("price_data"),
                "score_name": "TREND_SCORE",
                "expected_message": "TREND_SCORE入力データエラー",
                "name": "KeyError（入力データ不足）"
            },
            {
                "error": ValueError("Invalid value encountered in calculation"),
                "score_name": "VOLATILITY_SCORE",
                "expected_message": "VOLATILITY_SCORE計算エラー",
                "name": "ValueError（計算エラー）"
            },
            {
                "error": RuntimeError("memory allocation failed"),
                "score_name": "PATTERN_SCORE",
                "expected_message": "PATTERN_SCOREメモリ不足エラー",
                "name": "RuntimeError（メモリ不足）"
            },
            {
                "error": IndexError("index out of range"),
                "score_name": "TREND_SCORE",
                "expected_message": "TREND_SCOREデータ配列エラー",
                "name": "IndexError（配列エラー）"
            }
        ]

        print("MLスコアエラー分析結果:")
        print("-" * 60)

        results = []

        for test_case in test_cases:
            # テストデータ作成（MLには十分なデータが必要）
            test_data = pd.DataFrame({
                'Close': [100 + i for i in range(30)],  # 30日分
                'Volume': [1000 + i*10 for i in range(30)]
            })

            # エラー分析実行
            error_info = manager._analyze_ml_error(
                test_case["error"],
                test_case["score_name"],
                test_data
            )

            # 結果検証
            message_valid = test_case["expected_message"] in error_info['message']
            score_valid = 'score' in error_info and 0 <= error_info['score'] <= 100
            confidence_valid = 'confidence' in error_info and 0 <= error_info['confidence'] <= 1

            test_passed = message_valid and score_valid and confidence_valid
            results.append(test_passed)

            status = "OK PASS" if test_passed else "NG FAIL"
            print(f"{test_case['name']:30s}: {status}")
            print(f"  スコア: {error_info['score']}")
            print(f"  信頼度: {error_info['confidence']}")
            print(f"  メッセージ: {error_info['message']}")
            print()

        all_passed = all(results)
        print(f"MLスコアエラー分析: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"MLスコアエラー分析テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_error_handling():
    """統合エラーハンドリングテスト"""
    print("\n=== Issue #617 Integrated Error Handling Test ===\n")

    try:
        from src.day_trade.data.real_market_data import RealMarketDataManager

        manager = RealMarketDataManager()

        print("統合エラーハンドリングテスト:")
        print("-" * 60)

        # 1. RSI計算でのエラーハンドリング
        test_data = pd.DataFrame({
            'Close': [100, 105, 103, 108, 102],
            'Volume': [1000, 1200, 900, 1500, 800]
        })

        # データ不足エラーのシミュレート
        insufficient_data = pd.DataFrame({'Close': [100, 105]})  # 14日未満
        rsi_result = manager.calculate_rsi(insufficient_data)

        rsi_test = rsi_result is not None and isinstance(rsi_result, float)
        print(f"RSIデータ不足ハンドリング: {'OK PASS' if rsi_test else 'NG FAIL'}")
        print(f"  RSI値: {rsi_result}")

        # 2. MACD計算でのエラーハンドリング
        macd_result = manager.calculate_macd(insufficient_data)

        macd_test = macd_result is not None and isinstance(macd_result, float)
        print(f"MACDデータ不足ハンドリング: {'OK PASS' if macd_test else 'NG FAIL'}")
        print(f"  MACD値: {macd_result}")

        # 3. 出来高比率計算でのエラーハンドリング
        volume_result = manager.calculate_volume_ratio(insufficient_data)

        volume_test = volume_result is not None and isinstance(volume_result, float)
        print(f"出来高比率データ不足ハンドリング: {'OK PASS' if volume_test else 'NG FAIL'}")
        print(f"  出来高比率: {volume_result}")

        # 4. MLスコア計算でのエラーハンドリング
        trend_score, trend_conf = manager.generate_ml_trend_score(insufficient_data)

        ml_test = (trend_score is not None and
                  trend_conf is not None and
                  0 <= trend_score <= 100 and
                  0 <= trend_conf <= 1)
        print(f"MLトレンドスコアデータ不足ハンドリング: {'OK PASS' if ml_test else 'NG FAIL'}")
        print(f"  トレンドスコア: {trend_score}, 信頼度: {trend_conf}")

        # 5. モックを使用したエラーシミュレーション
        with patch('pandas.DataFrame.rolling', side_effect=KeyError("Close")):
            try:
                error_rsi = manager.calculate_rsi(test_data)
                mock_test = error_rsi is not None and isinstance(error_rsi, float)
                print(f"モックエラーハンドリング: {'OK PASS' if mock_test else 'NG FAIL'}")
                print(f"  エラー時RSI: {error_rsi}")
            except Exception as e:
                mock_test = False
                print(f"モックエラーハンドリング: NG FAIL - {e}")

        all_passed = rsi_test and macd_test and volume_test and ml_test and mock_test
        print(f"\n統合エラーハンドリング: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"統合エラーハンドリングテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_information_detail():
    """エラー情報詳細化テスト"""
    print("\n=== Issue #617 Error Information Detail Test ===\n")

    try:
        from src.day_trade.data.real_market_data import RealMarketDataManager, calculate_real_technical_indicators

        manager = RealMarketDataManager()

        print("エラー情報詳細化テスト:")
        print("-" * 60)

        # 1. データなし時の技術指標辞書テスト
        with patch.object(manager, 'get_stock_data', return_value=None):
            indicators = calculate_real_technical_indicators("TEST")

            none_test = (
                indicators.get("rsi") is None and
                indicators.get("macd") is None and
                indicators.get("volume_ratio") is None and
                indicators.get("price_change_1d") is None and
                "_error" in indicators
            )

            print(f"データ取得失敗時のNone値化: {'OK PASS' if none_test else 'NG FAIL'}")
            print(f"  指標辞書: {indicators}")

        # 2. エラー情報の詳細度テスト
        test_errors = [
            (KeyError("Close"), "データ列不足エラー"),
            (ValueError("empty"), "データ不足エラー"),
            (IndexError("index"), "データ配列アクセスエラー"),
            (ZeroDivisionError("zero"), "ゼロ除算エラー")
        ]

        detail_results = []
        for error, expected_keyword in test_errors:
            test_data = pd.DataFrame({'Close': [100], 'Volume': [1000]})
            error_info = manager._analyze_technical_error(error, "TEST", test_data)

            detail_valid = expected_keyword in error_info['message']
            detail_results.append(detail_valid)

            status = "OK PASS" if detail_valid else "NG FAIL"
            print(f"エラー詳細 {type(error).__name__}: {status}")
            print(f"  期待キーワード: {expected_keyword}")
            print(f"  実際のメッセージ: {error_info['message']}")

        detail_test = all(detail_results)

        # 3. 値の適切性テスト：エラータイプ別の適切なハンドリング確認
        error_cases = [
            (KeyError("test"), "RSI", 40.0),  # RSI基準値50.0の80% = 40.0
            (ValueError("empty"), "RSI", 25.0),  # RSI基準値50.0の50% = 25.0（データ不足）
            (IndexError("test"), "VOLUME_RATIO", 0.6)  # 基準値1.0の60% = 0.6
        ]

        value_results = []
        for error, indicator, expected_value in error_cases:
            test_data = pd.DataFrame({'Close': [100], 'Volume': [1000]})
            error_info = manager._analyze_technical_error(error, indicator, test_data)

            # 期待値と一致することを確認（適切なエラーハンドリングされていることを証明）
            value_correct = abs(error_info['value'] - expected_value) < 0.1
            value_results.append(value_correct)

            status = "OK PASS" if value_correct else "NG FAIL"
            print(f"値適正 {indicator}: {status}")
            print(f"  期待値: {expected_value} ← 実際値: {error_info['value']}")

        value_test = all(value_results)

        all_passed = none_test and detail_test and value_test
        print(f"\nエラー情報詳細化: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"エラー情報詳細化テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Issue #617 RealMarketDataManager Error Handling Improvements Test\n")

    tests = [
        ("テクニカル指標エラー分析", test_technical_error_analysis),
        ("MLスコアエラー分析", test_ml_error_analysis),
        ("統合エラーハンドリング", test_integrated_error_handling),
        ("エラー情報詳細化", test_error_information_detail)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"実行中: {test_name}")
            print('='*60)

            if test_func():
                print(f"OK {test_name}: PASS")
                passed += 1
            else:
                print(f"NG {test_name}: FAIL")
                failed += 1

        except Exception as e:
            print(f"NG {test_name}: ERROR - {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("OK Issue #617 RealMarketDataManager Error Handling: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #617 RealMarketDataManager Error Handling: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)