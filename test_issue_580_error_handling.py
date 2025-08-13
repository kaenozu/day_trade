#!/usr/bin/env python3
"""
Issue #580: Error Handling Improvements Test
RecommendationEngineエラーハンドリング改善テスト
"""

import sys
import time
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
    print("=== Issue #580 Technical Error Analysis Test ===\n")

    try:
        from src.day_trade.recommendation.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()

        # テストケース：各エラータイプの分析
        test_cases = [
            {
                "error": KeyError("終値"),
                "symbol": "7203",
                "expected_score_range": (25, 35),
                "expected_message": "データ列不足エラー",
                "name": "KeyError（終値列不足）"
            },
            {
                "error": ValueError("empty DataFrame"),
                "symbol": "8306",
                "expected_score_range": (15, 25),
                "expected_message": "データ不足エラー",
                "name": "ValueError（データ不足）"
            },
            {
                "error": IndexError("list index out of range"),
                "symbol": "9984",
                "expected_score_range": (20, 30),
                "expected_message": "データ配列アクセスエラー",
                "name": "IndexError（配列範囲外）"
            },
            {
                "error": AttributeError("'NoneType' object has no attribute 'calculate'"),
                "symbol": "6758",
                "expected_score_range": (30, 40),
                "expected_message": "システム構造エラー",
                "name": "AttributeError（メソッド未実装）"
            },
            {
                "error": ImportError("No module named 'talib'"),
                "symbol": "4689",
                "expected_score_range": (40, 50),
                "expected_message": "ライブラリ依存エラー",
                "name": "ImportError（ライブラリ不足）"
            },
            {
                "error": TimeoutError("Calculation timeout"),
                "symbol": "7267",
                "expected_score_range": (45, 55),
                "expected_message": "処理時間超過エラー",
                "name": "TimeoutError（処理時間超過）"
            }
        ]

        print("テクニカル指標エラー分析結果:")
        print("-" * 60)

        results = []

        for test_case in test_cases:
            # テストデータ作成
            test_data = pd.DataFrame({
                '終値': [100, 105, 103, 108, 102],
                '出来高': [1000, 1200, 900, 1500, 800]
            })

            # エラー分析実行
            error_info = engine._analyze_technical_error(
                test_case["error"],
                test_case["symbol"],
                test_data
            )

            # 結果検証
            score_valid = test_case["expected_score_range"][0] <= error_info['score'] <= test_case["expected_score_range"][1]
            message_valid = test_case["expected_message"] in error_info['message']
            reasons_valid = len(error_info['reasons']) > 0

            test_passed = score_valid and message_valid and reasons_valid
            results.append(test_passed)

            status = "OK PASS" if test_passed else "NG FAIL"
            print(f"{test_case['name']:30s}: {status}")
            print(f"  スコア: {error_info['score']} (期待範囲: {test_case['expected_score_range']})")
            print(f"  メッセージ: {error_info['message']}")
            print(f"  理由: {', '.join(error_info['reasons'])}")
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
    """ML予測エラー分析テスト"""
    print("\n=== Issue #580 ML Error Analysis Test ===\n")

    try:
        from src.day_trade.recommendation.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()

        # テストケース：ML特有のエラータイプ分析
        test_cases = [
            {
                "error": KeyError("price_features"),
                "symbol": "7203",
                "expected_score_range": (30, 40),
                "expected_message": "ML特徴量エラー",
                "name": "KeyError（特徴量不足）"
            },
            {
                "error": ValueError("Input contains NaN, infinity"),
                "symbol": "8306",
                "expected_score_range": (35, 45),
                "expected_message": "ML予測計算エラー",
                "name": "ValueError（計算エラー）"
            },
            {
                "error": ValueError("shape mismatch: (10,) vs (20,)"),
                "symbol": "9984",
                "expected_score_range": (25, 35),
                "expected_message": "MLデータ形状エラー",
                "name": "ValueError（形状エラー）"
            },
            {
                "error": RuntimeError("CUDA out of memory"),
                "symbol": "6758",
                "expected_score_range": (50, 60),
                "expected_message": "GPU利用不可",
                "name": "RuntimeError（CUDA/GPU）"
            },
            {
                "error": RuntimeError("Model prediction failed"),
                "symbol": "4689",
                "expected_score_range": (30, 40),
                "expected_message": "ML予測処理エラー",
                "name": "RuntimeError（予測失敗）"
            },
            {
                "error": MemoryError("Cannot allocate memory"),
                "symbol": "7267",
                "expected_score_range": (40, 50),
                "expected_message": "MLメモリ不足エラー",
                "name": "MemoryError（メモリ不足）"
            },
            {
                "error": ImportError("No module named 'torch'"),
                "symbol": "2914",
                "expected_score_range": (45, 55),
                "expected_message": "ML依存ライブラリエラー",
                "name": "ImportError（MLライブラリ）"
            }
        ]

        print("ML予測エラー分析結果:")
        print("-" * 60)

        results = []

        for test_case in test_cases:
            # テストデータ作成（MLには十分なデータが必要）
            test_data = pd.DataFrame({
                '終値': [100 + i for i in range(30)],  # 30日分
                '出来高': [1000 + i*10 for i in range(30)]
            })

            # エラー分析実行
            error_info = engine._analyze_ml_error(
                test_case["error"],
                test_case["symbol"],
                test_data
            )

            # 結果検証
            score_valid = test_case["expected_score_range"][0] <= error_info['score'] <= test_case["expected_score_range"][1]
            message_valid = test_case["expected_message"] in error_info['message']
            reasons_valid = len(error_info['reasons']) > 0

            test_passed = score_valid and message_valid and reasons_valid
            results.append(test_passed)

            status = "OK PASS" if test_passed else "NG FAIL"
            print(f"{test_case['name']:30s}: {status}")
            print(f"  スコア: {error_info['score']} (期待範囲: {test_case['expected_score_range']})")
            print(f"  メッセージ: {error_info['message']}")
            print(f"  理由: {', '.join(error_info['reasons'])}")
            print()

        all_passed = all(results)
        print(f"ML予測エラー分析: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"ML予測エラー分析テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_error_handling():
    """統合エラーハンドリングテスト"""
    print("\n=== Issue #580 Integrated Error Handling Test ===\n")

    try:
        from src.day_trade.recommendation.recommendation_engine import RecommendationEngine

        # テストデータ作成
        test_data = pd.DataFrame({
            '終値': [100, 105, 103, 108, 102],
            '出来高': [1000, 1200, 900, 1500, 800]
        })

        # エラーを発生させるモック設定
        engine = RecommendationEngine()

        print("統合エラーハンドリングテスト:")
        print("-" * 60)

        # 1. テクニカル指標計算でのエラーハンドリング
        with patch.object(engine.technical_manager, 'calculate_indicators', side_effect=KeyError("終値")):
            score, reasons = await_result(engine._calculate_technical_score("7203", test_data))

            technical_test = (
                20 <= score <= 40 and
                len(reasons) > 0 and
                any("データ構造エラー" in reason for reason in reasons)
            )

            print(f"テクニカル指標エラーハンドリング: {'OK PASS' if technical_test else 'NG FAIL'}")
            print(f"  スコア: {score}, 理由: {reasons[0] if reasons else 'なし'}")

        # 2. ML予測計算でのエラーハンドリング
        with patch.object(engine.ml_engine, 'calculate_advanced_technical_indicators', side_effect=ValueError("shape mismatch")):
            score, reasons = await_result(engine._calculate_ml_score("8306", test_data))

            ml_test = (
                25 <= score <= 40 and
                len(reasons) > 0 and
                any("MLモデルエラー" in reason for reason in reasons)
            )

            print(f"ML予測エラーハンドリング: {'OK PASS' if ml_test else 'NG FAIL'}")
            print(f"  スコア: {score}, 理由: {reasons[0] if reasons else 'なし'}")

        # 3. 複数エラーでの連続性テスト
        errors_handled = 0
        total_errors = 0

        error_types = [KeyError, ValueError, IndexError, AttributeError, ImportError]

        for error_type in error_types:
            try:
                total_errors += 1
                with patch.object(engine.technical_manager, 'calculate_indicators', side_effect=error_type("test")):
                    score, reasons = await_result(engine._calculate_technical_score("test", test_data))
                    if 0 <= score <= 100 and len(reasons) > 0:
                        errors_handled += 1
            except Exception:
                pass  # エラーハンドリングのテストなのでキャッチ

        continuity_test = errors_handled >= total_errors * 0.8  # 80%以上成功

        print(f"連続エラー処理: {'OK PASS' if continuity_test else 'NG FAIL'}")
        print(f"  処理成功率: {errors_handled}/{total_errors} ({errors_handled/total_errors*100:.1f}%)")

        all_passed = technical_test and ml_test and continuity_test
        print(f"\n統合エラーハンドリング: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"統合エラーハンドリングテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_accuracy():
    """スコア精度テスト"""
    print("\n=== Issue #580 Score Accuracy Test ===\n")

    try:
        from src.day_trade.recommendation.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()

        print("スコア精度テスト:")
        print("-" * 60)

        # 各エラータイプでスコアが適切な範囲にあることを確認
        test_cases = [
            ("深刻なデータ不足", ValueError("empty"), 15, 30),
            ("軽微なデータ不整合", KeyError("column"), 30, 45),
            ("システム制限", ImportError("library"), 40, 55),
            ("タイムアウト", TimeoutError("timeout"), 45, 55)
        ]

        results = []
        test_data = pd.DataFrame({'終値': [100], '出来高': [1000]})  # 最小データ

        for name, error, min_score, max_score in test_cases:
            # テクニカル指標エラー分析
            tech_info = engine._analyze_technical_error(error, "TEST", test_data)
            tech_valid = min_score <= tech_info['score'] <= max_score

            # ML予測エラー分析
            ml_info = engine._analyze_ml_error(error, "TEST", test_data)
            ml_valid = min_score <= ml_info['score'] <= max_score

            test_passed = tech_valid and ml_valid
            results.append(test_passed)

            status = "OK PASS" if test_passed else "NG FAIL"
            print(f"{name:20s}: {status}")
            print(f"  テクニカル: {tech_info['score']:.1f} (範囲: {min_score}-{max_score})")
            print(f"  ML: {ml_info['score']:.1f} (範囲: {min_score}-{max_score})")
            print()

        # スコア一貫性テスト：同じエラーは同じスコア範囲を返すか
        consistency_test = True
        same_error = KeyError("test_consistency")

        scores = []
        for _ in range(5):
            info = engine._analyze_technical_error(same_error, "TEST", test_data)
            scores.append(info['score'])

        consistency_test = len(set(scores)) == 1  # すべて同じスコア

        print(f"スコア一貫性: {'OK PASS' if consistency_test else 'NG FAIL'}")
        print(f"  同一エラーのスコア: {set(scores)}")

        all_passed = all(results) and consistency_test
        print(f"\nスコア精度テスト: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"スコア精度テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def await_result(coro):
    """async関数の結果を取得するヘルパー"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 既にイベントループが実行中の場合
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def main():
    """メインテスト実行"""
    print("Issue #580 Error Handling Improvements Test\n")

    tests = [
        ("テクニカル指標エラー分析", test_technical_error_analysis),
        ("ML予測エラー分析", test_ml_error_analysis),
        ("統合エラーハンドリング", test_integrated_error_handling),
        ("スコア精度", test_score_accuracy)
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
        print("OK Issue #580 Error Handling: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #580 Error Handling: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)