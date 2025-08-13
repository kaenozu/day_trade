#!/usr/bin/env python3
"""
Issue #628: Log Error Context Source Tracing Test
ログエラーコンテキストソーストレース機能テスト
"""

import sys
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_log_error_with_context_source_tracing():
    """ログエラーコンテキストソーストレース機能テスト"""
    print("=== Issue #628 Log Error Context Source Tracing Test ===\n")

    try:
        from src.day_trade.utils.logging_config import log_error_with_context, setup_logging

        # ロギング設定
        setup_logging()

        # ログ出力をキャプチャするためのStringIO
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        handler.setLevel(logging.ERROR)

        # テスト用のエラーと仮想的なソースモジュールでテスト
        test_cases = [
            {
                "name": "自動ソース検出テスト",
                "error": ValueError("Test error for source tracing"),
                "context": {"test_data": "automatic_detection"},
                "source_module": None,  # 自動検出
                "expected_in_log": ["__main__", "ValueError"]
            },
            {
                "name": "明示的ソース指定テスト",
                "error": KeyError("missing_key"),
                "context": {"operation": "data_access"},
                "source_module": "src.day_trade.data.stock_fetcher",
                "expected_in_log": ["stock_fetcher", "KeyError", "missing_key"]
            }
        ]

        print("ログエラーコンテキストソーストレース結果:")
        print("-" * 60)

        results = []

        for i, test_case in enumerate(test_cases):
            # ログ出力をクリア
            log_output.truncate(0)
            log_output.seek(0)

            # テスト用ロガーを設定
            test_logger_name = test_case.get("source_module", __name__)
            test_logger = logging.getLogger(test_logger_name)
            test_logger.addHandler(handler)
            test_logger.setLevel(logging.ERROR)

            # ログエラー関数を呼び出し
            log_error_with_context(
                test_case["error"],
                test_case["context"],
                test_case["source_module"]
            )

            # ログ出力を取得
            log_content = log_output.getvalue()

            # 検証
            all_expected_found = all(
                expected in log_content for expected in test_case["expected_in_log"]
            )

            results.append(all_expected_found)

            status = "OK PASS" if all_expected_found else "NG FAIL"
            print(f"{test_case['name']:30s}: {status}")
            print(f"  エラー: {type(test_case['error']).__name__}: {test_case['error']}")
            print(f"  期待キーワード: {test_case['expected_in_log']}")
            print(f"  ログ内容（一部）: {log_content[:100]}...")
            print()

            # クリーンアップ
            test_logger.removeHandler(handler)

        all_passed = all(results)
        print(f"ログエラーコンテキストソーストレース: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"ログエラーコンテキストソーストレーステストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_caller_info_functionality():
    """呼び出し元情報取得機能テスト"""
    print("\n=== Issue #628 Get Caller Info Functionality Test ===\n")

    try:
        from src.day_trade.utils.logging_config import get_caller_info

        def test_function_level_1():
            """テスト関数レベル1"""
            return test_function_level_2()

        def test_function_level_2():
            """テスト関数レベル2"""
            # この関数内でget_caller_infoを呼び出し
            return get_caller_info(skip_frames=2)  # test_function_level_1を取得するため

        print("呼び出し元情報取得テスト:")
        print("-" * 60)

        # 呼び出し元情報を取得
        caller_info = test_function_level_1()

        # 検証項目
        required_keys = ['module_name', 'function_name', 'filename', 'line_number']

        results = []

        # 1. 必要なキーが全て含まれているか
        keys_test = all(key in caller_info for key in required_keys)
        results.append(keys_test)

        status = "OK PASS" if keys_test else "NG FAIL"
        print(f"必要キー存在確認: {status}")
        print(f"  期待キー: {required_keys}")
        print(f"  実際のキー: {list(caller_info.keys())}")

        # 2. 関数名が正確に取得されているか
        function_name_test = caller_info.get('function_name') == 'test_function_level_1'
        results.append(function_name_test)

        status = "OK PASS" if function_name_test else "NG FAIL"
        print(f"関数名正確性確認: {status}")
        print(f"  期待関数名: test_function_level_1")
        print(f"  実際の関数名: {caller_info.get('function_name')}")

        # 3. モジュール名が適切に設定されているか
        module_name_test = __name__ in caller_info.get('module_name', '')
        results.append(module_name_test)

        status = "OK PASS" if module_name_test else "NG FAIL"
        print(f"モジュール名確認: {status}")
        print(f"  期待モジュール名に含まれる: {__name__}")
        print(f"  実際のモジュール名: {caller_info.get('module_name')}")

        # 4. 行番号が正の値であるか
        line_number_test = isinstance(caller_info.get('line_number'), int) and caller_info.get('line_number') > 0
        results.append(line_number_test)

        status = "OK PASS" if line_number_test else "NG FAIL"
        print(f"行番号妥当性確認: {status}")
        print(f"  行番号: {caller_info.get('line_number')}")

        all_passed = all(results)
        print(f"\n呼び出し元情報取得機能: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"呼び出し元情報取得機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_error_logging():
    """拡張エラーログ機能テスト"""
    print("\n=== Issue #628 Enhanced Error Logging Test ===\n")

    try:
        from src.day_trade.utils.logging_config import log_error_with_enhanced_context, setup_logging

        # ロギング設定
        setup_logging()

        # ログ出力をキャプチャ
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        handler.setLevel(logging.ERROR)

        # 現在のモジュールのロガーにハンドラーを追加
        test_logger = logging.getLogger(__name__)
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.ERROR)

        print("拡張エラーログ機能テスト:")
        print("-" * 60)

        # テスト実行
        test_error = RuntimeError("Enhanced logging test error")
        test_context = {
            "operation": "test_operation",
            "parameters": {"param1": "value1", "param2": 42},
            "timestamp": "2025-08-12T22:00:00"
        }

        # 拡張ログ機能を呼び出し
        log_error_with_enhanced_context(test_error, test_context, include_caller_info=True)

        # ログ出力を取得
        log_content = log_output.getvalue()

        # 検証項目
        expected_elements = [
            "RuntimeError",
            "Enhanced logging test error",
            "test_enhanced_error_logging",  # 関数名
            # "test_operation"  # コンテキストはextraに含まれるが、メッセージ本文には含まれない場合がある
        ]

        results = []

        for element in expected_elements:
            element_found = element in log_content
            results.append(element_found)

            status = "OK PASS" if element_found else "NG FAIL"
            print(f"要素「{element}」検出: {status}")

        # ログ内容が空でないことを確認
        content_not_empty = len(log_content.strip()) > 0
        results.append(content_not_empty)

        status = "OK PASS" if content_not_empty else "NG FAIL"
        print(f"ログ出力内容存在: {status}")

        print(f"\nログ出力例（最初の200文字）:")
        print(f"{log_content[:200]}...")

        all_passed = all(results)
        print(f"\n拡張エラーログ機能: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        # クリーンアップ
        test_logger.removeHandler(handler)

        return all_passed

    except Exception as e:
        print(f"拡張エラーログ機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Issue #628 Log Error Context Source Tracing Test\n")

    tests = [
        ("ログエラーコンテキストソーストレース", test_log_error_with_context_source_tracing),
        ("呼び出し元情報取得機能", test_get_caller_info_functionality),
        ("拡張エラーログ機能", test_enhanced_error_logging)
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
        print("OK Issue #628 Log Error Context Source Tracing: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #628 Log Error Context Source Tracing: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)