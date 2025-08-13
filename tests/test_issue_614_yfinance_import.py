#!/usr/bin/env python3
"""
Issue #614: yfinance統一インポート標準化テスト
yfinanceインポートユーティリティとエラーハンドリングの機能テスト
"""

import sys
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_yfinance_import_utility():
    """yfinance統一インポートユーティリティテスト"""
    print("=== Issue #614 yfinance Import Utility Test ===\n")

    try:
        from src.day_trade.utils.yfinance_import import (
            get_yfinance,
            is_yfinance_available,
            require_yfinance,
            get_yfinance_ticker,
            safe_yfinance_operation
        )

        print("yfinance統一インポートユーティリティテスト:")
        print("-" * 60)

        results = []

        # 1. 基本的な取得機能テスト
        print("1. yfinance取得機能テスト:")
        yf_module, available = get_yfinance()

        get_yfinance_test = yf_module is not None if available else yf_module is None
        results.append(get_yfinance_test)

        status = "OK PASS" if get_yfinance_test else "NG FAIL"
        print(f"   get_yfinance機能: {status}")
        print(f"   利用可能: {available}")
        if available and yf_module:
            print(f"   モジュール: {type(yf_module)}")

        # 2. 利用可能性チェックテスト
        print("\n2. 利用可能性チェックテスト:")
        availability_check = is_yfinance_available()
        availability_test = availability_check == available
        results.append(availability_test)

        status = "OK PASS" if availability_test else "NG FAIL"
        print(f"   is_yfinance_available機能: {status}")
        print(f"   チェック結果: {availability_check}")

        # 3. require_yfinance機能テスト
        print("\n3. require_yfinance機能テスト:")
        require_test_passed = False

        try:
            required_module = require_yfinance()
            if available:
                # yfinanceが利用可能な場合はモジュールが返される
                require_test_passed = required_module is not None
            else:
                # ここに到達すべきではない（ImportErrorが発生するはず）
                require_test_passed = False
        except ImportError as e:
            if not available:
                # yfinanceが利用できない場合はImportErrorが発生すべき
                require_test_passed = True
                print(f"   期待通りのImportError: {e}")
            else:
                require_test_passed = False

        results.append(require_test_passed)

        status = "OK PASS" if require_test_passed else "NG FAIL"
        print(f"   require_yfinance機能: {status}")

        # 4. Tickerオブジェクト作成テスト（yfinanceが利用可能な場合のみ）
        print("\n4. Tickerオブジェクト作成テスト:")
        ticker_test_passed = False

        if available:
            try:
                ticker = get_yfinance_ticker("AAPL")
                ticker_test_passed = ticker is not None
                print(f"   Tickerオブジェクト作成成功: {type(ticker)}")
            except Exception as e:
                ticker_test_passed = False
                print(f"   Tickerオブジェクト作成エラー: {e}")
        else:
            try:
                ticker = get_yfinance_ticker("AAPL")
                ticker_test_passed = False  # エラーが発生すべき
            except ImportError:
                ticker_test_passed = True
                print("   yfinance利用不可時の適切なエラー発生")
            except Exception as e:
                ticker_test_passed = False
                print(f"   予期しないエラー: {e}")

        results.append(ticker_test_passed)

        status = "OK PASS" if ticker_test_passed else "NG FAIL"
        print(f"   get_yfinance_ticker機能: {status}")

        # 5. セーフデコレータテスト
        print("\n5. safe_yfinance_operation デコレータテスト:")

        @safe_yfinance_operation("テスト操作")
        def test_yfinance_operation():
            if not available:
                raise Exception("yfinance操作シミュレーション")
            return "成功"

        decorator_result = test_yfinance_operation()

        if available:
            decorator_test_passed = decorator_result == "成功"
            print(f"   デコレータ結果: {decorator_result}")
        else:
            decorator_test_passed = decorator_result is None
            print("   yfinance利用不可時はNoneを返す")

        results.append(decorator_test_passed)

        status = "OK PASS" if decorator_test_passed else "NG FAIL"
        print(f"   safe_yfinance_operationデコレータ: {status}")

        all_passed = all(results)
        print(f"\nyfinance統一インポートユーティリティ: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"yfinance統一インポートユーティリティテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standardized_imports():
    """標準化されたインポートテスト"""
    print("\n=== Issue #614 Standardized Imports Test ===\n")

    try:
        # 更新されたファイルのインポートテスト
        test_modules = [
            ("real_market_data", "src.day_trade.data.real_market_data"),
            ("stock_fetcher", "src.day_trade.data.stock_fetcher"),
            ("unified_api_adapter", "src.day_trade.data.unified_api_adapter"),
            ("yfinance_fetcher", "src.day_trade.data.fetchers.yfinance_fetcher")
        ]

        print("標準化されたインポートテスト:")
        print("-" * 60)

        results = []

        for module_name, module_path in test_modules:
            print(f"\n{module_name}モジュールテスト:")

            try:
                # モジュールのインポート
                module = __import__(module_path, fromlist=[''])

                # YFINANCE_AVAILABLEが存在するかチェック
                yfinance_available_exists = hasattr(module, 'YFINANCE_AVAILABLE')

                # yfモジュールが存在するかチェック
                yf_module_exists = hasattr(module, 'yf')

                # 統一インポートパターンが使用されているかチェック
                import_test_passed = yfinance_available_exists and yf_module_exists

                results.append(import_test_passed)

                status = "OK PASS" if import_test_passed else "NG FAIL"
                print(f"   統一インポートパターン: {status}")
                print(f"   YFINANCE_AVAILABLE存在: {yfinance_available_exists}")
                print(f"   yfモジュール存在: {yf_module_exists}")

                if yfinance_available_exists:
                    print(f"   YFINANCE_AVAILABLE値: {module.YFINANCE_AVAILABLE}")

            except ImportError as e:
                print(f"   モジュールインポートエラー: {e}")
                results.append(False)
            except Exception as e:
                print(f"   予期しないエラー: {e}")
                results.append(False)

        all_passed = all(results)
        print(f"\n標準化されたインポート: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"標準化されたインポートテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_improvements():
    """エラーハンドリング改善テスト"""
    print("\n=== Issue #614 Error Handling Improvements Test ===\n")

    try:
        from src.day_trade.data.stock_fetcher import StockFetcher
        from src.day_trade.data.real_market_data import RealMarketDataManager

        print("エラーハンドリング改善テスト:")
        print("-" * 60)

        results = []

        # 1. RealMarketDataManagerのエラーハンドリングテスト
        print("\n1. RealMarketDataManager エラーハンドリング:")

        try:
            manager = RealMarketDataManager()

            # yfinanceが利用できない場合のテスト
            with patch('src.day_trade.data.real_market_data.YFINANCE_AVAILABLE', False):
                data = manager.get_stock_data("7203")

                # yfinanceが利用できない場合はNoneが返される
                error_handling_test1 = data is None

            results.append(error_handling_test1)

            status = "OK PASS" if error_handling_test1 else "NG FAIL"
            print(f"   yfinance利用不可時の適切な処理: {status}")

        except Exception as e:
            print(f"   RealMarketDataManagerテストエラー: {e}")
            results.append(False)

        # 2. StockFetcher のエラーハンドリングテスト
        print("\n2. StockFetcher エラーハンドリング:")

        try:
            fetcher = StockFetcher()

            # _create_tickerメソッドのテスト
            with patch('src.day_trade.data.stock_fetcher.YFINANCE_AVAILABLE', False):
                try:
                    ticker = fetcher._create_ticker("AAPL")
                    error_handling_test2 = False  # ImportErrorが発生すべき
                except ImportError:
                    error_handling_test2 = True  # 期待通りのエラー
                except Exception:
                    error_handling_test2 = False

            results.append(error_handling_test2)

            status = "OK PASS" if error_handling_test2 else "NG FAIL"
            print(f"   _create_tickerでの適切なエラー発生: {status}")

        except Exception as e:
            print(f"   StockFetcherテストエラー: {e}")
            results.append(False)

        all_passed = all(results)
        print(f"\nエラーハンドリング改善: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"エラーハンドリング改善テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Issue #614 yfinance Import Standardization Test\n")

    tests = [
        ("yfinance統一インポートユーティリティ", test_yfinance_import_utility),
        ("標準化されたインポート", test_standardized_imports),
        ("エラーハンドリング改善", test_error_handling_improvements)
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
        print("OK Issue #614 yfinance Import Standardization: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #614 yfinance Import Standardization: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)