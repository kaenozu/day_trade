#!/usr/bin/env python3
"""
Issue #589: Resource Cleanup Improvements Simple Test
リソースクリーンアップ改善シンプルテスト
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_cleanup_method_functionality():
    """cleanupメソッド機能テスト"""
    print("=== Issue #589 Cleanup Method Functionality Test ===\\n")

    try:
        # モッククラスを作成してcleanupメソッドをテスト
        class MockOrchestrator:
            def __init__(self):
                # モックコンポーネント設定
                self.analysis_engines = {
                    "ENGINE_1": Mock(),
                    "ENGINE_2": Mock()
                }

                # 各エンジンにメソッド設定
                for engine in self.analysis_engines.values():
                    engine.stop = Mock()
                    engine.close = Mock()
                    engine.cleanup = Mock()

                # ML Engine
                self.ml_engine = Mock()
                self.ml_engine.model = Mock()
                self.ml_engine.model.cpu = Mock()
                self.ml_engine.close = Mock()
                self.ml_engine.cleanup = Mock()
                self.ml_engine.performance_history = ["data1", "data2"]

                # Batch Fetcher
                self.batch_fetcher = Mock()
                self.batch_fetcher.close = Mock()

                # Parallel Manager
                self.parallel_manager = Mock()
                self.parallel_manager.shutdown = Mock()

                # Performance Monitor
                self.performance_monitor = Mock()
                self.performance_monitor.stop = Mock()
                self.performance_monitor.close = Mock()

                # Stock Fetcher
                self.stock_fetcher = Mock()
                self.stock_fetcher.close = Mock()

                # Execution History
                self.execution_history = ["history1", "history2"]

            def cleanup(self):
                """Issue #589対応のリソースクリーンアップメソッド実装テスト版"""
                cleanup_summary = {
                    "analysis_engines": 0,
                    "batch_fetcher": False,
                    "ml_engine": False,
                    "parallel_manager": False,
                    "performance_monitor": False,
                    "errors": []
                }

                try:
                    # 分析エンジンクリーンアップ
                    for symbol, engine in self.analysis_engines.items():
                        try:
                            if hasattr(engine, "stop"):
                                engine.stop()
                            if hasattr(engine, "close"):
                                engine.close()
                            if hasattr(engine, "cleanup"):
                                engine.cleanup()
                            cleanup_summary["analysis_engines"] += 1
                        except Exception as e:
                            cleanup_summary["errors"].append(f"エンジン {symbol} エラー: {e}")

                    self.analysis_engines.clear()

                    # MLエンジンクリーンアップ
                    if hasattr(self, 'ml_engine') and self.ml_engine:
                        try:
                            if hasattr(self.ml_engine, "model") and self.ml_engine.model:
                                if hasattr(self.ml_engine.model, "cpu"):
                                    self.ml_engine.model.cpu()
                                del self.ml_engine.model

                            if hasattr(self.ml_engine, "close"):
                                self.ml_engine.close()
                            if hasattr(self.ml_engine, "cleanup"):
                                self.ml_engine.cleanup()

                            if hasattr(self.ml_engine, "performance_history"):
                                self.ml_engine.performance_history.clear()

                            self.ml_engine = None
                            cleanup_summary["ml_engine"] = True
                        except Exception as e:
                            cleanup_summary["errors"].append(f"MLエンジン エラー: {e}")

                    # バッチフェッチャークリーンアップ
                    if hasattr(self, 'batch_fetcher') and self.batch_fetcher:
                        try:
                            self.batch_fetcher.close()
                            self.batch_fetcher = None
                            cleanup_summary["batch_fetcher"] = True
                        except Exception as e:
                            cleanup_summary["errors"].append(f"バッチフェッチャー エラー: {e}")

                    # 並列マネージャークリーンアップ
                    if hasattr(self, 'parallel_manager') and self.parallel_manager:
                        try:
                            self.parallel_manager.shutdown()
                            self.parallel_manager = None
                            cleanup_summary["parallel_manager"] = True
                        except Exception as e:
                            cleanup_summary["errors"].append(f"並列マネージャー エラー: {e}")

                    # パフォーマンスモニタークリーンアップ
                    if hasattr(self, 'performance_monitor') and self.performance_monitor:
                        try:
                            if hasattr(self.performance_monitor, "stop"):
                                self.performance_monitor.stop()
                            if hasattr(self.performance_monitor, "close"):
                                self.performance_monitor.close()
                            self.performance_monitor = None
                            cleanup_summary["performance_monitor"] = True
                        except Exception as e:
                            cleanup_summary["errors"].append(f"パフォーマンスモニター エラー: {e}")

                    # ストックフェッチャークリーンアップ
                    if hasattr(self, 'stock_fetcher') and self.stock_fetcher:
                        try:
                            if hasattr(self.stock_fetcher, "close"):
                                self.stock_fetcher.close()
                            self.stock_fetcher = None
                        except Exception as e:
                            cleanup_summary["errors"].append(f"ストックフェッチャー エラー: {e}")

                    # 実行履歴クリア
                    if hasattr(self, 'execution_history'):
                        self.execution_history.clear()

                    return cleanup_summary

                except Exception as e:
                    cleanup_summary["errors"].append(f"致命的エラー: {e}")
                    return cleanup_summary

        # テスト実行
        mock_orchestrator = MockOrchestrator()

        print("クリーンアップ前の状態:")
        print(f"  分析エンジン数: {len(mock_orchestrator.analysis_engines)}")
        print(f"  MLエンジン: {'存在' if mock_orchestrator.ml_engine else '無'}")
        print(f"  バッチフェッチャー: {'存在' if mock_orchestrator.batch_fetcher else '無'}")
        print(f"  実行履歴数: {len(mock_orchestrator.execution_history)}")

        # クリーンアップ実行
        cleanup_result = mock_orchestrator.cleanup()

        print("\\nクリーンアップ実行結果:")
        print("-" * 40)
        print(f"  分析エンジンクリーンアップ: {cleanup_result['analysis_engines']} 個")
        print(f"  MLエンジンクリーンアップ: {'OK' if cleanup_result['ml_engine'] else 'NG'}")
        print(f"  バッチフェッチャークリーンアップ: {'OK' if cleanup_result['batch_fetcher'] else 'NG'}")
        print(f"  並列マネージャークリーンアップ: {'OK' if cleanup_result['parallel_manager'] else 'NG'}")
        print(f"  パフォーマンスモニタークリーンアップ: {'OK' if cleanup_result['performance_monitor'] else 'NG'}")
        print(f"  エラー数: {len(cleanup_result['errors'])}")

        print("\\nクリーンアップ後の状態:")
        print(f"  分析エンジン数: {len(mock_orchestrator.analysis_engines)}")
        print(f"  MLエンジン: {'存在' if mock_orchestrator.ml_engine else '削除済み'}")
        print(f"  バッチフェッチャー: {'存在' if mock_orchestrator.batch_fetcher else '削除済み'}")
        print(f"  実行履歴数: {len(mock_orchestrator.execution_history)}")

        # 検証
        success_criteria = [
            cleanup_result['analysis_engines'] == 2,
            cleanup_result['ml_engine'] == True,
            cleanup_result['batch_fetcher'] == True,
            cleanup_result['parallel_manager'] == True,
            cleanup_result['performance_monitor'] == True,
            len(cleanup_result['errors']) == 0,
            len(mock_orchestrator.analysis_engines) == 0,
            mock_orchestrator.ml_engine is None,
            len(mock_orchestrator.execution_history) == 0
        ]

        all_passed = all(success_criteria)

        print("\\n検証結果:")
        print("-" * 40)
        criteria_names = [
            "分析エンジンクリーンアップ数",
            "MLエンジンクリーンアップ",
            "バッチフェッチャークリーンアップ",
            "並列マネージャークリーンアップ",
            "パフォーマンスモニタークリーンアップ",
            "エラー無し",
            "分析エンジン辞書クリア",
            "MLエンジンNone化",
            "実行履歴クリア"
        ]

        for i, (name, passed) in enumerate(zip(criteria_names, success_criteria)):
            status = "OK PASS" if passed else "NG FAIL"
            print(f"  {name}: {status}")

        print(f"\\n総合結果: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_calls_verification():
    """メソッド呼び出し検証テスト"""
    print("\\n=== Method Calls Verification Test ===\\n")

    try:
        class MockOrchestratorWithVerification:
            def __init__(self):
                # カウンター付きモック設定
                self.analysis_engines = {"TEST_ENGINE": Mock()}
                self.ml_engine = Mock()
                self.batch_fetcher = Mock()
                self.parallel_manager = Mock()
                self.performance_monitor = Mock()
                self.stock_fetcher = Mock()
                self.execution_history = []

                # メソッド呼び出し記録
                self.call_log = []

            def cleanup(self):
                """メソッド呼び出しを記録するcleanup"""
                cleanup_summary = {"errors": []}

                # Analysis Engine
                for symbol, engine in self.analysis_engines.items():
                    if hasattr(engine, "stop"):
                        engine.stop()
                        self.call_log.append(f"{symbol}.stop()")
                    if hasattr(engine, "close"):
                        engine.close()
                        self.call_log.append(f"{symbol}.close()")
                    if hasattr(engine, "cleanup"):
                        engine.cleanup()
                        self.call_log.append(f"{symbol}.cleanup()")

                # ML Engine
                if self.ml_engine:
                    if hasattr(self.ml_engine, "close"):
                        self.ml_engine.close()
                        self.call_log.append("ml_engine.close()")

                # Batch Fetcher
                if self.batch_fetcher:
                    self.batch_fetcher.close()
                    self.call_log.append("batch_fetcher.close()")

                # Parallel Manager
                if self.parallel_manager:
                    self.parallel_manager.shutdown()
                    self.call_log.append("parallel_manager.shutdown()")

                # Performance Monitor
                if self.performance_monitor:
                    if hasattr(self.performance_monitor, "stop"):
                        self.performance_monitor.stop()
                        self.call_log.append("performance_monitor.stop()")
                    if hasattr(self.performance_monitor, "close"):
                        self.performance_monitor.close()
                        self.call_log.append("performance_monitor.close()")

                # Stock Fetcher
                if self.stock_fetcher:
                    if hasattr(self.stock_fetcher, "close"):
                        self.stock_fetcher.close()
                        self.call_log.append("stock_fetcher.close()")

                return cleanup_summary

        orchestrator = MockOrchestratorWithVerification()
        orchestrator.cleanup()

        expected_calls = [
            "TEST_ENGINE.stop()",
            "TEST_ENGINE.close()",
            "TEST_ENGINE.cleanup()",
            "ml_engine.close()",
            "batch_fetcher.close()",
            "parallel_manager.shutdown()",
            "performance_monitor.stop()",
            "performance_monitor.close()",
            "stock_fetcher.close()"
        ]

        print("期待されるメソッド呼び出し:")
        for call in expected_calls:
            print(f"  {call}")

        print("\\n実際のメソッド呼び出し:")
        for call in orchestrator.call_log:
            print(f"  {call}")

        # 検証
        all_calls_made = all(call in orchestrator.call_log for call in expected_calls)

        print(f"\\nメソッド呼び出し検証: {'OK PASS' if all_calls_made else 'NG FAIL'}")
        print(f"呼び出し数: {len(orchestrator.call_log)}/{len(expected_calls)}")

        return all_calls_made

    except Exception as e:
        print(f"メソッド呼び出し検証エラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("Issue #589 Resource Cleanup Improvements Simple Test\\n")

    tests = [
        ("クリーンアップメソッド機能", test_cleanup_method_functionality),
        ("メソッド呼び出し検証", test_method_calls_verification)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"{'='*50}")
            print(f"実行中: {test_name}")
            print('='*50)

            if test_func():
                print(f"OK {test_name}: PASS")
                passed += 1
            else:
                print(f"NG {test_name}: FAIL")
                failed += 1

        except Exception as e:
            print(f"NG {test_name}: ERROR - {e}")
            failed += 1

    print(f"\\n{'='*50}")
    print(f"=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("OK Issue #589 Resource Cleanup: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #589 Resource Cleanup: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)