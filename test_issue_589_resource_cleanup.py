#!/usr/bin/env python3
"""
Issue #589: Resource Cleanup Improvements Test
リソースクリーンアップ改善テスト
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def create_mock_orchestrator():
    """モックオーケストレーターを作成"""
    mock_orchestrator = Mock()

    # モックコンポーネント設定
    mock_orchestrator.analysis_engines = {
        "TEST_STOCK1": Mock(),
        "TEST_STOCK2": Mock(),
    }

    # analysis_enginesの各コンポーネントにメソッド設定
    for engine in mock_orchestrator.analysis_engines.values():
        engine.stop = Mock()
        engine.close = Mock()
        engine.cleanup = Mock()

    # ML Engine モック
    mock_ml_engine = Mock()
    mock_ml_engine.model = Mock()
    mock_ml_engine.model.cpu = Mock()
    mock_ml_engine.close = Mock()
    mock_ml_engine.cleanup = Mock()
    mock_ml_engine.performance_history = ["test_data1", "test_data2"]
    mock_orchestrator.ml_engine = mock_ml_engine

    # Batch Fetcher モック
    mock_batch_fetcher = Mock()
    mock_batch_fetcher.close = Mock()
    mock_orchestrator.batch_fetcher = mock_batch_fetcher

    # Parallel Manager モック
    mock_parallel_manager = Mock()
    mock_parallel_manager.shutdown = Mock()
    mock_orchestrator.parallel_manager = mock_parallel_manager

    # Performance Monitor モック
    mock_performance_monitor = Mock()
    mock_performance_monitor.stop = Mock()
    mock_performance_monitor.close = Mock()
    mock_orchestrator.performance_monitor = mock_performance_monitor

    # Stock Fetcher モック
    mock_stock_fetcher = Mock()
    mock_stock_fetcher.close = Mock()
    mock_orchestrator.stock_fetcher = mock_stock_fetcher

    # Execution History モック
    mock_orchestrator.execution_history = ["history1", "history2"]

    return mock_orchestrator


def test_comprehensive_resource_cleanup():
    """包括的リソースクリーンアップテスト"""
    print("=== Issue #589 Comprehensive Resource Cleanup Test ===\\n")

    try:
        from src.day_trade.automation.orchestrator import NextGenAIOrchestrator

        # 実際のオーケストレーター初期化（テスト用設定）
        with patch.dict('os.environ', {'CI': 'true'}):
            orchestrator = NextGenAIOrchestrator(
                config_path='config/development.json'
            )

        # テスト用モックコンポーネント追加
        test_components = create_mock_orchestrator()

        # 実際のオーケストレーターにモックコンポーネントを設定
        orchestrator.analysis_engines = test_components.analysis_engines
        orchestrator.ml_engine = test_components.ml_engine
        orchestrator.batch_fetcher = test_components.batch_fetcher
        orchestrator.parallel_manager = test_components.parallel_manager
        orchestrator.performance_monitor = test_components.performance_monitor
        orchestrator.stock_fetcher = test_components.stock_fetcher
        orchestrator.execution_history = test_components.execution_history

        print("リソースクリーンアップテスト:")
        print("-" * 60)

        # クリーンアップ実行
        cleanup_summary = orchestrator.cleanup()

        print(f"分析エンジンクリーンアップ数: {cleanup_summary['analysis_engines']}")
        print(f"バッチフェッチャークリーンアップ: {cleanup_summary['batch_fetcher']}")
        print(f"MLエンジンクリーンアップ: {cleanup_summary['ml_engine']}")
        print(f"並列マネージャークリーンアップ: {cleanup_summary['parallel_manager']}")
        print(f"パフォーマンスモニタークリーンアップ: {cleanup_summary['performance_monitor']}")
        print(f"エラー数: {len(cleanup_summary['errors'])}")

        # 検証
        results = {
            "analysis_engines_cleaned": cleanup_summary["analysis_engines"] == 2,
            "batch_fetcher_cleaned": cleanup_summary["batch_fetcher"] == True,
            "ml_engine_cleaned": cleanup_summary["ml_engine"] == True,
            "parallel_manager_cleaned": cleanup_summary["parallel_manager"] == True,
            "performance_monitor_cleaned": cleanup_summary["performance_monitor"] == True,
            "no_errors": len(cleanup_summary["errors"]) == 0
        }

        # 結果検証
        all_passed = all(results.values())

        print("\\n検証結果:")
        print("-" * 40)
        for test_name, passed in results.items():
            status = "OK PASS" if passed else "NG FAIL"
            print(f"{test_name:30s}: {status}")

        print(f"\\n総合結果: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return False


def test_context_manager_cleanup():
    """コンテキストマネージャークリーンアップテスト"""
    print("\\n=== Context Manager Cleanup Test ===\\n")

    try:
        from src.day_trade.automation.orchestrator import NextGenAIOrchestrator

        cleanup_called = False

        # cleanupメソッドが呼ばれるかテスト
        with patch.dict('os.environ', {'CI': 'true'}):
            with patch.object(NextGenAIOrchestrator, 'cleanup') as mock_cleanup:
                mock_cleanup.return_value = {"test": "cleanup_called"}

                with NextGenAIOrchestrator(
                    config_path='config/development.json'
                ) as orchestrator:
                    pass  # withブロック終了時にcleanupが呼ばれるはず

                cleanup_called = mock_cleanup.called

        print(f"コンテキストマネージャー終了時のcleanup呼び出し: {'OK PASS' if cleanup_called else 'NG FAIL'}")

        return cleanup_called

    except Exception as e:
        print(f"コンテキストマネージャーテストエラー: {e}")
        return False


def test_error_handling_during_cleanup():
    """クリーンアップ中のエラーハンドリングテスト"""
    print("\\n=== Error Handling During Cleanup Test ===\\n")

    try:
        from src.day_trade.automation.orchestrator import NextGenAIOrchestrator

        with patch.dict('os.environ', {'CI': 'true'}):
            orchestrator = NextGenAIOrchestrator(
                config_path='config/development.json',
            )

        # エラーを発生させるモックコンポーネント設定
        error_engine = Mock()
        error_engine.stop.side_effect = Exception("Test stop error")
        error_engine.close = Mock()
        error_engine.cleanup = Mock()

        orchestrator.analysis_engines = {"ERROR_ENGINE": error_engine}

        # エラーML Engine設定
        error_ml_engine = Mock()
        error_ml_engine.close.side_effect = Exception("Test ML engine error")
        orchestrator.ml_engine = error_ml_engine

        # クリーンアップ実行（エラーが発生しても継続されるかテスト）
        cleanup_summary = orchestrator.cleanup()

        print(f"エラー発生時のクリーンアップ継続: {'OK PASS' if len(cleanup_summary['errors']) >= 2 else 'NG FAIL'}")
        print(f"エラー数: {len(cleanup_summary['errors'])}")

        for i, error in enumerate(cleanup_summary['errors'][:3]):  # 最初の3つのエラーを表示
            print(f"エラー {i+1}: {error[:50]}...")

        return len(cleanup_summary['errors']) >= 2

    except Exception as e:
        print(f"エラーハンドリングテストエラー: {e}")
        return False


def test_memory_leak_prevention():
    """メモリリーク防止テスト"""
    print("\\n=== Memory Leak Prevention Test ===\\n")

    try:
        import gc

        from src.day_trade.automation.orchestrator import NextGenAIOrchestrator

        # ガベージコレクション実行前の状態
        gc.collect()
        objects_before = len(gc.get_objects())

        with patch.dict('os.environ', {'CI': 'true'}):
            orchestrator = NextGenAIOrchestrator(
                config_path='config/development.json',
            )

        # 大量のテストデータ設定
        test_data = ["test_data"] * 1000
        orchestrator.execution_history = test_data

        # 大量の分析エンジン設定
        for i in range(100):
            mock_engine = Mock()
            mock_engine.stop = Mock()
            mock_engine.close = Mock()
            mock_engine.cleanup = Mock()
            orchestrator.analysis_engines[f"ENGINE_{i}"] = mock_engine

        # クリーンアップ実行
        cleanup_summary = orchestrator.cleanup()

        # ガベージコレクション実行後の状態
        objects_after = len(gc.get_objects())

        # メモリ使用量変化確認
        memory_cleaned = objects_after <= objects_before + 50  # 許容範囲内

        print(f"オブジェクト数（クリーンアップ前）: {objects_before}")
        print(f"オブジェクト数（クリーンアップ後）: {objects_after}")
        print(f"メモリリーク防止: {'OK PASS' if memory_cleaned else 'NG FAIL'}")

        return memory_cleaned

    except Exception as e:
        print(f"メモリリーク防止テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("Issue #589 Resource Cleanup Improvements Test\\n")

    tests = [
        ("包括的リソースクリーンアップ", test_comprehensive_resource_cleanup),
        ("コンテキストマネージャークリーンアップ", test_context_manager_cleanup),
        ("エラーハンドリング", test_error_handling_during_cleanup),
        ("メモリリーク防止", test_memory_leak_prevention)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\\n{'='*50}")
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