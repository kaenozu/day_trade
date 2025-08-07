"""
Phase 4 簡単な統合テスト
"""

import sys

def test_direct_imports():
    """ダイレクトインポートテスト"""
    print("=== Phase 4 ダイレクトインポートテスト ===")

    try:
        # 統合エラーハンドリングシステム
        from src.day_trade.core.unified_error_handler import (
            UnifiedErrorHandler,
            ErrorSeverity,
            ErrorCategory,
        )
        print("OK: unified_error_handlerインポート成功")

        # 統合データベースアクセス層
        from src.day_trade.models.unified_database import (
            UnifiedDatabaseManager,
            DatabaseMetrics,
        )
        print("OK: unified_databaseインポート成功")

        return True

    except Exception as e:
        print(f"NG: インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """基本機能テスト"""
    print("\n=== 基本機能テスト ===")

    try:
        from src.day_trade.core.unified_error_handler import (
            UnifiedErrorHandler,
            UnifiedErrorContext,
            ErrorSeverity,
            ErrorCategory,
        )

        # エラーハンドラーインスタンス作成
        handler = UnifiedErrorHandler()
        print("OK: エラーハンドラー作成成功")

        # テストエラーコンテキスト作成
        test_error = ValueError("テストエラー")
        context = UnifiedErrorContext(
            error=test_error,
            operation="test_operation",
            component="test_component",
        )
        print(f"OK: エラーコンテキスト作成: {context.severity.value}")

        # メトリクス確認
        handler.metrics.record_error(context)
        stats = handler.metrics.get_stats()
        print(f"OK: メトリクス機能: errors={stats['total_errors']}")

        return True

    except Exception as e:
        print(f"NG: 基本機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_basic():
    """データベース基本テスト"""
    print("\n=== データベース基本テスト ===")

    try:
        from src.day_trade.models.unified_database import (
            DatabaseMetrics,
            DatabaseConnectionPool,
        )

        # メトリクス作成テスト
        metrics = DatabaseMetrics()
        metrics.record_operation(0.5)
        stats = metrics.get_stats()
        print(f"OK: データベースメトリクス: operations={stats['total_operations']}")

        return True

    except Exception as e:
        print(f"NG: データベース基本テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト"""
    print("Phase 4: 簡単統合テスト開始\n")

    tests = [
        test_direct_imports,
        test_basic_functionality,
        test_database_basic,
    ]

    results = []
    for test in tests:
        results.append(test())

    passed = sum(results)
    total = len(results)

    print(f"\n=== テスト結果 ===")
    print(f"成功: {passed}/{total}")
    print(f"失敗: {total - passed}/{total}")

    if passed == total:
        print("OK: 全てのテストが成功しました！")
        return True
    else:
        print("NG: 一部のテストが失敗しました。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
