"""
Phase 4 統合テスト

統合エラーハンドリングシステムと統合データベースアクセス層のテスト
"""

import sys
import traceback
from typing import Any, Dict

def test_imports():
    """モジュールインポートテスト"""
    print("=== Phase 4 モジュールインポートテスト ===")

    try:
        # 統合エラーハンドリングシステム
        from src.day_trade.core.unified_error_handler import (
            UnifiedErrorHandler,
            UnifiedErrorContext,
            unified_error_handling,
            get_unified_error_handler,
        )
        print("OK: 統合エラーハンドリングシステムのインポート成功")

        # 統合データベースアクセス層
        from src.day_trade.models.unified_database import (
            UnifiedDatabaseManager,
            DatabaseConnectionPool,
            get_unified_database_manager,
            get_database_session,
        )
        print("OK: 統合データベースアクセス層のインポート成功")

        return True

    except Exception as e:
        print(f"NG: インポートエラー: {e}")
        traceback.print_exc()
        return False

def test_error_handler():
    """エラーハンドラーテスト"""
    print("\n=== 統合エラーハンドラーテスト ===")

    try:
        from src.day_trade.core.unified_error_handler import get_unified_error_handler

        # エラーハンドラー取得
        handler = get_unified_error_handler()
        print("OK: エラーハンドラーインスタンス取得成功")

        # テストエラーを処理
        test_error = ValueError("テストエラー")
        context = handler.handle_error(
            error=test_error,
            operation="test_operation",
            component="test_component",
            user_data={"test_data": "test_value"},
        )

        print(f"OK: エラー処理成功: {context.category.value}, {context.severity.value}")

        # メトリクス確認
        stats = handler.metrics.get_stats()
        print(f"OK: メトリクス収集: {stats['total_errors']} errors")

        return True

    except Exception as e:
        print(f"NG: エラーハンドラーテストエラー: {e}")
        traceback.print_exc()
        return False

def test_database_manager():
    """データベースマネージャーテスト"""
    print("\n=== 統合データベースマネージャーテスト ===")

    try:
        from src.day_trade.models.unified_database import get_unified_database_manager

        # データベースマネージャー取得
        manager = get_unified_database_manager()
        print("OK: データベースマネージャーインスタンス取得成功")

        # 健全性チェック
        health = manager.connection_pool.health_check()
        print(f"OK: データベース健全性チェック: {health['status']}")

        # 統計情報取得
        stats = manager.get_system_stats()
        print(f"OK: システム統計取得: {list(stats.keys())}")

        return True

    except Exception as e:
        print(f"NG: データベースマネージャーテストエラー: {e}")
        traceback.print_exc()
        return False

def test_decorator():
    """デコレーターテスト"""
    print("\n=== エラーハンドリングデコレーターテスト ===")

    try:
        from src.day_trade.core.unified_error_handler import unified_error_handling

        @unified_error_handling(
            operation="test_decorated_function",
            component="test_component",
            auto_recovery=True,
        )
        def test_function():
            return "成功"

        # 正常ケース
        result = test_function()
        print(f"OK: デコレーター正常動作: {result}")

        @unified_error_handling(
            operation="test_error_function",
            component="test_component",
            auto_recovery=False,
        )
        def error_function():
            raise ValueError("テストエラー")

        # エラーケース
        try:
            error_function()
        except ValueError:
            print("OK: デコレーターエラーハンドリング正常動作")

        return True

    except Exception as e:
        print(f"NG: デコレーターテストエラー: {e}")
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("Phase 4: エラーハンドリング統合とデータベース層統合 テスト開始\n")

    test_results = []

    # 各テストを実行
    test_results.append(test_imports())
    test_results.append(test_error_handler())
    test_results.append(test_database_manager())
    test_results.append(test_decorator())

    # 結果集計
    passed = sum(test_results)
    total = len(test_results)

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
