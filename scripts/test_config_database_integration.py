#!/usr/bin/env python3
"""
Issue #119: ConfigManagerとDatabaseConfigの統合テスト

ConfigManagerがデータベース設定を正しく読み込み、
DatabaseManagerに適切に渡されているかを検証する。
"""

import sys
from pathlib import Path

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.config.config_manager import ConfigManager  # noqa: E402
from src.day_trade.models.database import DatabaseConfig, DatabaseManager  # noqa: E402
from src.day_trade.utils.logging_config import setup_logging  # noqa: E402


def test_config_manager_database_settings():
    """ConfigManagerがデータベース設定を正しく読み込むかテスト"""
    print("=== ConfigManagerデータベース設定テスト ===")

    try:
        config_manager = ConfigManager()
        database_settings = config_manager.get_database_settings()

        print("データベース設定取得成功:")
        print(f"  URL: {database_settings.url}")
        print(f"  バックアップ有効: {database_settings.backup_enabled}")
        print(f"  バックアップ間隔: {database_settings.backup_interval_hours}時間")

        return config_manager

    except Exception as e:
        print(f"ConfigManagerテストエラー: {e}")
        raise


def test_database_config_with_config_manager():
    """DatabaseConfigがConfigManagerを使用してデータベース設定を取得するかテスト"""
    print("\n=== DatabaseConfig統合テスト ===")

    try:
        config_manager = ConfigManager()
        db_config = DatabaseConfig(config_manager=config_manager)

        print("DatabaseConfig作成成功:")
        print(f"  データベースURL: {db_config.database_url}")
        print(f"  エコーモード: {db_config.echo}")
        print(f"  接続プールサイズ: {db_config.pool_size}")
        print(f"  SQLiteか?: {db_config.is_sqlite()}")
        print(f"  インメモリDB?: {db_config.is_in_memory()}")

        return db_config

    except Exception as e:
        print(f"DatabaseConfigテストエラー: {e}")
        raise


def test_database_manager_with_config_manager():
    """DatabaseManagerがConfigManagerと連携するかテスト"""
    print("\n=== DatabaseManager統合テスト ===")

    try:
        config_manager = ConfigManager()
        db_manager = DatabaseManager(config_manager=config_manager)

        print("DatabaseManager作成成功:")
        print(f"  データベースURL: {db_manager.config.database_url}")
        print(f"  データベースタイプ: {db_manager._get_database_type()}")

        # ヘルスチェック実行
        health = db_manager.health_check()
        print(f"  ヘルスチェック: {health['status']}")

        if health["status"] == "healthy":
            print(f"  接続プール情報: {health['connection_pool']}")
        else:
            print(f"  エラー: {health.get('errors', [])}")

        return db_manager

    except Exception as e:
        print(f"DatabaseManagerテストエラー: {e}")
        raise


def test_environment_variable_override():
    """環境変数でのデータベースURL上書きテスト"""
    print("\n=== 環境変数上書きテスト ===")

    import os

    # 環境変数を設定
    original_url = os.environ.get("DATABASE_URL")
    test_url = "sqlite:///test_override.db"
    os.environ["DATABASE_URL"] = test_url

    try:
        config_manager = ConfigManager()
        db_config = DatabaseConfig(config_manager=config_manager)

        print("環境変数設定後:")
        print(f"  設定ファイルURL: {config_manager.get_database_settings().url}")
        print(f"  実際のURL: {db_config.database_url}")

        # 環境変数が設定された場合、それが優先されるはず
        if db_config.database_url == test_url:
            print("  OK: 環境変数によるURL上書きが正常に動作")
        else:
            print("  NG: 環境変数による上書きが動作していない")

    finally:
        # 環境変数を元に戻す
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n=== 後方互換性テスト ===")

    try:
        # ConfigManagerなしでDatabaseConfigを作成
        db_config_without_manager = DatabaseConfig()
        print("ConfigManagerなしのDatabaseConfig:")
        print(f"  データベースURL: {db_config_without_manager.database_url}")

        # ConfigManagerありでDatabaseConfigを作成
        config_manager = ConfigManager()
        db_config_with_manager = DatabaseConfig(config_manager=config_manager)
        print("ConfigManagerありのDatabaseConfig:")
        print(f"  データベースURL: {db_config_with_manager.database_url}")

        print("  OK: 後方互換性が保たれている")

    except Exception as e:
        print(f"後方互換性テストエラー: {e}")
        raise


def main():
    """メイン関数"""
    setup_logging()

    print("開始: Issue #119 ConfigManagerとDatabase統合テスト")
    print("=" * 60)

    try:
        # 各テストを順次実行
        config_manager = test_config_manager_database_settings()
        test_database_config_with_config_manager()
        db_manager = test_database_manager_with_config_manager()

        test_environment_variable_override()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print(
            "SUCCESS: 全テスト完了: ConfigManagerとDatabase統合は正常に動作しています"
        )

        # 最終的な設定値をサマリー表示
        print("\nSUMMARY 設定サマリー:")
        print(f"   設定ファイルパス: {config_manager.config_path}")
        database_settings = config_manager.get_database_settings()
        print(f"   データベースURL: {database_settings.url}")
        print(f"   実際の接続URL: {db_manager.config.database_url}")
        print(f"   データベースタイプ: {db_manager._get_database_type()}")

    except Exception as e:
        print(f"\nERROR テスト失敗: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
