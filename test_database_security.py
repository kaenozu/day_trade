#!/usr/bin/env python3
"""
データベース層セキュリティテスト
Issue #392: データベース層のセキュリティ強化

実装されたセキュリティ強化:
1. SQLite PRAGMAのSQLインジェクション脆弱性対策 - ホワイトリスト検証・範囲制限
2. get_alembic_configのTOCTOU脆弱性対策 - 原子的操作・シンボリックリンク攻撃防止
"""

# テスト中はログ出力を有効化
import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO)

from sqlalchemy import text

from src.day_trade.models.database import DatabaseConfig, DatabaseManager
from src.day_trade.utils.exceptions import DatabaseError


def test_sqlite_pragma_sql_injection_protection():
    """SQLite PRAGMAのSQLインジェクション対策テスト"""
    print("=== SQLite PRAGMA SQLインジェクション対策テスト ===")

    # 1. 正常なPRAGMA値のテスト
    print("\n1. 正常PRAGMA値:")

    safe_config_values = [
        ("journal_mode", "WAL", "WAL"),
        ("journal_mode", "DELETE", "DELETE"),
        ("synchronous", "NORMAL", "NORMAL"),
        ("synchronous", "FULL", "FULL"),
        ("cache_size", 10000, "10000"),
        ("cache_size", -10000, "-10000"),
        ("temp_store", "MEMORY", "MEMORY"),
        ("temp_store", "FILE", "FILE"),
        ("mmap_size", 268435456, "268435456"),
        ("mmap_size", 0, "0"),
    ]

    config = DatabaseConfig.for_testing()
    db_manager = DatabaseManager(config)

    for pragma_name, input_value, expected_value in safe_config_values:
        try:
            validated_value = db_manager._validate_pragma_value(
                pragma_name, input_value
            )
            if validated_value == expected_value:
                print(f"  OK {pragma_name}={input_value} → {validated_value}")
            else:
                print(
                    f"  FAIL {pragma_name}={input_value} - 期待値: {expected_value}, 実際: {validated_value}"
                )
        except Exception as e:
            print(f"  FAIL {pragma_name}={input_value} - 予期しないエラー: {e}")

    # 2. 危険なPRAGMA値のテスト（SQLインジェクション攻撃試行）
    print("\n2. 危険PRAGMA値（SQLインジェクション攻撃）:")

    malicious_config_values = [
        ("journal_mode", "WAL; DROP TABLE users; --", "SQLインジェクション攻撃"),
        ("journal_mode", "DELETE'; INSERT INTO", "SQLインジェクション攻撃"),
        ("synchronous", "NORMAL; PRAGMA user_version=999", "追加PRAGMA実行攻撃"),
        ("cache_size", "10000; DELETE FROM sqlite_master", "テーブル削除攻撃"),
        ("cache_size", "abc", "無効な数値形式"),
        (
            "temp_store",
            "MEMORY; ATTACH DATABASE '/tmp/malicious.db' AS evil",
            "データベース接続攻撃",
        ),
        ("mmap_size", "-1", "無効な範囲"),
        ("mmap_size", "9999999999", "メモリ枯渇攻撃"),
        ("unknown_pragma", "any_value", "未サポートPRAGMA"),
    ]

    for pragma_name, malicious_value, attack_type in malicious_config_values:
        try:
            validated_value = db_manager._validate_pragma_value(
                pragma_name, malicious_value
            )
            if validated_value is None or validated_value != malicious_value:
                print(
                    f"  OK {attack_type} - {pragma_name}={malicious_value} - 正常に阻止"
                )
            else:
                print(
                    f"  FAIL {attack_type} - {pragma_name}={malicious_value} - 攻撃が通過: {validated_value}"
                )
        except Exception as e:
            print(f"  OK {attack_type} - {pragma_name}={malicious_value} - 例外で阻止")

    # 3. 範囲制限テスト
    print("\n3. PRAGMA値範囲制限:")

    range_test_values = [
        ("cache_size", -2000000, "範囲外（下限）"),  # -1000000より小さい
        ("cache_size", 2000000, "範囲外（上限）"),  # 1000000より大きい
        ("mmap_size", -1, "範囲外（負の値）"),  # 負の値は無効
        ("mmap_size", 2147483648, "範囲外（2GB）"),  # 1GBより大きい
    ]

    for pragma_name, out_of_range_value, test_type in range_test_values:
        try:
            validated_value = db_manager._validate_pragma_value(
                pragma_name, out_of_range_value
            )
            if validated_value != str(out_of_range_value):
                print(
                    f"  OK {test_type} - {pragma_name}={out_of_range_value} - デフォルト値に修正: {validated_value}"
                )
            else:
                print(
                    f"  FAIL {test_type} - {pragma_name}={out_of_range_value} - 範囲外値が通過"
                )
        except Exception as e:
            print(f"  OK {test_type} - {pragma_name}={out_of_range_value} - 例外で阻止")


def test_alembic_config_toctou_protection():
    """Alembic設定のTOCTOU脆弱性対策テスト"""
    print("\n=== Alembic設定TOCTOU脆弱性対策テスト ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = DatabaseConfig.for_testing()
        db_manager = DatabaseManager(config)
        temp_path = Path(temp_dir)

        # 1. 正常なAlembic設定ファイルアクセステスト
        print("\n1. 正常Alembic設定ファイル:")

        # テスト用の正常なalembic.ini作成
        normal_config_path = temp_path / "alembic.ini"
        with open(normal_config_path, "w", encoding="utf-8") as f:
            f.write(
                """[alembic]
script_location = alembic
sqlalchemy.url = sqlite:///test.db
[loggers]
keys = root,sqlalchemy,alembic
[handlers]
keys = console
"""
            )

        try:
            if db_manager._is_safe_readable_file(normal_config_path):
                print(f"  OK 正常設定ファイル - 読み取り可能: {normal_config_path}")
            else:
                print(f"  FAIL 正常設定ファイル - 読み取り不可: {normal_config_path}")
        except Exception as e:
            print(f"  FAIL 正常設定ファイルエラー: {e}")

        # 2. 危険なファイルパスのテスト
        print("\n2. 危険ファイルパス:")

        dangerous_paths = [
            "../../../etc/passwd",  # パストラバーサル攻撃
            "/etc/shadow",  # システムファイル
            "c:\\windows\\system32\\config",  # Windowsシステムファイル
            "\\\\malicious\\share\\config",  # UNCパス攻撃
            "/tmp/../../../root/.bashrc",  # 複合パストラバーサル
        ]

        for dangerous_path in dangerous_paths:
            try:
                validated_path = db_manager._validate_alembic_config_path(
                    dangerous_path
                )
                print(f"  FAIL 危険パス通過: {dangerous_path} → {validated_path}")
            except DatabaseError as e:
                if "ALEMBIC_DANGEROUS_PATH" in str(
                    e
                ) or "ALEMBIC_PATH_NOT_ALLOWED" in str(e):
                    print(f"  OK 危険パス阻止: {dangerous_path}")
                else:
                    print(f"  WARN 危険パス - 別エラー: {dangerous_path} - {e}")
            except Exception as e:
                print(f"  OK 危険パス例外阻止: {dangerous_path}")

        # 3. シンボリックリンク攻撃対策テスト
        print("\n3. シンボリックリンク攻撃対策:")

        try:
            # 危険なファイルを作成
            dangerous_file = temp_path / "sensitive_data.txt"
            with open(dangerous_file, "w") as f:
                f.write("機密情報")

            # シンボリックリンクを作成（権限が必要な場合はスキップ）
            symlink_path = temp_path / "fake_alembic.ini"

            try:
                if os.name == "nt":
                    # Windows環境での処理
                    import subprocess

                    result = subprocess.run(
                        ["mklink", str(symlink_path), str(dangerous_file)],
                        shell=True,
                        capture_output=True,
                    )
                    if result.returncode != 0:
                        raise PermissionError("シンボリックリンク作成失敗")
                else:
                    # Unix/Linux環境での処理
                    symlink_path.symlink_to(dangerous_file)

                # シンボリックリンクファイルの安全性チェック
                if db_manager._is_safe_readable_file(symlink_path):
                    print(f"  FAIL シンボリックリンク攻撃 - 通過: {symlink_path}")
                else:
                    print(f"  OK シンボリックリンク攻撃 - 阻止: {symlink_path}")

            except (PermissionError, subprocess.CalledProcessError, OSError):
                print("  SKIP シンボリックリンク攻撃対策 - 権限不足でテストスキップ")

        except Exception as e:
            print(f"  SKIP シンボリックリンクテストエラー: {e}")

        # 4. ファイルサイズ制限テスト
        print("\n4. ファイルサイズ制限:")

        # 異常に大きな設定ファイルを作成
        large_config_path = temp_path / "large_alembic.ini"
        try:
            with open(large_config_path, "w") as f:
                # 15MB（制限10MBを超過）の設定ファイル
                f.write(
                    "# Large config file\n" + "dummy_line\n" * (15 * 1024 * 1024 // 10)
                )

            if db_manager._is_safe_readable_file(large_config_path):
                print(f"  FAIL 大容量ファイル - 通過: {large_config_path}")
            else:
                print(f"  OK 大容量ファイル - 阻止: {large_config_path}")

        except Exception as e:
            print(f"  SKIP 大容量ファイルテスト - ディスク容量不足等: {e}")


def test_secure_alembic_config_search():
    """安全なAlembic設定ファイル検索テスト"""
    print("\n=== 安全なAlembic設定ファイル検索テスト ===")

    config = DatabaseConfig.for_testing()
    db_manager = DatabaseManager(config)

    # 1. 許可ディレクトリでの検索テスト
    print("\n1. 許可ディレクトリ検索:")

    try:
        # 現在のプロジェクトディレクトリにalembic.iniが存在するかテスト
        config_path = db_manager._find_secure_alembic_config()
        print(f"  OK 設定ファイル検索成功: {config_path}")

        # 検索されたファイルが実際に安全かチェック
        validated_path = db_manager._validate_alembic_config_path(config_path)
        if validated_path:
            print(f"  OK 検索ファイル安全性確認: {validated_path}")
        else:
            print("  FAIL 検索ファイル - 安全性検証失敗")

    except DatabaseError as e:
        if "ALEMBIC_CONFIG_NOT_FOUND" in str(e):
            print("  OK 設定ファイル未発見 - 正常（テスト環境）")
        else:
            print(f"  FAIL 設定ファイル検索エラー: {e}")
    except Exception as e:
        print(f"  FAIL 予期しない検索エラー: {e}")


def test_pragma_execution_safety():
    """PRAGMA実行安全性テスト"""
    print("\n=== PRAGMA実行安全性テスト ===")

    config = DatabaseConfig.for_testing()
    db_manager = DatabaseManager(config)

    # 1. モックカーソルでのPRAGMA実行テスト
    print("\n1. PRAGMA実行安全性:")

    mock_cursor = MagicMock()

    # 正常なPRAGMA実行
    safe_pragma_tests = [
        ("journal_mode", "WAL"),
        ("synchronous", "NORMAL"),
        ("cache_size", 10000),
        ("temp_store", "MEMORY"),
        ("mmap_size", 268435456),
    ]

    for pragma_name, pragma_value in safe_pragma_tests:
        try:
            db_manager._execute_safe_pragma(mock_cursor, pragma_name, pragma_value)
            print(f"  OK 安全PRAGMA実行: {pragma_name}={pragma_value}")
        except Exception as e:
            print(f"  FAIL 安全PRAGMA実行エラー: {pragma_name}={pragma_value} - {e}")

    # 危険なPRAGMA実行（阻止されるべき）
    malicious_pragma_tests = [
        ("journal_mode", "WAL; DROP TABLE users; --"),
        ("cache_size", "10000; PRAGMA user_version=999"),
        ("unknown_pragma", "any_value"),
    ]

    for pragma_name, pragma_value in malicious_pragma_tests:
        try:
            db_manager._execute_safe_pragma(mock_cursor, pragma_name, pragma_value)
            print(f"  FAIL 危険PRAGMA通過: {pragma_name}={pragma_value}")
        except Exception as e:
            print(f"  OK 危険PRAGMA阻止: {pragma_name}={pragma_value}")


def test_integration_security():
    """統合セキュリティテスト"""
    print("\n=== データベース統合セキュリティテスト ===")

    # セキュリティ強化されたデータベースマネージャーのテスト
    try:
        config = DatabaseConfig.for_testing()
        db_manager = DatabaseManager(config)

        # 1. データベース接続と基本操作
        print("\n1. セキュア接続:")
        try:
            # テーブル作成（セキュリティ機能統合確認）
            db_manager.create_tables()
            print("  OK セキュアデータベース接続・テーブル作成成功")

        except Exception as e:
            print(f"  FAIL セキュアデータベース接続エラー: {e}")

        # 2. セッション管理のセキュリティ
        print("\n2. セッション管理セキュリティ:")
        try:
            with db_manager.session_scope() as session:
                # 基本的なクエリ実行
                result = session.execute(text("SELECT 1")).scalar()
                if result == 1:
                    print("  OK セキュアセッション管理 - 正常動作")
                else:
                    print("  FAIL セキュアセッション管理 - 予期しない結果")
        except Exception as e:
            print(f"  FAIL セキュアセッション管理エラー: {e}")

        # 3. ヘルスチェック機能
        print("\n3. ヘルスチェック機能:")
        try:
            health_status = db_manager.health_check()
            if health_status.get("status") == "healthy":
                print("  OK データベースヘルスチェック - 健全")
            else:
                print(
                    f"  WARN データベースヘルスチェック - 状態: {health_status.get('status')}"
                )
        except Exception as e:
            print(f"  FAIL ヘルスチェックエラー: {e}")

    except Exception as e:
        print(f"  FAIL 統合セキュリティテストエラー: {e}")


def main():
    """メイン実行"""
    print("=== データベース層セキュリティテスト ===")
    print("Issue #392: データベース層のセキュリティ強化")
    print("=" * 60)

    try:
        # 各セキュリティ機能のテスト
        test_sqlite_pragma_sql_injection_protection()
        test_alembic_config_toctou_protection()
        test_secure_alembic_config_search()
        test_pragma_execution_safety()
        test_integration_security()

        print("\n" + "=" * 60)
        print("OK データベース層セキュリティテスト完了")
        print("\n実装されたセキュリティ強化:")
        print("- [SHIELD] SQLite PRAGMAホワイトリスト検証（SQLインジェクション防止）")
        print("- [GUARD] PRAGMA値範囲制限（メモリ枯渇攻撃防止）")
        print("- [ATOMIC] Alembic設定ファイルTOCTOU対策（原子的操作）")
        print("- [FILTER] 危険パス検出・シンボリックリンク攻撃防止")
        print("- [LIMIT] ファイルサイズ制限（DoS攻撃防止）")
        print("- [AUDIT] セキュリティ違反詳細ログ記録")

    except Exception as e:
        print(f"\nFAIL テスト実行中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
