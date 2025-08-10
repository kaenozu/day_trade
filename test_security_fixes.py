#!/usr/bin/env python3
"""
セキュリティ修正のテストスクリプト
Issue #388 と #387 の修正を検証
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_exception_sanitization():
    """例外サニタイズ機能のテスト"""
    print("=== 例外サニタイズ機能テスト ===")

    from src.day_trade.utils.exceptions import (
        DayTradeError,
        _sanitize_sensitive_info
    )

    # 機密情報を含むテストケース（セキュリティスキャナー回避のためダミーキー使用）
    # セキュリティテスト用：実際の機密情報は含まないダミーデータ
    test_cases = [
        "Database error: password=[DUMMY_PWD] connection failed",
        "API error: api_key=[DUMMY_KEY] token=bearer_[DUMMY_TOKEN]",
        "Connection failed to postgresql://user:[DUMMY_PASS]@localhost:5432/db",
        "Normal error message without sensitive data"
    ]

    for i, test_input in enumerate(test_cases, 1):
        sanitized = _sanitize_sensitive_info(test_input)
        print(f"テストケース #{i}: [機密情報を含むエラーメッセージ]")
        print(f"サニタイズ結果: {sanitized}")
        print()

    # DayTradeError のテスト
    error = DayTradeError(
        message="Database connection failed: password=[REDACTED]",
        error_code="DB_ERROR",
        details={"connection_string": "postgresql://user:[REDACTED]@host/db"}
    )

    print("DayTradeError サニタイズテスト:")
    print("安全なメッセージ:", error.get_safe_message())
    print("安全な詳細:", error.get_safe_details())

    return True

def test_password_validation():
    """パスワード強度検証のテスト"""
    print("\n=== パスワード強度検証テスト ===")

    from src.day_trade.core.security_config import SecureConfigManager

    config_manager = SecureConfigManager()

    # パスワードテストケース
    passwords = [
        "weak",                    # 短すぎる
        "password123",            # 弱い
        "StrongPass123!",         # 良い
        "VeryStr0ng&Secure!2024", # 非常に良い
        "aaaa1111",              # 同じ文字の連続
    ]

    for i, pwd in enumerate(passwords, 1):
        issues = config_manager._validate_password_strength(pwd)
        print(f"パスワード #{i}: {'*' * len(pwd)} (長さ: {len(pwd)})")
        if issues:
            print(f"  問題: {', '.join(issues)}")
        else:
            print("  OK: 強度十分")
        print()

    return True

def test_security_config():
    """セキュリティ設定のテスト"""
    print("=== セキュリティ設定テスト ===")

    from src.day_trade.core.security_config import SecureConfigManager

    # テスト設定（セキュリティスキャナー回避のためダミーデータ使用）
    test_config = {
        "api_key": "dummy_test_key_123",
        "password": "test_weak_pwd",
        "database_url": "postgresql://testuser:testpwd@localhost/testdb",
        "normal_setting": "normal_value"
    }

    config_manager = SecureConfigManager()

    # セキュリティ検証
    warnings = config_manager.validate_config_security(test_config)
    print("セキュリティ警告:")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    # 改善提案
    suggestions = config_manager.suggest_security_improvements(test_config)
    print("\n改善提案:")
    for suggestion in suggestions:
        print(f"  SUGGESTION: {suggestion}")

    return True

def main():
    """メインテスト実行"""
    print("セキュリティ修正テスト開始")
    print("=" * 50)

    try:
        # テスト実行
        test_exception_sanitization()
        test_password_validation()
        test_security_config()

        print("\n[SUCCESS] 全てのセキュリティ修正テストが完了しました")
        return 0

    except Exception as e:
        print(f"\n[ERROR] テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
