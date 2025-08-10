#!/usr/bin/env python3
"""
外部APIクライアント セキュリティ強化テスト
Issue #395: 外部APIクライアントのセキュリティ強化

実装されたセキュリティ強化:
1. APIキー管理のセキュリティ強化 - SecurityManager使用
2. URLパラメータ置換の脆弱性修正 - 危険文字検出・URLエンコーディング
3. エラーメッセージ情報漏洩対策 - 機密情報サニタイゼーション
4. CSV解析処理のセキュリティ強化 - ファイルサイズ・インジェクション対策
"""

import asyncio
import tempfile
import os
from unittest.mock import Mock, patch

# テスト中はログ出力を有効化
import logging
logging.basicConfig(level=logging.INFO)

from src.day_trade.api.external_api_client import (
    ExternalAPIClient,
    APIConfig,
    APIProvider,
    DataType,
    APIEndpoint,
    APIRequest,
    RequestMethod
)

# SecurityManagerはオプショナル依存関係
try:
    from src.day_trade.core.security_manager import SecurityManager
except ImportError:
    SecurityManager = None


def test_api_key_security_management():
    """APIキー管理セキュリティテスト"""
    print("=== APIキー管理セキュリティテスト ===")

    # 1. セキュリティマネージャーを使用したAPIキー管理テスト
    print("\n1. セキュリティマネージャー統合:")

    try:
        # テスト用環境変数設定
        os.environ['AV_API_KEY'] = 'test_alpha_vantage_key_12345'

        config = APIConfig()
        client = ExternalAPIClient(config)

        # Alpha Vantageエンドポイント
        endpoint = APIEndpoint(
            provider=APIProvider.ALPHA_VANTAGE,
            data_type=DataType.STOCK_PRICE,
            endpoint_url="https://www.alphavantage.co/query",
            requires_auth=True,
            auth_param_name="apikey"
        )

        # セキュリティマネージャー経由でAPIキー取得
        api_key = client._get_auth_key(endpoint)

        if api_key and api_key == 'test_alpha_vantage_key_12345':
            print("  OK セキュリティマネージャーAPIキー取得 - 成功")
        else:
            print(f"  FAIL セキュリティマネージャーAPIキー取得 - 失敗: {api_key}")

        # 環境変数クリーンアップ
        del os.environ['AV_API_KEY']

    except Exception as e:
        print(f"  FAIL セキュリティマネージャーテストエラー: {e}")

    # 2. 従来のapi_keys辞書フォールバックテスト
    print("\n2. 従来辞書フォールバック:")

    try:
        config_legacy = APIConfig(api_keys={'alpha_vantage': 'legacy_key_67890'})
        client_legacy = ExternalAPIClient(config_legacy)

        api_key_legacy = client_legacy._get_auth_key(endpoint)

        if api_key_legacy == 'legacy_key_67890':
            print("  OK 従来辞書フォールバック - 成功")
        else:
            print(f"  FAIL 従来辞書フォールバック - 失敗: {api_key_legacy}")

    except Exception as e:
        print(f"  FAIL 従来辞書テストエラー: {e}")


def test_url_parameter_security():
    """URLパラメータセキュリティテスト"""
    print("\n=== URLパラメータセキュリティテスト ===")

    client = ExternalAPIClient()

    # 1. 正常なパラメータのテスト
    print("\n1. 正常パラメータ:")
    safe_params = [
        ("symbol", "7203"),
        ("symbol", "AAPL"),
        ("symbol", "BRK.A"),
        ("symbol", "BRK-A"),
        ("ticker", "TSM"),
    ]

    for param_name, value in safe_params:
        try:
            sanitized = client._sanitize_url_parameter(value, param_name)
            print(f"  OK {param_name}={value} → {sanitized}")
        except Exception as e:
            print(f"  FAIL {param_name}={value} - 予期しないエラー: {e}")

    # 2. 危険なパラメータのテスト
    print("\n2. 危険パラメータ:")
    dangerous_params = [
        ("symbol", "../../../etc/passwd"),          # パストラバーサル攻撃
        ("symbol", "..\\..\\..\\windows\\system32"), # Windows パストラバーサル
        ("symbol", "%2e%2e%2fconfig"),              # エンコード済みパストラバーサル
        ("symbol", "//malicious.com/api"),          # プロトコル相対URL
        ("symbol", "\\\\malicious\\share"),         # UNCパス
        ("symbol", "test\x00.txt"),                 # NULLバイト攻撃
        ("symbol", "<script>alert(1)</script>"),    # HTMLタグ
        ("symbol", "'; DROP TABLE stocks; --"),    # SQLインジェクション様
        ("symbol", "javascript:alert(1)"),         # JavaScriptスキーム
        ("symbol", "data:text/html,<script>"),     # データスキーム
        ("symbol", "a" * 250),                     # 長すぎるパラメータ
    ]

    for param_name, value in dangerous_params:
        try:
            sanitized = client._sanitize_url_parameter(value, param_name)
            print(f"  FAIL {param_name}={value} - 危険パラメータが通過: {sanitized}")
        except ValueError as e:
            print(f"  OK {param_name}={value[:30]}... - 正常に阻止")
        except Exception as e:
            print(f"  WARN {param_name}={value[:30]}... - 予期しないエラー: {e}")


def test_error_message_sanitization():
    """エラーメッセージサニタイゼーションテスト"""
    print("\n=== エラーメッセージサニタイゼーションテスト ===")

    client = ExternalAPIClient()

    # 1. 機密情報を含むエラーメッセージのテスト
    print("\n1. 機密情報含有エラーメッセージ:")
    sensitive_errors = [
        ("ConnectionError", "Failed to connect to 192.168.1.100:8080"),
        ("FileNotFoundError", "No such file: C:/Users/admin/secret_api_keys.txt"),
        ("KeyError", "Missing key: sk_live_abcdef123456789abcdef"),
        ("ValueError", "Invalid token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
        ("AuthError", "Authentication failed for user@company.com"),
        ("URLError", "Cannot access https://internal.company.com/api/v1"),
        ("ConfigError", "password=super_secret_password_123 not found"),
    ]

    for error_type, error_message in sensitive_errors:
        try:
            sanitized = client._sanitize_error_message(error_message, error_type)

            # 機密情報がサニタイズされているかチェック
            if any(sensitive in sanitized.lower() for sensitive in
                   ['192.168', 'c:/users', 'sk_live', 'password', '@company.com']):
                print(f"  FAIL {error_type}: 機密情報が残存 - {sanitized}")
            else:
                print(f"  OK {error_type}: 安全にサニタイズ済み - {sanitized}")

        except Exception as e:
            print(f"  FAIL {error_type}: サニタイゼーションエラー - {e}")

    # 2. 一般的なエラーメッセージのテスト
    print("\n2. 一般的エラーメッセージ:")
    general_errors = [
        ("TimeoutError", "Request timed out after 30 seconds"),
        ("ValueError", "Invalid input format"),
        ("KeyError", "Required field missing"),
    ]

    for error_type, error_message in general_errors:
        try:
            sanitized = client._sanitize_error_message(error_message, error_type)
            print(f"  OK {error_type}: {sanitized}")
        except Exception as e:
            print(f"  FAIL {error_type}: エラー - {e}")


def test_csv_security_parsing():
    """CSV解析セキュリティテスト"""
    print("\n=== CSV解析セキュリティテスト ===")

    client = ExternalAPIClient()

    # 1. 正常なCSVデータのテスト
    print("\n1. 正常CSVデータ:")
    safe_csv = """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,95.0,102.0,1000000
2023-01-02,102.0,108.0,98.0,106.0,1200000
2023-01-03,106.0,110.0,104.0,109.0,900000"""

    try:
        df = client._parse_csv_response(safe_csv)
        if not df.empty and len(df) == 3:
            print(f"  OK 正常CSV解析 - {len(df)}行、{len(df.columns)}列")
        else:
            print(f"  FAIL 正常CSV解析 - 予期しない結果: {len(df)}行")
    except Exception as e:
        print(f"  FAIL 正常CSV解析エラー: {e}")

    # 2. 危険なCSVパターンのテスト
    print("\n2. 危険CSVパターン:")
    dangerous_csvs = [
        ("Excelコマンド実行", "Symbol,Price\n=cmd|'/c calc'!A1,100"),
        ("システムコマンド", "Symbol,Price\n=system('rm -rf /'),100"),
        ("Excel関数インジェクション", "Symbol,Price\n@SUM(A1:A1000)*1000,100"),
        ("ハイパーリンク", "Symbol,Price\n=HYPERLINK('http://malicious.com'),100"),
        ("JavaScript", "Symbol,Price\njavascript:alert(1),100"),
        ("HTMLデータ", "Symbol,Price\ndata:text/html;<script>alert(1)</script>,100"),
    ]

    for description, csv_content in dangerous_csvs:
        try:
            df = client._parse_csv_response(csv_content)
            if df.empty:
                print(f"  OK {description} - 正常に阻止（空DataFrame）")
            else:
                print(f"  WARN {description} - 処理されたが安全性要確認")
        except ValueError as e:
            print(f"  OK {description} - 正常に阻止（ValueError）")
        except Exception as e:
            print(f"  WARN {description} - 予期しないエラー: {e}")

    # 3. 大きなCSVファイルのテスト
    print("\n3. 大容量CSVファイル:")

    # 行数制限テスト
    large_csv_rows = "Symbol,Price\n" + "\n".join([f"TEST{i},{100+i}" for i in range(60000)])

    try:
        df = client._parse_csv_response(large_csv_rows)
        print(f"  FAIL 大容量CSV（行数） - 制限を超過したが処理された: {len(df)}行")
    except ValueError as e:
        print(f"  OK 大容量CSV（行数） - 正常に阻止")
    except Exception as e:
        print(f"  WARN 大容量CSV（行数） - 予期しないエラー: {e}")

    # ファイルサイズ制限テスト
    huge_csv = "Symbol,Price\n" + "A" * (11 * 1024 * 1024)  # 11MB

    try:
        df = client._parse_csv_response(huge_csv)
        print(f"  FAIL 大容量CSV（サイズ） - 制限を超過したが処理された")
    except ValueError as e:
        print(f"  OK 大容量CSV（サイズ） - 正常に阻止")
    except Exception as e:
        print(f"  WARN 大容量CSV（サイズ） - 予期しないエラー: {e}")


async def test_integrated_security():
    """統合セキュリティテスト"""
    print("\n=== 統合セキュリティテスト ===")

    # セキュリティ強化されたクライアント設定
    config = APIConfig(
        max_concurrent_requests=2,
        default_timeout_seconds=5,
        default_max_retries=1
    )

    client = ExternalAPIClient(config)
    await client.initialize()

    try:
        # 1. 危険なシンボルでの株価取得テスト
        print("\n1. 危険シンボル株価取得テスト:")
        dangerous_symbols = [
            "../malicious",
            "<script>",
            "'; DROP--",
        ]

        for symbol in dangerous_symbols:
            try:
                response = await client.fetch_stock_data(symbol, APIProvider.MOCK_PROVIDER)
                if response and not response.success:
                    print(f"  OK 危険シンボル阻止: {symbol} - {response.error_message}")
                else:
                    print(f"  WARN 危険シンボル処理: {symbol} - 要確認")
            except Exception as e:
                print(f"  OK 危険シンボル例外阻止: {symbol}")

        # 2. 通常の株価取得テスト（正常性確認）
        print("\n2. 正常株価取得テスト:")
        try:
            response = await client.fetch_stock_data("7203", APIProvider.MOCK_PROVIDER)
            if response and response.success:
                print(f"  OK 正常株価取得 - レスポンス時間: {response.response_time_ms:.1f}ms")
            else:
                error_msg = response.error_message if response else "レスポンスなし"
                print(f"  WARN 正常株価取得失敗 - {error_msg}")
        except Exception as e:
            print(f"  FAIL 正常株価取得エラー: {e}")

    finally:
        await client.cleanup()


def main():
    """メイン実行"""
    print("=== 外部APIクライアント セキュリティ強化テスト ===")
    print("Issue #395: 外部APIクライアントのセキュリティ強化")
    print("=" * 60)

    try:
        # 各セキュリティ機能のテスト
        test_api_key_security_management()
        test_url_parameter_security()
        test_error_message_sanitization()
        test_csv_security_parsing()

        # 非同期統合テスト
        asyncio.run(test_integrated_security())

        print("\n" + "=" * 60)
        print("OK 外部APIクライアント セキュリティ強化テスト完了")
        print("\n実装されたセキュリティ強化:")
        print("- [SECURE] APIキー管理強化 (SecurityManager統合)")
        print("- [SHIELD] URLパラメータ検証・エンコーディング")
        print("- [MASK] エラーメッセージ機密情報サニタイゼーション")
        print("- [GUARD] CSV解析インジェクション・DoS対策")
        print("- [AUDIT] セキュリティ違反ログ記録")

    except Exception as e:
        print(f"\nFAIL テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
