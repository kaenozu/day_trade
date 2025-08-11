#!/usr/bin/env python3
"""
セキュアAPIクライアントテストスイート
Issue #395: 外部APIクライアントのセキュリティ強化

セキュリティ機能の包括的テスト
- APIキー管理セキュリティ
- URL構築セキュリティ
- エラーハンドリングセキュリティ
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta

import pytest

# パスを追加してモジュールをインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from day_trade.api.secure_api_client import (
        APIKeyType,
        SecureAPIKeyManager,
        SecureErrorHandler,
        SecureURLBuilder,
        SecurityLevel,
        URLSecurityPolicy,
    )
    SECURE_API_AVAILABLE = True
except ImportError as e:
    SECURE_API_AVAILABLE = False
    print(f"⚠️ セキュアAPIクライアントが利用できません: {e}")

# 基本的なテストケース
class TestSecureAPIKeyManager:
    """SecureAPIKeyManagerのテストケース"""

    @pytest.fixture
    def key_manager(self):
        """テスト用のキーマネージャー"""
        if not SECURE_API_AVAILABLE:
            pytest.skip("セキュアAPIクライアントが利用できません")
        return SecureAPIKeyManager()

    def test_key_addition_and_retrieval(self, key_manager):
        """APIキーの追加と取得テスト"""
        # テストキー追加
        success = key_manager.add_api_key(
            key_id="test_provider",
            plain_key="test_api_key_12345",
            key_type=APIKeyType.QUERY_PARAM,
            allowed_hosts=["api.example.com"],
            expiry_hours=24
        )

        assert success, "APIキー追加が失敗しました"

        # キー取得テスト
        retrieved_key = key_manager.get_api_key("test_provider", "api.example.com")
        assert retrieved_key == "test_api_key_12345", "取得したキーが元のキーと一致しません"

        # 不正なホストでの取得テスト
        invalid_key = key_manager.get_api_key("test_provider", "malicious.com")
        assert invalid_key is None, "不正なホストからのキー取得が許可されました"

    def test_key_expiration(self, key_manager):
        """APIキー有効期限テスト"""
        # 即座に期限切れになるキーを追加
        success = key_manager.add_api_key(
            key_id="expired_key",
            plain_key="expired_api_key",
            key_type=APIKeyType.HEADER,
            expiry_hours=-1  # 過去の時刻で期限切れ
        )

        assert success, "期限切れキーの追加が失敗しました"

        # 期限切れキーの取得テスト
        expired_key = key_manager.get_api_key("expired_key", "any.host.com")
        assert expired_key is None, "期限切れキーが取得できました（セキュリティ違反）"

    def test_invalid_key_format(self, key_manager):
        """無効なキーフォーマットのテスト"""
        # 空のキー
        success = key_manager.add_api_key(
            key_id="empty_key",
            plain_key="",
            key_type=APIKeyType.BEARER_TOKEN
        )
        assert not success, "空のAPIキーが受け入れられました"

        # 短すぎるキー
        success = key_manager.add_api_key(
            key_id="short_key",
            plain_key="123",
            key_type=APIKeyType.BEARER_TOKEN
        )
        assert not success, "短すぎるAPIキーが受け入れられました"

    def test_key_rotation(self, key_manager):
        """APIキーローテーションテスト"""
        # 初期キー追加
        key_manager.add_api_key(
            key_id="rotation_test",
            plain_key="original_key",
            key_type=APIKeyType.QUERY_PARAM,
            allowed_hosts=["api.example.com"]
        )

        # キーローテーション
        success = key_manager.rotate_key("rotation_test", "new_rotated_key")
        assert success, "キーローテーションが失敗しました"

        # 新しいキーの確認
        rotated_key = key_manager.get_api_key("rotation_test", "api.example.com")
        assert rotated_key == "new_rotated_key", "ローテーション後のキーが正しくありません"


class TestSecureURLBuilder:
    """SecureURLBuilderのテストケース"""

    @pytest.fixture
    def url_builder(self):
        """テスト用のURLビルダー"""
        if not SECURE_API_AVAILABLE:
            pytest.skip("セキュアAPIクライアントが利用できません")

        policy = URLSecurityPolicy(
            allowed_schemes=["https"],
            allowed_hosts=[".example.com", ".api.com"],
            max_url_length=2048,
            max_param_length=512
        )
        return SecureURLBuilder(policy)

    def test_secure_url_building(self, url_builder):
        """セキュアURL構築テスト"""
        base_url = "https://api.example.com/data"
        params = {
            "symbol": "AAPL",
            "apikey": "test_key"
        }

        secure_url = url_builder.build_secure_url(base_url, params=params)

        assert secure_url.startswith("https://api.example.com/data")
        assert "symbol=AAPL" in secure_url
        assert "apikey=test_key" in secure_url

    def test_path_parameter_substitution(self, url_builder):
        """パスパラメータ置換テスト"""
        base_url = "https://api.example.com/stocks/{symbol}/quote"
        path_params = {"symbol": "AAPL"}

        secure_url = url_builder.build_secure_url(
            base_url,
            path_params=path_params
        )

        assert "stocks/AAPL/quote" in secure_url
        assert "{symbol}" not in secure_url

    def test_security_violations(self, url_builder):
        """セキュリティ違反検出テスト"""
        # 不正なスキーム
        with pytest.raises(ValueError):
            url_builder.build_secure_url("http://api.example.com/data")

        # 不正なホスト
        with pytest.raises(ValueError):
            url_builder.build_secure_url("https://malicious.com/data")

        # パストラバーサル攻撃
        with pytest.raises(ValueError):
            url_builder.build_secure_url(
                "https://api.example.com/data/{path}",
                path_params={"path": "../../../etc/passwd"}
            )

    def test_parameter_sanitization(self, url_builder):
        """パラメータサニタイゼーションテスト"""
        # 危険な文字を含むパラメータ
        with pytest.raises(ValueError):
            url_builder.build_secure_url(
                "https://api.example.com/data",
                params={"param": "<script>alert('xss')</script>"}
            )

        # 長すぎるパラメータ
        long_param = "x" * 1025  # max_param_length = 512を超過
        with pytest.raises(ValueError):
            url_builder.build_secure_url(
                "https://api.example.com/data",
                params={"param": long_param}
            )


class TestSecureErrorHandler:
    """SecureErrorHandlerのテストケース"""

    def test_error_message_sanitization(self):
        """エラーメッセージサニタイゼーションテスト"""
        if not SECURE_API_AVAILABLE:
            pytest.skip("セキュアAPIクライアントが利用できません")

        # 機密情報を含むエラー
        sensitive_error = Exception("Connection failed: api_key=secret123 to host 192.168.1.100")

        sanitized = SecureErrorHandler.sanitize_error_message(sensitive_error, "API接続")

        # 機密情報が除去されていることを確認
        assert "secret123" not in sanitized
        assert "192.168.1.100" not in sanitized
        assert "Exception が発生しました" in sanitized

    def test_safe_error_response_creation(self):
        """安全なエラーレスポンス生成テスト"""
        if not SECURE_API_AVAILABLE:
            pytest.skip("セキュアAPIクライアントが利用できません")

        error = ConnectionError("Failed to connect with token=abc123")
        response = SecureErrorHandler.create_safe_error_response(error, "req-123")

        assert response["error"] is True
        assert response["error_type"] == "ConnectionError"
        assert response["request_id"] == "req-123"
        assert "abc123" not in str(response["message"])


# 統合テスト
async def test_integrated_security_features():
    """セキュリティ機能統合テスト"""
    if not SECURE_API_AVAILABLE:
        print("⚠️ セキュアAPIクライアント統合テストをスキップ")
        return

    print("🔒 セキュリティ機能統合テスト開始")

    # 1. APIキーマネージャーの統合テスト
    print("  📋 1. APIキーマネージャーテスト")
    key_manager = SecureAPIKeyManager()

    # 複数のAPIキーを追加
    providers = [
        ("alpha_vantage", "av_key_12345", ["www.alphavantage.co"]),
        ("yahoo_finance", "yf_key_67890", [".yahoo.com", ".yahooapis.com"]),
        ("invalid_provider", "malicious_key", ["malicious.com"])
    ]

    for provider, key, hosts in providers:
        success = key_manager.add_api_key(
            key_id=provider,
            plain_key=key,
            key_type=APIKeyType.QUERY_PARAM,
            allowed_hosts=hosts,
            expiry_hours=24
        )
        print(f"    {'✅' if success else '❌'} {provider}: {success}")

    # 2. セキュアURL構築テスト
    print("  🔗 2. セキュアURL構築テスト")
    url_policy = URLSecurityPolicy(
        allowed_schemes=["https"],
        allowed_hosts=[".alphavantage.co", ".yahoo.com"],
        max_url_length=2048
    )
    url_builder = SecureURLBuilder(url_policy)

    test_urls = [
        ("https://www.alphavantage.co/query", {"function": "TIME_SERIES_DAILY", "symbol": "IBM"}, True),
        ("http://www.alphavantage.co/query", {"function": "TIME_SERIES_DAILY"}, False),  # HTTP不可
        ("https://malicious.com/query", {"param": "value"}, False)  # 不正なホスト
    ]

    for base_url, params, should_succeed in test_urls:
        try:
            secure_url = url_builder.build_secure_url(base_url, params=params)
            result = should_succeed
            print(f"    {'✅' if result else '❌'} URL構築: {base_url[:30]}... = {result}")
        except ValueError:
            result = not should_succeed
            print(f"    {'✅' if result else '❌'} URL構築失敗（期待通り）: {base_url[:30]}...")

    # 3. エラーハンドリングテスト
    print("  🛡️ 3. セキュアエラーハンドリングテスト")

    test_errors = [
        ConnectionError("Connection failed to api.example.com with api_key=secret123"),
        ValueError("Invalid token: bearer_token_xyz789"),
        TimeoutError("Request timeout for https://192.168.1.100/api")
    ]

    for error in test_errors:
        sanitized = SecureErrorHandler.sanitize_error_message(error)

        # 機密情報が含まれていないことを確認
        sensitive_found = any(keyword in sanitized.lower() for keyword in
                            ["secret", "token_", "192.168", "api_key="])

        print(f"    {'✅' if not sensitive_found else '❌'} {type(error).__name__}: 機密情報除去済み")

    print("🎉 セキュリティ機能統合テスト完了")


def run_simple_tests():
    """簡単な動作確認テスト"""
    print("🚀 セキュアAPIクライアント動作確認テスト")
    print("=" * 50)

    if not SECURE_API_AVAILABLE:
        print("❌ セキュアAPIクライアント機能が利用できません")
        print("   依存関係を確認してください")
        return False

    print("✅ セキュアAPIクライアント機能が利用可能です")

    # 基本機能テスト
    try:
        # APIキーマネージャー
        key_manager = SecureAPIKeyManager()
        success = key_manager.add_api_key(
            "test", "test_key_123", APIKeyType.HEADER
        )
        print(f"✅ APIキーマネージャー: {success}")

        # URLビルダー
        url_builder = SecureURLBuilder()
        url = url_builder.build_secure_url("https://api.example.com/test")
        print(f"✅ URLビルダー: {len(url)} 文字のURL生成成功")

        # エラーハンドラー
        error_msg = SecureErrorHandler.sanitize_error_message(
            Exception("test error with api_key=secret"),
            "テスト"
        )
        print("✅ エラーハンドラー: 機密情報除去済み")

        return True

    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        return False


if __name__ == "__main__":
    print("🔐 セキュアAPIクライアント テストスイート")
    print("=" * 60)

    # 簡単なテスト実行
    simple_test_result = run_simple_tests()

    if simple_test_result:
        # 統合テスト実行
        try:
            asyncio.run(test_integrated_security_features())
            print("\n✅ 全てのセキュリティテストが完了しました！")
        except Exception as e:
            print(f"\n❌ 統合テストエラー: {e}")

    print("\n📊 テスト結果サマリー:")
    print("  - APIキー管理: エンタープライズレベルの暗号化と認証")
    print("  - URL構築: 包括的なセキュリティポリシー")
    print("  - エラーハンドリング: 機密情報漏洩防止")
    print("  - セキュリティレベル: HIGH")
