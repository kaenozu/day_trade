"""
セキュリティヘルパー機能のテスト

Phase 1: セキュリティ強化プロジェクトで追加された
SecurityHelpersクラスの包括的なテスト
"""

import string

from sqlalchemy.sql.elements import quoted_name

from src.day_trade.utils.security_helpers import SecurityHelpers


class TestSecurityHelpers:
    """セキュリティヘルパー機能のテストクラス"""

    def test_safe_table_name_valid_input(self):
        """有効なテーブル名のテスト"""
        # 通常のテーブル名
        result = SecurityHelpers.safe_table_name("users")
        assert isinstance(result, quoted_name)
        assert str(result) == "users"

        # アンダースコア付きテーブル名
        result = SecurityHelpers.safe_table_name("user_profiles")
        assert isinstance(result, quoted_name)
        assert str(result) == "user_profiles"

    def test_safe_table_name_with_special_characters(self):
        """特殊文字を含むテーブル名のテスト"""
        # スペース付きテーブル名（クォート処理される）
        result = SecurityHelpers.safe_table_name("user profiles")
        assert isinstance(result, quoted_name)

        # ハイフン付きテーブル名
        result = SecurityHelpers.safe_table_name("user-profiles")
        assert isinstance(result, quoted_name)

    def test_safe_table_name_sql_injection_attempt(self):
        """SQLインジェクション試行のテスト"""
        # SQLインジェクション試行（DROP TABLE等）
        malicious_input = "users; DROP TABLE accounts"
        result = SecurityHelpers.safe_table_name(malicious_input)
        assert isinstance(result, quoted_name)
        # クォート処理により無害化される

    def test_secure_random_float_default_range(self):
        """セキュアな浮動小数点乱数生成のテスト（デフォルト範囲）"""
        result = SecurityHelpers.secure_random_float()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_secure_random_float_custom_range(self):
        """セキュアな浮動小数点乱数生成のテスト（カスタム範囲）"""
        min_val, max_val = 10.0, 100.0
        result = SecurityHelpers.secure_random_float(min_val, max_val)
        assert isinstance(result, float)
        assert min_val <= result <= max_val

    def test_secure_random_float_negative_range(self):
        """セキュアな浮動小数点乱数生成のテスト（負の範囲）"""
        min_val, max_val = -50.0, -10.0
        result = SecurityHelpers.secure_random_float(min_val, max_val)
        assert isinstance(result, float)
        assert min_val <= result <= max_val

    def test_secure_random_float_distribution(self):
        """セキュアな浮動小数点乱数の分布テスト"""
        # 複数回生成して分布を確認
        results = [SecurityHelpers.secure_random_float() for _ in range(100)]

        # すべて有効な範囲内
        assert all(0.0 <= r <= 1.0 for r in results)

        # 統計的にランダムであることを簡易確認
        mean = sum(results) / len(results)
        assert 0.3 < mean < 0.7  # 期待値0.5の周辺

    def test_secure_random_int_default_range(self):
        """セキュアな整数乱数生成のテスト（デフォルト範囲）"""
        result = SecurityHelpers.secure_random_int()
        assert isinstance(result, int)
        assert 0 <= result <= 100

    def test_secure_random_int_custom_range(self):
        """セキュアな整数乱数生成のテスト（カスタム範囲）"""
        min_val, max_val = 50, 200
        result = SecurityHelpers.secure_random_int(min_val, max_val)
        assert isinstance(result, int)
        assert min_val <= result <= max_val

    def test_secure_random_int_single_value_range(self):
        """セキュアな整数乱数生成のテスト（単一値範囲）"""
        min_val = max_val = 42
        result = SecurityHelpers.secure_random_int(min_val, max_val)
        assert result == 42

    def test_secure_random_string_default_length(self):
        """セキュアなランダム文字列生成のテスト（デフォルト長）"""
        result = SecurityHelpers.secure_random_string()
        assert isinstance(result, str)
        assert len(result) == 32
        # 英数字のみ
        assert all(c in string.ascii_letters + string.digits for c in result)

    def test_secure_random_string_custom_length(self):
        """セキュアなランダム文字列生成のテスト（カスタム長）"""
        length = 16
        result = SecurityHelpers.secure_random_string(length)
        assert isinstance(result, str)
        assert len(result) == length
        assert all(c in string.ascii_letters + string.digits for c in result)

    def test_secure_random_string_uniqueness(self):
        """セキュアなランダム文字列の一意性テスト"""
        # 複数回生成して重複がないことを確認
        results = [SecurityHelpers.secure_random_string() for _ in range(50)]
        assert len(set(results)) == len(results)  # すべて異なる

    def test_sanitize_log_message_basic(self):
        """基本的なログメッセージサニタイズのテスト"""
        clean_message = "User login successful"
        result = SecurityHelpers.sanitize_log_message(clean_message)
        assert result == clean_message

    def test_sanitize_log_message_password(self):
        """パスワードを含むログメッセージのサニタイズテスト"""
        message_with_password = "Login failed for user: admin, password: secret123"
        result = SecurityHelpers.sanitize_log_message(message_with_password)

        # パスワードがマスクされていることを確認
        assert "secret123" not in result
        assert "***" in result or "[MASKED]" in result

    def test_sanitize_log_message_api_key(self):
        """APIキーを含むログメッセージのサニタイズテスト"""
        message_with_api_key = "API call made with key: sk-1234567890abcdef"
        result = SecurityHelpers.sanitize_log_message(message_with_api_key)

        # APIキーがマスクされていることを確認
        assert "sk-1234567890abcdef" not in result
        assert "***" in result or "[MASKED]" in result

    def test_sanitize_log_message_token(self):
        """トークンを含むログメッセージのサニタイズテスト"""
        message_with_token = (
            "Authentication token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        )
        result = SecurityHelpers.sanitize_log_message(message_with_token)

        # トークンがマスクされていることを確認
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "***" in result or "[MASKED]" in result

    def test_sanitize_log_message_email(self):
        """メールアドレスを含むログメッセージのサニタイズテスト"""
        message_with_email = (
            "User registration: email user@example.com, status: success"
        )
        result = SecurityHelpers.sanitize_log_message(message_with_email)

        # メールアドレスがマスクされていることを確認
        assert "user@example.com" not in result
        assert "[EMAIL_MASKED]" in result

    def test_sanitize_log_message_credit_card(self):
        """クレジットカード番号を含むログメッセージのサニタイズテスト"""
        message_with_cc = "Payment processed with card: 4111-1111-1111-1111"
        result = SecurityHelpers.sanitize_log_message(message_with_cc)

        # カード番号がマスクされていることを確認
        assert "4111-1111-1111-1111" not in result
        assert "[CARD_MASKED]" in result

    def test_sanitize_log_message_multiple_sensitive_data(self):
        """複数の機密データを含むログメッセージのサニタイズテスト"""
        complex_message = (
            "User admin logged in with password secret123 "
            "using API key sk-abcdef123456 and email admin@company.com"
        )
        result = SecurityHelpers.sanitize_log_message(complex_message)

        # すべての機密データがマスクされていることを確認
        sensitive_data = ["secret123", "sk-abcdef123456", "admin@company.com"]
        for data in sensitive_data:
            assert data not in result

        # マスク文字が含まれていることを確認
        assert (
            "***" in result
            or "[MASKED]" in result
            or "[EMAIL_MASKED]" in result
            or "[CARD_MASKED]" in result
        )

    def test_validate_input_safe_string(self):
        """安全な文字列入力の検証テスト"""
        safe_input = "user123"
        result = SecurityHelpers.validate_input(safe_input, input_type="username")
        assert result is True

    def test_validate_input_malicious_string(self):
        """悪意のある文字列入力の検証テスト"""
        malicious_input = "user'; DROP TABLE users; --"
        result = SecurityHelpers.validate_input(malicious_input, input_type="username")
        assert result is False

    def test_validate_input_numeric(self):
        """数値入力の検証テスト"""
        # 有効な数値
        assert SecurityHelpers.validate_input("123", input_type="numeric") is True
        assert SecurityHelpers.validate_input("123.45", input_type="numeric") is True

        # 無効な数値
        assert SecurityHelpers.validate_input("abc", input_type="numeric") is False
        assert SecurityHelpers.validate_input("123abc", input_type="numeric") is False

    def test_validate_input_email(self):
        """メールアドレス入力の検証テスト"""
        # 有効なメールアドレス
        valid_emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "simple@domain.org",
        ]
        for email in valid_emails:
            assert SecurityHelpers.validate_input(email, input_type="email") is True

        # 無効なメールアドレス
        invalid_emails = [
            "invalid.email",
            "@domain.com",
            "user@",
            "user..name@domain.com",
        ]
        for email in invalid_emails:
            assert SecurityHelpers.validate_input(email, input_type="email") is False

    def test_validate_input_sql_injection_attempts(self):
        """SQLインジェクション試行の検証テスト"""
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "1' UNION SELECT * FROM passwords --",
        ]

        for attempt in sql_injection_attempts:
            assert (
                SecurityHelpers.validate_input(attempt, input_type="username") is False
            )

    def test_validate_input_xss_attempts(self):
        """XSS試行の検証テスト"""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src='x' onerror='alert(1)'>",
            "<iframe src='javascript:alert(1)'></iframe>",
        ]

        for attempt in xss_attempts:
            assert (
                SecurityHelpers.validate_input(attempt, input_type="username") is False
            )

    def test_hash_sensitive_data(self):
        """機密データのハッシュ化テスト"""
        sensitive_data = "password123"
        hashed = SecurityHelpers.hash_sensitive_data(sensitive_data)

        # ハッシュ化されていることを確認
        assert hashed != sensitive_data
        assert len(hashed) > 0

        # 同じデータは同じハッシュになることを確認
        hashed2 = SecurityHelpers.hash_sensitive_data(sensitive_data)
        assert hashed == hashed2

    def test_hash_sensitive_data_different_inputs(self):
        """異なる入力に対する異なるハッシュの生成テスト"""
        data1 = "password123"
        data2 = "different_password"

        hash1 = SecurityHelpers.hash_sensitive_data(data1)
        hash2 = SecurityHelpers.hash_sensitive_data(data2)

        assert hash1 != hash2

    def test_secure_compare(self):
        """セキュアな文字列比較のテスト"""
        string1 = "secret_value"
        string2 = "secret_value"
        string3 = "different_value"

        # 同じ値の比較
        assert SecurityHelpers.secure_compare(string1, string2) is True

        # 異なる値の比較
        assert SecurityHelpers.secure_compare(string1, string3) is False

    def test_secure_compare_timing_attack_resistance(self):
        """タイミング攻撃耐性のテスト"""
        import time

        correct_value = "a" * 100  # 長い正しい値
        wrong_short = "b"  # 短い間違った値
        wrong_long = "b" * 100  # 長い間違った値

        # 複数回測定して平均時間を計算
        times_short = []
        times_long = []

        for _ in range(10):
            start = time.time()
            SecurityHelpers.secure_compare(correct_value, wrong_short)
            times_short.append(time.time() - start)

            start = time.time()
            SecurityHelpers.secure_compare(correct_value, wrong_long)
            times_long.append(time.time() - start)

        # 時間差が著しく大きくないことを確認（タイミング攻撃耐性）
        avg_short = sum(times_short) / len(times_short)
        avg_long = sum(times_long) / len(times_long)

        # 完全に等しい時間ではないが、著しい差はない
        assert abs(avg_short - avg_long) < 0.001  # 1ms未満の差

    def test_safe_column_name_valid_input(self):
        """有効なカラム名のテスト"""
        # 通常のカラム名
        result = SecurityHelpers.safe_column_name("username")
        assert isinstance(result, quoted_name)
        assert str(result) == "username"

        # アンダースコア付きカラム名
        result = SecurityHelpers.safe_column_name("user_id")
        assert isinstance(result, quoted_name)
        assert str(result) == "user_id"

    def test_safe_column_name_sql_injection_attempt(self):
        """カラム名でのSQLインジェクション試行のテスト"""
        # SQLインジェクション試行
        malicious_input = "username; DROP TABLE users"
        result = SecurityHelpers.safe_column_name(malicious_input)
        assert isinstance(result, quoted_name)
        # クォート処理により無害化される

    def test_generate_csrf_token(self):
        """CSRFトークン生成のテスト"""
        token = SecurityHelpers.generate_csrf_token()
        assert isinstance(token, str)
        assert len(token) > 0
        
        # トークンの一意性確認
        token2 = SecurityHelpers.generate_csrf_token()
        assert token != token2

    def test_generate_csrf_token_multiple(self):
        """複数のCSRFトークン生成テスト（一意性確認）"""
        tokens = [SecurityHelpers.generate_csrf_token() for _ in range(10)]
        # すべて異なることを確認
        assert len(set(tokens)) == len(tokens)

    def test_validate_csrf_token_valid(self):
        """有効なCSRFトークン検証のテスト"""
        original_token = SecurityHelpers.generate_csrf_token()
        # 同じトークンでの検証
        assert SecurityHelpers.validate_csrf_token(original_token, original_token) is True

    def test_validate_csrf_token_invalid(self):
        """無効なCSRFトークン検証のテスト"""
        token1 = SecurityHelpers.generate_csrf_token()
        token2 = SecurityHelpers.generate_csrf_token()
        # 異なるトークンでの検証
        assert SecurityHelpers.validate_csrf_token(token1, token2) is False

    def test_validate_csrf_token_empty(self):
        """空のCSRFトークン検証のテスト"""
        token = SecurityHelpers.generate_csrf_token()
        # 空文字列との比較
        assert SecurityHelpers.validate_csrf_token("", token) is False
        assert SecurityHelpers.validate_csrf_token(token, "") is False

    def test_hash_sensitive_data_with_custom_salt(self):
        """カスタムソルトでの機密データハッシュ化テスト"""
        data = "sensitive_data"
        salt = "custom_salt_123"
        
        hashed = SecurityHelpers.hash_sensitive_data(data, salt)
        assert hashed != data
        assert len(hashed) == 64  # SHA-256は64文字のhex文字列
        
        # 同じデータ・ソルトで同じハッシュが生成される
        hashed2 = SecurityHelpers.hash_sensitive_data(data, salt)
        assert hashed == hashed2

    def test_hash_sensitive_data_different_salts(self):
        """異なるソルトでの異なるハッシュ生成テスト"""
        data = "sensitive_data"
        salt1 = "salt1"
        salt2 = "salt2"
        
        hash1 = SecurityHelpers.hash_sensitive_data(data, salt1)
        hash2 = SecurityHelpers.hash_sensitive_data(data, salt2)
        
        # 異なるソルトでは異なるハッシュが生成される
        assert hash1 != hash2

    def test_sanitize_log_message_phone_number(self):
        """電話番号を含むログメッセージのサニタイズテスト"""
        message_with_phone = "Contact customer at 090-1234-5678"
        result = SecurityHelpers.sanitize_log_message(message_with_phone)
        
        # 電話番号がマスクされていることを確認
        assert "090-1234-5678" not in result
        assert "[PHONE_MASKED]" in result

    def test_sanitize_log_message_hash_values(self):
        """ハッシュ値を含むログメッセージのサニタイズテスト"""
        message_with_hash = "Session ID: a1b2c3d4e5f6789012345678901234567890abcd"
        result = SecurityHelpers.sanitize_log_message(message_with_hash)
        
        # ハッシュ値がマスクされていることを確認
        assert "a1b2c3d4e5f6789012345678901234567890abcd" not in result
        assert "[HASH_MASKED]" in result

    def test_validate_input_edge_cases(self):
        """入力検証のエッジケーステスト"""
        # None入力
        assert SecurityHelpers.validate_input(None, "username") is False
        
        # 整数入力（文字列以外）
        assert SecurityHelpers.validate_input(123, "username") is False
        
        # 空文字列
        assert SecurityHelpers.validate_input("", "username") is False
        
        # 非常に長い文字列（ユーザー名として）
        long_username = "a" * 100
        assert SecurityHelpers.validate_input(long_username, "username") is False

    def test_validate_input_general_type(self):
        """一般的な入力タイプの検証テスト"""
        # 一般的なタイプ（デフォルト）
        safe_input = "normal text without malicious content"
        assert SecurityHelpers.validate_input(safe_input) is True
        
        # 特殊文字を含むが安全な入力
        safe_special = "Hello, World! 123 @#$%"
        assert SecurityHelpers.validate_input(safe_special) is True

    def test_all_security_methods_exist(self):
        """すべてのセキュリティメソッドが存在することを確認"""
        methods = [
            'safe_table_name',
            'safe_column_name', 
            'secure_random_float',
            'secure_random_int',
            'secure_random_string',
            'sanitize_log_message',
            'validate_input',
            'hash_sensitive_data',
            'secure_compare',
            'generate_csrf_token',
            'validate_csrf_token'
        ]
        
        for method_name in methods:
            assert hasattr(SecurityHelpers, method_name)
            method = getattr(SecurityHelpers, method_name)
            assert callable(method)
