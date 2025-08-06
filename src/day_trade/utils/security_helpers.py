"""
セキュリティヘルパーユーティリティ

SQLインジェクション防止、セキュアな乱数生成、入力検証、
ログメッセージのサニタイズなど、包括的なセキュリティ機能を提供。

Phase 1: セキュリティ強化プロジェクト対応
"""

import hashlib
import hmac
import re
import secrets
import string
from typing import Optional

from sqlalchemy.sql.elements import quoted_name


class SecurityHelpers:
    """セキュリティ関連のヘルパー機能を提供するクラス"""

    @staticmethod
    def safe_table_name(table_name: str) -> quoted_name:
        """
        SQLインジェクション攻撃を防ぐ安全なテーブル名の生成

        Args:
            table_name: テーブル名

        Returns:
            quoted_name: SQLAlchemyの安全なテーブル名オブジェクト
        """
        # SQLAlchemyのquoted_nameを使用してSQLインジェクションを防止
        return quoted_name(table_name, quote=True)

    @staticmethod
    def safe_column_name(column_name: str) -> quoted_name:
        """
        SQLインジェクション攻撃を防ぐ安全なカラム名の生成

        Args:
            column_name: カラム名

        Returns:
            quoted_name: SQLAlchemyの安全なカラム名オブジェクト
        """
        return quoted_name(column_name, quote=True)

    @staticmethod
    def secure_random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        暗号学的に安全な浮動小数点乱数を生成

        Args:
            min_val: 最小値
            max_val: 最大値

        Returns:
            float: セキュアな乱数
        """
        # cryptographically secure random number generator
        random_bytes = secrets.randbits(32)
        # 0.0 to 1.0 の範囲に正規化
        normalized = random_bytes / (2**32 - 1)
        # 指定範囲にスケール
        return min_val + (normalized * (max_val - min_val))

    @staticmethod
    def secure_random_int(min_val: int = 0, max_val: int = 100) -> int:
        """
        暗号学的に安全な整数乱数を生成

        Args:
            min_val: 最小値
            max_val: 最大値

        Returns:
            int: セキュアな乱数
        """
        return secrets.randbelow(max_val - min_val + 1) + min_val

    @staticmethod
    def secure_random_string(length: int = 32) -> str:
        """
        暗号学的に安全なランダム文字列を生成

        Args:
            length: 文字列の長さ

        Returns:
            str: セキュアなランダム文字列
        """
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """
        ログメッセージから機密情報を除去

        Args:
            message: 元のログメッセージ

        Returns:
            str: サニタイズされたメッセージ
        """
        # 機密情報のパターンを定義
        patterns = [
            # パスワード
            (r"password[:\s=]+[^\s,;]+", "password: ***"),
            (r"pwd[:\s=]+[^\s,;]+", "pwd: ***"),
            # APIキー・トークン
            (r"api[_\s]?key[:\s=]+[^\s,;]+", "api_key: ***"),
            (r"key[:\s=]+sk-[^\s,;]+", "key: ***"),
            (r"token[:\s=]+[^\s,;]+", "token: ***"),
            (r"bearer[:\s=]+[^\s,;]+", "bearer: ***"),
            # JWT トークン
            (r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*", "[JWT_TOKEN]"),
            # メールアドレス
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_MASKED]"),
            # クレジットカード番号
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_MASKED]"),
            # 電話番号
            (r"\b\d{3}-?\d{4}-?\d{4}\b", "[PHONE_MASKED]"),
            # セッションID・ハッシュ値
            (r"\b[a-fA-F0-9]{32,}\b", "[HASH_MASKED]"),
        ]

        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    @staticmethod
    def validate_input(input_value: str, input_type: str = "general") -> bool:
        """
        入力値の検証（SQLインジェクション、XSS等の攻撃検出）

        Args:
            input_value: 検証する入力値
            input_type: 入力タイプ（username, email, numeric等）

        Returns:
            bool: 安全な入力の場合True
        """
        if not isinstance(input_value, str):
            return False

        # 基本的な悪意のあるパターンをチェック
        malicious_patterns = [
            # SQLインジェクション
            r"(\';|\";|\/\*|\*\/|xp_|sp_)",
            r"(union\s+select|drop\s+table|insert\s+into)",
            r"(delete\s+from|update\s+.*set)",
            r"(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
            # XSS
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                return False

        # 入力タイプ別の追加検証
        if input_type == "username":
            # ユーザー名：英数字、アンダースコア、ハイフンのみ
            return re.match(r"^[a-zA-Z0-9_-]{3,50}$", input_value) is not None

        elif input_type == "email":
            # 基本的なメール形式チェック（連続ドットを禁止）
            if ".." in input_value:
                return False
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return re.match(email_pattern, input_value) is not None

        elif input_type == "numeric":
            # 数値のみ
            try:
                float(input_value)
                return True
            except ValueError:
                return False

        return True

    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """
        機密データのハッシュ化

        Args:
            data: ハッシュ化するデータ
            salt: ソルト（指定しない場合は固定ソルトを使用）

        Returns:
            str: ハッシュ化されたデータ
        """
        if salt is None:
            # テスト用に固定ソルトを使用（実際の運用では動的ソルトを推奨）
            salt = "default_salt_for_testing"

        # SHA-256でハッシュ化
        return hashlib.sha256((data + salt).encode("utf-8")).hexdigest()

    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """
        タイミング攻撃耐性のある文字列比較

        Args:
            a: 比較する文字列1
            b: 比較する文字列2

        Returns:
            bool: 同じ文字列の場合True
        """
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

    @staticmethod
    def generate_csrf_token() -> str:
        """
        CSRF攻撃防止用のトークンを生成

        Returns:
            str: CSRFトークン
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def validate_csrf_token(token: str, expected_token: str) -> bool:
        """
        CSRFトークンの検証

        Args:
            token: 受信したトークン
            expected_token: 期待されるトークン

        Returns:
            bool: 有効なトークンの場合True
        """
        return SecurityHelpers.secure_compare(token, expected_token)
