"""
セキュリティヘルパー関数 - セキュリティ強化プロジェクト対応

SQLインジェクション対策、安全な乱数生成、
その他のセキュリティ機能を提供する。
"""

import secrets
import string
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.sql import quoted_name


class SecurityHelpers:
    """セキュリティ関連のヘルパー関数を提供するクラス"""

    @staticmethod
    def safe_table_name(table_name: str) -> quoted_name:
        """
        安全なテーブル名を作成する（SQLインジェクション対策）

        Args:
            table_name: テーブル名

        Returns:
            quoted_name: 安全にクォートされたテーブル名
        """
        # テーブル名の妥当性チェック
        if not isinstance(table_name, str):
            raise ValueError("テーブル名は文字列である必要があります")

        # 危険な文字のチェック
        forbidden_chars = [";", "--", "/*", "*/", "\\", "'", '"']
        if any(char in table_name for char in forbidden_chars):
            raise ValueError("テーブル名に危険な文字が含まれています")

        # 英数字とアンダースコアのみ許可
        if not table_name.replace("_", "").isalnum():
            raise ValueError("テーブル名は英数字とアンダースコアのみ使用できます")

        return quoted_name(table_name, True)

    @staticmethod
    def safe_sql_text(sql_template: str, **params: Any) -> text:
        """
        安全なSQLテキストを作成する（パラメータ化クエリ）

        Args:
            sql_template: SQLテンプレート（名前付きパラメータを使用）
            **params: パラメータ

        Returns:
            text: 安全なSQLテキスト

        Example:
            safe_sql_text("SELECT * FROM :table_name WHERE id = :id",
                         table_name="users", id=123)
        """
        return text(sql_template).params(**params)

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        暗号学的に安全なトークンを生成する

        Args:
            length: トークン長（デフォルト: 32文字）

        Returns:
            str: 安全に生成されたトークン
        """
        if length <= 0:
            raise ValueError("トークン長は正の整数である必要があります")

        # 英数字からなる安全なトークンを生成
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def secure_random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        暗号学的に安全な浮動小数点乱数を生成する

        Args:
            min_val: 最小値（デフォルト: 0.0）
            max_val: 最大値（デフォルト: 1.0）

        Returns:
            float: 安全に生成された浮動小数点数
        """
        if min_val >= max_val:
            raise ValueError("最小値は最大値より小さい必要があります")

        # secrets.randbits()を使用して安全な乱数を生成
        random_bits = secrets.randbits(32)
        normalized = random_bits / (2**32)  # 0.0-1.0に正規化

        # 指定範囲にスケール
        return min_val + normalized * (max_val - min_val)

    @staticmethod
    def secure_random_int(min_val: int, max_val: int) -> int:
        """
        暗号学的に安全な整数乱数を生成する

        Args:
            min_val: 最小値
            max_val: 最大値（含む）

        Returns:
            int: 安全に生成された整数
        """
        if min_val > max_val:
            raise ValueError("最小値は最大値以下である必要があります")

        return secrets.randbelow(max_val - min_val + 1) + min_val

    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """
        ログメッセージをサニタイズする（機密情報の漏洩防止）

        Args:
            message: 元のメッセージ

        Returns:
            str: サニタイズされたメッセージ
        """
        if not isinstance(message, str):
            return str(message)

        # 機密情報パターンをマスク
        sensitive_patterns = [
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD-MASKED]"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL-MASKED]"),
            (r"\bpassword\s*[:=]\s*[^\s]+", "password=[MASKED]"),
            (r"\bapi[_-]?key\s*[:=]\s*[^\s]+", "api_key=[MASKED]"),
            (r"\btoken\s*[:=]\s*[^\s]+", "token=[MASKED]"),
            (r"\bsecret\s*[:=]\s*[^\s]+", "secret=[MASKED]"),
        ]

        import re

        sanitized = message
        for pattern, replacement in sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    @staticmethod
    def validate_input_string(
        input_str: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
    ) -> str:
        """
        入力文字列の妥当性を検証する

        Args:
            input_str: 入力文字列
            max_length: 最大長（オプション）
            allowed_chars: 許可文字セット（オプション）

        Returns:
            str: 検証済み文字列

        Raises:
            ValueError: 妥当性検証に失敗した場合
        """
        if not isinstance(input_str, str):
            raise ValueError("入力は文字列である必要があります")

        if max_length is not None and len(input_str) > max_length:
            raise ValueError(f"入力文字列が最大長{max_length}を超えています")

        if allowed_chars is not None and not all(c in allowed_chars for c in input_str):
            raise ValueError("許可されていない文字が含まれています")

        return input_str
