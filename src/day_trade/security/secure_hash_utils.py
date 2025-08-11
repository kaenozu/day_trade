#!/usr/bin/env python3
"""
セキュアハッシュユーティリティ
Issue #419対応 - 弱いハッシュアルゴリズムの代替実装
"""

import hashlib
import secrets
from typing import Optional, Union


class SecureHashUtils:
    """セキュアなハッシュ化ユーティリティクラス"""

    @staticmethod
    def generate_cache_key(
        data: Union[str, bytes], prefix: str = "", length: Optional[int] = 16
    ) -> str:
        """
        キャッシュキー用のセキュアなハッシュ生成

        Args:
            data: ハッシュ化するデータ
            prefix: キーのプレフィックス
            length: ハッシュの長さ（None で完全長）

        Returns:
            セキュアなハッシュキー
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        # SHA256使用（セキュリティ用途でないことを明示）
        hash_obj = hashlib.sha256(data, usedforsecurity=False)
        hash_digest = hash_obj.hexdigest()

        if length is not None:
            hash_digest = hash_digest[:length]

        return f"{prefix}{hash_digest}" if prefix else hash_digest

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        セキュアなランダムトークン生成

        Args:
            length: トークンの長さ

        Returns:
            セキュアなランダムトークン
        """
        return secrets.token_hex(length)

    @staticmethod
    def generate_content_hash(content: Union[str, bytes]) -> str:
        """
        コンテンツの整合性確認用ハッシュ生成

        Args:
            content: ハッシュ化するコンテンツ

        Returns:
            SHA256ハッシュ
        """
        if isinstance(content, str):
            content = content.encode("utf-8")

        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def verify_content_hash(content: Union[str, bytes], expected_hash: str) -> bool:
        """
        コンテンツハッシュの検証

        Args:
            content: 検証するコンテンツ
            expected_hash: 期待されるハッシュ値

        Returns:
            ハッシュが一致するかどうか
        """
        actual_hash = SecureHashUtils.generate_content_hash(content)
        return secrets.compare_digest(actual_hash, expected_hash)


def secure_cache_key_generator(
    data: Union[str, bytes], prefix: str = "", length: int = 16
) -> str:
    """
    レガシーコード互換のキャッシュキー生成関数

    Args:
        data: ハッシュ化するデータ
        prefix: キーのプレフィックス
        length: ハッシュの長さ

    Returns:
        セキュアなキャッシュキー
    """
    return SecureHashUtils.generate_cache_key(data, prefix, length)


def replace_md5_hash(data: Union[str, bytes], length: Optional[int] = None) -> str:
    """
    MD5ハッシュの安全な代替

    Args:
        data: ハッシュ化するデータ
        length: ハッシュの長さ（None で完全長）

    Returns:
        SHA256ハッシュ（非セキュリティ用途）
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    hash_digest = hashlib.sha256(data, usedforsecurity=False).hexdigest()

    if length is not None:
        hash_digest = hash_digest[:length]

    return hash_digest
