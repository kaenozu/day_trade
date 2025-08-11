#!/usr/bin/env python3
"""
セキュア外部APIクライアント
Issue #395: 外部APIクライアントのセキュリティ強化

API キー管理・URL 構築・エラー詳細の改善
- エンタープライズグレードのAPIキー管理
- 完全なURL構築セキュリティ
- 機密情報漏洩防止エラーハンドリング
"""

import asyncio
import hashlib
import hmac
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientError, ClientTimeout

from ..utils.logging_config import get_context_logger
from ..utils.security_helpers import SecurityHelpers

logger = get_context_logger(__name__)


class SecurityLevel(Enum):
    """セキュリティレベル定義"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class APIKeyType(Enum):
    """APIキータイプ"""

    HEADER = "header"
    QUERY_PARAM = "query_param"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    HMAC_SIGNATURE = "hmac_signature"


@dataclass
class SecureAPIKey:
    """セキュアAPIキー管理"""

    key_id: str
    key_type: APIKeyType
    encrypted_key: bytes
    encryption_salt: bytes
    creation_time: datetime
    expiry_time: Optional[datetime] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    allowed_hosts: List[str] = field(default_factory=list)
    rate_limit_per_hour: int = 1000

    def is_expired(self) -> bool:
        """キーの有効期限チェック"""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time

    def can_use(self, host: str) -> bool:
        """使用可能性チェック"""
        if self.is_expired():
            return False
        return not (self.allowed_hosts and host not in self.allowed_hosts)


@dataclass
class URLSecurityPolicy:
    """URL構築セキュリティポリシー"""

    allowed_schemes: List[str] = field(default_factory=lambda: ["https"])
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(
        default_factory=lambda: ["../", "..\\", "%2e%2e", ".git", ".env"]
    )
    max_url_length: int = 2048
    max_param_length: int = 1024
    allow_unicode: bool = False
    strict_encoding: bool = True


class SecureAPIKeyManager:
    """エンタープライズグレードAPIキー管理システム"""

    def __init__(self, master_key: Optional[bytes] = None):
        """
        初期化

        Args:
            master_key: マスター暗号化キー（環境変数から取得推奨）
        """
        self.master_key = master_key or self._generate_master_key()
        self.api_keys: Dict[str, SecureAPIKey] = {}
        self.key_rotation_log: List[Dict] = []

    def _generate_master_key(self) -> bytes:
        """マスターキー生成"""
        import os

        key = os.environ.get("API_MASTER_KEY")
        if key:
            return key.encode()

        # 新しいキーを生成（本番環境では環境変数に保存）
        new_key = SecurityHelpers.generate_secure_random_string(64)
        logger.warning(
            "新しいマスターキーを生成しました。環境変数に保存してください: API_MASTER_KEY=[GENERATED_KEY]"
        )
        return new_key.encode()

    def add_api_key(
        self,
        key_id: str,
        plain_key: str,
        key_type: APIKeyType,
        allowed_hosts: Optional[List[str]] = None,
        expiry_hours: Optional[int] = None,
    ) -> bool:
        """
        APIキーの安全な追加

        Args:
            key_id: キーの識別子
            plain_key: 平文のAPIキー
            key_type: キーのタイプ
            allowed_hosts: 許可されたホストリスト
            expiry_hours: 有効期限（時間）

        Returns:
            bool: 追加成功フラグ
        """
        try:
            # セキュリティ検証
            if not self._validate_key_format(plain_key, key_type):
                logger.error(f"APIキーフォーマットが無効: {key_id}")
                return False

            # 暗号化
            salt = secrets.token_bytes(32)  # セキュアなソルト生成
            encrypted_key = self._encrypt_api_key(plain_key, salt)

            # 有効期限設定
            expiry_time = None
            if expiry_hours:
                expiry_time = datetime.now() + timedelta(hours=expiry_hours)

            # キー作成
            secure_key = SecureAPIKey(
                key_id=key_id,
                key_type=key_type,
                encrypted_key=encrypted_key,
                encryption_salt=salt,
                creation_time=datetime.now(),
                expiry_time=expiry_time,
                allowed_hosts=allowed_hosts or [],
            )

            self.api_keys[key_id] = secure_key

            # ログ記録
            self.key_rotation_log.append(
                {
                    "action": "key_added",
                    "key_id": key_id,
                    "timestamp": datetime.now().isoformat(),
                    "key_type": key_type.value,
                }
            )

            logger.info(f"APIキー安全に追加完了: {key_id}")
            return True

        except Exception as e:
            logger.error(f"APIキー追加エラー: {e}")
            return False

    def get_api_key(self, key_id: str, host: str) -> Optional[str]:
        """
        APIキーの安全な取得

        Args:
            key_id: キー識別子
            host: 接続先ホスト

        Returns:
            Optional[str]: 復号化されたAPIキー
        """
        try:
            if key_id not in self.api_keys:
                logger.warning(f"未登録のAPIキー要求: {key_id}")
                return None

            secure_key = self.api_keys[key_id]

            # 使用可能性チェック
            if not secure_key.can_use(host):
                logger.warning(f"APIキー使用不可: {key_id} for {host}")
                return None

            # 使用統計更新
            secure_key.usage_count += 1
            secure_key.last_used = datetime.now()

            # 復号化
            plain_key = self._decrypt_api_key(
                secure_key.encrypted_key, secure_key.encryption_salt
            )

            logger.debug(f"APIキー取得成功: {key_id}")
            return plain_key

        except Exception as e:
            logger.error(f"APIキー取得エラー: {e}")
            return None

    def rotate_key(self, key_id: str, new_plain_key: str) -> bool:
        """APIキーローテーション"""
        if key_id not in self.api_keys:
            return False

        old_key = self.api_keys[key_id]

        # 新しいキーで置換
        salt = secrets.token_bytes(32)  # セキュアなソルト生成
        encrypted_key = self._encrypt_api_key(new_plain_key, salt)

        old_key.encrypted_key = encrypted_key
        old_key.encryption_salt = salt
        old_key.creation_time = datetime.now()
        old_key.usage_count = 0

        # ローテーションログ
        self.key_rotation_log.append(
            {
                "action": "key_rotated",
                "key_id": key_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"APIキーローテーション完了: {key_id}")
        return True

    def _validate_key_format(self, key: str, key_type: APIKeyType) -> bool:
        """APIキーフォーマット検証"""
        if not key or len(key.strip()) == 0:
            return False

        # 基本的な長さチェック
        if len(key) < 8:
            logger.warning("APIキーが短すぎます")
            return False

        if len(key) > 512:
            logger.warning("APIキーが長すぎます")
            return False

        # タイプ別検証
        if key_type == APIKeyType.BEARER_TOKEN:
            # Bearer tokenの形式チェック
            return re.match(r"^[A-Za-z0-9\-._~+/]+=*$", key) is not None
        elif key_type == APIKeyType.OAUTH2:
            # OAuth2 tokenの基本チェック
            return len(key) >= 16 and re.match(r"^[A-Za-z0-9\-._~]+$", key) is not None

        # デフォルト: 基本的なAPIキー文字セット
        return re.match(r"^[A-Za-z0-9\-._~+=]+$", key) is not None

    def _encrypt_api_key(self, plain_key: str, salt: bytes) -> bytes:
        """APIキー暗号化"""
        try:
            import base64

            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            # パスワードベース派生キー
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            cipher = Fernet(key)

            return cipher.encrypt(plain_key.encode())

        except ImportError:
            # フォールバック: より基本的な暗号化（本番では推奨しない）
            logger.warning("cryptography ライブラリが利用不可。基本的な暗号化を使用")
            return self._basic_encrypt(plain_key, salt)

    def _decrypt_api_key(self, encrypted_key: bytes, salt: bytes) -> str:
        """APIキー復号化"""
        try:
            import base64

            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            cipher = Fernet(key)

            return cipher.decrypt(encrypted_key).decode()

        except ImportError:
            # フォールバック復号化
            return self._basic_decrypt(encrypted_key, salt)

    def _basic_encrypt(self, plain_key: str, salt: bytes) -> bytes:
        """基本暗号化（フォールバック）"""
        import base64

        # XORベースの基本暗号化（デモ用）
        key_bytes = plain_key.encode()
        encrypted = bytes(
            a ^ b for a, b in zip(key_bytes, salt * (len(key_bytes) // len(salt) + 1))
        )
        return base64.b64encode(encrypted)

    def _basic_decrypt(self, encrypted_key: bytes, salt: bytes) -> str:
        """基本復号化（フォールバック）"""
        import base64

        key_bytes = base64.b64decode(encrypted_key)
        decrypted = bytes(
            a ^ b for a, b in zip(key_bytes, salt * (len(key_bytes) // len(salt) + 1))
        )
        return decrypted.decode()


class SecureURLBuilder:
    """セキュアURL構築システム"""

    def __init__(self, security_policy: Optional[URLSecurityPolicy] = None):
        self.policy = security_policy or URLSecurityPolicy()

    def build_secure_url(
        self,
        base_url: str,
        params: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        セキュアURL構築

        Args:
            base_url: ベースURL
            params: クエリパラメータ
            path_params: パスパラメータ（{key}形式の置換）

        Returns:
            str: セキュアに構築されたURL

        Raises:
            ValueError: セキュリティポリシー違反時
        """
        try:
            # ベースURLの検証
            parsed_base = urllib.parse.urlparse(base_url)

            if not self._validate_scheme(parsed_base.scheme):
                raise ValueError(f"許可されていないスキーム: {parsed_base.scheme}")

            if not self._validate_host(parsed_base.hostname):
                raise ValueError(f"許可されていないホスト: {parsed_base.hostname}")

            # パスパラメータの安全な置換
            url = base_url
            if path_params:
                for key, value in path_params.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in url:
                        safe_value = self._sanitize_path_parameter(value)
                        url = url.replace(placeholder, safe_value)

            # クエリパラメータの安全な追加
            if params:
                safe_params = {}
                for key, value in params.items():
                    safe_key = self._sanitize_query_key(key)
                    safe_value = self._sanitize_query_value(str(value))
                    safe_params[safe_key] = safe_value

                # URLエンコーディング
                encoded_params = urllib.parse.urlencode(safe_params, safe="")
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{encoded_params}"

            # 最終URL検証
            self._validate_final_url(url)

            return url

        except Exception as e:
            logger.error(f"セキュアURL構築エラー: {e}")
            raise ValueError("URL構築に失敗しました: セキュリティポリシー違反")

    def _validate_scheme(self, scheme: str) -> bool:
        """スキーム検証"""
        return scheme.lower() in [s.lower() for s in self.policy.allowed_schemes]

    def _validate_host(self, hostname: Optional[str]) -> bool:
        """ホスト検証"""
        if not hostname:
            return False

        # 許可リストが空の場合は全て許可
        if not self.policy.allowed_hosts:
            return True

        # 完全一致またはサブドメイン一致
        for allowed_host in self.policy.allowed_hosts:
            if hostname == allowed_host:
                return True
            if allowed_host.startswith(".") and hostname.endswith(allowed_host):
                return True

        return False

    def _sanitize_path_parameter(self, value: str) -> str:
        """パスパラメータのサニタイゼーション"""
        # 長さチェック
        if len(value) > self.policy.max_param_length:
            raise ValueError(
                f"パスパラメータが長すぎます: {len(value)} > {self.policy.max_param_length}"
            )

        # 危険パターンチェック
        for blocked_path in self.policy.blocked_paths:
            if blocked_path in value.lower():
                raise ValueError(f"ブロックされたパスパターンを検出: {blocked_path}")

        # 危険文字チェック
        dangerous_chars = ["<", ">", '"', "'", "&", "\0", "\n", "\r"]
        for char in dangerous_chars:
            if char in value:
                raise ValueError(f"危険な文字を検出: {char}")

        # URLエンコーディング
        encoded = urllib.parse.quote(value, safe="")

        return encoded

    def _sanitize_query_key(self, key: str) -> str:
        """クエリキーのサニタイゼーション"""
        # 基本的なキー検証
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", key):
            raise ValueError(f"無効なクエリキー: {key}")

        if len(key) > 128:
            raise ValueError(f"クエリキーが長すぎます: {len(key)}")

        return key

    def _sanitize_query_value(self, value: str) -> str:
        """クエリ値のサニタイゼーション"""
        # 長さチェック
        if len(value) > self.policy.max_param_length:
            raise ValueError(f"クエリ値が長すぎます: {len(value)}")

        # Unicode文字チェック
        if not self.policy.allow_unicode:
            if any(ord(c) > 127 for c in value):
                raise ValueError("Unicode文字は許可されていません")

        # 危険文字除去
        dangerous_chars = ["\0", "\n", "\r"]
        for char in dangerous_chars:
            if char in value:
                raise ValueError(f"危険な制御文字を検出: {repr(char)}")

        return value

    def _validate_final_url(self, url: str) -> None:
        """最終URL検証"""
        if len(url) > self.policy.max_url_length:
            raise ValueError(
                f"URL長が制限を超過: {len(url)} > {self.policy.max_url_length}"
            )

        # 再パース検証
        parsed = urllib.parse.urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            raise ValueError("URLフォーマットが無効です")


class SecureErrorHandler:
    """機密情報漏洩防止エラーハンドラー"""

    @staticmethod
    def sanitize_error_message(error: Exception, context: str = "") -> str:
        """
        エラーメッセージの機密情報除去

        Args:
            error: 元の例外
            context: エラー文脈

        Returns:
            str: サニタイズ済みエラーメッセージ
        """
        original_msg = str(error)

        # 機密情報パターン
        sensitive_patterns = [
            # APIキー・トークン
            r"api[_-]?key[:\s=]+[a-zA-Z0-9\-._~+=]+",
            r"token[:\s=]+[a-zA-Z0-9\-._~+=]+",
            r"secret[:\s=]+[a-zA-Z0-9\-._~+=]+",
            r"password[:\s=]+[^\s]+",
            # 認証ヘッダー
            r"authorization:\s*bearer\s+[a-zA-Z0-9\-._~+=]+",
            r"authorization:\s*basic\s+[a-zA-Z0-9+=]+",
            # URL内の機密情報
            r"https?://[^/]*:[^@]*@",  # URL内の認証情報
            # ファイルパス
            r"[a-zA-Z]:[\\\/][^\\\/\s]*[\\\/][^\\\/\s]*",  # Windows絶対パス
            r"\/[^\/\s]*\/[^\/\s]*\/[^\/\s]*",  # Unix絶対パス
            # IPアドレス・ホスト
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            # その他の機密情報パターン
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # メール
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",  # UUID
        ]

        # パターンマッチングによる除去
        sanitized_msg = original_msg
        for pattern in sensitive_patterns:
            sanitized_msg = re.sub(
                pattern, "[REDACTED]", sanitized_msg, flags=re.IGNORECASE
            )

        # 基本的なエラータイプ分類
        error_type = type(error).__name__

        # コンテキスト付きの安全なメッセージ
        if context:
            safe_msg = f"{context}: {error_type} が発生しました"
        else:
            safe_msg = f"{error_type} が発生しました"

        # 元のメッセージが安全と判断されれば含める
        if len(sanitized_msg) < len(original_msg):
            # 機密情報が除去された場合は汎用メッセージ
            logger.warning(f"エラーメッセージから機密情報を除去: {error_type}")
            return f"{safe_msg}（詳細はシステムログを確認してください）"
        else:
            # 安全と判断される場合は詳細を含める
            return f"{safe_msg}: {sanitized_msg}"

    @staticmethod
    def create_safe_error_response(
        error: Exception, request_id: str = None
    ) -> Dict[str, Any]:
        """
        安全なエラーレスポンス生成

        Args:
            error: 元の例外
            request_id: リクエストID

        Returns:
            Dict: 機密情報が除去されたエラーレスポンス
        """
        return {
            "error": True,
            "error_type": type(error).__name__,
            "message": SecureErrorHandler.sanitize_error_message(error),
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            # デバッグ情報は本番環境では除去される
            "debug_info": None,  # ローカル開発時のみ詳細情報を含める設定
        }


# 使用例とテスト用のスタンドアロン関数
def create_secure_api_client_example():
    """セキュアAPIクライアント使用例"""

    # APIキーマネージャー初期化
    key_manager = SecureAPIKeyManager()

    # APIキー追加
    success = key_manager.add_api_key(
        key_id="alpha_vantage",
        plain_key="[DEMO_API_KEY]",
        key_type=APIKeyType.QUERY_PARAM,
        allowed_hosts=["www.alphavantage.co"],
        expiry_hours=24 * 30,  # 30日間有効
    )

    if success:
        logger.info("APIキー登録成功")

    # セキュアURL構築
    url_builder = SecureURLBuilder(
        URLSecurityPolicy(
            allowed_schemes=["https"],
            allowed_hosts=[".alphavantage.co", ".yahoo.com"],
            max_url_length=2048,
        )
    )

    try:
        secure_url = url_builder.build_secure_url(
            base_url="https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": "IBM",
                "apikey": "[DEMO_KEY]",
            },
        )
        logger.info(f"セキュアURL構築成功: {secure_url}")

    except ValueError as e:
        logger.error(f"URL構築失敗: {e}")

    # エラーハンドリング例
    try:
        # 意図的なエラー発生
        raise ConnectionError(
            "Connection failed with api_key=[REDACTED] to host [REDACTED]"
        )
    except ConnectionError as e:
        safe_error = SecureErrorHandler.sanitize_error_message(e, "API接続")
        logger.info(f"サニタイズ済みエラー: {safe_error}")


if __name__ == "__main__":
    create_secure_api_client_example()
