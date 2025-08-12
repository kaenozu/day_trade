#!/usr/bin/env python3
"""
強化データ暗号化システム
Issue #419対応 - データ保護強化
"""

import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class EncryptionMetadata:
    """暗号化メタデータ"""

    algorithm: str
    key_id: str
    created_at: datetime
    encrypted_by: str
    version: str = "1.0"


class SecureKeyManager:
    """セキュアキー管理システム"""

    def __init__(self, key_storage_path: Path = Path("security/keys")):
        self.key_storage_path = key_storage_path
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, bytes] = {}

    def generate_key(self, key_id: str = None) -> str:
        """新しい暗号化キー生成"""
        if key_id is None:
            key_id = f"key_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"

        # Fernet用のキー生成
        key = Fernet.generate_key()
        self._keys[key_id] = key

        # キーファイルに保存（実運用では環境変数やVault使用推奨）
        key_file = self.key_storage_path / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(key)

        # ファイル権限設定（Unix系のみ）
        try:
            os.chmod(key_file, 0o600)
        except (OSError, AttributeError):
            pass  # Windowsでは無視

        logger.info(f"暗号化キー生成: {key_id}")
        return key_id

    def get_key(self, key_id: str) -> bytes:
        """キー取得"""
        if key_id in self._keys:
            return self._keys[key_id]

        # ファイルからキー読み込み
        key_file = self.key_storage_path / f"{key_id}.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
                self._keys[key_id] = key
                return key

        raise ValueError(f"暗号化キーが見つかりません: {key_id}")

    def derive_key_from_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """パスワードからキー導出"""
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt


class EnhancedDataEncryption:
    """強化データ暗号化システム"""

    def __init__(self, key_manager: SecureKeyManager = None):
        self.key_manager = key_manager or SecureKeyManager()
        self.default_key_id = None

    def initialize_default_key(self) -> str:
        """デフォルトキー初期化"""
        self.default_key_id = self.key_manager.generate_key("default_encryption_key")
        return self.default_key_id

    def encrypt_data(
        self,
        data: Union[str, dict, list],
        key_id: str = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """データ暗号化"""
        if key_id is None:
            if self.default_key_id is None:
                self.initialize_default_key()
            key_id = self.default_key_id

        # データをJSON文字列に変換
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, ensure_ascii=False)
        else:
            data_str = str(data)

        # 暗号化実行
        key = self.key_manager.get_key(key_id)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data_str.encode("utf-8"))

        result = {
            "encrypted_data": base64.b64encode(encrypted_data).decode("ascii"),
            "key_id": key_id,
        }

        if include_metadata:
            metadata = EncryptionMetadata(
                algorithm="Fernet",
                key_id=key_id,
                created_at=datetime.now(timezone.utc),
                encrypted_by="day_trade_security_system",
            )
            result["metadata"] = {
                "algorithm": metadata.algorithm,
                "key_id": metadata.key_id,
                "created_at": metadata.created_at.isoformat(),
                "encrypted_by": metadata.encrypted_by,
                "version": metadata.version,
            }

        logger.debug(f"データ暗号化完了: key_id={key_id}")
        return result

    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> Any:
        """データ復号化"""
        key_id = encrypted_data.get("key_id")
        if not key_id:
            raise ValueError("キーIDが指定されていません")

        encrypted_bytes = base64.b64decode(encrypted_data["encrypted_data"])

        key = self.key_manager.get_key(key_id)
        fernet = Fernet(key)

        try:
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode("utf-8")

            # JSONとして解析を試行
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logger.error(f"復号化エラー: {e}")
            raise ValueError(f"復号化に失敗しました: {e}")

    def encrypt_file(self, file_path: Path, output_path: Path = None) -> Path:
        """ファイル暗号化"""
        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + ".encrypted")

        with open(file_path, "rb") as f:
            file_data = f.read()

        # ファイルデータを暗号化
        encrypted_result = self.encrypt_data(base64.b64encode(file_data).decode("ascii"))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(encrypted_result, f, indent=2)

        logger.info(f"ファイル暗号化完了: {file_path} -> {output_path}")
        return output_path

    def decrypt_file(self, encrypted_file_path: Path, output_path: Path = None) -> Path:
        """ファイル復号化"""
        if output_path is None:
            output_path = encrypted_file_path.with_suffix("")
            if output_path.suffix == ".encrypted":
                output_path = output_path.with_suffix("")

        with open(encrypted_file_path, encoding="utf-8") as f:
            encrypted_data = json.load(f)

        # ファイルデータを復号化
        decrypted_data = self.decrypt_data(encrypted_data)
        file_data = base64.b64decode(decrypted_data)

        with open(output_path, "wb") as f:
            f.write(file_data)

        logger.info(f"ファイル復号化完了: {encrypted_file_path} -> {output_path}")
        return output_path


class SensitiveDataProtector:
    """機密データ保護システム"""

    def __init__(self, encryption_system: EnhancedDataEncryption = None):
        self.encryption = encryption_system or EnhancedDataEncryption()
        self.sensitive_keys = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "secret_key",
            "private_key",
            "token",
            "access_token",
            "refresh_token",
            "api_key",
            "apikey",
            "auth_key",
            "database_url",
            "db_url",
            "connection_string",
        }

    def protect_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定ファイルの機密データ保護"""
        protected_config = {}

        for key, value in config.items():
            if self._is_sensitive_key(key) and isinstance(value, str) and value.strip():
                # 機密データを暗号化
                encrypted_value = self.encryption.encrypt_data(value)
                protected_config[key] = {"_encrypted": True, **encrypted_value}
                logger.info(f"機密データ暗号化: {key}")
            elif isinstance(value, dict):
                # 再帰的に処理
                protected_config[key] = self.protect_config(value)
            else:
                protected_config[key] = value

        return protected_config

    def unprotect_config(self, protected_config: Dict[str, Any]) -> Dict[str, Any]:
        """暗号化設定の復号化"""
        config = {}

        for key, value in protected_config.items():
            if isinstance(value, dict) and value.get("_encrypted"):
                # 暗号化データを復号化
                encrypted_data = {k: v for k, v in value.items() if k != "_encrypted"}
                config[key] = self.encryption.decrypt_data(encrypted_data)
                logger.debug(f"機密データ復号化: {key}")
            elif isinstance(value, dict):
                config[key] = self.unprotect_config(value)
            else:
                config[key] = value

        return config

    def _is_sensitive_key(self, key: str) -> bool:
        """機密キーかどうか判定"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def scan_for_sensitive_data(self, data: Any, path: str = "root") -> List[str]:
        """機密データスキャン"""
        findings = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                if self._is_sensitive_key(key) and isinstance(value, str) and value.strip():
                    findings.append(f"機密データ検出: {current_path}")
                findings.extend(self.scan_for_sensitive_data(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                findings.extend(self.scan_for_sensitive_data(item, f"{path}[{i}]"))

        return findings


def demonstrate_encryption():
    """暗号化システムデモ"""
    print("=== データ暗号化システムデモ ===")

    # 暗号化システム初期化
    encryption = EnhancedDataEncryption()
    protector = SensitiveDataProtector(encryption)

    # テストデータ
    sensitive_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "password": "super_secret_password",
            "api_key": "test_api_key_12345",
        },
        "app": {"name": "day_trade", "debug": False},
    }

    print("\n1. 機密データスキャン:")
    findings = protector.scan_for_sensitive_data(sensitive_config)
    for finding in findings:
        print(f"  {finding}")

    print("\n2. 設定の暗号化:")
    protected_config = protector.protect_config(sensitive_config)
    print("  暗号化後の設定構造:")
    for key in protected_config:
        print(
            f"    {key}: {'[暗号化済み]' if isinstance(protected_config[key], dict) and any('_encrypted' in str(v) for v in protected_config[key].values()) else '[平文]'}"
        )

    print("\n3. 設定の復号化:")
    restored_config = protector.unprotect_config(protected_config)
    print("  復号化成功:", restored_config == sensitive_config)

    print("\n4. 直接データ暗号化テスト:")
    test_data = {"secret": "very_confidential_data", "user": "admin"}
    encrypted = encryption.encrypt_data(test_data)
    print(f"  暗号化データサイズ: {len(encrypted['encrypted_data'])}文字")

    decrypted = encryption.decrypt_data(encrypted)
    print(f"  復号化成功: {decrypted == test_data}")

    print("\n=== デモ完了 ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_encryption()
