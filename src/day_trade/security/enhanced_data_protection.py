"""
強化データ保護・暗号化・秘密管理システム
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

AES-256-GCM暗号化、TOTP多要素認証、HashiCorp Vault風の秘密管理機能。
データの暗号化（data at rest）・転送時暗号化（data in transit）の実装。
"""

import base64
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import pyotp
    import qrcode
    from PIL import Image

    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False


class EncryptionAlgorithm(Enum):
    """暗号化アルゴリズム"""

    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class SecretType(Enum):
    """秘密の種類"""

    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_TOKEN = "oauth_token"
    PRIVATE_KEY = "private_key"
    CERTIFICATE = "certificate"
    GENERIC_SECRET = "generic_secret"


@dataclass
class EncryptedData:
    """暗号化データ"""

    algorithm: EncryptionAlgorithm
    ciphertext: bytes
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecretEntry:
    """秘密エントリ"""

    id: str
    name: str
    secret_type: SecretType
    encrypted_value: EncryptedData
    description: str = ""
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: Optional[datetime] = None
    access_count: int = 0


class AESGCMEncryption:
    """AES-256-GCM暗号化サービス"""

    def __init__(self, key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptographyライブラリが必要です")

        if key is None:
            # 256ビット（32バイト）のランダムキーを生成
            key = secrets.token_bytes(32)
        elif len(key) != 32:
            raise ValueError("AES-256には32バイトのキーが必要です")

        self.aesgcm = AESGCM(key)
        self.key = key

    def encrypt(self, plaintext: Union[str, bytes], additional_data: bytes = None) -> EncryptedData:
        """データを暗号化"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        # 96ビット（12バイト）のnonceを生成
        nonce = secrets.token_bytes(12)

        # 暗号化実行
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, additional_data)

        return EncryptedData(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            ciphertext=ciphertext,
            nonce=nonce,
        )

    def decrypt(self, encrypted_data: EncryptedData, additional_data: bytes = None) -> bytes:
        """データを復号化"""
        if encrypted_data.algorithm != EncryptionAlgorithm.AES_256_GCM:
            raise ValueError("AES-GCMで暗号化されたデータではありません")

        if not encrypted_data.nonce:
            raise ValueError("nonceが見つかりません")

        # 復号化実行
        plaintext = self.aesgcm.decrypt(
            encrypted_data.nonce, encrypted_data.ciphertext, additional_data
        )

        return plaintext

    @classmethod
    def derive_key_from_password(
        cls, password: str, salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """パスワードから暗号化キーを導出"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptographyライブラリが必要です")

        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # 適切なイテレーション数
        )

        key = kdf.derive(password.encode("utf-8"))
        return key, salt


class FernetEncryption:
    """Fernet暗号化サービス（対称暗号）"""

    def __init__(self, key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptographyライブラリが必要です")

        if key is None:
            key = Fernet.generate_key()

        self.fernet = Fernet(key)
        self.key = key

    def encrypt(self, plaintext: Union[str, bytes]) -> EncryptedData:
        """データを暗号化"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        ciphertext = self.fernet.encrypt(plaintext)

        return EncryptedData(algorithm=EncryptionAlgorithm.FERNET, ciphertext=ciphertext)

    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """データを復号化"""
        if encrypted_data.algorithm != EncryptionAlgorithm.FERNET:
            raise ValueError("Fernetで暗号化されたデータではありません")

        plaintext = self.fernet.decrypt(encrypted_data.ciphertext)
        return plaintext


class TOTPManager:
    """TOTP（Time-based One-time Password）多要素認証管理"""

    def __init__(self):
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp, qrcodeライブラリが必要です")

    def generate_secret(self) -> str:
        """TOTPシークレットキーを生成"""
        return pyotp.random_base32()

    def generate_qr_code(self, secret: str, name: str, issuer: str = "DayTrade") -> bytes:
        """QRコードを生成"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=name, issuer_name=issuer)

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # PNGバイトを返す
        from io import BytesIO

        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

    def verify_token(self, secret: str, token: str, valid_window: int = 1) -> bool:
        """TOTPトークンを検証"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=valid_window)

    def get_current_token(self, secret: str) -> str:
        """現在のTOTPトークンを取得"""
        totp = pyotp.TOTP(secret)
        return totp.now()


class SecretManager:
    """秘密管理システム（HashiCorp Vault風）"""

    def __init__(self, master_key: Optional[str] = None, db_path: str = "secrets.db"):
        self.db_path = db_path

        # マスターキーから暗号化サービスを初期化
        if master_key:
            derived_key, salt = AESGCMEncryption.derive_key_from_password(master_key)
            self.encryption_service = AESGCMEncryption(derived_key)
            self.master_key_salt = salt
        else:
            # 開発環境用：環境変数またはランダムキー
            env_key = os.getenv("SECRET_MANAGER_KEY")
            if env_key:
                derived_key, salt = AESGCMEncryption.derive_key_from_password(env_key)
                self.encryption_service = AESGCMEncryption(derived_key)
                self.master_key_salt = salt
            else:
                self.encryption_service = AESGCMEncryption()
                self.master_key_salt = None

        self.totp_manager = TOTPManager() if TOTP_AVAILABLE else None
        self._initialize_database()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS secrets (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    secret_type TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    ciphertext BLOB NOT NULL,
                    nonce BLOB,
                    salt BLOB,
                    metadata TEXT,
                    description TEXT,
                    tags TEXT,
                    expires_at DATETIME,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    accessed_at DATETIME,
                    access_count INTEGER DEFAULT 0
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    secret_id TEXT NOT NULL,
                    secret_name TEXT NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    access_type TEXT NOT NULL,  -- read, create, update, delete
                    client_info TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT
                )
            """
            )

            # インデックス作成
            conn.execute("CREATE INDEX IF NOT EXISTS idx_secrets_name ON secrets(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_secrets_type ON secrets(secret_type)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_logs_secret ON access_logs(secret_id)"
            )

            conn.commit()

    def store_secret(
        self,
        name: str,
        value: Union[str, bytes],
        secret_type: SecretType = SecretType.GENERIC_SECRET,
        description: str = "",
        tags: List[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> str:
        """秘密を保存"""
        if tags is None:
            tags = []

        try:
            # 暗号化
            encrypted_data = self.encryption_service.encrypt(value)

            secret_id = f"secret_{secrets.token_hex(8)}"
            current_time = datetime.utcnow()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO secrets
                    (id, name, secret_type, algorithm, ciphertext, nonce, salt,
                     metadata, description, tags, expires_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        secret_id,
                        name,
                        secret_type.value,
                        encrypted_data.algorithm.value,
                        encrypted_data.ciphertext,
                        encrypted_data.nonce,
                        encrypted_data.salt,
                        json.dumps(encrypted_data.metadata),
                        description,
                        json.dumps(tags),
                        expires_at.isoformat() if expires_at else None,
                        current_time.isoformat(),
                        current_time.isoformat(),
                    ),
                )
                conn.commit()

            # アクセスログ記録
            self._log_access(secret_id, name, "create", True)

            return secret_id

        except Exception as e:
            self._log_access("", name, "create", False, str(e))
            raise

    def retrieve_secret(self, name: str) -> Optional[str]:
        """秘密を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, algorithm, ciphertext, nonce, salt, metadata, expires_at
                    FROM secrets
                    WHERE name = ?
                """,
                    (name,),
                )

                row = cursor.fetchone()
                if not row:
                    self._log_access("", name, "read", False, "Secret not found")
                    return None

                (
                    secret_id,
                    algorithm,
                    ciphertext,
                    nonce,
                    salt,
                    metadata_json,
                    expires_at,
                ) = row

                # 有効期限チェック
                if expires_at:
                    expires_datetime = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expires_datetime:
                        self._log_access(secret_id, name, "read", False, "Secret expired")
                        return None

                # 復号化データの構築
                metadata = json.loads(metadata_json) if metadata_json else {}
                encrypted_data = EncryptedData(
                    algorithm=EncryptionAlgorithm(algorithm),
                    ciphertext=ciphertext,
                    nonce=nonce,
                    salt=salt,
                    metadata=metadata,
                )

                # 復号化
                plaintext = self.encryption_service.decrypt(encrypted_data)

                # アクセス統計更新
                conn.execute(
                    """
                    UPDATE secrets
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE id = ?
                """,
                    (datetime.utcnow().isoformat(), secret_id),
                )
                conn.commit()

                # アクセスログ記録
                self._log_access(secret_id, name, "read", True)

                return plaintext.decode("utf-8")

        except Exception as e:
            self._log_access("", name, "read", False, str(e))
            raise

    def delete_secret(self, name: str) -> bool:
        """秘密を削除"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id FROM secrets WHERE name = ?", (name,))
                row = cursor.fetchone()

                if not row:
                    self._log_access("", name, "delete", False, "Secret not found")
                    return False

                secret_id = row[0]

                conn.execute("DELETE FROM secrets WHERE name = ?", (name,))
                conn.commit()

                # アクセスログ記録
                self._log_access(secret_id, name, "delete", True)

                return True

        except Exception as e:
            self._log_access("", name, "delete", False, str(e))
            raise

    def list_secrets(self) -> List[Dict[str, Any]]:
        """秘密一覧を取得（値は含まず）"""
        secrets_list = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, name, secret_type, description, tags, expires_at,
                       created_at, updated_at, accessed_at, access_count
                FROM secrets
                ORDER BY created_at DESC
            """
            )

            for row in cursor.fetchall():
                secret_info = {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "tags": json.loads(row[4]) if row[4] else [],
                    "expires_at": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                    "accessed_at": row[8],
                    "access_count": row[9],
                }
                secrets_list.append(secret_info)

        return secrets_list

    def _log_access(
        self,
        secret_id: str,
        secret_name: str,
        access_type: str,
        success: bool,
        error_message: str = None,
    ):
        """アクセスログを記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO access_logs
                    (secret_id, secret_name, accessed_at, access_type, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        secret_id,
                        secret_name,
                        datetime.utcnow().isoformat(),
                        access_type,
                        success,
                        error_message,
                    ),
                )
                conn.commit()
        except:
            # ログ記録の失敗は本処理に影響させない
            pass

    def get_access_audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        """アクセス監査ログを取得"""
        logs = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT secret_id, secret_name, accessed_at, access_type,
                       success, error_message
                FROM access_logs
                ORDER BY accessed_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            for row in cursor.fetchall():
                log_entry = {
                    "secret_id": row[0],
                    "secret_name": row[1],
                    "accessed_at": row[2],
                    "access_type": row[3],
                    "success": bool(row[4]),
                    "error_message": row[5],
                }
                logs.append(log_entry)

        return logs

    def rotate_key(self, new_master_key: str):
        """暗号化キーのローテーション"""
        # 既存の秘密をすべて取得・復号化
        secrets_data = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, name, secret_type, algorithm, ciphertext, nonce, salt,
                       metadata, description, tags, expires_at, created_at, access_count
                FROM secrets
            """
            )

            for row in cursor.fetchall():
                (
                    secret_id,
                    name,
                    secret_type,
                    algorithm,
                    ciphertext,
                    nonce,
                    salt,
                    metadata_json,
                    description,
                    tags,
                    expires_at,
                    created_at,
                    access_count,
                ) = row

                try:
                    # 現在のキーで復号化
                    encrypted_data = EncryptedData(
                        algorithm=EncryptionAlgorithm(algorithm),
                        ciphertext=ciphertext,
                        nonce=nonce,
                        salt=salt,
                        metadata=json.loads(metadata_json) if metadata_json else {},
                    )

                    plaintext = self.encryption_service.decrypt(encrypted_data)

                    secrets_data.append(
                        {
                            "id": secret_id,
                            "name": name,
                            "type": SecretType(secret_type),
                            "value": plaintext,
                            "description": description,
                            "tags": json.loads(tags) if tags else [],
                            "expires_at": (
                                datetime.fromisoformat(expires_at) if expires_at else None
                            ),
                            "created_at": datetime.fromisoformat(created_at),
                            "access_count": access_count,
                        }
                    )

                except Exception as e:
                    print(f"秘密の復号化に失敗: {name} - {e}")

        # 新しい暗号化サービスを初期化
        new_key, new_salt = AESGCMEncryption.derive_key_from_password(new_master_key)
        new_encryption_service = AESGCMEncryption(new_key)

        # すべての秘密を新しいキーで再暗号化
        with sqlite3.connect(self.db_path) as conn:
            for secret_data in secrets_data:
                try:
                    new_encrypted = new_encryption_service.encrypt(secret_data["value"])

                    conn.execute(
                        """
                        UPDATE secrets
                        SET algorithm = ?, ciphertext = ?, nonce = ?, salt = ?,
                            metadata = ?, updated_at = ?
                        WHERE id = ?
                    """,
                        (
                            new_encrypted.algorithm.value,
                            new_encrypted.ciphertext,
                            new_encrypted.nonce,
                            new_encrypted.salt,
                            json.dumps(new_encrypted.metadata),
                            datetime.utcnow().isoformat(),
                            secret_data["id"],
                        ),
                    )

                except Exception as e:
                    print(f"秘密の再暗号化に失敗: {secret_data['name']} - {e}")

            conn.commit()

        # 新しい暗号化サービスを適用
        self.encryption_service = new_encryption_service
        self.master_key_salt = new_salt


class DataProtectionManager:
    """統合データ保護管理システム"""

    def __init__(self, master_key: Optional[str] = None):
        self.secret_manager = SecretManager(master_key)
        self.totp_manager = TOTPManager() if TOTP_AVAILABLE else None

        # デフォルト暗号化サービス
        self.default_encryption = AESGCMEncryption()

    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """機密データの暗号化"""
        encrypted = self.default_encryption.encrypt(data)

        # Base64エンコードで文字列として保存可能な形式に
        encoded_data = {
            "algorithm": encrypted.algorithm.value,
            "ciphertext": base64.b64encode(encrypted.ciphertext).decode("ascii"),
            "nonce": base64.b64encode(encrypted.nonce).decode("ascii") if encrypted.nonce else None,
            "created_at": encrypted.created_at.isoformat(),
        }

        return base64.b64encode(json.dumps(encoded_data).encode("utf-8")).decode("ascii")

    def decrypt_sensitive_data(self, encrypted_string: str) -> str:
        """機密データの復号化"""
        try:
            # Base64デコードしてJSONパース
            json_data = json.loads(base64.b64decode(encrypted_string).decode("utf-8"))

            encrypted_data = EncryptedData(
                algorithm=EncryptionAlgorithm(json_data["algorithm"]),
                ciphertext=base64.b64decode(json_data["ciphertext"]),
                nonce=base64.b64decode(json_data["nonce"]) if json_data["nonce"] else None,
            )

            plaintext = self.default_encryption.decrypt(encrypted_data)
            return plaintext.decode("utf-8")

        except Exception as e:
            raise ValueError(f"復号化に失敗しました: {e}")

    def setup_2fa(self, user_name: str) -> Dict[str, Any]:
        """2要素認証のセットアップ"""
        if not self.totp_manager:
            raise ImportError("TOTP機能が利用できません")

        # TOTPシークレットを生成
        secret = self.totp_manager.generate_secret()

        # QRコードを生成
        qr_code_bytes = self.totp_manager.generate_qr_code(secret, user_name)

        # シークレットを安全に保存
        secret_name = f"totp_secret_{user_name}"
        self.secret_manager.store_secret(
            name=secret_name,
            value=secret,
            secret_type=SecretType.ENCRYPTION_KEY,
            description=f"TOTP secret for user {user_name}",
        )

        return {
            "secret": secret,
            "qr_code": base64.b64encode(qr_code_bytes).decode("ascii"),
            "backup_codes": [secrets.token_hex(4) for _ in range(10)],  # バックアップコード生成
        }

    def verify_2fa_token(self, user_name: str, token: str) -> bool:
        """2要素認証トークンの検証"""
        if not self.totp_manager:
            return False

        try:
            secret_name = f"totp_secret_{user_name}"
            secret = self.secret_manager.retrieve_secret(secret_name)

            if not secret:
                return False

            return self.totp_manager.verify_token(secret, token)

        except Exception:
            return False

    def get_security_metrics(self) -> Dict[str, Any]:
        """セキュリティメトリクスを取得"""
        secrets_list = self.secret_manager.list_secrets()
        access_logs = self.secret_manager.get_access_audit(limit=100)

        # 統計計算
        total_secrets = len(secrets_list)
        secrets_by_type = {}
        expired_secrets = 0

        for secret in secrets_list:
            secret_type = secret["type"]
            secrets_by_type[secret_type] = secrets_by_type.get(secret_type, 0) + 1

            if secret["expires_at"]:
                expires_at = datetime.fromisoformat(secret["expires_at"])
                if datetime.utcnow() > expires_at:
                    expired_secrets += 1

        # アクセス統計
        successful_accesses = len([log for log in access_logs if log["success"]])
        failed_accesses = len([log for log in access_logs if not log["success"]])

        return {
            "secrets_summary": {
                "total": total_secrets,
                "by_type": secrets_by_type,
                "expired": expired_secrets,
            },
            "access_summary": {
                "successful": successful_accesses,
                "failed": failed_accesses,
                "total": len(access_logs),
            },
            "security_features": {
                "encryption_algorithm": "AES-256-GCM",
                "totp_available": TOTP_AVAILABLE,
                "audit_logging": True,
                "key_rotation_supported": True,
            },
            "last_updated": datetime.utcnow().isoformat(),
        }


# グローバルインスタンス
_data_protection_manager = None


def get_data_protection_manager() -> DataProtectionManager:
    """グローバルデータ保護管理を取得"""
    global _data_protection_manager
    if _data_protection_manager is None:
        master_key = os.getenv("DATA_PROTECTION_MASTER_KEY")
        _data_protection_manager = DataProtectionManager(master_key)
    return _data_protection_manager
