#!/usr/bin/env python3
"""
データ保護・暗号化システム
Issue #419: セキュリティ強化 - データ保護の強化

データ暗号化、機密情報管理、セキュアストレージ、
アクセス制御を統合したデータ保護システム
"""

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 暗号化ライブラリ
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "cryptography library not available. Some features will be disabled."
    )

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class EncryptionAlgorithm(Enum):
    """暗号化アルゴリズム"""

    AES_256_GCM = "aes-256-gcm"
    FERNET = "fernet"
    RSA_2048 = "rsa-2048"
    CHACHA20_POLY1305 = "chacha20-poly1305"


class DataClassification(Enum):
    """データ分類レベル"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class EncryptionConfig:
    """暗号化設定"""

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_size: int = 256
    key_rotation_days: int = 90
    backup_encryption: bool = True
    compression_enabled: bool = True

    # セキュリティパラメータ
    pbkdf2_iterations: int = 100000
    salt_length: int = 32
    iv_length: int = 16
    tag_length: int = 16


@dataclass
class EncryptedData:
    """暗号化されたデータ"""

    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 暗号化パラメータ
    salt: Optional[bytes] = None
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None

    # メタデータ
    encrypted_at: datetime = field(default_factory=datetime.utcnow)
    key_id: Optional[str] = None
    compression_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("utf-8"),
            "algorithm": self.algorithm.value,
            "metadata": self.metadata,
            "salt": base64.b64encode(self.salt).decode("utf-8") if self.salt else None,
            "iv": base64.b64encode(self.iv).decode("utf-8") if self.iv else None,
            "tag": base64.b64encode(self.tag).decode("utf-8") if self.tag else None,
            "encrypted_at": self.encrypted_at.isoformat(),
            "key_id": self.key_id,
            "compression_used": self.compression_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedData":
        """辞書から復元"""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            metadata=data.get("metadata", {}),
            salt=base64.b64decode(data["salt"]) if data.get("salt") else None,
            iv=base64.b64decode(data["iv"]) if data.get("iv") else None,
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            encrypted_at=(
                datetime.fromisoformat(data["encrypted_at"])
                if data.get("encrypted_at")
                else datetime.utcnow()
            ),
            key_id=data.get("key_id"),
            compression_used=data.get("compression_used", False),
        )


class EncryptionProvider(ABC):
    """暗号化プロバイダー基底クラス"""

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """データ暗号化"""
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """データ復号化"""
        pass

    @abstractmethod
    def generate_key(self) -> bytes:
        """暗号化キー生成"""
        pass


class AESGCMProvider(EncryptionProvider):
    """AES-256-GCM暗号化プロバイダー"""

    def __init__(self, config: EncryptionConfig):
        self.config = config

    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """AES-GCM暗号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        # IV生成
        iv = os.urandom(self.config.iv_length)

        # AES-GCM暗号化
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        tag = encryptor.tag

        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            iv=iv,
            tag=tag,
        )

    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """AES-GCM復号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        if encrypted_data.algorithm != EncryptionAlgorithm.AES_256_GCM:
            raise ValueError("Invalid algorithm for AES-GCM provider")

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data.iv, encrypted_data.tag),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

        return plaintext

    def generate_key(self) -> bytes:
        """AESキー生成（256-bit）"""
        return os.urandom(32)  # 256 bits


class FernetProvider(EncryptionProvider):
    """Fernet暗号化プロバイダー（より簡単な実装）"""

    def __init__(self, config: EncryptionConfig):
        self.config = config

    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Fernet暗号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        fernet = Fernet(key)
        ciphertext = fernet.encrypt(plaintext)

        return EncryptedData(
            ciphertext=ciphertext, algorithm=EncryptionAlgorithm.FERNET
        )

    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Fernet復号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        if encrypted_data.algorithm != EncryptionAlgorithm.FERNET:
            raise ValueError("Invalid algorithm for Fernet provider")

        fernet = Fernet(key)
        plaintext = fernet.decrypt(encrypted_data.ciphertext)

        return plaintext

    def generate_key(self) -> bytes:
        """Fernetキー生成"""
        return Fernet.generate_key()


class KeyManager:
    """暗号化キー管理システム"""

    def __init__(
        self,
        key_storage_path: str = "security/keys",
        master_key_env: str = "DAYTRADE_MASTER_KEY",
    ):
        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        self.master_key_env = master_key_env

        # キーレジストリ
        self.key_registry: Dict[str, Dict[str, Any]] = {}

        # マスターキー取得/生成
        self.master_key = self._get_or_create_master_key()

        # キーレジストリ読み込み
        self._load_key_registry()

    def _get_or_create_master_key(self) -> bytes:
        """マスターキー取得/生成"""
        # 環境変数から取得を試行
        env_key = os.getenv(self.master_key_env)
        if env_key:
            try:
                return base64.b64decode(env_key)
            except Exception:
                logger.warning(
                    "環境変数のマスターキーが無効です。新しいキーを生成します。"
                )

        # ファイルから取得を試行
        master_key_file = self.key_storage_path / "master.key"
        if master_key_file.exists():
            try:
                with open(master_key_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"マスターキーファイル読み込みエラー: {e}")

        # 新しいマスターキー生成
        master_key = os.urandom(32)  # 256-bit

        # セキュアに保存
        try:
            with open(master_key_file, "wb") as f:
                f.write(master_key)

            # ファイル権限を600に設定（Unix系）
            if os.name == "posix":
                os.chmod(master_key_file, 0o600)

            logger.info(f"新しいマスターキーを生成しました: {master_key_file}")
            logger.warning("本番環境では環境変数でマスターキーを管理してください。")

        except Exception as e:
            logger.error(f"マスターキー保存エラー: {e}")

        return master_key

    def _load_key_registry(self):
        """キーレジストリ読み込み"""
        registry_file = self.key_storage_path / "key_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, encoding="utf-8") as f:
                    self.key_registry = json.load(f)
                logger.info(f"キーレジストリ読み込み完了: {len(self.key_registry)}キー")
            except Exception as e:
                logger.error(f"キーレジストリ読み込みエラー: {e}")
                self.key_registry = {}

    def _save_key_registry(self):
        """キーレジストリ保存"""
        registry_file = self.key_storage_path / "key_registry.json"
        try:
            with open(registry_file, "w", encoding="utf-8") as f:
                json.dump(self.key_registry, f, indent=2)
        except Exception as e:
            logger.error(f"キーレジストリ保存エラー: {e}")

    def create_data_key(
        self,
        key_id: str,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """データキー作成"""
        if key_id in self.key_registry:
            raise ValueError(f"Key ID already exists: {key_id}")

        # データキー生成
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            data_key = os.urandom(32)  # 256-bit
        elif algorithm == EncryptionAlgorithm.FERNET:
            data_key = Fernet.generate_key()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # マスターキーで暗号化
        if CRYPTO_AVAILABLE:
            fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
            encrypted_data_key = fernet.encrypt(data_key)
        else:
            # Fallback: XOR暗号化（本番環境では使用しない）
            encrypted_data_key = bytes(a ^ b for a, b in zip(data_key, self.master_key))

        # キー情報保存
        key_info = {
            "key_id": key_id,
            "algorithm": algorithm.value,
            "encrypted_data_key": base64.b64encode(encrypted_data_key).decode("utf-8"),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "rotation_due": (datetime.utcnow() + timedelta(days=90)).isoformat(),
        }

        self.key_registry[key_id] = key_info
        self._save_key_registry()

        logger.info(f"データキー作成完了: {key_id}")
        return key_id

    def get_data_key(self, key_id: str) -> bytes:
        """データキー取得"""
        if key_id not in self.key_registry:
            raise KeyError(f"Key not found: {key_id}")

        key_info = self.key_registry[key_id]
        encrypted_data_key = base64.b64decode(key_info["encrypted_data_key"])

        # マスターキーで復号化
        if CRYPTO_AVAILABLE:
            fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
            data_key = fernet.decrypt(encrypted_data_key)
        else:
            # Fallback: XOR復号化（本番環境では使用しない）
            data_key = bytes(a ^ b for a, b in zip(encrypted_data_key, self.master_key))

        # キーローテーションチェック
        rotation_due = datetime.fromisoformat(key_info["rotation_due"])
        if datetime.utcnow() > rotation_due:
            logger.warning(f"キーローテーションが必要です: {key_id}")

        return data_key

    def rotate_key(self, key_id: str) -> str:
        """キーローテーション"""
        if key_id not in self.key_registry:
            raise KeyError(f"Key not found: {key_id}")

        old_key_info = self.key_registry[key_id]

        # 新しいキーID生成
        new_key_id = f"{key_id}_rotated_{int(time.time())}"

        # 新しいキー作成
        algorithm = EncryptionAlgorithm(old_key_info["algorithm"])
        self.create_data_key(new_key_id, algorithm, old_key_info["metadata"])

        # 古いキーに終了マーク
        old_key_info["rotated_at"] = datetime.utcnow().isoformat()
        old_key_info["replaced_by"] = new_key_id
        old_key_info["status"] = "rotated"

        self._save_key_registry()

        logger.info(f"キーローテーション完了: {key_id} -> {new_key_id}")
        return new_key_id

    def list_keys(self) -> List[Dict[str, Any]]:
        """キー一覧取得"""
        return [
            {
                "key_id": key_id,
                "algorithm": info["algorithm"],
                "created_at": info["created_at"],
                "rotation_due": info["rotation_due"],
                "status": info.get("status", "active"),
            }
            for key_id, info in self.key_registry.items()
        ]


class DataProtectionManager:
    """
    データ保護管理システム

    暗号化、復号化、キー管理、アクセス制御を統合し、
    機密データの包括的な保護を提供
    """

    def __init__(
        self,
        storage_path: str = "security/data_protection",
        key_storage_path: str = "security/keys",
        default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    ):
        """
        初期化

        Args:
            storage_path: データ保護設定保存パス
            key_storage_path: キー保存パス
            default_algorithm: デフォルト暗号化アルゴリズム
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # キー管理システム
        self.key_manager = KeyManager(key_storage_path)

        # 暗号化設定
        self.encryption_config = EncryptionConfig(algorithm=default_algorithm)

        # 暗号化プロバイダー
        self.providers: Dict[EncryptionAlgorithm, EncryptionProvider] = {}
        if CRYPTO_AVAILABLE:
            self.providers[EncryptionAlgorithm.AES_256_GCM] = AESGCMProvider(
                self.encryption_config
            )
            self.providers[EncryptionAlgorithm.FERNET] = FernetProvider(
                self.encryption_config
            )

        # データ分類ポリシー
        self.data_classification_policies: Dict[DataClassification, Dict[str, Any]] = {
            DataClassification.PUBLIC: {
                "encryption_required": False,
                "access_logging": False,
                "retention_days": 365,
            },
            DataClassification.INTERNAL: {
                "encryption_required": False,
                "access_logging": True,
                "retention_days": 730,
            },
            DataClassification.CONFIDENTIAL: {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 2555,  # 7年
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
            },
            DataClassification.RESTRICTED: {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 3650,  # 10年
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_rotation_days": 30,
            },
            DataClassification.TOP_SECRET: {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 7300,  # 20年
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_rotation_days": 7,
            },
        }

        # アクセスログ
        self.access_logs: List[Dict[str, Any]] = []

        logger.info("DataProtectionManager初期化完了")

    def encrypt_data(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        key_id: Optional[str] = None,
        algorithm: Optional[EncryptionAlgorithm] = None,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
    ) -> EncryptedData:
        """データ暗号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Encryption not available. Install cryptography library."
            )

        # データをバイト列に変換
        if isinstance(data, str):
            plaintext = data.encode("utf-8")
        elif isinstance(data, dict):
            plaintext = json.dumps(data, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, bytes):
            plaintext = data
        else:
            plaintext = str(data).encode("utf-8")

        # アルゴリズム決定
        if algorithm is None:
            policy = self.data_classification_policies[classification]
            algorithm = policy.get("algorithm", self.encryption_config.algorithm)

        # キー取得/作成
        if key_id is None:
            key_id = f"auto_key_{int(time.time())}"
            self.key_manager.create_data_key(key_id, algorithm)

        data_key = self.key_manager.get_data_key(key_id)

        # 暗号化実行
        provider = self.providers.get(algorithm)
        if not provider:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        encrypted_data = provider.encrypt(plaintext, data_key)
        encrypted_data.key_id = key_id
        encrypted_data.metadata.update(
            {
                "classification": classification.value,
                "original_type": type(data).__name__,
            }
        )

        # アクセスログ記録
        self._log_access("encrypt", key_id, classification)

        logger.info(f"データ暗号化完了: key_id={key_id}, algorithm={algorithm.value}")
        return encrypted_data

    def decrypt_data(
        self, encrypted_data: EncryptedData, output_type: str = "auto"
    ) -> Union[str, bytes, Dict[str, Any]]:
        """データ復号化"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "Decryption not available. Install cryptography library."
            )

        if not encrypted_data.key_id:
            raise ValueError("Key ID not specified in encrypted data")

        # キー取得
        data_key = self.key_manager.get_data_key(encrypted_data.key_id)

        # 復号化実行
        provider = self.providers.get(encrypted_data.algorithm)
        if not provider:
            raise ValueError(f"Unsupported algorithm: {encrypted_data.algorithm}")

        plaintext = provider.decrypt(encrypted_data, data_key)

        # データタイプに応じて変換
        original_type = encrypted_data.metadata.get("original_type", "str")
        classification = DataClassification(
            encrypted_data.metadata.get("classification", "confidential")
        )

        if output_type == "auto":
            if original_type == "dict":
                result = json.loads(plaintext.decode("utf-8"))
            elif original_type in ["str", "NoneType"]:
                result = plaintext.decode("utf-8")
            else:
                result = plaintext
        elif output_type == "str":
            result = plaintext.decode("utf-8")
        elif output_type == "bytes":
            result = plaintext
        elif output_type == "dict":
            result = json.loads(plaintext.decode("utf-8"))
        else:
            result = plaintext

        # アクセスログ記録
        self._log_access("decrypt", encrypted_data.key_id, classification)

        logger.info(f"データ復号化完了: key_id={encrypted_data.key_id}")
        return result

    def _log_access(
        self, operation: str, key_id: str, classification: DataClassification
    ):
        """アクセスログ記録"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "key_id": key_id,
            "classification": classification.value,
            "process_id": os.getpid(),
        }

        self.access_logs.append(log_entry)

        # ログファイルに記録
        log_file = self.storage_path / "access.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"アクセスログ記録エラー: {e}")

    def encrypt_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
    ) -> str:
        """ファイル暗号化"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # ファイル読み込み
        with open(file_path, "rb") as f:
            file_data = f.read()

        # 暗号化
        encrypted_data = self.encrypt_data(file_data, classification=classification)

        # 出力ファイルパス決定
        if output_path is None:
            output_path = str(file_path) + ".encrypted"

        # 暗号化データ保存
        encrypted_file_data = {
            "encrypted_data": encrypted_data.to_dict(),
            "original_filename": file_path.name,
            "original_size": len(file_data),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(encrypted_file_data, f, indent=2)

        logger.info(f"ファイル暗号化完了: {file_path} -> {output_path}")
        return output_path

    def decrypt_file(
        self, encrypted_file_path: str, output_path: Optional[str] = None
    ) -> str:
        """ファイル復号化"""
        encrypted_file_path = Path(encrypted_file_path)
        if not encrypted_file_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")

        # 暗号化ファイル読み込み
        with open(encrypted_file_path, encoding="utf-8") as f:
            encrypted_file_data = json.load(f)

        encrypted_data = EncryptedData.from_dict(encrypted_file_data["encrypted_data"])

        # 復号化
        decrypted_data = self.decrypt_data(encrypted_data, output_type="bytes")

        # 出力ファイルパス決定
        if output_path is None:
            original_filename = encrypted_file_data.get(
                "original_filename", "decrypted_file"
            )
            output_path = encrypted_file_path.parent / original_filename

        # 復号化データ保存
        with open(output_path, "wb") as f:
            f.write(decrypted_data)

        logger.info(f"ファイル復号化完了: {encrypted_file_path} -> {output_path}")
        return str(output_path)

    def get_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        key_stats = self.key_manager.list_keys()

        # キー統計
        active_keys = [k for k in key_stats if k["status"] == "active"]
        rotation_needed = []

        for key in active_keys:
            rotation_due = datetime.fromisoformat(key["rotation_due"])
            if datetime.utcnow() > rotation_due:
                rotation_needed.append(key["key_id"])

        # アクセスログ統計
        recent_accesses = [
            log
            for log in self.access_logs
            if datetime.fromisoformat(log["timestamp"])
            > datetime.utcnow() - timedelta(days=7)
        ]

        access_stats = {}
        for log in recent_accesses:
            classification = log["classification"]
            access_stats[classification] = access_stats.get(classification, 0) + 1

        report = {
            "report_id": f"data_protection_report_{int(time.time())}",
            "generated_at": datetime.utcnow().isoformat(),
            "key_management": {
                "total_keys": len(key_stats),
                "active_keys": len(active_keys),
                "rotation_needed": len(rotation_needed),
                "keys_needing_rotation": rotation_needed,
            },
            "access_statistics": {
                "recent_accesses_7days": len(recent_accesses),
                "access_by_classification": access_stats,
            },
            "encryption_status": {
                "crypto_library_available": CRYPTO_AVAILABLE,
                "supported_algorithms": [algo.value for algo in self.providers.keys()],
                "default_algorithm": self.encryption_config.algorithm.value,
            },
            "recommendations": self._generate_security_recommendations(
                rotation_needed, recent_accesses
            ),
        }

        return report

    def _generate_security_recommendations(
        self, rotation_needed: List[str], recent_accesses: List[Dict[str, Any]]
    ) -> List[str]:
        """セキュリティ推奨事項生成"""
        recommendations = []

        if rotation_needed:
            recommendations.append(
                f"🔄 {len(rotation_needed)}個のキーのローテーションが必要です。"
            )

        if not CRYPTO_AVAILABLE:
            recommendations.append(
                "⚠️ cryptographyライブラリをインストールしてください。"
            )

        # 高頻度アクセスチェック
        high_classification_accesses = [
            a
            for a in recent_accesses
            if a["classification"] in ["restricted", "top_secret"]
        ]

        if len(high_classification_accesses) > 100:
            recommendations.append(
                "🔍 機密度の高いデータへのアクセス頻度が高いです。アクセス制御を確認してください。"
            )

        if not recommendations:
            recommendations.append("✅ 現在のデータ保護設定は適切に構成されています。")

        return recommendations


# Factory function
def create_data_protection_manager(
    storage_path: str = "security/data_protection", **config
) -> DataProtectionManager:
    """DataProtectionManagerファクトリ関数"""
    return DataProtectionManager(storage_path=storage_path, **config)


if __name__ == "__main__":
    # テスト実行
    def main():
        print("=== Issue #419 データ保護・暗号化システムテスト ===")

        manager = None
        try:
            # データ保護システム初期化
            manager = create_data_protection_manager()

            print("\n1. 暗号化システム状態")
            print(f"暗号化ライブラリ利用可能: {CRYPTO_AVAILABLE}")
            print(
                f"サポートアルゴリズム: {[algo.value for algo in manager.providers.keys()]}"
            )

            if CRYPTO_AVAILABLE:
                print("\n2. データ暗号化テスト")

                # 文字列データ暗号化
                test_string = "これは機密データです。API Key: secret_12345"
                encrypted_string = manager.encrypt_data(
                    test_string, classification=DataClassification.CONFIDENTIAL
                )

                print(f"暗号化完了: key_id={encrypted_string.key_id}")
                print(f"アルゴリズム: {encrypted_string.algorithm.value}")

                # 復号化
                decrypted_string = manager.decrypt_data(encrypted_string)
                print(f"復号化完了: {decrypted_string == test_string}")

                print("\n3. JSON データ暗号化テスト")
                test_json = {
                    "api_key": "secret_api_key_12345",
                    "password": "super_secure_password",
                    "user_info": {
                        "name": "テストユーザー",
                        "email": "test@example.com",
                    },
                }

                encrypted_json = manager.encrypt_data(
                    test_json, classification=DataClassification.RESTRICTED
                )

                decrypted_json = manager.decrypt_data(encrypted_json)
                print(f"JSON暗号化・復号化: {decrypted_json == test_json}")

                print("\n4. キー管理テスト")
                keys = manager.key_manager.list_keys()
                print(f"管理されているキー数: {len(keys)}")
                for key in keys[:3]:  # 最初の3つを表示
                    print(
                        f"  - {key['key_id']}: {key['algorithm']} (作成: {key['created_at'][:10]})"
                    )

            print("\n5. セキュリティレポート生成")
            report = manager.get_security_report()

            print(f"レポートID: {report['report_id']}")
            print("キー管理状況:")
            key_mgmt = report["key_management"]
            print(f"  総キー数: {key_mgmt['total_keys']}")
            print(f"  アクティブキー: {key_mgmt['active_keys']}")
            print(f"  ローテーション必要: {key_mgmt['rotation_needed']}")

            print("アクセス統計:")
            access_stats = report["access_statistics"]
            print(f"  過去7日間のアクセス: {access_stats['recent_accesses_7days']}")

            print("推奨事項:")
            for rec in report["recommendations"]:
                print(f"  {rec}")

            print("\n6. データ分類ポリシー")
            for classification, policy in manager.data_classification_policies.items():
                print(f"{classification.value}:")
                print(f"  暗号化必須: {policy['encryption_required']}")
                print(f"  アクセスログ: {policy['access_logging']}")
                print(f"  保持期間: {policy['retention_days']}日")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        print("\n=== データ保護・暗号化システムテスト完了 ===")

    main()
