#!/usr/bin/env python3
"""
Issue #800 Phase 5: 暗号化・シークレット管理システム
Day Trade ML System データ暗号化・保護
"""

import os
import base64
import secrets
import logging
from typing import Dict, Optional, Union, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import redis
import json
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EncryptionManager:
    """暗号化管理システム"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_CRYPTO_DB', 4)),
            decode_responses=False  # バイナリデータ用
        )

        # マスターキー（環境変数から取得、なければ生成）
        self.master_key = self._get_or_create_master_key()

        # データ暗号化用Fernet
        self.fernet = Fernet(self.master_key)

        # RSA キーペア（公開鍵暗号用）
        self.rsa_private_key, self.rsa_public_key = self._get_or_create_rsa_keys()

    def encrypt_data(self, data: Union[str, bytes], key_rotation: bool = False) -> str:
        """データ暗号化（対称暗号）"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        # キーローテーション対応
        if key_rotation:
            # 新しいキーで暗号化
            rotation_key = self._generate_rotation_key()
            encrypted = Fernet(rotation_key).encrypt(data)

            # ローテーションキーを保存
            key_id = self._store_rotation_key(rotation_key)

            # キーIDを含む形式で返却
            return f"ROTATED:{key_id}:{base64.b64encode(encrypted).decode()}"
        else:
            # マスターキーで暗号化
            encrypted = self.fernet.encrypt(data)
            return base64.b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> bytes:
        """データ復号化"""
        try:
            # ローテーションキー確認
            if encrypted_data.startswith('ROTATED:'):
                parts = encrypted_data.split(':', 2)
                if len(parts) == 3:
                    key_id = parts[1]
                    encrypted_bytes = base64.b64decode(parts[2])

                    # ローテーションキー取得
                    rotation_key = self._get_rotation_key(key_id)
                    if rotation_key:
                        return Fernet(rotation_key).decrypt(encrypted_bytes)
                    else:
                        raise ValueError(f"Rotation key not found: {key_id}")

            # マスターキーで復号化
            encrypted_bytes = base64.b64decode(encrypted_data)
            return self.fernet.decrypt(encrypted_bytes)

        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    def encrypt_with_public_key(self, data: Union[str, bytes]) -> str:
        """公開鍵暗号化（小サイズデータ用）"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self.rsa_public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return base64.b64encode(encrypted).decode()

    def decrypt_with_private_key(self, encrypted_data: str) -> bytes:
        """秘密鍵復号化"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)

            decrypted = self.rsa_private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return decrypted

        except Exception as e:
            logger.error(f"RSA decryption failed: {str(e)}")
            raise

    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """ファイル暗号化"""
        if not output_path:
            output_path = f"{file_path}.encrypted"

        with open(file_path, 'rb') as infile:
            file_data = infile.read()

        encrypted_data = self.encrypt_data(file_data)

        with open(output_path, 'w') as outfile:
            outfile.write(encrypted_data)

        logger.info(f"File encrypted: {file_path} -> {output_path}")
        return output_path

    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """ファイル復号化"""
        if not output_path:
            if encrypted_file_path.endswith('.encrypted'):
                output_path = encrypted_file_path[:-10]  # .encrypted除去
            else:
                output_path = f"{encrypted_file_path}.decrypted"

        with open(encrypted_file_path, 'r') as infile:
            encrypted_data = infile.read()

        decrypted_data = self.decrypt_data(encrypted_data)

        with open(output_path, 'wb') as outfile:
            outfile.write(decrypted_data)

        logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
        return output_path

    def generate_secure_token(self, length: int = 32) -> str:
        """セキュアトークン生成"""
        return secrets.token_urlsafe(length)

    def hash_password(self, password: str, salt: bytes = None) -> Tuple[str, str]:
        """パスワードハッシュ化（PBKDF2）"""
        if salt is None:
            salt = os.urandom(32)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = kdf.derive(password.encode())

        return (
            base64.b64encode(key).decode(),
            base64.b64encode(salt).decode()
        )

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """パスワード検証"""
        try:
            salt_bytes = base64.b64decode(salt)
            hash_bytes = base64.b64decode(password_hash)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )

            kdf.verify(password.encode(), hash_bytes)
            return True

        except Exception:
            return False

    def rotate_master_key(self) -> bool:
        """マスターキーローテーション"""
        try:
            # 新しいマスターキー生成
            new_master_key = Fernet.generate_key()

            # 古いキーをバックアップ
            old_key_id = self._backup_master_key(self.master_key)

            # 新しいキーを設定
            self.master_key = new_master_key
            self.fernet = Fernet(new_master_key)

            # 環境変数更新（実際の運用では外部キー管理サービス使用）
            self.redis_client.set('master_key_current', new_master_key)

            # ローテーション記録
            rotation_record = {
                'rotated_at': datetime.utcnow().isoformat(),
                'old_key_id': old_key_id,
                'new_key_active': True
            }

            self.redis_client.set(
                'key_rotation_latest',
                json.dumps(rotation_record)
            )

            logger.info(f"Master key rotated successfully. Old key backed up: {old_key_id}")
            return True

        except Exception as e:
            logger.error(f"Master key rotation failed: {str(e)}")
            return False

    def _get_or_create_master_key(self) -> bytes:
        """マスターキー取得または生成"""
        # 環境変数から取得試行
        key_b64 = os.getenv('MASTER_ENCRYPTION_KEY')
        if key_b64:
            try:
                return base64.b64decode(key_b64)
            except Exception:
                logger.warning("Invalid master key in environment variable")

        # Redisから取得試行
        key_bytes = self.redis_client.get('master_key_current')
        if key_bytes:
            return key_bytes

        # 新規生成
        new_key = Fernet.generate_key()
        self.redis_client.set('master_key_current', new_key)

        logger.info("New master encryption key generated")
        logger.warning("Master key stored in Redis. For production, use external key management service.")

        return new_key

    def _get_or_create_rsa_keys(self) -> Tuple:
        """RSAキーペア取得または生成"""
        # Redisから取得試行
        private_key_pem = self.redis_client.get('rsa_private_key')
        public_key_pem = self.redis_client.get('rsa_public_key')

        if private_key_pem and public_key_pem:
            try:
                private_key = serialization.load_pem_private_key(
                    private_key_pem, password=None
                )
                public_key = serialization.load_pem_public_key(public_key_pem)

                return private_key, public_key

            except Exception:
                logger.warning("Invalid RSA keys in Redis, generating new ones")

        # 新規生成
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        # PEM形式でシリアライズ
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Redisに保存
        self.redis_client.set('rsa_private_key', private_pem)
        self.redis_client.set('rsa_public_key', public_pem)

        logger.info("New RSA key pair generated")
        return private_key, public_key

    def _generate_rotation_key(self) -> bytes:
        """ローテーション用キー生成"""
        return Fernet.generate_key()

    def _store_rotation_key(self, key: bytes) -> str:
        """ローテーションキー保存"""
        key_id = secrets.token_hex(16)
        self.redis_client.setex(
            f"rotation_key:{key_id}",
            timedelta(days=90).total_seconds(),  # 90日間保持
            key
        )
        return key_id

    def _get_rotation_key(self, key_id: str) -> Optional[bytes]:
        """ローテーションキー取得"""
        return self.redis_client.get(f"rotation_key:{key_id}")

    def _backup_master_key(self, key: bytes) -> str:
        """マスターキーバックアップ"""
        key_id = secrets.token_hex(16)
        backup_record = {
            'key': base64.b64encode(key).decode(),
            'backed_up_at': datetime.utcnow().isoformat(),
            'key_id': key_id
        }

        self.redis_client.set(
            f"master_key_backup:{key_id}",
            json.dumps(backup_record)
        )

        return key_id

class SecretManager:
    """シークレット管理システム"""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_SECRETS_DB', 5)),
            decode_responses=True
        )

        # 重要なシークレット定義
        self.critical_secrets = {
            'database_password',
            'api_keys',
            'jwt_secret',
            'oauth_client_secret',
            'encryption_keys',
            'certificate_keys'
        }

    def store_secret(self, key: str, value: str, description: str = "", ttl: int = None) -> bool:
        """シークレット保存"""
        try:
            # 暗号化
            encrypted_value = self.encryption_manager.encrypt_data(value)

            # メタデータ
            metadata = {
                'encrypted_value': encrypted_value,
                'description': description,
                'created_at': datetime.utcnow().isoformat(),
                'accessed_count': 0,
                'is_critical': key in self.critical_secrets
            }

            # Redisに保存
            secret_key = f"secret:{key}"
            self.redis_client.hset(secret_key, mapping=metadata)

            # TTL設定
            if ttl:
                self.redis_client.expire(secret_key, ttl)

            # アクセスログ
            self._log_secret_access(key, 'STORE', 'System')

            logger.info(f"Secret stored: {key} ({'critical' if key in self.critical_secrets else 'standard'})")
            return True

        except Exception as e:
            logger.error(f"Failed to store secret {key}: {str(e)}")
            return False

    def get_secret(self, key: str, requester: str = "Unknown") -> Optional[str]:
        """シークレット取得"""
        try:
            secret_key = f"secret:{key}"
            metadata = self.redis_client.hgetall(secret_key)

            if not metadata:
                logger.warning(f"Secret not found: {key}")
                return None

            # 復号化
            encrypted_value = metadata['encrypted_value']
            decrypted_value = self.encryption_manager.decrypt_data(encrypted_value)

            # アクセス記録更新
            self.redis_client.hincrby(secret_key, 'accessed_count', 1)
            self.redis_client.hset(secret_key, 'last_accessed', datetime.utcnow().isoformat())

            # アクセスログ
            self._log_secret_access(key, 'ACCESS', requester)

            return decrypted_value.decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to get secret {key}: {str(e)}")
            # 失敗もログに記録
            self._log_secret_access(key, 'ACCESS_FAILED', requester)
            return None

    def delete_secret(self, key: str, requester: str = "Unknown") -> bool:
        """シークレット削除"""
        try:
            secret_key = f"secret:{key}"

            # 存在確認
            if not self.redis_client.exists(secret_key):
                return False

            # Critical シークレットの削除は特別な確認が必要
            if key in self.critical_secrets:
                logger.warning(f"Attempt to delete critical secret: {key} by {requester}")
                # 実際の運用では追加の認証が必要

            # 削除実行
            self.redis_client.delete(secret_key)

            # 削除ログ
            self._log_secret_access(key, 'DELETE', requester)

            logger.info(f"Secret deleted: {key} by {requester}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {str(e)}")
            return False

    def list_secrets(self, include_critical: bool = False) -> List[Dict]:
        """シークレット一覧"""
        secrets_list = []

        for key in self.redis_client.keys("secret:*"):
            secret_name = key.replace("secret:", "")
            metadata = self.redis_client.hgetall(key)

            # Critical シークレット除外オプション
            if not include_critical and metadata.get('is_critical') == 'True':
                continue

            secret_info = {
                'name': secret_name,
                'description': metadata.get('description', ''),
                'created_at': metadata.get('created_at', ''),
                'last_accessed': metadata.get('last_accessed', ''),
                'accessed_count': int(metadata.get('accessed_count', 0)),
                'is_critical': metadata.get('is_critical') == 'True'
            }

            secrets_list.append(secret_info)

        return secrets_list

    def rotate_secret(self, key: str, new_value: str, requester: str = "Unknown") -> bool:
        """シークレットローテーション"""
        try:
            # 既存シークレット確認
            secret_key = f"secret:{key}"
            old_metadata = self.redis_client.hgetall(secret_key)

            if not old_metadata:
                logger.warning(f"Cannot rotate non-existent secret: {key}")
                return False

            # 古いバージョンをバックアップ
            backup_key = f"secret_backup:{key}:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.redis_client.hset(backup_key, mapping=old_metadata)
            self.redis_client.expire(backup_key, timedelta(days=30).total_seconds())

            # 新しい値で更新
            encrypted_value = self.encryption_manager.encrypt_data(new_value)

            update_data = {
                'encrypted_value': encrypted_value,
                'rotated_at': datetime.utcnow().isoformat(),
                'rotated_by': requester
            }

            self.redis_client.hset(secret_key, mapping=update_data)

            # ローテーションログ
            self._log_secret_access(key, 'ROTATE', requester)

            logger.info(f"Secret rotated: {key} by {requester}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secret {key}: {str(e)}")
            return False

    def _log_secret_access(self, secret_name: str, action: str, requester: str):
        """シークレットアクセスログ"""
        log_entry = {
            'secret_name': secret_name,
            'action': action,
            'requester': requester,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': 'local'  # 実際の実装では実IPを記録
        }

        # 日別ログキー
        log_date = datetime.utcnow().strftime('%Y-%m-%d')
        log_key = f"secret_access_log:{log_date}"

        self.redis_client.lpush(log_key, json.dumps(log_entry))
        self.redis_client.expire(log_key, timedelta(days=90).total_seconds())

    def get_access_logs(self, secret_name: str = None, days: int = 7) -> List[Dict]:
        """アクセスログ取得"""
        logs = []

        for i in range(days):
            log_date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
            log_key = f"secret_access_log:{log_date}"

            day_logs = self.redis_client.lrange(log_key, 0, -1)
            for log_entry in day_logs:
                log_data = json.loads(log_entry)

                # 特定シークレットでフィルタ
                if secret_name and log_data['secret_name'] != secret_name:
                    continue

                logs.append(log_data)

        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

if __name__ == '__main__':
    # テスト用
    encryption_manager = EncryptionManager()
    secret_manager = SecretManager(encryption_manager)

    # サンプル暗号化
    test_data = "Sensitive data: API Key ABC123"
    encrypted = encryption_manager.encrypt_data(test_data)
    decrypted = encryption_manager.decrypt_data(encrypted)

    print(f"Original: {test_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted.decode()}")

    # サンプルシークレット管理
    secret_manager.store_secret('test_api_key', 'abc123xyz789', 'Test API key')
    retrieved_secret = secret_manager.get_secret('test_api_key', 'test_user')
    print(f"Retrieved Secret: {retrieved_secret}")