#!/usr/bin/env python3
"""
セキュリティ管理システム
Phase E: セキュリティ強化実装

認証・認可・監査ログ・データ保護機能
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jwt

from ..utils.logging_config import get_context_logger
from .optimization_strategy import OptimizationConfig

logger = get_context_logger(__name__)


class SecurityLevel(Enum):
    """セキュリティレベル"""

    LOW = "low"  # 基本認証のみ
    MEDIUM = "medium"  # API認証 + 監査ログ
    HIGH = "high"  # 高度認証 + 暗号化
    ENTERPRISE = "enterprise"  # エンタープライズ対応


class AccessLevel(Enum):
    """アクセスレベル"""

    READ_ONLY = "read_only"  # 読み取り専用
    ANALYST = "analyst"  # 分析実行権限
    ADMINISTRATOR = "administrator"  # 設定変更権限
    SYSTEM = "system"  # システム管理権限


@dataclass
class User:
    """ユーザー情報"""

    user_id: str
    username: str
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    api_key: Optional[str] = None


@dataclass
class AuditLog:
    """監査ログ"""

    log_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None


class TokenManager:
    """トークン管理"""

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        self.refresh_token_expiry = timedelta(days=30)

    def _generate_secret_key(self) -> str:
        """秘密キー生成"""
        return secrets.token_urlsafe(32)

    def generate_access_token(self, user: User) -> str:
        """アクセストークン生成"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "access_level": user.access_level.value,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def generate_refresh_token(self, user: User) -> str:
        """リフレッシュトークン生成"""
        payload = {
            "user_id": user.user_id,
            "exp": datetime.utcnow() + self.refresh_token_expiry,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """トークン検証"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("トークン期限切れ")
            return None
        except jwt.InvalidTokenError:
            logger.warning("無効なトークン")
            return None

    def generate_api_key(self, user: User) -> str:
        """APIキー生成"""
        # ユーザーIDとタイムスタンプから一意のAPIキー生成
        data = f"{user.user_id}:{datetime.utcnow().isoformat()}:{secrets.token_urlsafe(16)}"
        api_key = hashlib.sha256(data.encode()).hexdigest()
        return f"dt_{api_key[:32]}"

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """APIキー検証"""
        return hmac.compare_digest(api_key, stored_hash)


class PermissionManager:
    """権限管理"""

    def __init__(self):
        self.permissions = {
            AccessLevel.READ_ONLY: [
                "view_dashboard",
                "read_analysis_results",
                "download_reports",
            ],
            AccessLevel.ANALYST: [
                "view_dashboard",
                "read_analysis_results",
                "download_reports",
                "execute_analysis",
                "modify_parameters",
                "create_reports",
            ],
            AccessLevel.ADMINISTRATOR: [
                "view_dashboard",
                "read_analysis_results",
                "download_reports",
                "execute_analysis",
                "modify_parameters",
                "create_reports",
                "manage_users",
                "modify_configuration",
                "view_audit_logs",
            ],
            AccessLevel.SYSTEM: [
                "view_dashboard",
                "read_analysis_results",
                "download_reports",
                "execute_analysis",
                "modify_parameters",
                "create_reports",
                "manage_users",
                "modify_configuration",
                "view_audit_logs",
                "system_administration",
                "security_configuration",
            ],
        }

    def has_permission(self, access_level: AccessLevel, action: str) -> bool:
        """権限チェック"""
        user_permissions = self.permissions.get(access_level, [])
        return action in user_permissions

    def get_user_permissions(self, access_level: AccessLevel) -> List[str]:
        """ユーザー権限一覧取得"""
        return self.permissions.get(access_level, [])


class AuditLogger:
    """監査ログ管理"""

    def __init__(self, log_file_path: Optional[Path] = None):
        self.log_file = log_file_path or Path("logs/audit.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 監査ログ用のロガー設定
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)

        # ファイルハンドラー追加
        if not self.audit_logger.handlers:
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - AUDIT - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            self.audit_logger.addHandler(file_handler)

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """アクション記録"""
        audit_log = AuditLog(
            log_id=self._generate_log_id(),
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            request_data=self._sanitize_request_data(request_data),
        )

        # ログ出力
        log_message = self._format_audit_log(audit_log)
        self.audit_logger.info(log_message)

        return audit_log

    def _generate_log_id(self) -> str:
        """ログID生成"""
        import uuid

        return f"AUDIT_{uuid.uuid4().hex[:12]}"

    def _sanitize_request_data(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """リクエストデータのサニタイズ（機密情報除去）"""
        if not data:
            return None

        sensitive_keys = ["password", "token", "api_key", "secret", "auth"]
        sanitized = {}

        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value

        return sanitized

    def _format_audit_log(self, audit_log: AuditLog) -> str:
        """監査ログフォーマット"""
        return json.dumps(
            {
                "log_id": audit_log.log_id,
                "user_id": audit_log.user_id,
                "action": audit_log.action,
                "resource": audit_log.resource,
                "timestamp": audit_log.timestamp.isoformat(),
                "success": audit_log.success,
                "ip_address": audit_log.ip_address,
                "user_agent": audit_log.user_agent,
                "error_message": audit_log.error_message,
                "request_data": audit_log.request_data,
            },
            ensure_ascii=False,
        )

    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[AuditLog]:
        """監査ログ検索"""
        # 簡易実装（実運用では専用DBを使用）
        logs = []

        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if "AUDIT" in line:
                        try:
                            # JSON部分を抽出
                            json_start = line.find("{")
                            if json_start != -1:
                                json_data = json.loads(line[json_start:])

                                # フィルタリング
                                log_time = datetime.fromisoformat(json_data["timestamp"])

                                if start_date and log_time < start_date:
                                    continue
                                if end_date and log_time > end_date:
                                    continue
                                if user_id and json_data["user_id"] != user_id:
                                    continue
                                if action and json_data["action"] != action:
                                    continue

                                # AuditLogオブジェクト作成
                                audit_log = AuditLog(
                                    log_id=json_data["log_id"],
                                    user_id=json_data["user_id"],
                                    action=json_data["action"],
                                    resource=json_data["resource"],
                                    timestamp=log_time,
                                    ip_address=json_data.get("ip_address"),
                                    user_agent=json_data.get("user_agent"),
                                    success=json_data["success"],
                                    error_message=json_data.get("error_message"),
                                    request_data=json_data.get("request_data"),
                                )
                                logs.append(audit_log)

                        except json.JSONDecodeError:
                            continue

        except FileNotFoundError:
            logger.warning("監査ログファイルが見つかりません")

        return sorted(logs, key=lambda x: x.timestamp, reverse=True)


class DataEncryption:
    """データ暗号化"""

    def __init__(self, key: Optional[bytes] = None):
        try:
            from cryptography.fernet import Fernet

            self.cipher_suite = Fernet(key or Fernet.generate_key())
            self.encryption_available = True
        except ImportError:
            logger.warning("cryptographyライブラリが未インストール: データ暗号化無効")
            self.encryption_available = False

    def encrypt_data(self, data: str) -> str:
        """データ暗号化"""
        if not self.encryption_available:
            return data  # 暗号化不可の場合はプレーンテキスト

        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return encrypted_data.decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """データ復号化"""
        if not self.encryption_available:
            return encrypted_data

        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"復号化エラー: {e}")
            raise

    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """機密データハッシュ化"""
        if salt is None:
            salt = secrets.token_urlsafe(16)

        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            data.encode(),
            salt.encode(),
            100000,  # iterations
        )

        return hashed.hex(), salt


class SecurityManager:
    """セキュリティ管理システム"""

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
    ):
        self.config = config or OptimizationConfig()
        self.security_level = security_level

        # コンポーネント初期化
        self.token_manager = TokenManager()
        self.permission_manager = PermissionManager()
        self.audit_logger = AuditLogger()
        self.data_encryption = DataEncryption()

        # ユーザーストレージ（実運用では専用DB使用）
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # デフォルト管理者ユーザー作成
        self._create_default_admin()

        logger.info(f"セキュリティ管理システム初期化完了 (レベル: {security_level.value})")

    def _create_default_admin(self):
        """デフォルト管理者ユーザー作成"""
        admin_user = User(
            user_id="admin_001",
            username="admin",
            access_level=AccessLevel.ADMINISTRATOR,
            created_at=datetime.utcnow(),
            is_active=True,
        )
        admin_user.api_key = self.token_manager.generate_api_key(admin_user)
        self.users[admin_user.user_id] = admin_user

        logger.info("デフォルト管理者ユーザー作成完了")

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """ユーザー認証"""
        # 簡易認証実装（実運用では適切なパスワードハッシュ検証）
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break

        if not user:
            self.audit_logger.log_action(
                user_id="unknown",
                action="login_attempt",
                resource="authentication",
                success=False,
                error_message="ユーザーが見つかりません",
                ip_address=ip_address,
                user_agent=user_agent,
            )
            return None

        # パスワード検証（環境変数または設定ファイルから取得）
        admin_password = os.getenv("ADMIN_PASSWORD", "default_admin_password")
        if password != admin_password:  # 実運用では適切なハッシュ検証
            self.audit_logger.log_action(
                user_id=user.user_id,
                action="login_attempt",
                resource="authentication",
                success=False,
                error_message="パスワードが正しくありません",
                ip_address=ip_address,
                user_agent=user_agent,
            )
            return None

        # 認証成功
        user.last_login = datetime.utcnow()
        access_token = self.token_manager.generate_access_token(user)
        refresh_token = self.token_manager.generate_refresh_token(user)

        # セッション管理
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip_address": ip_address,
            "user_agent": user_agent,
        }

        self.audit_logger.log_action(
            user_id=user.user_id,
            action="login_success",
            resource="authentication",
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return access_token, refresh_token

    def authorize_action(
        self, token: str, action: str, resource: str, ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[User]]:
        """アクション認可"""
        # トークン検証
        payload = self.token_manager.verify_token(token)
        if not payload:
            return False, None

        user_id = payload.get("user_id")
        if not user_id or user_id not in self.users:
            return False, None

        user = self.users[user_id]
        if not user.is_active:
            return False, None

        # 権限チェック
        access_level = AccessLevel(payload.get("access_level"))
        if not self.permission_manager.has_permission(access_level, action):
            self.audit_logger.log_action(
                user_id=user_id,
                action=action,
                resource=resource,
                success=False,
                error_message="権限不足",
                ip_address=ip_address,
            )
            return False, None

        # 認可成功
        self.audit_logger.log_action(
            user_id=user_id,
            action=action,
            resource=resource,
            success=True,
            ip_address=ip_address,
        )

        return True, user

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """APIキー検証"""
        for user in self.users.values():
            if user.api_key and user.api_key == api_key and user.is_active:
                return user
        return None

    def create_user(self, username: str, access_level: AccessLevel, creator_user_id: str) -> User:
        """ユーザー作成"""
        user_id = f"user_{secrets.token_urlsafe(8)}"
        user = User(
            user_id=user_id,
            username=username,
            access_level=access_level,
            created_at=datetime.utcnow(),
            is_active=True,
        )
        user.api_key = self.token_manager.generate_api_key(user)

        self.users[user_id] = user

        self.audit_logger.log_action(
            user_id=creator_user_id,
            action="create_user",
            resource=f"user:{user_id}",
            success=True,
            request_data={"username": username, "access_level": access_level.value},
        )

        return user

    def get_security_status(self) -> Dict[str, Any]:
        """セキュリティ状態取得"""
        return {
            "security_level": self.security_level.value,
            "encryption_enabled": self.data_encryption.encryption_available,
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "active_sessions": len(self.active_sessions),
            "audit_logs_enabled": True,
            "token_expiry_hours": self.token_manager.token_expiry.total_seconds() / 3600,
            "last_security_check": datetime.utcnow().isoformat(),
        }

    def cleanup_expired_sessions(self):
        """期限切れセッション削除"""
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data["last_activity"]
            if current_time - last_activity > timedelta(hours=24):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            logger.info(f"期限切れセッション削除: {len(expired_sessions)}件")

    def get_api_key(self, env_var_name: str) -> Optional[str]:
        """安全なAPIキー取得"""
        try:
            # 環境変数から取得
            api_key = os.environ.get(env_var_name)

            if not api_key:
                logger.debug(f"環境変数が設定されていません: {env_var_name}")
                return None

            # 基本的な形式検証
            if len(api_key.strip()) == 0:
                logger.warning(f"空のAPIキーが設定されています: {env_var_name}")
                return None

            if len(api_key) < 8:
                logger.warning(f"APIキーが短すぎます: {env_var_name}")
                return None

            if len(api_key) > 500:
                logger.warning(f"APIキーが長すぎます: {env_var_name}")
                return None

            # 監査ログ記録
            self.audit_logger.log_action(
                user_id="system",
                action="api_key_access",
                resource=f"env_var:{env_var_name}",
                success=True,
            )

            logger.debug(f"APIキー取得成功: {env_var_name}")
            return api_key.strip()

        except Exception as e:
            # 監査ログ記録（失敗）
            self.audit_logger.log_action(
                user_id="system",
                action="api_key_access",
                resource=f"env_var:{env_var_name}",
                success=False,
                error_message=str(e),
            )
            logger.error(f"APIキー取得エラー: {env_var_name}, error={e}")
            return None

    def generate_security_report(self, days: int = 30) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # 監査ログ取得
        audit_logs = self.audit_logger.get_audit_logs(start_date, end_date)

        # 統計計算
        total_actions = len(audit_logs)
        successful_actions = len([log for log in audit_logs if log.success])
        failed_actions = total_actions - successful_actions

        # アクション別集計
        action_counts = {}
        for log in audit_logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1

        # ユーザー別集計
        user_activity = {}
        for log in audit_logs:
            user_activity[log.user_id] = user_activity.get(log.user_id, 0) + 1

        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "summary": {
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "success_rate": successful_actions / total_actions if total_actions > 0 else 0,
            },
            "action_breakdown": action_counts,
            "user_activity": user_activity,
            "security_level": self.security_level.value,
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "generated_at": datetime.utcnow().isoformat(),
        }


# セキュリティデコレータ
def require_authentication(action: str, resource: str = "system"):
    """認証必須デコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # セキュリティマネージャー取得（グローバルインスタンスまたは引数から）
            security_manager = kwargs.pop("security_manager", None)
            token = kwargs.pop("auth_token", None)

            if not security_manager or not token:
                raise PermissionError("認証情報が不足しています")

            # 認可チェック
            authorized, user = security_manager.authorize_action(token, action, resource)
            if not authorized:
                raise PermissionError(f"アクション '{action}' への権限がありません")

            # 認証済みユーザー情報を追加
            kwargs["authenticated_user"] = user
            return func(*args, **kwargs)

        return wrapper

    return decorator


# グローバルセキュリティマネージャー
_global_security_manager: Optional[SecurityManager] = None


def get_security_manager(
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
) -> SecurityManager:
    """グローバルセキュリティマネージャー取得"""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager(security_level=security_level)
    return _global_security_manager
