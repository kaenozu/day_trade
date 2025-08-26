#!/usr/bin/env python3
"""
アクセス制御システム - メインのアクセス制御管理

このモジュールは、アクセス制御システムのメインクラスである
AccessControlManagerを提供します。ユーザー管理、認証、
セッション管理の基本機能を担当します。
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .enums import (
    AccessLogEntry,
    AuthenticationMethod,
    Session,
    SessionStatus,
    User,
    UserRole,
)
from .mfa_manager import MFAManager
from .role_manager import RolePermissionManager
from .validators import PasswordValidator

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AccessControlManager:
    """
    アクセス制御管理システム
    
    ユーザー管理、認証、認可、セッション管理、
    監査ログを統合したセキュリティ管理システム
    """

    def __init__(
        self,
        storage_path: str = "security/access_control",
        session_timeout_minutes: int = 30,
    ):
        """
        初期化
        
        Args:
            storage_path: データ保存パス
            session_timeout_minutes: セッションタイムアウト（分）
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 管理システム初期化
        self.password_validator = PasswordValidator()
        self.role_permission_manager = RolePermissionManager()
        self.mfa_manager = MFAManager()

        # データストレージ
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.access_logs: List[AccessLogEntry] = []

        # 設定
        self.session_timeout_minutes = session_timeout_minutes
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30

        # セキュリティ設定
        self.require_mfa_for_admin = True
        self.password_expiry_days = 90
        self.session_rotation_enabled = True

        # データ読み込み
        self._load_users()
        self._load_access_logs()

        # デフォルト管理者作成
        self._create_default_admin()

        logger.info("AccessControlManager初期化完了")

    def _load_users(self):
        """ユーザーデータ読み込み"""
        users_file = self.storage_path / "users.json"
        if users_file.exists():
            try:
                with open(users_file, encoding="utf-8") as f:
                    data = json.load(f)

                for user_data in data.get("users", []):
                    user = User(
                        user_id=user_data["user_id"],
                        username=user_data["username"],
                        email=user_data["email"],
                        role=UserRole(user_data.get("role", "viewer")),
                        password_hash=user_data.get("password_hash"),
                        salt=user_data.get("salt"),
                        totp_secret=user_data.get("totp_secret"),
                        mfa_enabled=user_data.get("mfa_enabled", False),
                        mfa_methods=[
                            AuthenticationMethod(m)
                            for m in user_data.get("mfa_methods", [])
                        ],
                        backup_codes=user_data.get("backup_codes", []),
                        is_active=user_data.get("is_active", True),
                        is_locked=user_data.get("is_locked", False),
                        failed_login_attempts=user_data.get("failed_login_attempts", 0),
                        last_login=(
                            datetime.fromisoformat(user_data["last_login"])
                            if user_data.get("last_login")
                            else None
                        ),
                        created_at=datetime.fromisoformat(
                            user_data.get("created_at", datetime.utcnow().isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            user_data.get("updated_at", datetime.utcnow().isoformat())
                        ),
                        password_changed_at=(
                            datetime.fromisoformat(user_data["password_changed_at"])
                            if user_data.get("password_changed_at")
                            else None
                        ),
                        require_password_change=user_data.get(
                            "require_password_change", False
                        ),
                        session_timeout_minutes=user_data.get(
                            "session_timeout_minutes", self.session_timeout_minutes
                        ),
                        allowed_ip_addresses=user_data.get("allowed_ip_addresses", []),
                    )
                    self.users[user.user_id] = user

                logger.info(f"ユーザーデータ読み込み完了: {len(self.users)}ユーザー")

            except Exception as e:
                logger.error(f"ユーザーデータ読み込みエラー: {e}")

    def _save_users(self):
        """ユーザーデータ保存"""
        users_file = self.storage_path / "users.json"
        try:
            data = {
                "users": [user.to_dict() for user in self.users.values()],
                "last_updated": datetime.utcnow().isoformat(),
            }

            # センシティブ情報は除外
            for user_data in data["users"]:
                if "password_hash" in user_data:
                    user_data["password_hash"] = "***"
                if "salt" in user_data:
                    user_data["salt"] = "***"
                if "totp_secret" in user_data:
                    user_data["totp_secret"] = "***"
                if "backup_codes" in user_data:
                    user_data["backup_codes"] = ["***"] * len(user_data["backup_codes"])

            with open(users_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"ユーザーデータ保存エラー: {e}")

    def _load_access_logs(self):
        """アクセスログ読み込み"""
        logs_file = self.storage_path / "access_logs.jsonl"
        if logs_file.exists():
            try:
                with open(logs_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            log_data = json.loads(line)
                            log_entry = AccessLogEntry(
                                timestamp=datetime.fromisoformat(log_data["timestamp"]),
                                user_id=log_data["user_id"],
                                session_id=log_data.get("session_id"),
                                action=log_data["action"],
                                resource=log_data["resource"],
                                ip_address=log_data["ip_address"],
                                user_agent=log_data["user_agent"],
                                success=log_data["success"],
                                details=log_data.get("details", {}),
                            )
                            self.access_logs.append(log_entry)

                # メモリ使用量制限のため最新1000件のみ保持
                if len(self.access_logs) > 1000:
                    self.access_logs = self.access_logs[-1000:]

                logger.info(
                    f"アクセスログ読み込み完了: {len(self.access_logs)}エントリ"
                )

            except Exception as e:
                logger.error(f"アクセスログ読み込みエラー: {e}")

    def _create_default_admin(self):
        """デフォルト管理者作成"""
        admin_id = "admin-default"
        if admin_id not in self.users:
            # 初期パスワードは環境変数から取得、なければ生成
            default_password = os.getenv("DAYTRADE_ADMIN_PASSWORD", "TempAdmin123!")

            admin_user = self.create_user(
                username="admin",
                email="admin@daytrade.local",
                password=default_password,
                role=UserRole.SUPER_ADMIN,
                require_password_change=True,
            )

            if admin_user:
                logger.warning("デフォルト管理者アカウントを作成しました: admin")
                logger.warning("初期パスワードを変更してください")

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        require_password_change: bool = False,
    ) -> Optional[User]:
        """
        ユーザー作成
        
        Args:
            username: ユーザー名
            email: メールアドレス
            password: パスワード
            role: ユーザーロール
            require_password_change: パスワード変更を必須とするか
            
        Returns:
            Optional[User]: 作成されたユーザー（失敗時はNone）
        """
        # 重複チェック
        for user in self.users.values():
            if user.username == username or user.email == email:
                logger.error(
                    f"ユーザー名またはメールアドレスが既に使用されています: {username}"
                )
                return None

        # パスワード強度チェック
        is_valid, errors = self.password_validator.validate(password)
        if not is_valid:
            logger.error(f"パスワードが要件を満たしていません: {', '.join(errors)}")
            return None

        # ユーザーID生成
        user_id = f"user-{secrets.token_hex(8)}"

        # パスワードハッシュ化
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        # ユーザー作成
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            password_hash=password_hash,
            salt=salt,
            password_changed_at=datetime.utcnow(),
            require_password_change=require_password_change,
        )

        self.users[user_id] = user
        self._save_users()

        logger.info(f"ユーザー作成完了: {username} ({role.value})")
        return user

    def create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """
        セッション作成
        
        Args:
            user: ユーザーオブジェクト
            ip_address: IPアドレス
            user_agent: ユーザーエージェント
            
        Returns:
            Session: 作成されたセッション
        """
        session_id = secrets.token_urlsafe(32)

        # セッション有効期限設定
        expires_at = datetime.utcnow() + timedelta(minutes=user.session_timeout_minutes)

        # 権限設定
        permissions = self.role_permission_manager.get_permissions(user.role)

        # リスクスコア計算（security_manager.pyから移動）
        risk_score = self._calculate_risk_score(user, ip_address, user_agent)

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            mfa_verified=user.mfa_enabled,
            permissions=permissions,
            risk_score=risk_score,
        )

        self.sessions[session_id] = session

        self._log_access(
            user.user_id,
            session_id,
            "session",
            "create",
            ip_address,
            user_agent,
            True,
            {"risk_score": risk_score, "permissions_count": len(permissions)},
        )

        logger.info(f"セッション作成: {user.username} (リスクスコア: {risk_score:.2f})")
        return session

    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        セッション検証
        
        Args:
            session_id: セッションID
            
        Returns:
            Optional[Session]: 有効なセッション（無効時はNone）
        """
        session = self.sessions.get(session_id)
        if not session or not session.is_valid():
            if session:
                session.status = SessionStatus.EXPIRED
                self._log_access(
                    session.user_id,
                    session_id,
                    "session",
                    "expired",
                    session.ip_address,
                    session.user_agent,
                    False,
                    {"reason": "session_expired"},
                )
            return None

        # 最終アクティビティ更新
        session.last_activity = datetime.utcnow()

        return session

    def terminate_session(self, session_id: str, reason: str = "user_logout"):
        """
        セッション終了
        
        Args:
            session_id: 終了するセッションID
            reason: 終了理由
        """
        session = self.sessions.get(session_id)
        if session:
            session.status = SessionStatus.TERMINATED

            self._log_access(
                session.user_id,
                session_id,
                "session",
                "terminate",
                session.ip_address,
                session.user_agent,
                True,
                {"reason": reason},
            )

            del self.sessions[session_id]

    def _hash_password(self, password: str, salt: str) -> str:
        """
        パスワードハッシュ化
        
        Args:
            password: 平文パスワード
            salt: ソルト
            
        Returns:
            str: ハッシュ化されたパスワード
        """
        return hashlib.pbkdf2_hex(
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # 反復回数
            32,  # ハッシュ長
        )

    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        パスワード検証
        
        Args:
            password: 検証するパスワード
            stored_hash: 保存されているハッシュ
            salt: ソルト
            
        Returns:
            bool: 検証結果
        """
        return hmac.compare_digest(self._hash_password(password, salt), stored_hash)

    def _calculate_risk_score(
        self, user: User, ip_address: str, user_agent: str
    ) -> float:
        """
        リスクスコア計算
        
        Args:
            user: ユーザーオブジェクト
            ip_address: IPアドレス
            user_agent: ユーザーエージェント
            
        Returns:
            float: リスクスコア（0.0-1.0）
        """
        risk_score = 0.0

        # MFA未使用でのリスク増加
        if not user.mfa_enabled and user.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            risk_score += 0.3

        # 管理者ロールでのリスク
        if user.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            risk_score += 0.2

        # IP制限未設定でのリスク
        if not user.allowed_ip_addresses:
            risk_score += 0.1

        # パスワード変更期限切れ
        if (
            user.password_changed_at
            and datetime.utcnow() - user.password_changed_at
            > timedelta(days=self.password_expiry_days)
        ):
            risk_score += 0.2

        return min(risk_score, 1.0)

    def _log_access(
        self,
        user_id: str,
        session_id: Optional[str],
        action: str,
        resource: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        アクセスログ記録
        
        Args:
            user_id: ユーザーID
            session_id: セッションID
            action: アクション
            resource: リソース
            ip_address: IPアドレス
            user_agent: ユーザーエージェント
            success: 成功/失敗
            details: 詳細情報
        """
        log_entry = AccessLogEntry(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
        )

        self.access_logs.append(log_entry)

        # ファイルに即座に記録
        logs_file = self.storage_path / "access_logs.jsonl"
        try:
            with open(logs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"アクセスログ記録エラー: {e}")

    def setup_mfa(self, user_id: str) -> Tuple[bool, str, Optional[str]]:
        """
        MFA設定
        
        Args:
            user_id: ユーザーID
            
        Returns:
            Tuple[bool, str, Optional[str]]: (成功/失敗, メッセージ, QRコード)
        """
        user = self.users.get(user_id)
        if not user:
            return False, "ユーザーが見つかりません", None

        if user.mfa_enabled:
            return False, "MFAは既に有効です", None

        # TOTP秘密鍵生成
        secret = self.mfa_manager.generate_totp_secret()
        user.totp_secret = secret

        # QRコード生成
        qr_code = self.mfa_manager.generate_qr_code(user.username, secret)

        # バックアップコード生成
        backup_codes = self.mfa_manager.generate_backup_codes()
        user.backup_codes = backup_codes

        # MFA方法追加
        user.mfa_methods.append(AuthenticationMethod.TOTP)
        user.updated_at = datetime.utcnow()

        self._save_users()

        logger.info(f"MFA設定開始: {user.username}")
        return True, "MFA設定が開始されました", qr_code

    def enable_mfa(self, user_id: str, totp_code: str) -> Tuple[bool, str]:
        """
        MFA有効化
        
        Args:
            user_id: ユーザーID
            totp_code: TOTPコード
            
        Returns:
            Tuple[bool, str]: (成功/失敗, メッセージ)
        """
        user = self.users.get(user_id)
        if not user or not user.totp_secret:
            return False, "MFA設定が見つかりません"

        # TOTPコード検証
        if not self.mfa_manager.verify_totp(user.totp_secret, totp_code):
            return False, "TOTPコードが正しくありません"

        # MFA有効化
        user.mfa_enabled = True
        user.updated_at = datetime.utcnow()
        self._save_users()

        logger.info(f"MFA有効化完了: {user.username}")
        return True, "MFAが有効化されました"