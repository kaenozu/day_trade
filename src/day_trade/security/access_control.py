#!/usr/bin/env python3
"""
アクセス制御・認証システム
Issue #419: セキュリティ強化 - アクセス制御と認証システムの強化

ロールベースアクセス制御(RBAC)、多要素認証(MFA)、
セッション管理、監査ログを統合したアクセス制御システム
"""

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pyotp
import qrcode

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class UserRole(Enum):
    """ユーザーロール"""

    GUEST = "guest"
    VIEWER = "viewer"
    TRADER = "trader"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """権限"""

    # データアクセス
    VIEW_MARKET_DATA = "view_market_data"
    VIEW_HISTORICAL_DATA = "view_historical_data"
    VIEW_REPORTS = "view_reports"

    # 取引関連
    PLACE_ORDERS = "place_orders"
    MODIFY_ORDERS = "modify_orders"
    CANCEL_ORDERS = "cancel_orders"
    VIEW_POSITIONS = "view_positions"

    # 分析機能
    RUN_ANALYSIS = "run_analysis"
    CREATE_STRATEGIES = "create_strategies"
    MODIFY_STRATEGIES = "modify_strategies"
    BACKTEST_STRATEGIES = "backtest_strategies"

    # システム管理
    MANAGE_USERS = "manage_users"
    MANAGE_SETTINGS = "manage_settings"
    VIEW_LOGS = "view_logs"
    MANAGE_SECURITY = "manage_security"

    # 高リスク操作
    EXECUTE_BULK_OPERATIONS = "execute_bulk_operations"
    MODIFY_RISK_LIMITS = "modify_risk_limits"
    ACCESS_API_KEYS = "access_api_keys"
    EXPORT_DATA = "export_data"


class AuthenticationMethod(Enum):
    """認証方法"""

    PASSWORD = "password"
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"


class SessionStatus(Enum):
    """セッションステータス"""

    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPICIOUS = "suspicious"


@dataclass
class User:
    """ユーザー情報"""

    user_id: str
    username: str
    email: str
    role: UserRole = UserRole.VIEWER

    # 認証情報
    password_hash: Optional[str] = None
    salt: Optional[str] = None
    totp_secret: Optional[str] = None

    # MFA設定
    mfa_enabled: bool = False
    mfa_methods: List[AuthenticationMethod] = field(default_factory=list)
    backup_codes: List[str] = field(default_factory=list)

    # アカウント状態
    is_active: bool = True
    is_locked: bool = False
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None

    # 追跡情報
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    password_changed_at: Optional[datetime] = None

    # セキュリティ設定
    require_password_change: bool = False
    session_timeout_minutes: int = 30
    allowed_ip_addresses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "mfa_enabled": self.mfa_enabled,
            "mfa_methods": [method.value for method in self.mfa_methods],
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "failed_login_attempts": self.failed_login_attempts,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "password_changed_at": (
                self.password_changed_at.isoformat() if self.password_changed_at else None
            ),
            "require_password_change": self.require_password_change,
            "session_timeout_minutes": self.session_timeout_minutes,
            "allowed_ip_addresses": self.allowed_ip_addresses,
        }


@dataclass
class Session:
    """セッション情報"""

    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    status: SessionStatus = SessionStatus.ACTIVE

    # セキュリティ情報
    mfa_verified: bool = False
    permissions: Set[Permission] = field(default_factory=set)
    risk_score: float = 0.0

    def is_valid(self) -> bool:
        """セッション有効性チェック"""
        if self.status != SessionStatus.ACTIVE:
            return False

        return not (self.expires_at and datetime.utcnow() > self.expires_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "mfa_verified": self.mfa_verified,
            "permissions": [perm.value for perm in self.permissions],
            "risk_score": self.risk_score,
        }


@dataclass
class AccessLogEntry:
    """アクセスログエントリ"""

    timestamp: datetime
    user_id: str
    session_id: Optional[str]
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "resource": self.resource,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details,
        }


class PasswordValidator:
    """パスワード強度バリデーター"""

    def __init__(self):
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_symbols = True
        self.common_passwords = self._load_common_passwords()

    def _load_common_passwords(self) -> Set[str]:
        """一般的な脆弱パスワードリスト"""
        return {
            "password",
            "123456",
            "password123",
            "admin",
            "qwerty",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
            "master",
            "password1",
            "123456789",
            "12345678",
            "12345",
            "1234567890",
            "trading",
            "finance",
            "money",
            "profit",
            "investment",
            "daytrading",
            "stock",
            "market",
            "trader",
            "analyst",
        }

    def validate(self, password: str) -> tuple[bool, List[str]]:
        """パスワード強度チェック"""
        errors = []

        if len(password) < self.min_length:
            errors.append(f"パスワードは{self.min_length}文字以上である必要があります")

        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("大文字を含む必要があります")

        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("小文字を含む必要があります")

        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("数字を含む必要があります")

        if self.require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("記号を含む必要があります")

        if password.lower() in self.common_passwords:
            errors.append("一般的すぎるパスワードです")

        # 連続文字チェック
        if self._has_sequential_chars(password):
            errors.append("連続した文字の使用は避けてください")

        return len(errors) == 0, errors

    def _has_sequential_chars(self, password: str) -> bool:
        """連続文字チェック"""
        for i in range(len(password) - 2):
            if (
                ord(password[i + 1]) == ord(password[i]) + 1
                and ord(password[i + 2]) == ord(password[i]) + 2
            ):
                return True
        return False


class RolePermissionManager:
    """ロール権限管理"""

    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()

    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """ロール別権限の初期化"""
        return {
            UserRole.GUEST: set(),
            UserRole.VIEWER: {Permission.VIEW_MARKET_DATA, Permission.VIEW_REPORTS},
            UserRole.TRADER: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.PLACE_ORDERS,
                Permission.MODIFY_ORDERS,
                Permission.CANCEL_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.RUN_ANALYSIS,
            },
            UserRole.ANALYST: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.RUN_ANALYSIS,
                Permission.CREATE_STRATEGIES,
                Permission.MODIFY_STRATEGIES,
                Permission.BACKTEST_STRATEGIES,
                Permission.EXPORT_DATA,
            },
            UserRole.ADMIN: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.PLACE_ORDERS,
                Permission.MODIFY_ORDERS,
                Permission.CANCEL_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.RUN_ANALYSIS,
                Permission.CREATE_STRATEGIES,
                Permission.MODIFY_STRATEGIES,
                Permission.BACKTEST_STRATEGIES,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SETTINGS,
                Permission.VIEW_LOGS,
                Permission.EXPORT_DATA,
            },
            UserRole.SUPER_ADMIN: set(Permission),  # 全権限
        }

    def get_permissions(self, role: UserRole) -> Set[Permission]:
        """ロールの権限取得"""
        return self.role_permissions.get(role, set()).copy()

    def has_permission(self, role: UserRole, permission: Permission) -> bool:
        """権限チェック"""
        return permission in self.role_permissions.get(role, set())


class MFAManager:
    """多要素認証管理"""

    def __init__(self):
        self.issuer_name = "DayTrade Security"

    def generate_totp_secret(self) -> str:
        """TOTP秘密鍵生成"""
        return pyotp.random_base32()

    def generate_qr_code(self, username: str, secret: str) -> str:
        """QRコード生成（Base64エンコード済み画像）"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username, issuer_name=self.issuer_name
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return base64.b64encode(img_buffer.getvalue()).decode()

    def verify_totp(self, secret: str, token: str) -> bool:
        """TOTP検証"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # 30秒の許容範囲
        except Exception:
            return False

    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """バックアップコード生成"""
        return [secrets.token_hex(4).upper() for _ in range(count)]


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
                            AuthenticationMethod(m) for m in user_data.get("mfa_methods", [])
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
                        require_password_change=user_data.get("require_password_change", False),
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

                logger.info(f"アクセスログ読み込み完了: {len(self.access_logs)}エントリ")

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
        """ユーザー作成"""
        # 重複チェック
        for user in self.users.values():
            if user.username == username or user.email == email:
                logger.error(f"ユーザー名またはメールアドレスが既に使用されています: {username}")
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

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        totp_code: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[User]]:
        """ユーザー認証"""
        # ユーザー検索
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            self._log_access(
                "UNKNOWN_USER",
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "user_not_found", "username": username},
            )
            return False, "認証に失敗しました", None

        # アカウント状態チェック
        if not user.is_active:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "account_inactive"},
            )
            return False, "アカウントが無効です", None

        if user.is_locked:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "account_locked"},
            )
            return False, "アカウントがロックされています", None

        # IP制限チェック
        if user.allowed_ip_addresses and ip_address not in user.allowed_ip_addresses:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "ip_restricted"},
            )
            return False, "許可されていないIPアドレスです", None

        # パスワード検証
        if not self._verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1

            # アカウントロック判定
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.is_locked = True
                logger.warning(
                    f"アカウントロック: {username} (失敗回数: {user.failed_login_attempts})"
                )

            self._save_users()
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {
                    "reason": "invalid_password",
                    "failed_attempts": user.failed_login_attempts,
                },
            )
            return False, "認証に失敗しました", None

        # MFA検証（有効な場合）
        if user.mfa_enabled and AuthenticationMethod.TOTP in user.mfa_methods:
            if not totp_code:
                return False, "TOTPコードが必要です", None

            if not self.mfa_manager.verify_totp(user.totp_secret, totp_code):
                self._log_access(
                    user.user_id,
                    None,
                    "authentication",
                    "mfa_failure",
                    ip_address,
                    user_agent,
                    False,
                    {"reason": "invalid_totp"},
                )
                return False, "TOTPコードが正しくありません", None

        # 認証成功
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        self._save_users()

        self._log_access(
            user.user_id,
            None,
            "authentication",
            "login",
            ip_address,
            user_agent,
            True,
            {"mfa_used": user.mfa_enabled},
        )

        logger.info(f"ユーザー認証成功: {username}")
        return True, "認証成功", user

    def create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """セッション作成"""
        session_id = secrets.token_urlsafe(32)

        # セッション有効期限設定
        expires_at = datetime.utcnow() + timedelta(minutes=user.session_timeout_minutes)

        # 権限設定
        permissions = self.role_permission_manager.get_permissions(user.role)

        # リスクスコア計算
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
        """セッション検証"""
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

    def check_permission(
        self, session: Session, permission: Permission, resource: Optional[str] = None
    ) -> bool:
        """権限チェック"""
        has_permission = permission in session.permissions

        self._log_access(
            session.user_id,
            session.session_id,
            "authorization",
            "check_permission",
            session.ip_address,
            session.user_agent,
            has_permission,
            {"permission": permission.value, "resource": resource},
        )

        return has_permission

    def terminate_session(self, session_id: str, reason: str = "user_logout"):
        """セッション終了"""
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

    def setup_mfa(self, user_id: str) -> tuple[bool, str, Optional[str]]:
        """MFA設定"""
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

    def enable_mfa(self, user_id: str, totp_code: str) -> tuple[bool, str]:
        """MFA有効化"""
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

    def _hash_password(self, password: str, salt: str) -> str:
        """パスワードハッシュ化"""
        return hashlib.pbkdf2_hex(
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # 反復回数
            32,  # ハッシュ長
        )

    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """パスワード検証"""
        return hmac.compare_digest(self._hash_password(password, salt), stored_hash)

    def _calculate_risk_score(self, user: User, ip_address: str, user_agent: str) -> float:
        """リスクスコア計算"""
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
        if user.password_changed_at and datetime.utcnow() - user.password_changed_at > timedelta(
            days=self.password_expiry_days
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
        """アクセスログ記録"""
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

    def get_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        now = datetime.utcnow()

        # ユーザー統計
        user_stats = {
            "total_users": len(self.users),
            "active_users": sum(1 for u in self.users.values() if u.is_active),
            "locked_users": sum(1 for u in self.users.values() if u.is_locked),
            "mfa_enabled_users": sum(1 for u in self.users.values() if u.mfa_enabled),
            "password_expiry_soon": sum(
                1
                for u in self.users.values()
                if u.password_changed_at
                and now - u.password_changed_at > timedelta(days=self.password_expiry_days - 7)
            ),
        }

        # セッション統計
        active_sessions = [s for s in self.sessions.values() if s.is_valid()]
        session_stats = {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "high_risk_sessions": sum(1 for s in active_sessions if s.risk_score > 0.5),
        }

        # ログ統計（過去24時間）
        recent_logs = [log for log in self.access_logs if log.timestamp > now - timedelta(hours=24)]

        log_stats = {
            "total_events_24h": len(recent_logs),
            "failed_logins_24h": sum(
                1 for log in recent_logs if log.action == "login" and not log.success
            ),
            "successful_logins_24h": sum(
                1 for log in recent_logs if log.action == "login" and log.success
            ),
        }

        report = {
            "report_id": f"access-control-report-{int(time.time())}",
            "generated_at": now.isoformat(),
            "user_statistics": user_stats,
            "session_statistics": session_stats,
            "log_statistics": log_stats,
            "recommendations": self._generate_security_recommendations(
                user_stats, session_stats, log_stats
            ),
        }

        return report

    def _generate_security_recommendations(
        self,
        user_stats: Dict[str, Any],
        session_stats: Dict[str, Any],
        log_stats: Dict[str, Any],
    ) -> List[str]:
        """セキュリティ推奨事項生成"""
        recommendations = []

        if user_stats["locked_users"] > 0:
            recommendations.append(
                f"🔒 {user_stats['locked_users']}個のアカウントがロックされています。調査してください。"
            )

        mfa_ratio = user_stats["mfa_enabled_users"] / max(user_stats["total_users"], 1)
        if mfa_ratio < 0.8:
            recommendations.append(
                f"🔐 MFA有効化率が{mfa_ratio:.1%}です。全ユーザーでのMFA有効化を推奨します。"
            )

        if user_stats["password_expiry_soon"] > 0:
            recommendations.append(
                f"⏰ {user_stats['password_expiry_soon']}ユーザーのパスワード期限が近づいています。"
            )

        if session_stats["high_risk_sessions"] > 0:
            recommendations.append(
                f"⚠️ {session_stats['high_risk_sessions']}個の高リスクセッションが検出されています。"
            )

        if log_stats["failed_logins_24h"] > 10:
            recommendations.append(
                f"🚨 過去24時間で{log_stats['failed_logins_24h']}回のログイン失敗が発生しています。攻撃の可能性があります。"
            )

        if not recommendations:
            recommendations.append("✅ アクセス制御システムは正常に稼働しています。")

        return recommendations


# Factory function
def create_access_control_manager(
    storage_path: str = "security/access_control", **config
) -> AccessControlManager:
    """AccessControlManagerファクトリ関数"""
    return AccessControlManager(storage_path=storage_path, **config)


if __name__ == "__main__":
    # テスト実行
    def main():
        print("=== Issue #419 アクセス制御・認証システムテスト ===")

        manager = None
        try:
            # アクセス制御システム初期化
            manager = create_access_control_manager()

            print("\n1. システム初期化状態")
            print(f"登録ユーザー数: {len(manager.users)}")
            print(f"アクティブセッション数: {len(manager.sessions)}")

            print("\n2. テストユーザー作成")
            test_user = manager.create_user(
                username="test_trader",
                email="trader@test.com",
                password="SecureTrading2024!",
                role=UserRole.TRADER,
            )

            if test_user:
                print(f"テストユーザー作成成功: {test_user.username} ({test_user.role.value})")

                print("\n3. MFA設定テスト")
                success, message, qr_code = manager.setup_mfa(test_user.user_id)
                if success:
                    print("MFA設定開始成功")
                    print(f"QRコード生成: {len(qr_code) if qr_code else 0} bytes")

                    # テスト用のTOTPコード生成
                    import pyotp

                    totp = pyotp.TOTP(test_user.totp_secret)
                    test_code = totp.now()

                    mfa_success, mfa_message = manager.enable_mfa(test_user.user_id, test_code)
                    print(f"MFA有効化: {mfa_success} - {mfa_message}")

                print("\n4. 認証テスト")
                auth_success, auth_message, auth_user = manager.authenticate_user(
                    username="test_trader",
                    password="SecureTrading2024!",
                    ip_address="127.0.0.1",
                    user_agent="TestClient/1.0",
                    totp_code=totp.now() if test_user.mfa_enabled else None,
                )

                print(f"認証結果: {auth_success} - {auth_message}")

                if auth_success and auth_user:
                    print("\n5. セッション作成・権限テスト")
                    session = manager.create_session(auth_user, "127.0.0.1", "TestClient/1.0")

                    print(f"セッション作成: {session.session_id[:16]}...")
                    print(f"権限数: {len(session.permissions)}")
                    print(f"リスクスコア: {session.risk_score:.2f}")

                    # 権限チェックテスト
                    can_trade = manager.check_permission(session, Permission.PLACE_ORDERS)
                    can_manage = manager.check_permission(session, Permission.MANAGE_USERS)

                    print(f"取引権限: {can_trade}")
                    print(f"管理権限: {can_manage}")

            print("\n6. セキュリティレポート生成")
            report = manager.get_security_report()

            print(f"レポートID: {report['report_id']}")
            print("ユーザー統計:")
            user_stats = report["user_statistics"]
            print(f"  総ユーザー数: {user_stats['total_users']}")
            print(f"  アクティブ: {user_stats['active_users']}")
            print(f"  MFA有効: {user_stats['mfa_enabled_users']}")

            print("セッション統計:")
            session_stats = report["session_statistics"]
            print(f"  アクティブセッション: {session_stats['active_sessions']}")
            print(f"  高リスクセッション: {session_stats['high_risk_sessions']}")

            print("推奨事項:")
            for rec in report["recommendations"]:
                print(f"  {rec}")

        except Exception as e:
            print(f"テスト実行エラー: {e}")
            import traceback

            traceback.print_exc()

        print("\n=== アクセス制御・認証システムテスト完了 ===")

    main()
