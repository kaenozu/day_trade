#!/usr/bin/env python3
"""
アクセス制御システム - 列挙型とデータクラス定義

このモジュールは、アクセス制御システムで使用される
列挙型とデータクラスを定義します。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


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
                self.password_changed_at.isoformat()
                if self.password_changed_at
                else None
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
        """辞書形式変換"""
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
        """辞書形式変換"""
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