#!/usr/bin/env python3
"""
セキュリティ関連の型定義とEnum
Issue #918 項目9対応: セキュリティ強化

基本的な型、Enum、データクラスの定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum


class SecurityLevel(Enum):
    """セキュリティレベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """脅威レベル"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """アクション種別"""
    LOGIN = "login"
    LOGOUT = "logout"
    API_ACCESS = "api_access"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    TRADE_EXECUTION = "trade_execution"
    FILE_ACCESS = "file_access"
    SYSTEM_ADMIN = "system_admin"


@dataclass
class SecurityEvent:
    """セキュリティイベント"""
    event_id: str
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    action: Optional[ActionType] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    sanitized_value: Any = None
    error_message: Optional[str] = None
    threat_level: ThreatLevel = ThreatLevel.INFO


@dataclass
class AuthenticationResult:
    """認証結果"""
    is_authenticated: bool
    user_id: Optional[str] = None
    session_token: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class RateLimitInfo:
    """レート制限情報"""
    limit: int
    window_seconds: int
    current_count: int
    reset_time: datetime
    is_exceeded: bool = False