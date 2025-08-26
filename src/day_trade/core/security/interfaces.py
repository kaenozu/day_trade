#!/usr/bin/env python3
"""
セキュリティサービスのインターフェース定義
Issue #918 項目9対応: セキュリティ強化

全セキュリティサービスのインターフェース抽象クラス定義
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union

from .types import (
    ValidationResult, AuthenticationResult, RateLimitInfo, SecurityEvent,
    ActionType, ThreatLevel
)


class IInputValidationService(ABC):
    """入力検証サービスインターフェース"""

    @abstractmethod
    def validate_string(self, value: str, max_length: int = 1000,
                       allow_special_chars: bool = False) -> ValidationResult:
        """文字列検証"""
        pass

    @abstractmethod
    def validate_number(self, value: Union[str, int, float],
                       min_value: float = None, max_value: float = None) -> ValidationResult:
        """数値検証"""
        pass

    @abstractmethod
    def validate_email(self, email: str) -> ValidationResult:
        """メールアドレス検証"""
        pass

    @abstractmethod
    def validate_ip_address(self, ip: str) -> ValidationResult:
        """IPアドレス検証"""
        pass

    @abstractmethod
    def sanitize_sql_input(self, value: str) -> ValidationResult:
        """SQL入力サニタイゼーション"""
        pass

    @abstractmethod
    def validate_file_path(self, path: str, allowed_extensions: Set[str] = None) -> ValidationResult:
        """ファイルパス検証"""
        pass


class IAuthenticationService(ABC):
    """認証サービスインターフェース"""

    @abstractmethod
    def authenticate_user(self, username: str, password: str) -> AuthenticationResult:
        """ユーザー認証"""
        pass

    @abstractmethod
    def create_session(self, user_id: str, permissions: Set[str]) -> AuthenticationResult:
        """セッション作成"""
        pass

    @abstractmethod
    def validate_session(self, session_token: str) -> AuthenticationResult:
        """セッション検証"""
        pass

    @abstractmethod
    def revoke_session(self, session_token: str) -> bool:
        """セッション無効化"""
        pass

    @abstractmethod
    def hash_password(self, password: str) -> str:
        """パスワードハッシュ化"""
        pass

    @abstractmethod
    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワード検証"""
        pass


class IAuthorizationService(ABC):
    """認可サービスインターフェース"""

    @abstractmethod
    def check_permission(self, user_id: str, action: ActionType, resource: str = None) -> bool:
        """権限チェック"""
        pass

    @abstractmethod
    def grant_permission(self, user_id: str, permission: str) -> bool:
        """権限付与"""
        pass

    @abstractmethod
    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """権限剥奪"""
        pass

    @abstractmethod
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """ユーザー権限取得"""
        pass


class IRateLimitService(ABC):
    """レート制限サービスインターフェース"""

    @abstractmethod
    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitInfo:
        """レート制限チェック"""
        pass

    @abstractmethod
    def increment_counter(self, key: str) -> int:
        """カウンター増分"""
        pass

    @abstractmethod
    def reset_counter(self, key: str) -> bool:
        """カウンターリセット"""
        pass


class ISecurityAuditService(ABC):
    """セキュリティ監査サービスインターフェース"""

    @abstractmethod
    def log_security_event(self, event: SecurityEvent) -> str:
        """セキュリティイベントログ"""
        pass

    @abstractmethod
    def get_security_events(self, start_time: datetime = None,
                          end_time: datetime = None,
                          threat_level: ThreatLevel = None) -> List[SecurityEvent]:
        """セキュリティイベント取得"""
        pass

    @abstractmethod
    def analyze_threats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """脅威分析"""
        pass

    @abstractmethod
    def generate_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        pass