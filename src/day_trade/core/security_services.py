#!/usr/bin/env python3
"""
セキュリティサービス - 依存性注入版
Issue #918 項目9対応: セキュリティ強化

包括的なセキュリティ機能とセキュリティポリシーの実装
"""

import re
import hashlib
import secrets
import hmac
import time
import json
import ipaddress
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from enum import Enum
import threading
from pathlib import Path

from .dependency_injection import (
    IConfigurationService, ILoggingService, injectable, singleton, get_container
)
from ..utils.logging_config import get_context_logger


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


@singleton(IInputValidationService)
@injectable
class InputValidationService(IInputValidationService):
    """入力検証サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "InputValidationService")
        
        # 危険なパターン定義
        self._sql_injection_patterns = [
            r'(\bunion\b|\bselect\b|\binsert\b|\bdelete\b|\bupdate\b|\bdrop\b)',
            r'(--|\*\/|\*)',
            r'(\bor\b|\band\b)\s+\d+\s*=\s*\d+',
            r'(\bor\b|\band\b)\s+[\'"].*[\'"]',
        ]
        
        self._xss_patterns = [
            r'<\s*script[^>]*>.*?</\s*script\s*>',
            r'<\s*iframe[^>]*>.*?</\s*iframe\s*>',
            r'javascript:',
            r'on\w+\s*=',
        ]
        
        self._path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'\.\./\.\.',
            r'\.\.\\\.\.', 
        ]

    def validate_string(self, value: str, max_length: int = 1000, 
                       allow_special_chars: bool = False) -> ValidationResult:
        """文字列検証"""
        try:
            if not isinstance(value, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="Value must be a string",
                    threat_level=ThreatLevel.LOW
                )
            
            # 長さチェック
            if len(value) > max_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"String length exceeds maximum of {max_length}",
                    threat_level=ThreatLevel.MEDIUM
                )
            
            # XSS攻撃パターンチェック
            for pattern in self._xss_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    self.logger.warning(f"XSS pattern detected: {pattern}")
                    return ValidationResult(
                        is_valid=False,
                        error_message="Potentially malicious script detected",
                        threat_level=ThreatLevel.HIGH
                    )
            
            # 特殊文字チェック
            if not allow_special_chars:
                if re.search(r'[<>&"\']', value):
                    sanitized = re.sub(r'[<>&"\']', '', value)
                    return ValidationResult(
                        is_valid=True,
                        sanitized_value=sanitized,
                        error_message="Special characters removed",
                        threat_level=ThreatLevel.LOW
                    )
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=value,
                threat_level=ThreatLevel.INFO
            )
            
        except Exception as e:
            self.logger.error(f"String validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_number(self, value: Union[str, int, float], 
                       min_value: float = None, max_value: float = None) -> ValidationResult:
        """数値検証"""
        try:
            # 数値への変換試行
            if isinstance(value, str):
                # 危険な文字列パターンチェック
                if re.search(r'[^\d\.\-\+e]', value.lower()):
                    return ValidationResult(
                        is_valid=False,
                        error_message="Invalid characters in numeric string",
                        threat_level=ThreatLevel.MEDIUM
                    )
                numeric_value = float(value)
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message="Value must be numeric",
                    threat_level=ThreatLevel.LOW
                )
            
            # 範囲チェック
            if min_value is not None and numeric_value < min_value:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} is below minimum {min_value}",
                    threat_level=ThreatLevel.LOW
                )
            
            if max_value is not None and numeric_value > max_value:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} exceeds maximum {max_value}",
                    threat_level=ThreatLevel.LOW
                )
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=numeric_value,
                threat_level=ThreatLevel.INFO
            )
            
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid numeric value: {e}",
                threat_level=ThreatLevel.MEDIUM
            )
        except Exception as e:
            self.logger.error(f"Number validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_email(self, email: str) -> ValidationResult:
        """メールアドレス検証"""
        try:
            # 基本的なフォーマット検証
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not re.match(pattern, email):
                return ValidationResult(
                    is_valid=False,
                    error_message="Invalid email format",
                    threat_level=ThreatLevel.LOW
                )
            
            # 長さ制限
            if len(email) > 254:  # RFC 5321 制限
                return ValidationResult(
                    is_valid=False,
                    error_message="Email address too long",
                    threat_level=ThreatLevel.LOW
                )
            
            # 危険パターンチェック
            if any(re.search(pattern, email, re.IGNORECASE) for pattern in self._xss_patterns):
                return ValidationResult(
                    is_valid=False,
                    error_message="Potentially malicious email format",
                    threat_level=ThreatLevel.HIGH
                )
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=email.lower(),
                threat_level=ThreatLevel.INFO
            )
            
        except Exception as e:
            self.logger.error(f"Email validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_ip_address(self, ip: str) -> ValidationResult:
        """IPアドレス検証"""
        try:
            # IPv4/IPv6アドレス検証
            ip_obj = ipaddress.ip_address(ip)
            
            # プライベートアドレスの検証
            is_private = ip_obj.is_private
            is_loopback = ip_obj.is_loopback
            is_multicast = ip_obj.is_multicast
            
            # ValidationResultは詳細情報をサポートしていないため、基本情報のみ返す
            return ValidationResult(
                is_valid=True,
                sanitized_value=str(ip_obj),
                threat_level=ThreatLevel.INFO
            )
            
        except ValueError:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid IP address format",
                threat_level=ThreatLevel.MEDIUM
            )
        except Exception as e:
            self.logger.error(f"IP validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def sanitize_sql_input(self, value: str) -> ValidationResult:
        """SQL入力サニタイゼーション"""
        try:
            # SQLインジェクション攻撃パターンチェック
            threat_level = ThreatLevel.INFO
            sanitized = value
            pattern_found = False
            
            for pattern in self._sql_injection_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    self.logger.warning(f"SQL injection pattern detected: {pattern}")
                    threat_level = ThreatLevel.HIGH
                    pattern_found = True
                    # 危険なパターンを削除
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # パストラバーサルパターンもチェック
            for pattern in self._path_traversal_patterns:
                if re.search(pattern, value):
                    threat_level = ThreatLevel.HIGH
                    pattern_found = True
                    sanitized = re.sub(pattern, '', sanitized)
            
            # 基本的なエスケープ処理
            sanitized = sanitized.replace("'", "''")  # シングルクオートのエスケープ
            sanitized = sanitized.replace('"', '""')  # ダブルクオートのエスケープ
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized,
                threat_level=threat_level,
                error_message="SQL input sanitized" if pattern_found else None
            )
            
        except Exception as e:
            self.logger.error(f"SQL sanitization error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Sanitization error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_file_path(self, path: str, allowed_extensions: Set[str] = None) -> ValidationResult:
        """ファイルパス検証"""
        try:
            # パストラバーサル攻撃チェック
            for pattern in self._path_traversal_patterns:
                if re.search(pattern, path):
                    self.logger.warning(f"Path traversal pattern detected: {pattern}")
                    return ValidationResult(
                        is_valid=False,
                        error_message="Path traversal attack detected",
                        threat_level=ThreatLevel.HIGH
                    )
            
            # パスの正規化
            normalized_path = Path(path).resolve()
            
            # 拡張子チェック
            if allowed_extensions is not None:
                file_extension = normalized_path.suffix.lower()
                if file_extension not in allowed_extensions:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"File extension {file_extension} not allowed",
                        threat_level=ThreatLevel.MEDIUM
                    )
            
            # 危険な文字チェック
            dangerous_chars = ['<', '>', '|', '&', ';']
            if any(char in str(normalized_path) for char in dangerous_chars):
                return ValidationResult(
                    is_valid=False,
                    error_message="Dangerous characters in file path",
                    threat_level=ThreatLevel.HIGH
                )
            
            return ValidationResult(
                is_valid=True,
                sanitized_value=str(normalized_path),
                threat_level=ThreatLevel.INFO
            )
            
        except Exception as e:
            self.logger.error(f"File path validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )


@singleton(IAuthenticationService)
@injectable
class AuthenticationService(IAuthenticationService):
    """認証サービス実装"""

    def __init__(self, 
                 logging_service: ILoggingService,
                 config_service: IConfigurationService):
        self.logging_service = logging_service
        self.config_service = config_service
        self.logger = logging_service.get_logger(__name__, "AuthenticationService")
        
        # セッション管理
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = threading.RLock()
        
        # ユーザーデータベース（実装では外部DBを使用）
        self._users: Dict[str, Dict[str, Any]] = {}
        
        # 設定値
        config = config_service.get_config()
        security_config = config.get('security', {})
        self._session_timeout_minutes = security_config.get('session_timeout_minutes', 60)
        self._max_login_attempts = security_config.get('max_login_attempts', 5)
        self._password_min_length = security_config.get('password_min_length', 8)
        
        # ログイン試行回数追跡
        self._login_attempts: Dict[str, Dict[str, Any]] = {}
        self._attempt_lock = threading.RLock()

    def authenticate_user(self, username: str, password: str) -> AuthenticationResult:
        """ユーザー認証"""
        try:
            with self._attempt_lock:
                # ログイン試行回数チェック
                if self._is_account_locked(username):
                    return AuthenticationResult(
                        is_authenticated=False,
                        error_message="Account temporarily locked due to too many failed attempts"
                    )
                
                # ユーザー存在確認
                if username not in self._users:
                    self._record_failed_attempt(username)
                    return AuthenticationResult(
                        is_authenticated=False,
                        error_message="Invalid username or password"
                    )
                
                user_data = self._users[username]
                
                # パスワード検証
                if not self.verify_password(password, user_data['password_hash']):
                    self._record_failed_attempt(username)
                    return AuthenticationResult(
                        is_authenticated=False,
                        error_message="Invalid username or password"
                    )
                
                # 認証成功
                self._clear_failed_attempts(username)
                
                # セッション作成
                permissions = set(user_data.get('permissions', []))
                session_result = self.create_session(username, permissions)
                
                self.logger.info(f"User authenticated successfully: {username}")
                return session_result
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Authentication service error"
            )

    def create_session(self, user_id: str, permissions: Set[str]) -> AuthenticationResult:
        """セッション作成"""
        try:
            with self._session_lock:
                # セッショントークン生成
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(minutes=self._session_timeout_minutes)
                
                session_data = {
                    'user_id': user_id,
                    'permissions': permissions,
                    'created_at': datetime.now(),
                    'expires_at': expires_at,
                    'last_accessed': datetime.now()
                }
                
                self._sessions[session_token] = session_data
                
                self.logger.info(f"Session created for user: {user_id}")
                
                return AuthenticationResult(
                    is_authenticated=True,
                    user_id=user_id,
                    session_token=session_token,
                    permissions=permissions,
                    expires_at=expires_at
                )
                
        except Exception as e:
            self.logger.error(f"Session creation error: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Session creation failed"
            )

    def validate_session(self, session_token: str) -> AuthenticationResult:
        """セッション検証"""
        try:
            with self._session_lock:
                if session_token not in self._sessions:
                    return AuthenticationResult(
                        is_authenticated=False,
                        error_message="Invalid session token"
                    )
                
                session_data = self._sessions[session_token]
                
                # 有効期限チェック
                if datetime.now() > session_data['expires_at']:
                    del self._sessions[session_token]
                    return AuthenticationResult(
                        is_authenticated=False,
                        error_message="Session expired"
                    )
                
                # 最終アクセス時刻更新
                session_data['last_accessed'] = datetime.now()
                
                return AuthenticationResult(
                    is_authenticated=True,
                    user_id=session_data['user_id'],
                    session_token=session_token,
                    permissions=session_data['permissions'],
                    expires_at=session_data['expires_at']
                )
                
        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return AuthenticationResult(
                is_authenticated=False,
                error_message="Session validation failed"
            )

    def revoke_session(self, session_token: str) -> bool:
        """セッション無効化"""
        try:
            with self._session_lock:
                if session_token in self._sessions:
                    user_id = self._sessions[session_token]['user_id']
                    del self._sessions[session_token]
                    self.logger.info(f"Session revoked for user: {user_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Session revocation error: {e}")
            return False

    def hash_password(self, password: str) -> str:
        """パスワードハッシュ化"""
        try:
            # ソルト生成
            salt = secrets.token_hex(16)
            
            # パスワードハッシュ化（PBKDF2）
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000  # 100,000回のイテレーション
            )
            
            # ソルトとハッシュを結合
            return f"{salt}:{password_hash.hex()}"
            
        except Exception as e:
            self.logger.error(f"Password hashing error: {e}")
            raise

    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワード検証"""
        try:
            # ソルトとハッシュを分離
            salt, stored_hash = hashed.split(':')
            
            # 入力パスワードのハッシュ化
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            
            # タイミング攻撃を防ぐため、hmac.compare_digestを使用
            return hmac.compare_digest(password_hash.hex(), stored_hash)
            
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False

    def _is_account_locked(self, username: str) -> bool:
        """アカウントロック状態チェック"""
        if username not in self._login_attempts:
            return False
        
        attempt_data = self._login_attempts[username]
        
        # 最大試行回数チェック
        if attempt_data['count'] >= self._max_login_attempts:
            # ロック時間チェック（15分）
            lock_duration = timedelta(minutes=15)
            if datetime.now() < attempt_data['locked_until']:
                return True
            else:
                # ロック期間終了、リセット
                del self._login_attempts[username]
                return False
        
        return False

    def _record_failed_attempt(self, username: str):
        """失敗試行記録"""
        now = datetime.now()
        
        if username not in self._login_attempts:
            self._login_attempts[username] = {
                'count': 1,
                'last_attempt': now,
                'locked_until': now
            }
        else:
            attempt_data = self._login_attempts[username]
            attempt_data['count'] += 1
            attempt_data['last_attempt'] = now
            
            # ロック時間設定
            if attempt_data['count'] >= self._max_login_attempts:
                attempt_data['locked_until'] = now + timedelta(minutes=15)

    def _clear_failed_attempts(self, username: str):
        """失敗試行クリア"""
        if username in self._login_attempts:
            del self._login_attempts[username]

    def register_user(self, username: str, password: str, permissions: Set[str] = None) -> bool:
        """ユーザー登録（テスト用）"""
        try:
            if username in self._users:
                return False
            
            # パスワード強度チェック
            if len(password) < self._password_min_length:
                return False
            
            password_hash = self.hash_password(password)
            
            self._users[username] = {
                'password_hash': password_hash,
                'permissions': list(permissions or set()),
                'created_at': datetime.now(),
                'is_active': True
            }
            
            self.logger.info(f"User registered: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"User registration error: {e}")
            return False


@singleton(IAuthorizationService)
@injectable
class AuthorizationService(IAuthorizationService):
    """認可サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "AuthorizationService")
        
        # ユーザー権限管理
        self._user_permissions: Dict[str, Set[str]] = {}
        self._permission_lock = threading.RLock()
        
        # デフォルト権限定義
        self._default_permissions = {
            'admin': {
                'system.admin', 'data.read', 'data.write', 'data.delete',
                'trade.execute', 'trade.view', 'config.read', 'config.write'
            },
            'trader': {
                'data.read', 'trade.execute', 'trade.view', 'config.read'
            },
            'viewer': {
                'data.read', 'trade.view'
            }
        }

    def check_permission(self, user_id: str, action: ActionType, resource: str = None) -> bool:
        """権限チェック"""
        try:
            with self._permission_lock:
                user_permissions = self._user_permissions.get(user_id, set())
                
                # アクション種別に応じた権限マッピング
                required_permissions = self._get_required_permissions(action, resource)
                
                # 権限チェック
                has_permission = any(perm in user_permissions for perm in required_permissions)
                
                if not has_permission:
                    self.logger.warning(f"Permission denied for user {user_id}: {action.value}")
                
                return has_permission
                
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return False

    def grant_permission(self, user_id: str, permission: str) -> bool:
        """権限付与"""
        try:
            with self._permission_lock:
                if user_id not in self._user_permissions:
                    self._user_permissions[user_id] = set()
                
                self._user_permissions[user_id].add(permission)
                self.logger.info(f"Permission granted to {user_id}: {permission}")
                return True
                
        except Exception as e:
            self.logger.error(f"Permission grant error: {e}")
            return False

    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """権限剥奪"""
        try:
            with self._permission_lock:
                if user_id in self._user_permissions:
                    self._user_permissions[user_id].discard(permission)
                    self.logger.info(f"Permission revoked from {user_id}: {permission}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Permission revoke error: {e}")
            return False

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """ユーザー権限取得"""
        with self._permission_lock:
            return self._user_permissions.get(user_id, set()).copy()

    def assign_role(self, user_id: str, role: str) -> bool:
        """ロール割り当て"""
        try:
            if role not in self._default_permissions:
                self.logger.error(f"Unknown role: {role}")
                return False
            
            with self._permission_lock:
                self._user_permissions[user_id] = self._default_permissions[role].copy()
                self.logger.info(f"Role assigned to {user_id}: {role}")
                return True
                
        except Exception as e:
            self.logger.error(f"Role assignment error: {e}")
            return False

    def _get_required_permissions(self, action: ActionType, resource: str = None) -> Set[str]:
        """必要権限取得"""
        permission_map = {
            ActionType.LOGIN: {'system.login'},
            ActionType.LOGOUT: {'system.logout'},
            ActionType.API_ACCESS: {'api.access'},
            ActionType.DATA_ACCESS: {'data.read'},
            ActionType.CONFIG_CHANGE: {'config.write'},
            ActionType.TRADE_EXECUTION: {'trade.execute'},
            ActionType.FILE_ACCESS: {'file.access'},
            ActionType.SYSTEM_ADMIN: {'system.admin'}
        }
        
        base_permissions = permission_map.get(action, {'generic.access'})
        
        # リソース特化権限の追加
        if resource:
            resource_permission = f"{action.value}.{resource}"
            base_permissions.add(resource_permission)
        
        return base_permissions


@singleton(IRateLimitService)
@injectable
class RateLimitService(IRateLimitService):
    """レート制限サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "RateLimitService")
        
        # カウンター管理
        self._counters: Dict[str, Dict[str, Any]] = {}
        self._counter_lock = threading.RLock()

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitInfo:
        """レート制限チェック"""
        try:
            with self._counter_lock:
                now = datetime.now()
                
                # カウンター初期化または期限切れリセット
                if key not in self._counters:
                    self._counters[key] = {
                        'count': 0,
                        'window_start': now,
                        'window_seconds': window_seconds
                    }
                
                counter_data = self._counters[key]
                
                # ウィンドウリセットチェック
                if now >= counter_data['window_start'] + timedelta(seconds=window_seconds):
                    counter_data['count'] = 0
                    counter_data['window_start'] = now
                    counter_data['window_seconds'] = window_seconds
                
                # レート制限チェック
                current_count = counter_data['count']
                reset_time = counter_data['window_start'] + timedelta(seconds=window_seconds)
                is_exceeded = current_count >= limit
                
                if is_exceeded:
                    self.logger.warning(f"Rate limit exceeded for key: {key}")
                
                return RateLimitInfo(
                    limit=limit,
                    window_seconds=window_seconds,
                    current_count=current_count,
                    reset_time=reset_time,
                    is_exceeded=is_exceeded
                )
                
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return RateLimitInfo(
                limit=limit,
                window_seconds=window_seconds,
                current_count=0,
                reset_time=datetime.now(),
                is_exceeded=False
            )

    def increment_counter(self, key: str) -> int:
        """カウンター増分"""
        try:
            with self._counter_lock:
                if key in self._counters:
                    self._counters[key]['count'] += 1
                    return self._counters[key]['count']
                return 0
                
        except Exception as e:
            self.logger.error(f"Counter increment error: {e}")
            return 0

    def reset_counter(self, key: str) -> bool:
        """カウンターリセット"""
        try:
            with self._counter_lock:
                if key in self._counters:
                    self._counters[key]['count'] = 0
                    self._counters[key]['window_start'] = datetime.now()
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Counter reset error: {e}")
            return False


@singleton(ISecurityAuditService)
@injectable
class SecurityAuditService(ISecurityAuditService):
    """セキュリティ監査サービス実装"""

    def __init__(self, 
                 logging_service: ILoggingService,
                 config_service: IConfigurationService):
        self.logging_service = logging_service
        self.config_service = config_service
        self.logger = logging_service.get_logger(__name__, "SecurityAuditService")
        
        # イベントストレージ
        self._events: List[SecurityEvent] = []
        self._event_lock = threading.RLock()
        
        # 設定
        config = config_service.get_config()
        security_config = config.get('security', {})
        self._max_events = security_config.get('max_audit_events', 10000)
        self._audit_log_file = security_config.get('audit_log_file', 'security_audit.log')

    def log_security_event(self, event: SecurityEvent) -> str:
        """セキュリティイベントログ"""
        try:
            with self._event_lock:
                # イベントID生成
                if not event.event_id:
                    event.event_id = f"evt_{int(time.time())}_{secrets.token_hex(4)}"
                
                # イベント追加
                self._events.append(event)
                
                # イベント数制限
                if len(self._events) > self._max_events:
                    self._events = self._events[-self._max_events:]
                
                # ログ出力
                self._write_audit_log(event)
                
                self.logger.info(f"Security event logged: {event.event_id}")
                return event.event_id
                
        except Exception as e:
            self.logger.error(f"Security event logging error: {e}")
            return ""

    def get_security_events(self, start_time: datetime = None, 
                          end_time: datetime = None, 
                          threat_level: ThreatLevel = None) -> List[SecurityEvent]:
        """セキュリティイベント取得"""
        try:
            with self._event_lock:
                filtered_events = self._events.copy()
                
                # 時間範囲フィルター
                if start_time:
                    filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
                
                if end_time:
                    filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
                
                # 脅威レベルフィルター
                if threat_level:
                    filtered_events = [e for e in filtered_events if e.threat_level == threat_level]
                
                return filtered_events
                
        except Exception as e:
            self.logger.error(f"Security event retrieval error: {e}")
            return []

    def analyze_threats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """脅威分析"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            events = self.get_security_events(start_time, end_time)
            
            # 統計計算
            total_events = len(events)
            threat_counts = {}
            event_type_counts = {}
            blocked_count = 0
            source_ips = set()
            
            for event in events:
                # 脅威レベル集計
                threat_level = event.threat_level.value
                threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
                
                # イベント種別集計
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
                
                # ブロック数集計
                if event.blocked:
                    blocked_count += 1
                
                # ソースIP集計
                if event.source_ip:
                    source_ips.add(event.source_ip)
            
            # リスク評価
            risk_score = self._calculate_risk_score(threat_counts, total_events)
            
            return {
                'analysis_period_hours': time_window_hours,
                'total_events': total_events,
                'threat_level_distribution': threat_counts,
                'event_type_distribution': event_type_counts,
                'blocked_events': blocked_count,
                'unique_source_ips': len(source_ips),
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'top_threat_sources': list(source_ips)[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Threat analysis error: {e}")
            return {}

    def generate_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        try:
            # 24時間、7日間、30日間の分析
            analysis_24h = self.analyze_threats(24)
            analysis_7d = self.analyze_threats(24 * 7)
            analysis_30d = self.analyze_threats(24 * 30)
            
            # 全体統計
            with self._event_lock:
                total_events_all_time = len(self._events)
            
            return {
                'report_generated_at': datetime.now().isoformat(),
                'total_events_all_time': total_events_all_time,
                'analysis_24_hours': analysis_24h,
                'analysis_7_days': analysis_7d,
                'analysis_30_days': analysis_30d,
                'recommendations': self._generate_recommendations(analysis_24h, analysis_7d)
            }
            
        except Exception as e:
            self.logger.error(f"Security report generation error: {e}")
            return {}

    def _write_audit_log(self, event: SecurityEvent):
        """監査ログ書き込み"""
        try:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_id': event.event_id,
                'event_type': event.event_type,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'action': event.action.value if event.action else None,
                'resource': event.resource,
                'details': event.details,
                'blocked': event.blocked
            }
            
            # ファイルに追記（実装では外部ログシステムを使用することを推奨）
            log_line = json.dumps(log_entry, ensure_ascii=False)
            
            # 簡易ファイル書き込み（実装では適切なログローテーションを実装）
            with open(self._audit_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
                
        except Exception as e:
            self.logger.error(f"Audit log write error: {e}")

    def _calculate_risk_score(self, threat_counts: Dict[str, int], total_events: int) -> float:
        """リスクスコア計算"""
        if total_events == 0:
            return 0.0
        
        # 脅威レベル重み
        weights = {
            'info': 0.1,
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        weighted_score = 0.0
        for threat_level, count in threat_counts.items():
            weight = weights.get(threat_level, 0.5)
            weighted_score += (count / total_events) * weight
        
        return min(weighted_score * 100, 100.0)  # 0-100スケール

    def _get_risk_level(self, risk_score: float) -> str:
        """リスクレベル取得"""
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _generate_recommendations(self, analysis_24h: Dict[str, Any], 
                                analysis_7d: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        try:
            # 24時間の高脅威イベント数チェック
            high_threats_24h = analysis_24h.get('threat_level_distribution', {})
            high_count = high_threats_24h.get('high', 0) + high_threats_24h.get('critical', 0)
            
            if high_count > 10:
                recommendations.append("高脅威イベントが頻発しています。セキュリティ対策の見直しを推奨します。")
            
            # ブロック率チェック
            total_24h = analysis_24h.get('total_events', 0)
            blocked_24h = analysis_24h.get('blocked_events', 0)
            
            if total_24h > 0 and (blocked_24h / total_24h) > 0.1:
                recommendations.append("ブロック率が高くなっています。攻撃者の活動が活発化している可能性があります。")
            
            # IP多様性チェック
            unique_ips = analysis_24h.get('unique_source_ips', 0)
            if unique_ips > 100:
                recommendations.append("多数のIPアドレスからのアクセスが検出されています。分散攻撃の可能性を調査してください。")
            
            # 長期トレンドチェック
            events_24h = analysis_24h.get('total_events', 0)
            events_7d = analysis_7d.get('total_events', 0)
            
            if events_24h > 0 and events_7d > 0:
                daily_avg_7d = events_7d / 7
                if events_24h > daily_avg_7d * 2:
                    recommendations.append("イベント数が平常時より大幅に増加しています。")
            
            if not recommendations:
                recommendations.append("現在、セキュリティ状況は安定しています。定期的な監視を継続してください。")
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {e}")
            recommendations.append("推奨事項の生成中にエラーが発生しました。")
        
        return recommendations


def register_security_services():
    """セキュリティサービスを登録"""
    container = get_container()
    
    # 入力検証サービス
    if not container.is_registered(IInputValidationService):
        container.register_singleton(IInputValidationService, InputValidationService)
    
    # 認証サービス
    if not container.is_registered(IAuthenticationService):
        container.register_singleton(IAuthenticationService, AuthenticationService)
    
    # 認可サービス
    if not container.is_registered(IAuthorizationService):
        container.register_singleton(IAuthorizationService, AuthorizationService)
    
    # レート制限サービス
    if not container.is_registered(IRateLimitService):
        container.register_singleton(IRateLimitService, RateLimitService)
    
    # セキュリティ監査サービス
    if not container.is_registered(ISecurityAuditService):
        container.register_singleton(ISecurityAuditService, SecurityAuditService)


# 便利な関数
def get_security_services():
    """セキュリティサービス取得"""
    container = get_container()
    return {
        'validation': container.resolve(IInputValidationService),
        'authentication': container.resolve(IAuthenticationService),
        'authorization': container.resolve(IAuthorizationService),
        'rate_limit': container.resolve(IRateLimitService),
        'audit': container.resolve(ISecurityAuditService)
    }


def create_security_event(event_type: str, threat_level: ThreatLevel, **kwargs) -> SecurityEvent:
    """セキュリティイベント作成ヘルパー"""
    return SecurityEvent(
        event_id="",  # サービスで自動生成
        timestamp=datetime.now(),
        event_type=event_type,
        threat_level=threat_level,
        source_ip=kwargs.get('source_ip'),
        user_id=kwargs.get('user_id'),
        action=kwargs.get('action'),
        resource=kwargs.get('resource'),
        details=kwargs.get('details', {}),
        blocked=kwargs.get('blocked', False)
    )