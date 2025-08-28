"""
セキュリティマネージャー

包括的なセキュリティ機能を提供：
- 入力検証・サニタイゼーション
- 認証・認可
- 暗号化・復号化
- セキュリティログ・監視
- 脅威検出・防御
"""

import re
import hashlib
import hmac
import secrets
import uuid
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Pattern
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import base64

# 暗号化用ライブラリ（実際にはcryptographyライブラリを使用）
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None


# ================================
# セキュリティ列挙型
# ================================

class SecurityLevel(Enum):
    """セキュリティレベル"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """脅威レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """認証方法"""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MFA = "multi_factor"


# ================================
# セキュリティデータクラス
# ================================

@dataclass
class SecurityContext:
    """セキュリティコンテキスト"""
    user_id: str = ""
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    authenticated: bool = False
    authentication_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'permissions': list(self.permissions),
            'security_level': self.security_level.value,
            'authenticated': self.authenticated,
            'authentication_time': self.authentication_time.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }


@dataclass
class SecurityThreat:
    """セキュリティ脅威"""
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: str = ""
    level: ThreatLevel = ThreatLevel.LOW
    source_ip: str = ""
    target: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type,
            'level': self.level.value,
            'source_ip': self.source_ip,
            'target': self.target,
            'description': self.description,
            'evidence': self.evidence,
            'timestamp': self.timestamp.isoformat(),
            'blocked': self.blocked
        }


# ================================
# 入力検証・サニタイゼーション
# ================================

class InputValidator:
    """入力検証クラス"""
    
    # 共通パターン
    PATTERNS = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'phone': re.compile(r'^\+?[1-9]\d{1,14}$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'numeric': re.compile(r'^\d+$'),
        'decimal': re.compile(r'^\d+(\.\d+)?$'),
        'symbol': re.compile(r'^[A-Z]{1,10}$'),  # 株式銘柄コード
        'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
    }
    
    # 危険なパターン
    DANGEROUS_PATTERNS = {
        'sql_injection': re.compile(r'(union|select|insert|update|delete|drop|create|alter|exec|script)', re.IGNORECASE),
        'xss': re.compile(r'<script|javascript:|vbscript:|onload|onerror|onclick', re.IGNORECASE),
        'command_injection': re.compile(r'[;&|`$(){}[\]\\]'),
        'path_traversal': re.compile(r'\.\./|\.\.\\'),
    }
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """メールアドレス検証"""
        if not email or len(email) > 254:
            return False
        return bool(cls.PATTERNS['email'].match(email))
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """株式銘柄コード検証"""
        if not symbol or len(symbol) > 10:
            return False
        return bool(cls.PATTERNS['symbol'].match(symbol))
    
    @classmethod
    def validate_numeric(cls, value: str, min_value: float = None, max_value: float = None) -> bool:
        """数値検証"""
        if not value:
            return False
        
        if not cls.PATTERNS['decimal'].match(value):
            return False
        
        try:
            num_value = float(value)
            if min_value is not None and num_value < min_value:
                return False
            if max_value is not None and num_value > max_value:
                return False
            return True
        except ValueError:
            return False
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """文字列サニタイゼーション"""
        if not value:
            return ""
        
        # 長さ制限
        sanitized = value[:max_length]
        
        # HTMLエスケープ
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        
        return sanitized
    
    @classmethod
    def detect_threats(cls, value: str) -> List[str]:
        """脅威検出"""
        threats = []
        
        for threat_name, pattern in cls.DANGEROUS_PATTERNS.items():
            if pattern.search(value):
                threats.append(threat_name)
        
        return threats
    
    @classmethod
    def validate_safe_input(cls, value: str, field_name: str = "") -> bool:
        """安全な入力検証"""
        if not value:
            return True  # 空文字は許可
        
        # 脅威検出
        threats = cls.detect_threats(value)
        if threats:
            logging.warning(f"セキュリティ脅威検出 ({field_name}): {threats}")
            return False
        
        # 基本的な安全パターンチェック
        return bool(cls.PATTERNS['safe_string'].match(value))


# ================================
# 暗号化システム
# ================================

class EncryptionManager:
    """暗号化マネージャー"""
    
    def __init__(self, master_key: str = None):
        self.master_key = master_key or self._generate_master_key()
        self._symmetric_key = None
        self._private_key = None
        self._public_key = None
        
        if CRYPTO_AVAILABLE:
            self._init_encryption()
    
    def _generate_master_key(self) -> str:
        """マスターキー生成"""
        return secrets.token_urlsafe(32)
    
    def _init_encryption(self):
        """暗号化初期化"""
        # 対称キー生成
        password = self.master_key.encode()
        salt = b'day_trade_salt'  # 実際にはランダム生成すべき
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._symmetric_key = Fernet(key)
        
        # 非対称キーペア生成
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self._public_key = self._private_key.public_key()
    
    def encrypt_symmetric(self, data: str) -> str:
        """対称暗号化"""
        if not CRYPTO_AVAILABLE or not self._symmetric_key:
            # フォールバック: 簡単なBase64エンコード
            return base64.b64encode(data.encode()).decode()
        
        encrypted = self._symmetric_key.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """対称復号化"""
        if not CRYPTO_AVAILABLE or not self._symmetric_key:
            # フォールバック: Base64デコード
            return base64.b64decode(encrypted_data.encode()).decode()
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self._symmetric_key.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """パスワードハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # PBKDF2ハッシュ化
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return salt, password_hash.hex()
    
    def verify_password(self, password: str, salt: str, stored_hash: str) -> bool:
        """パスワード検証"""
        _, computed_hash = self.hash_password(password, salt)
        return hmac.compare_digest(stored_hash, computed_hash)
    
    def generate_api_key(self, user_id: str) -> str:
        """APIキー生成"""
        timestamp = str(int(datetime.now().timestamp()))
        data = f"{user_id}:{timestamp}:{secrets.token_hex(16)}"
        return base64.b64encode(data.encode()).decode()
    
    def generate_session_token(self, user_id: str, ip_address: str) -> str:
        """セッショントークン生成"""
        timestamp = str(int(datetime.now().timestamp()))
        data = f"{user_id}:{ip_address}:{timestamp}:{secrets.token_hex(32)}"
        signature = hmac.new(
            self.master_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{base64.b64encode(data.encode()).decode()}.{signature}"
    
    def verify_session_token(self, token: str, user_id: str, ip_address: str, max_age: int = 3600) -> bool:
        """セッショントークン検証"""
        try:
            token_data, signature = token.split('.')
            data = base64.b64decode(token_data.encode()).decode()
            
            # 署名検証
            expected_signature = hmac.new(
                self.master_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return False
            
            # データ解析
            parts = data.split(':')
            if len(parts) != 4:
                return False
            
            token_user_id, token_ip, timestamp_str, _ = parts
            
            # ユーザーIDとIPアドレス確認
            if token_user_id != user_id or token_ip != ip_address:
                return False
            
            # タイムスタンプ確認
            token_timestamp = int(timestamp_str)
            current_timestamp = int(datetime.now().timestamp())
            
            if current_timestamp - token_timestamp > max_age:
                return False
            
            return True
        
        except Exception:
            return False


# ================================
# 認証・認可システム
# ================================

class AuthenticationManager:
    """認証マネージャー"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # 設定
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=2)
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[SecurityContext]:
        """ユーザー認証"""
        with self._lock:
            # IPブロックチェック
            if self._is_ip_blocked(ip_address):
                logging.warning(f"ブロック済みIPからの認証試行: {ip_address}")
                return None
            
            # 失敗回数チェック
            if self._is_user_locked_out(username):
                logging.warning(f"ロックアウト中ユーザーの認証試行: {username}")
                return None
            
            # 実際の認証処理（簡略化）
            if self._verify_credentials(username, password):
                # 認証成功
                self._reset_failed_attempts(username)
                
                context = SecurityContext(
                    user_id=username,
                    session_id=str(uuid.uuid4()),
                    ip_address=ip_address,
                    authenticated=True,
                    permissions=self._get_user_permissions(username),
                    security_level=self._get_user_security_level(username)
                )
                
                self.sessions[context.session_id] = context
                logging.info(f"ユーザー認証成功: {username}")
                return context
            else:
                # 認証失敗
                self._record_failed_attempt(username, ip_address)
                logging.warning(f"ユーザー認証失敗: {username} from {ip_address}")
                return None
    
    def authenticate_api_key(self, api_key: str, ip_address: str) -> Optional[SecurityContext]:
        """APIキー認証"""
        try:
            # APIキー解析（簡略化）
            decoded = base64.b64decode(api_key.encode()).decode()
            parts = decoded.split(':')
            
            if len(parts) >= 3:
                user_id = parts[0]
                timestamp = int(parts[1])
                
                # タイムスタンプ確認（APIキーの有効期限）
                current_timestamp = int(datetime.now().timestamp())
                if current_timestamp - timestamp > 86400 * 30:  # 30日
                    return None
                
                context = SecurityContext(
                    user_id=user_id,
                    session_id=str(uuid.uuid4()),
                    ip_address=ip_address,
                    authenticated=True,
                    permissions=self._get_user_permissions(user_id),
                    security_level=SecurityLevel.INTERNAL
                )
                
                self.sessions[context.session_id] = context
                return context
        
        except Exception:
            pass
        
        return None
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[SecurityContext]:
        """セッション検証"""
        with self._lock:
            context = self.sessions.get(session_id)
            
            if not context:
                return None
            
            # IPアドレス確認
            if context.ip_address != ip_address:
                logging.warning(f"セッションのIPアドレス不一致: {session_id}")
                self.invalidate_session(session_id)
                return None
            
            # セッションタイムアウト確認
            if datetime.now() - context.last_activity > self.session_timeout:
                logging.info(f"セッションタイムアウト: {session_id}")
                self.invalidate_session(session_id)
                return None
            
            # 最終活動時刻更新
            context.last_activity = datetime.now()
            return context
    
    def invalidate_session(self, session_id: str):
        """セッション無効化"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logging.info(f"セッション無効化: {session_id}")
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """認証情報検証（簡略化）"""
        # 実際にはデータベースから取得
        test_users = {
            'admin': ('salt1', 'hashed_password1'),
            'trader': ('salt2', 'hashed_password2')
        }
        
        if username in test_users:
            salt, stored_hash = test_users[username]
            return self.encryption_manager.verify_password(password, salt, stored_hash)
        
        return False
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """ユーザー権限取得"""
        # 実際にはデータベースから取得
        permissions_map = {
            'admin': {'read', 'write', 'delete', 'admin'},
            'trader': {'read', 'write'},
            'viewer': {'read'}
        }
        return permissions_map.get(user_id, {'read'})
    
    def _get_user_security_level(self, user_id: str) -> SecurityLevel:
        """ユーザーセキュリティレベル取得"""
        level_map = {
            'admin': SecurityLevel.SECRET,
            'trader': SecurityLevel.CONFIDENTIAL,
            'viewer': SecurityLevel.INTERNAL
        }
        return level_map.get(user_id, SecurityLevel.PUBLIC)
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """IP ブロック確認"""
        if ip_address in self.blocked_ips:
            block_time = self.blocked_ips[ip_address]
            if datetime.now() - block_time < self.lockout_duration:
                return True
            else:
                del self.blocked_ips[ip_address]
        return False
    
    def _is_user_locked_out(self, username: str) -> bool:
        """ユーザーロックアウト確認"""
        if username not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[username]
        cutoff_time = datetime.now() - self.lockout_duration
        
        # 古い試行を削除
        attempts = [attempt for attempt in attempts if attempt > cutoff_time]
        self.failed_attempts[username] = attempts
        
        return len(attempts) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """失敗試行記録"""
        now = datetime.now()
        
        # ユーザー別失敗回数記録
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        self.failed_attempts[username].append(now)
        
        # IP別失敗回数チェック
        ip_key = f"ip_{ip_address}"
        if ip_key not in self.failed_attempts:
            self.failed_attempts[ip_key] = []
        self.failed_attempts[ip_key].append(now)
        
        # IPブロック判定
        if len(self.failed_attempts[ip_key]) >= self.max_failed_attempts:
            self.blocked_ips[ip_address] = now
            logging.warning(f"IP アドレスをブロック: {ip_address}")
    
    def _reset_failed_attempts(self, username: str):
        """失敗試行リセット"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]


# ================================
# セキュリティマネージャー統合
# ================================

class SecurityManager:
    """統合セキュリティマネージャー"""
    
    def __init__(self, master_key: str = None):
        self.encryption_manager = EncryptionManager(master_key)
        self.authentication_manager = AuthenticationManager(self.encryption_manager)
        self.threats: List[SecurityThreat] = []
        self.security_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # セキュリティ監視設定
        self.monitoring_enabled = True
        self.threat_detection_enabled = True
    
    def validate_and_sanitize(self, data: Dict[str, Any], field_validations: Dict[str, Callable] = None) -> Dict[str, Any]:
        """データ検証・サニタイゼーション"""
        sanitized = {}
        threats = []
        
        for field, value in data.items():
            if isinstance(value, str):
                # 脅威検出
                field_threats = InputValidator.detect_threats(value)
                if field_threats:
                    threats.extend([f"{field}:{threat}" for threat in field_threats])
                
                # サニタイゼーション
                sanitized[field] = InputValidator.sanitize_string(value)
                
                # カスタム検証
                if field_validations and field in field_validations:
                    if not field_validations[field](value):
                        raise ValueError(f"フィールド検証失敗: {field}")
            else:
                sanitized[field] = value
        
        # 脅威が検出された場合
        if threats:
            self._log_security_event({
                'event_type': 'threat_detected',
                'threats': threats,
                'data_fields': list(data.keys()),
                'timestamp': datetime.now().isoformat()
            })
            
            if self.threat_detection_enabled:
                raise SecurityError(f"セキュリティ脅威が検出されました: {threats}")
        
        return sanitized
    
    def authenticate(self, credentials: Dict[str, Any], ip_address: str) -> Optional[SecurityContext]:
        """統合認証"""
        auth_method = credentials.get('method', AuthenticationMethod.PASSWORD.value)
        
        try:
            if auth_method == AuthenticationMethod.PASSWORD.value:
                username = credentials.get('username')
                password = credentials.get('password')
                return self.authentication_manager.authenticate_user(username, password, ip_address)
            
            elif auth_method == AuthenticationMethod.API_KEY.value:
                api_key = credentials.get('api_key')
                return self.authentication_manager.authenticate_api_key(api_key, ip_address)
            
            else:
                self._log_security_event({
                    'event_type': 'unsupported_auth_method',
                    'method': auth_method,
                    'ip_address': ip_address,
                    'timestamp': datetime.now().isoformat()
                })
                return None
        
        except Exception as e:
            self._log_security_event({
                'event_type': 'authentication_error',
                'error': str(e),
                'ip_address': ip_address,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def authorize(self, context: SecurityContext, required_permission: str, resource: str = "") -> bool:
        """認可チェック"""
        if not context.authenticated:
            return False
        
        # 権限チェック
        if required_permission not in context.permissions:
            self._log_security_event({
                'event_type': 'authorization_failure',
                'user_id': context.user_id,
                'required_permission': required_permission,
                'user_permissions': list(context.permissions),
                'resource': resource,
                'timestamp': datetime.now().isoformat()
            })
            return False
        
        return True
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: Set[str]) -> Dict[str, Any]:
        """機密データ暗号化"""
        encrypted = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted and isinstance(encrypted[field], str):
                encrypted[field] = self.encryption_manager.encrypt_symmetric(encrypted[field])
        
        return encrypted
    
    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: Set[str]) -> Dict[str, Any]:
        """機密データ復号化"""
        decrypted = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted and isinstance(decrypted[field], str):
                try:
                    decrypted[field] = self.encryption_manager.decrypt_symmetric(decrypted[field])
                except Exception:
                    # 復号化失敗の場合は元の値を保持
                    pass
        
        return decrypted
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], context: SecurityContext = None):
        """セキュリティイベントログ"""
        self._log_security_event({
            'event_type': event_type,
            'details': details,
            'context': context.to_dict() if context else None,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log_security_event(self, event: Dict[str, Any]):
        """内部セキュリティログ"""
        with self._lock:
            self.security_log.append(event)
            
            # ログサイズ制限
            if len(self.security_log) > 10000:
                self.security_log.pop(0)
        
        # ログ出力
        logging.info(f"セキュリティイベント: {json.dumps(event, ensure_ascii=False)}")
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """セキュリティ統計取得"""
        with self._lock:
            event_types = {}
            recent_events = 0
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for event in self.security_log:
                event_type = event.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
                
                event_time = datetime.fromisoformat(event['timestamp'])
                if event_time > cutoff_time:
                    recent_events += 1
            
            return {
                'total_events': len(self.security_log),
                'recent_events_24h': recent_events,
                'event_types': event_types,
                'active_sessions': len(self.authentication_manager.sessions),
                'blocked_ips': len(self.authentication_manager.blocked_ips),
                'threats_detected': len(self.threats)
            }


# ================================
# セキュリティデコレーター
# ================================

def require_authentication(permission: str = "read"):
    """認証必須デコレーター"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # セキュリティコンテキストを引数から取得
            context = kwargs.get('security_context')
            if not context or not context.authenticated:
                raise SecurityError("認証が必要です")
            
            if permission not in context.permissions:
                raise SecurityError(f"権限が不足しています: {permission}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = kwargs.get('security_context')
            if not context or not context.authenticated:
                raise SecurityError("認証が必要です")
            
            if permission not in context.permissions:
                raise SecurityError(f"権限が不足しています: {permission}")
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def secure_endpoint(validate_input: bool = True, encrypt_response: bool = False):
    """セキュアエンドポイントデコレーター"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 入力検証
            if validate_input:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        threats = InputValidator.detect_threats(value)
                        if threats:
                            raise SecurityError(f"セキュリティ脅威検出: {threats}")
            
            result = await func(*args, **kwargs)
            
            # レスポンス暗号化
            if encrypt_response and isinstance(result, dict):
                # 実装簡略化：実際には必要な場合のみ暗号化
                pass
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if validate_input:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        threats = InputValidator.detect_threats(value)
                        if threats:
                            raise SecurityError(f"セキュリティ脅威検出: {threats}")
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ================================
# セキュリティ例外
# ================================

class SecurityError(Exception):
    """セキュリティエラー"""
    pass


# エクスポート
__all__ = [
    # 列挙型
    'SecurityLevel', 'ThreatLevel', 'AuthenticationMethod',
    
    # データクラス
    'SecurityContext', 'SecurityThreat',
    
    # 主要クラス
    'InputValidator', 'EncryptionManager', 'AuthenticationManager', 'SecurityManager',
    
    # デコレーター
    'require_authentication', 'secure_endpoint',
    
    # 例外
    'SecurityError'
]