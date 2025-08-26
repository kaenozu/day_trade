#!/usr/bin/env python3
"""
認証サービス実装
Issue #918 項目9対応: セキュリティ強化

ユーザー認証、セッション管理、パスワード処理機能
"""

import hashlib
import secrets
import hmac
import threading
from datetime import datetime, timedelta
from typing import Dict, Set, Any

from ..dependency_injection import ILoggingService, IConfigurationService, injectable, singleton
from .interfaces import IAuthenticationService
from .types import AuthenticationResult


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