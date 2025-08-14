# Authentication & Authorization Service
# Day Trade ML System - Issue #803

import jwt
import bcrypt
import secrets
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from enum import Enum
import re

# FastAPI
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database
from ..database.connection import database
from ..database.models import User, UserSession, Permission, Role

# Services
from .notification_service import NotificationService
from .cache_service import CacheService

# Config
from ..config import settings

# Utils
from ..utils.security import hash_password, verify_password, generate_token
from ..utils.rate_limit import RateLimiter
from ..utils.validation import validate_email, validate_password_strength


class AuthMethod(Enum):
    """認証方式"""
    PASSWORD = "password"
    GOOGLE_OAUTH = "google_oauth"
    BIOMETRIC = "biometric"
    API_KEY = "api_key"
    MFA = "mfa"


class PermissionLevel(Enum):
    """権限レベル"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPERUSER = "superuser"


@dataclass
class AuthResult:
    """認証結果"""
    success: bool
    user: Optional[User] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    error_message: Optional[str] = None
    requires_mfa: bool = False
    mfa_token: Optional[str] = None


@dataclass
class TokenPayload:
    """JWTトークンペイロード"""
    user_id: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime
    token_type: str  # "access" | "refresh" | "mfa"


class AuthService:
    """認証・認可サービス"""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.notification_service: Optional[NotificationService] = None
        self.cache_service: Optional[CacheService] = None
        self.security = HTTPBearer()

        # JWT設定
        self.jwt_algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
        self.mfa_token_expire = timedelta(minutes=5)

        # パスワードポリシー
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special": True,
            "max_age_days": 90,
            "history_count": 5,
        }

        # セッション管理
        self.max_sessions_per_user = 5
        self.session_timeout = timedelta(hours=24)

        logging.info("AuthService initialized")

    def set_dependencies(
        self,
        notification_service: NotificationService,
        cache_service: CacheService
    ):
        """依存関係を設定"""
        self.notification_service = notification_service
        self.cache_service = cache_service

    async def authenticate_user(
        self,
        email: str,
        password: str,
        auth_method: AuthMethod = AuthMethod.PASSWORD,
        client_info: Optional[Dict[str, Any]] = None
    ) -> AuthResult:
        """ユーザー認証"""
        try:
            # レート制限チェック
            if not await self.rate_limiter.check_rate_limit(f"auth_{email}", max_requests=5, window=300):
                return AuthResult(
                    success=False,
                    error_message="認証試行回数が上限を超えました。5分後に再試行してください。"
                )

            # 入力検証
            if not validate_email(email):
                return AuthResult(success=False, error_message="無効なメールアドレスです")

            # ユーザー取得
            user = await self._get_user_by_email(email)
            if not user:
                await self._log_auth_attempt(email, False, "user_not_found", client_info)
                return AuthResult(success=False, error_message="認証に失敗しました")

            # アカウント状態チェック
            if not user.is_active:
                return AuthResult(success=False, error_message="アカウントが無効化されています")

            if user.is_locked:
                return AuthResult(success=False, error_message="アカウントがロックされています")

            # パスワード認証
            if auth_method == AuthMethod.PASSWORD:
                if not verify_password(password, user.password_hash):
                    await self._handle_failed_login(user, client_info)
                    return AuthResult(success=False, error_message="認証に失敗しました")

            # MFA チェック
            if user.mfa_enabled and auth_method != AuthMethod.MFA:
                mfa_token = await self._generate_mfa_token(user)
                await self._send_mfa_code(user)

                return AuthResult(
                    success=False,
                    requires_mfa=True,
                    mfa_token=mfa_token,
                    error_message="多要素認証が必要です"
                )

            # 認証成功
            await self._handle_successful_login(user, client_info)

            # トークン生成
            access_token, refresh_token = await self._generate_tokens(user, client_info)

            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=int(self.access_token_expire.total_seconds())
            )

        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return AuthResult(success=False, error_message="認証中にエラーが発生しました")

    async def verify_mfa_code(
        self,
        mfa_token: str,
        mfa_code: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> AuthResult:
        """MFA コード検証"""
        try:
            # MFA トークン検証
            payload = await self._verify_token(mfa_token, token_type="mfa")
            if not payload:
                return AuthResult(success=False, error_message="無効なMFAトークンです")

            # ユーザー取得
            user = await self._get_user_by_id(payload.user_id)
            if not user:
                return AuthResult(success=False, error_message="ユーザーが見つかりません")

            # MFA コード検証
            if not await self._verify_mfa_code(user, mfa_code):
                return AuthResult(success=False, error_message="無効なMFAコードです")

            # 認証成功
            await self._handle_successful_login(user, client_info)

            # トークン生成
            access_token, refresh_token = await self._generate_tokens(user, client_info)

            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=int(self.access_token_expire.total_seconds())
            )

        except Exception as e:
            logging.error(f"MFA verification error: {e}")
            return AuthResult(success=False, error_message="MFA検証中にエラーが発生しました")

    async def refresh_access_token(self, refresh_token: str) -> AuthResult:
        """アクセストークン更新"""
        try:
            # リフレッシュトークン検証
            payload = await self._verify_token(refresh_token, token_type="refresh")
            if not payload:
                return AuthResult(success=False, error_message="無効なリフレッシュトークンです")

            # セッション確認
            session = await self._get_session(payload.session_id)
            if not session or not session.is_active:
                return AuthResult(success=False, error_message="セッションが無効です")

            # ユーザー取得
            user = await self._get_user_by_id(payload.user_id)
            if not user or not user.is_active:
                return AuthResult(success=False, error_message="ユーザーが無効です")

            # 新しいアクセストークン生成
            access_token = await self._generate_access_token(user, session.id)

            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                expires_in=int(self.access_token_expire.total_seconds())
            )

        except Exception as e:
            logging.error(f"Token refresh error: {e}")
            return AuthResult(success=False, error_message="トークン更新中にエラーが発生しました")

    async def logout(self, token: str) -> bool:
        """ログアウト"""
        try:
            payload = await self._verify_token(token)
            if payload:
                # セッション無効化
                await self._invalidate_session(payload.session_id)

                # キャッシュからトークンを削除
                if self.cache_service:
                    await self.cache_service.delete(f"token_{token}")

                logging.info(f"User logged out: {payload.user_id}")

            return True

        except Exception as e:
            logging.error(f"Logout error: {e}")
            return False

    async def verify_token(self, token: str) -> Optional[User]:
        """トークン検証とユーザー取得"""
        try:
            # キャッシュから確認
            if self.cache_service:
                cached_user = await self.cache_service.get(f"token_{token}")
                if cached_user:
                    return User(**cached_user)

            # トークン検証
            payload = await self._verify_token(token)
            if not payload:
                return None

            # ユーザー取得
            user = await self._get_user_by_id(payload.user_id)
            if not user or not user.is_active:
                return None

            # セッション確認
            session = await self._get_session(payload.session_id)
            if not session or not session.is_active:
                return None

            # キャッシュに保存
            if self.cache_service:
                await self.cache_service.set(
                    f"token_{token}",
                    user.dict(),
                    ttl=300  # 5分間キャッシュ
                )

            return user

        except Exception as e:
            logging.error(f"Token verification error: {e}")
            return None

    async def check_permission(
        self,
        user: User,
        resource: str,
        action: str
    ) -> bool:
        """権限チェック"""
        try:
            # スーパーユーザーは全ての権限を持つ
            if user.is_superuser:
                return True

            # ユーザーの役割と権限を取得
            user_permissions = await self._get_user_permissions(user.id)

            # 権限チェック
            permission_key = f"{resource}:{action}"
            return permission_key in user_permissions

        except Exception as e:
            logging.error(f"Permission check error: {e}")
            return False

    async def create_api_key(
        self,
        user: User,
        name: str,
        permissions: List[str],
        expires_at: Optional[datetime] = None
    ) -> Optional[str]:
        """APIキー作成"""
        try:
            # APIキー生成
            api_key = f"dtm_{secrets.token_urlsafe(32)}"

            # データベースに保存
            query = """
            INSERT INTO api_keys (user_id, name, key_hash, permissions, expires_at, created_at)
            VALUES (:user_id, :name, :key_hash, :permissions, :expires_at, :created_at)
            """

            await database.execute(query, {
                "user_id": user.id,
                "name": name,
                "key_hash": hash_password(api_key),
                "permissions": ",".join(permissions),
                "expires_at": expires_at,
                "created_at": datetime.utcnow(),
            })

            logging.info(f"API key created for user {user.id}: {name}")
            return api_key

        except Exception as e:
            logging.error(f"API key creation error: {e}")
            return None

    async def revoke_api_key(self, user: User, api_key_id: str) -> bool:
        """APIキー取り消し"""
        try:
            query = """
            UPDATE api_keys
            SET is_active = FALSE, revoked_at = :revoked_at
            WHERE id = :api_key_id AND user_id = :user_id
            """

            await database.execute(query, {
                "api_key_id": api_key_id,
                "user_id": user.id,
                "revoked_at": datetime.utcnow(),
            })

            return True

        except Exception as e:
            logging.error(f"API key revocation error: {e}")
            return False

    # プライベートメソッド
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """メールアドレスでユーザー取得"""
        query = "SELECT * FROM users WHERE email = :email"
        result = await database.fetch_one(query, {"email": email})
        return User(**result) if result else None

    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """IDでユーザー取得"""
        query = "SELECT * FROM users WHERE id = :user_id"
        result = await database.fetch_one(query, {"user_id": user_id})
        return User(**result) if result else None

    async def _generate_tokens(
        self,
        user: User,
        client_info: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """アクセストークンとリフレッシュトークンを生成"""
        # セッション作成
        session_id = await self._create_session(user, client_info)

        # ユーザー権限取得
        permissions = await self._get_user_permissions(user.id)
        roles = await self._get_user_roles(user.id)

        # アクセストークン
        access_payload = TokenPayload(
            user_id=user.id,
            email=user.email,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.access_token_expire,
            token_type="access"
        )

        access_token = jwt.encode(
            access_payload.__dict__,
            settings.JWT_SECRET_KEY,
            algorithm=self.jwt_algorithm
        )

        # リフレッシュトークン
        refresh_payload = TokenPayload(
            user_id=user.id,
            email=user.email,
            roles=[],
            permissions=[],
            session_id=session_id,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.refresh_token_expire,
            token_type="refresh"
        )

        refresh_token = jwt.encode(
            refresh_payload.__dict__,
            settings.JWT_SECRET_KEY,
            algorithm=self.jwt_algorithm
        )

        return access_token, refresh_token

    async def _generate_access_token(self, user: User, session_id: str) -> str:
        """アクセストークン生成"""
        permissions = await self._get_user_permissions(user.id)
        roles = await self._get_user_roles(user.id)

        payload = TokenPayload(
            user_id=user.id,
            email=user.email,
            roles=roles,
            permissions=permissions,
            session_id=session_id,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.access_token_expire,
            token_type="access"
        )

        return jwt.encode(
            payload.__dict__,
            settings.JWT_SECRET_KEY,
            algorithm=self.jwt_algorithm
        )

    async def _verify_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[TokenPayload]:
        """トークン検証"""
        try:
            payload_dict = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[self.jwt_algorithm]
            )

            payload = TokenPayload(**payload_dict)

            # トークンタイプ確認
            if payload.token_type != token_type:
                return None

            # 有効期限確認
            if datetime.utcnow() > payload.expires_at:
                return None

            return payload

        except jwt.InvalidTokenError:
            return None

    async def _create_session(
        self,
        user: User,
        client_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """セッション作成"""
        session_id = secrets.token_urlsafe(32)

        query = """
        INSERT INTO user_sessions (id, user_id, ip_address, user_agent, expires_at, created_at, is_active)
        VALUES (:session_id, :user_id, :ip_address, :user_agent, :expires_at, :created_at, TRUE)
        """

        await database.execute(query, {
            "session_id": session_id,
            "user_id": user.id,
            "ip_address": client_info.get("ip_address") if client_info else None,
            "user_agent": client_info.get("user_agent") if client_info else None,
            "expires_at": datetime.utcnow() + self.session_timeout,
            "created_at": datetime.utcnow(),
        })

        # 古いセッションを削除
        await self._cleanup_user_sessions(user.id)

        return session_id

    async def _get_session(self, session_id: str) -> Optional[dict]:
        """セッション取得"""
        query = "SELECT * FROM user_sessions WHERE id = :session_id"
        return await database.fetch_one(query, {"session_id": session_id})

    async def _invalidate_session(self, session_id: str) -> None:
        """セッション無効化"""
        query = """
        UPDATE user_sessions
        SET is_active = FALSE, logged_out_at = :logged_out_at
        WHERE id = :session_id
        """

        await database.execute(query, {
            "session_id": session_id,
            "logged_out_at": datetime.utcnow(),
        })

    async def _get_user_permissions(self, user_id: str) -> List[str]:
        """ユーザー権限取得"""
        query = """
        SELECT DISTINCT p.resource, p.action
        FROM permissions p
        JOIN role_permissions rp ON p.id = rp.permission_id
        JOIN user_roles ur ON rp.role_id = ur.role_id
        WHERE ur.user_id = :user_id
        """

        results = await database.fetch_all(query, {"user_id": user_id})
        return [f"{row['resource']}:{row['action']}" for row in results]

    async def _get_user_roles(self, user_id: str) -> List[str]:
        """ユーザー役割取得"""
        query = """
        SELECT r.name
        FROM roles r
        JOIN user_roles ur ON r.id = ur.role_id
        WHERE ur.user_id = :user_id
        """

        results = await database.fetch_all(query, {"user_id": user_id})
        return [row['name'] for row in results]

    async def _handle_successful_login(
        self,
        user: User,
        client_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """ログイン成功処理"""
        # ログイン回数をリセット
        await database.execute(
            "UPDATE users SET failed_login_attempts = 0, last_login_at = :last_login WHERE id = :user_id",
            {"user_id": user.id, "last_login": datetime.utcnow()}
        )

        # ログ記録
        await self._log_auth_attempt(user.email, True, "success", client_info)

        # 通知送信（必要に応じて）
        if self.notification_service and client_info:
            await self.notification_service.send_login_notification(user, client_info)

    async def _handle_failed_login(
        self,
        user: User,
        client_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """ログイン失敗処理"""
        # 失敗回数増加
        failed_attempts = user.failed_login_attempts + 1

        # アカウントロック判定
        if failed_attempts >= 5:
            await database.execute(
                "UPDATE users SET is_locked = TRUE, locked_at = :locked_at WHERE id = :user_id",
                {"user_id": user.id, "locked_at": datetime.utcnow()}
            )

            # ロック通知
            if self.notification_service:
                await self.notification_service.send_account_locked_notification(user)

        await database.execute(
            "UPDATE users SET failed_login_attempts = :attempts WHERE id = :user_id",
            {"user_id": user.id, "attempts": failed_attempts}
        )

        # ログ記録
        await self._log_auth_attempt(user.email, False, "invalid_password", client_info)

    async def _log_auth_attempt(
        self,
        email: str,
        success: bool,
        reason: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """認証試行ログ記録"""
        query = """
        INSERT INTO auth_logs (email, success, reason, ip_address, user_agent, attempted_at)
        VALUES (:email, :success, :reason, :ip_address, :user_agent, :attempted_at)
        """

        await database.execute(query, {
            "email": email,
            "success": success,
            "reason": reason,
            "ip_address": client_info.get("ip_address") if client_info else None,
            "user_agent": client_info.get("user_agent") if client_info else None,
            "attempted_at": datetime.utcnow(),
        })

    async def _cleanup_user_sessions(self, user_id: str) -> None:
        """古いセッションをクリーンアップ"""
        # 期限切れセッションを無効化
        await database.execute(
            "UPDATE user_sessions SET is_active = FALSE WHERE user_id = :user_id AND expires_at < :now",
            {"user_id": user_id, "now": datetime.utcnow()}
        )

        # セッション数制限
        query = """
        SELECT id FROM user_sessions
        WHERE user_id = :user_id AND is_active = TRUE
        ORDER BY created_at DESC
        OFFSET :max_sessions
        """

        old_sessions = await database.fetch_all(query, {
            "user_id": user_id,
            "max_sessions": self.max_sessions_per_user
        })

        for session in old_sessions:
            await self._invalidate_session(session['id'])


# グローバルインスタンス
auth_service = AuthService()