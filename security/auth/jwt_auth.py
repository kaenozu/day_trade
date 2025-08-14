#!/usr/bin/env python3
"""
Issue #800 Phase 5: JWT認証システム
Day Trade ML System セキュアアクセス管理
"""

import os
import jwt
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from functools import wraps
from flask import request, jsonify, current_app
import redis
import bcrypt

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class User:
    """ユーザー情報"""
    id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

@dataclass
class Role:
    """ロール情報"""
    name: str
    permissions: List[str]
    description: str = ""

class JWTAuthManager:
    """JWT認証管理システム"""

    def __init__(self):
        self.secret_key = os.getenv('JWT_SECRET_KEY', 'day-trade-ml-system-secret')
        self.algorithm = 'HS256'
        self.access_token_expire = timedelta(hours=24)
        self.refresh_token_expire = timedelta(days=7)

        # Redis接続（ブラックリスト管理）
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_AUTH_DB', 1)),
            decode_responses=True
        )

        # 事前定義ロール
        self.roles = {
            'admin': Role(
                name='admin',
                permissions=[
                    'ml.predict', 'ml.train', 'ml.deploy',
                    'data.read', 'data.write', 'data.delete',
                    'scheduler.read', 'scheduler.write',
                    'monitoring.read', 'monitoring.write',
                    'security.read', 'security.write',
                    'system.admin'
                ],
                description='システム管理者'
            ),
            'ml_engineer': Role(
                name='ml_engineer',
                permissions=[
                    'ml.predict', 'ml.train', 'ml.deploy',
                    'data.read', 'data.write',
                    'scheduler.read',
                    'monitoring.read'
                ],
                description='ML エンジニア'
            ),
            'data_analyst': Role(
                name='data_analyst',
                permissions=[
                    'ml.predict',
                    'data.read',
                    'scheduler.read',
                    'monitoring.read'
                ],
                description='データ分析者'
            ),
            'viewer': Role(
                name='viewer',
                permissions=[
                    'ml.predict',
                    'data.read',
                    'monitoring.read'
                ],
                description='読み取り専用'
            ),
            'api_service': Role(
                name='api_service',
                permissions=[
                    'ml.predict',
                    'data.read'
                ],
                description='API サービス'
            )
        }

    def generate_password_hash(self, password: str) -> str:
        """パスワードハッシュ生成"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str, password_hash: str) -> bool:
        """パスワード検証"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    def create_access_token(self, user: User) -> str:
        """アクセストークン生成"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'roles': user.roles,
            'permissions': user.permissions,
            'type': 'access',
            'exp': datetime.utcnow() + self.access_token_expire,
            'iat': datetime.utcnow(),
            'iss': 'day-trade-ml-system'
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # トークンをRedisに記録（有効性管理）
        self.redis_client.setex(
            f"access_token:{user.id}:{self._get_token_id(token)}",
            int(self.access_token_expire.total_seconds()),
            'valid'
        )

        return token

    def create_refresh_token(self, user: User) -> str:
        """リフレッシュトークン生成"""
        payload = {
            'user_id': user.id,
            'type': 'refresh',
            'exp': datetime.utcnow() + self.refresh_token_expire,
            'iat': datetime.utcnow(),
            'iss': 'day-trade-ml-system'
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # リフレッシュトークンをRedisに記録
        self.redis_client.setex(
            f"refresh_token:{user.id}:{self._get_token_id(token)}",
            int(self.refresh_token_expire.total_seconds()),
            'valid'
        )

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """トークン検証"""
        try:
            # JWT デコード
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # ブラックリスト確認
            token_id = self._get_token_id(token)
            if self.redis_client.get(f"blacklist:{token_id}"):
                logger.warning(f"Blacklisted token used: {token_id}")
                return None

            # トークン有効性確認
            if payload['type'] == 'access':
                if not self.redis_client.get(f"access_token:{payload['user_id']}:{token_id}"):
                    logger.warning(f"Invalid access token: {token_id}")
                    return None
            elif payload['type'] == 'refresh':
                if not self.redis_client.get(f"refresh_token:{payload['user_id']}:{token_id}"):
                    logger.warning(f"Invalid refresh token: {token_id}")
                    return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None

    def revoke_token(self, token: str) -> bool:
        """トークン無効化（ブラックリスト追加）"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            token_id = self._get_token_id(token)

            # ブラックリストに追加
            remaining_time = payload['exp'] - datetime.utcnow().timestamp()
            if remaining_time > 0:
                self.redis_client.setex(
                    f"blacklist:{token_id}",
                    int(remaining_time),
                    'revoked'
                )

            # 対応するトークンをRedisから削除
            if payload['type'] == 'access':
                self.redis_client.delete(f"access_token:{payload['user_id']}:{token_id}")
            elif payload['type'] == 'refresh':
                self.redis_client.delete(f"refresh_token:{payload['user_id']}:{token_id}")

            logger.info(f"Token revoked: {token_id}")
            return True

        except Exception as e:
            logger.error(f"Token revocation failed: {str(e)}")
            return False

    def revoke_all_user_tokens(self, user_id: str) -> bool:
        """ユーザーの全トークン無効化"""
        try:
            # ユーザーの全トークンを検索・削除
            access_keys = self.redis_client.keys(f"access_token:{user_id}:*")
            refresh_keys = self.redis_client.keys(f"refresh_token:{user_id}:*")

            for key in access_keys + refresh_keys:
                self.redis_client.delete(key)

            logger.info(f"All tokens revoked for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke all tokens for user {user_id}: {str(e)}")
            return False

    def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """権限確認"""
        # 管理者は全権限
        if 'system.admin' in user_permissions:
            return True

        # 完全一致確認
        if required_permission in user_permissions:
            return True

        # ワイルドカード権限確認（例: ml.* → ml.predict）
        for permission in user_permissions:
            if permission.endswith('.*'):
                prefix = permission[:-1]
                if required_permission.startswith(prefix):
                    return True

        return False

    def _get_token_id(self, token: str) -> str:
        """トークンID生成（ハッシュ）"""
        return hashlib.sha256(token.encode()).hexdigest()[:16]

class AuthDecorator:
    """認証デコレータ"""

    def __init__(self, auth_manager: JWTAuthManager):
        self.auth_manager = auth_manager

    def require_auth(self, required_permission: str = None):
        """認証必須デコレータ"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Authorization ヘッダー確認
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Authorization header missing or invalid'}), 401

                token = auth_header.split(' ')[1]

                # トークン検証
                payload = self.auth_manager.verify_token(token)
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401

                # 権限確認
                if required_permission:
                    user_permissions = payload.get('permissions', [])
                    if not self.auth_manager.has_permission(user_permissions, required_permission):
                        return jsonify({
                            'error': 'Insufficient permissions',
                            'required': required_permission
                        }), 403

                # リクエストにユーザー情報追加
                request.current_user = payload

                return f(*args, **kwargs)

            return decorated_function
        return decorator

    def require_role(self, required_roles: Union[str, List[str]]):
        """ロール必須デコレータ"""
        if isinstance(required_roles, str):
            required_roles = [required_roles]

        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Authorization header missing or invalid'}), 401

                token = auth_header.split(' ')[1]
                payload = self.auth_manager.verify_token(token)
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401

                # ロール確認
                user_roles = payload.get('roles', [])
                if not any(role in user_roles for role in required_roles):
                    return jsonify({
                        'error': 'Insufficient role',
                        'required_roles': required_roles,
                        'user_roles': user_roles
                    }), 403

                request.current_user = payload
                return f(*args, **kwargs)

            return decorated_function
        return decorator

# ユーザー管理クラス
class UserManager:
    """ユーザー管理システム"""

    def __init__(self, auth_manager: JWTAuthManager):
        self.auth_manager = auth_manager

        # デフォルト管理者ユーザー作成
        self.users = {
            'admin': User(
                id='admin',
                username='admin',
                email='admin@company.com',
                roles=['admin'],
                permissions=auth_manager.roles['admin'].permissions,
                created_at=datetime.utcnow()
            ),
            'ml_service': User(
                id='ml_service',
                username='ml_service',
                email='ml-service@system.internal',
                roles=['api_service'],
                permissions=auth_manager.roles['api_service'].permissions,
                created_at=datetime.utcnow()
            )
        }

        # パスワードハッシュ（実際の運用では外部データベース使用）
        self.password_hashes = {
            'admin': auth_manager.generate_password_hash('DayTrade2025!Admin'),
            'ml_service': auth_manager.generate_password_hash('MLService2025!Token')
        }

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """ユーザー認証"""
        user = self.users.get(username)
        if not user or not user.is_active:
            return None

        password_hash = self.password_hashes.get(username)
        if not password_hash:
            return None

        if self.auth_manager.verify_password(password, password_hash):
            # 最終ログイン時刻更新
            user.last_login = datetime.utcnow()
            logger.info(f"User authenticated successfully: {username}")
            return user

        logger.warning(f"Authentication failed for user: {username}")
        return None

    def create_user(self, username: str, email: str, password: str, roles: List[str]) -> Optional[User]:
        """ユーザー作成"""
        if username in self.users:
            return None

        # ロール権限集約
        permissions = []
        for role_name in roles:
            role = self.auth_manager.roles.get(role_name)
            if role:
                permissions.extend(role.permissions)

        permissions = list(set(permissions))  # 重複除去

        user = User(
            id=username,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=datetime.utcnow()
        )

        self.users[username] = user
        self.password_hashes[username] = self.auth_manager.generate_password_hash(password)

        logger.info(f"User created: {username} with roles: {roles}")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """ユーザー取得"""
        return self.users.get(user_id)

    def update_user_roles(self, user_id: str, new_roles: List[str]) -> bool:
        """ユーザーロール更新"""
        user = self.users.get(user_id)
        if not user:
            return False

        # 新しい権限計算
        permissions = []
        for role_name in new_roles:
            role = self.auth_manager.roles.get(role_name)
            if role:
                permissions.extend(role.permissions)

        user.roles = new_roles
        user.permissions = list(set(permissions))

        # 既存トークン無効化（権限変更反映）
        self.auth_manager.revoke_all_user_tokens(user_id)

        logger.info(f"User roles updated: {user_id} -> {new_roles}")
        return True

if __name__ == '__main__':
    # テスト用サンプル
    auth_manager = JWTAuthManager()
    user_manager = UserManager(auth_manager)
    auth_decorator = AuthDecorator(auth_manager)

    # サンプルユーザー認証
    user = user_manager.authenticate_user('admin', 'DayTrade2025!Admin')
    if user:
        access_token = auth_manager.create_access_token(user)
        refresh_token = auth_manager.create_refresh_token(user)

        print(f"Access Token: {access_token}")
        print(f"Refresh Token: {refresh_token}")

        # トークン検証テスト
        payload = auth_manager.verify_token(access_token)
        print(f"Token Payload: {payload}")