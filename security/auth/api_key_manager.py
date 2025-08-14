#!/usr/bin/env python3
"""
Issue #800 Phase 5: API Key管理システム
Day Trade ML System API認証・管理
"""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import redis
from enum import Enum

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIKeyStatus(Enum):
    """API Key ステータス"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"

@dataclass
class APIKey:
    """API Key 情報"""
    key_id: str
    key_hash: str
    name: str
    description: str
    permissions: List[str]
    rate_limit: int  # requests per minute
    status: APIKeyStatus
    owner: str
    service: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    ip_whitelist: Optional[List[str]] = None

class APIKeyManager:
    """API Key 管理システム"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_APIKEY_DB', 2)),
            decode_responses=True
        )

        # レート制限用Redis
        self.rate_limit_db = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_RATELIMIT_DB', 3)),
            decode_responses=True
        )

        # プリセット権限セット
        self.permission_sets = {
            'ml_full': [
                'ml.predict', 'ml.train', 'ml.model_info',
                'data.read', 'monitoring.read'
            ],
            'ml_predict_only': [
                'ml.predict', 'ml.model_info'
            ],
            'data_read': [
                'data.read', 'monitoring.read'
            ],
            'monitoring_only': [
                'monitoring.read'
            ],
            'admin': [
                'ml.*', 'data.*', 'scheduler.*',
                'monitoring.*', 'security.*', 'system.*'
            ]
        }

        # 初期システムAPIキー作成
        self._create_default_api_keys()

    def generate_api_key(
        self,
        name: str,
        description: str,
        permissions: List[str],
        owner: str,
        service: str = "external",
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None,
        ip_whitelist: Optional[List[str]] = None
    ) -> Tuple[str, APIKey]:
        """API Key 生成"""

        # キー生成
        key_prefix = "dt"  # Day Trade
        key_body = secrets.token_urlsafe(32)
        api_key = f"{key_prefix}_{key_body}"

        # キーハッシュ
        key_hash = self._hash_key(api_key)
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # 有効期限設定
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # API Key オブジェクト作成
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            permissions=permissions,
            rate_limit=rate_limit,
            status=APIKeyStatus.ACTIVE,
            owner=owner,
            service=service,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_whitelist=ip_whitelist
        )

        # Redisに保存
        self._store_api_key(api_key_obj)

        logger.info(f"API Key generated: {name} for {owner}")
        return api_key, api_key_obj

    def verify_api_key(self, api_key: str, required_permission: str = None, client_ip: str = None) -> Optional[APIKey]:
        """API Key 検証"""

        key_hash = self._hash_key(api_key)

        # Redis から取得
        api_key_obj = self._get_api_key_by_hash(key_hash)
        if not api_key_obj:
            logger.warning(f"Invalid API key used from IP: {client_ip}")
            return None

        # ステータス確認
        if api_key_obj.status != APIKeyStatus.ACTIVE:
            logger.warning(f"Inactive API key used: {api_key_obj.name} (Status: {api_key_obj.status})")
            return None

        # 有効期限確認
        if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
            api_key_obj.status = APIKeyStatus.EXPIRED
            self._store_api_key(api_key_obj)
            logger.warning(f"Expired API key used: {api_key_obj.name}")
            return None

        # IP ホワイトリスト確認
        if api_key_obj.ip_whitelist and client_ip:
            if not any(self._ip_in_range(client_ip, allowed_ip) for allowed_ip in api_key_obj.ip_whitelist):
                logger.warning(f"API key used from unauthorized IP: {client_ip} for key {api_key_obj.name}")
                return None

        # 権限確認
        if required_permission and not self._has_permission(api_key_obj.permissions, required_permission):
            logger.warning(f"Insufficient permission for API key: {api_key_obj.name}, required: {required_permission}")
            return None

        # レート制限確認
        if not self._check_rate_limit(api_key_obj.key_id, api_key_obj.rate_limit):
            logger.warning(f"Rate limit exceeded for API key: {api_key_obj.name}")
            return None

        # 使用記録更新
        api_key_obj.last_used = datetime.utcnow()
        api_key_obj.usage_count += 1
        self._store_api_key(api_key_obj)

        return api_key_obj

    def revoke_api_key(self, key_id: str, reason: str = "Manual revocation") -> bool:
        """API Key 無効化"""
        api_key_obj = self._get_api_key_by_id(key_id)
        if not api_key_obj:
            return False

        api_key_obj.status = APIKeyStatus.REVOKED
        self._store_api_key(api_key_obj)

        # 無効化ログ記録
        self.redis_client.hset(
            f"api_key_revocation:{key_id}",
            mapping={
                'revoked_at': datetime.utcnow().isoformat(),
                'reason': reason
            }
        )

        logger.info(f"API Key revoked: {api_key_obj.name} (Reason: {reason})")
        return True

    def suspend_api_key(self, key_id: str, reason: str = "Manual suspension") -> bool:
        """API Key 一時停止"""
        api_key_obj = self._get_api_key_by_id(key_id)
        if not api_key_obj:
            return False

        api_key_obj.status = APIKeyStatus.SUSPENDED
        self._store_api_key(api_key_obj)

        logger.info(f"API Key suspended: {api_key_obj.name} (Reason: {reason})")
        return True

    def reactivate_api_key(self, key_id: str) -> bool:
        """API Key 再有効化"""
        api_key_obj = self._get_api_key_by_id(key_id)
        if not api_key_obj or api_key_obj.status == APIKeyStatus.REVOKED:
            return False

        api_key_obj.status = APIKeyStatus.ACTIVE
        self._store_api_key(api_key_obj)

        logger.info(f"API Key reactivated: {api_key_obj.name}")
        return True

    def update_api_key_permissions(self, key_id: str, new_permissions: List[str]) -> bool:
        """API Key 権限更新"""
        api_key_obj = self._get_api_key_by_id(key_id)
        if not api_key_obj:
            return False

        old_permissions = api_key_obj.permissions.copy()
        api_key_obj.permissions = new_permissions
        self._store_api_key(api_key_obj)

        logger.info(f"API Key permissions updated: {api_key_obj.name} - {old_permissions} -> {new_permissions}")
        return True

    def get_api_key_usage_stats(self, key_id: str) -> Optional[Dict]:
        """API Key 使用統計取得"""
        api_key_obj = self._get_api_key_by_id(key_id)
        if not api_key_obj:
            return None

        # 時間別使用統計
        hourly_stats = {}
        daily_stats = {}

        for hour in range(24):
            count_key = f"usage_hourly:{key_id}:{hour}"
            hourly_stats[hour] = int(self.rate_limit_db.get(count_key) or 0)

        for day in range(7):
            count_key = f"usage_daily:{key_id}:{day}"
            daily_stats[day] = int(self.rate_limit_db.get(count_key) or 0)

        return {
            'key_info': asdict(api_key_obj),
            'hourly_usage': hourly_stats,
            'daily_usage': daily_stats,
            'total_usage': api_key_obj.usage_count
        }

    def list_api_keys(self, owner: str = None, service: str = None, status: APIKeyStatus = None) -> List[APIKey]:
        """API Key 一覧取得"""
        all_keys = []

        # 全キー取得
        for key in self.redis_client.keys("api_key:*"):
            api_key_data = self.redis_client.hgetall(key)
            if api_key_data:
                api_key_obj = self._deserialize_api_key(api_key_data)
                all_keys.append(api_key_obj)

        # フィルタリング
        filtered_keys = all_keys
        if owner:
            filtered_keys = [k for k in filtered_keys if k.owner == owner]
        if service:
            filtered_keys = [k for k in filtered_keys if k.service == service]
        if status:
            filtered_keys = [k for k in filtered_keys if k.status == status]

        return filtered_keys

    def rotate_api_key(self, key_id: str) -> Tuple[str, APIKey]:
        """API Key ローテーション（新しいキー生成・古いキー無効化）"""
        old_key = self._get_api_key_by_id(key_id)
        if not old_key:
            raise ValueError(f"API Key not found: {key_id}")

        # 新しいキー生成
        new_api_key, new_key_obj = self.generate_api_key(
            name=f"{old_key.name}_rotated",
            description=f"Rotated from {old_key.name}",
            permissions=old_key.permissions,
            owner=old_key.owner,
            service=old_key.service,
            rate_limit=old_key.rate_limit,
            ip_whitelist=old_key.ip_whitelist
        )

        # 古いキー無効化
        self.revoke_api_key(key_id, "Key rotation")

        logger.info(f"API Key rotated: {old_key.name} -> {new_key_obj.name}")
        return new_api_key, new_key_obj

    def _create_default_api_keys(self):
        """デフォルトシステムAPIキー作成"""
        default_keys = [
            {
                'name': 'ml_service_internal',
                'description': 'ML Service internal API key',
                'permissions': self.permission_sets['ml_full'],
                'owner': 'system',
                'service': 'ml-service',
                'rate_limit': 10000,
                'expires_in_days': None
            },
            {
                'name': 'data_service_internal',
                'description': 'Data Service internal API key',
                'permissions': self.permission_sets['data_read'],
                'owner': 'system',
                'service': 'data-service',
                'rate_limit': 5000,
                'expires_in_days': None
            },
            {
                'name': 'scheduler_service_internal',
                'description': 'Scheduler Service internal API key',
                'permissions': ['ml.predict', 'data.read', 'scheduler.read'],
                'owner': 'system',
                'service': 'scheduler-service',
                'rate_limit': 2000,
                'expires_in_days': None
            }
        ]

        for key_config in default_keys:
            # 既存確認
            existing_keys = self.list_api_keys(
                owner=key_config['owner'],
                service=key_config['service']
            )

            if not any(k.name == key_config['name'] for k in existing_keys):
                api_key, api_key_obj = self.generate_api_key(**key_config)
                logger.info(f"Default API Key created: {key_config['name']} = {api_key}")

    def _hash_key(self, api_key: str) -> str:
        """API Key ハッシュ化"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _store_api_key(self, api_key_obj: APIKey):
        """API Key Redis保存"""
        key = f"api_key:{api_key_obj.key_id}"
        data = {
            'key_id': api_key_obj.key_id,
            'key_hash': api_key_obj.key_hash,
            'name': api_key_obj.name,
            'description': api_key_obj.description,
            'permissions': json.dumps(api_key_obj.permissions),
            'rate_limit': str(api_key_obj.rate_limit),
            'status': api_key_obj.status.value,
            'owner': api_key_obj.owner,
            'service': api_key_obj.service,
            'created_at': api_key_obj.created_at.isoformat(),
            'expires_at': api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else '',
            'last_used': api_key_obj.last_used.isoformat() if api_key_obj.last_used else '',
            'usage_count': str(api_key_obj.usage_count),
            'ip_whitelist': json.dumps(api_key_obj.ip_whitelist) if api_key_obj.ip_whitelist else ''
        }

        self.redis_client.hset(key, mapping=data)

        # ハッシュによる逆引きインデックス
        self.redis_client.set(f"key_hash_to_id:{api_key_obj.key_hash}", api_key_obj.key_id)

    def _get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """ハッシュによるAPI Key取得"""
        key_id = self.redis_client.get(f"key_hash_to_id:{key_hash}")
        if not key_id:
            return None

        return self._get_api_key_by_id(key_id)

    def _get_api_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """IDによるAPI Key取得"""
        key = f"api_key:{key_id}"
        data = self.redis_client.hgetall(key)

        if not data:
            return None

        return self._deserialize_api_key(data)

    def _deserialize_api_key(self, data: Dict[str, str]) -> APIKey:
        """Redis データからAPI Key オブジェクト復元"""
        return APIKey(
            key_id=data['key_id'],
            key_hash=data['key_hash'],
            name=data['name'],
            description=data['description'],
            permissions=json.loads(data['permissions']),
            rate_limit=int(data['rate_limit']),
            status=APIKeyStatus(data['status']),
            owner=data['owner'],
            service=data['service'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None,
            usage_count=int(data['usage_count']),
            ip_whitelist=json.loads(data['ip_whitelist']) if data['ip_whitelist'] else None
        )

    def _has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """権限確認"""
        # 完全一致
        if required_permission in user_permissions:
            return True

        # ワイルドカード権限
        for permission in user_permissions:
            if permission.endswith('.*'):
                prefix = permission[:-1]
                if required_permission.startswith(prefix):
                    return True

        return False

    def _check_rate_limit(self, key_id: str, rate_limit: int) -> bool:
        """レート制限確認"""
        current_minute = datetime.utcnow().strftime('%Y-%m-%d-%H-%M')
        count_key = f"rate_limit:{key_id}:{current_minute}"

        current_count = self.rate_limit_db.get(count_key)
        if current_count is None:
            current_count = 0
        else:
            current_count = int(current_count)

        if current_count >= rate_limit:
            return False

        # カウンタ更新
        pipe = self.rate_limit_db.pipeline()
        pipe.incr(count_key)
        pipe.expire(count_key, 60)  # 1分後に期限切れ
        pipe.execute()

        return True

    def _ip_in_range(self, client_ip: str, allowed_range: str) -> bool:
        """IP範囲確認（簡易版）"""
        # 完全一致
        if client_ip == allowed_range:
            return True

        # CIDR記法対応（簡易版）
        if '/' in allowed_range:
            # 実際の実装ではipaddressモジュール使用
            return client_ip.startswith(allowed_range.split('/')[0].rsplit('.', 1)[0])

        return False

if __name__ == '__main__':
    # テスト用
    api_key_manager = APIKeyManager()

    # サンプルAPIキー生成
    api_key, key_obj = api_key_manager.generate_api_key(
        name='test_key',
        description='Test API key',
        permissions=['ml.predict', 'data.read'],
        owner='test_user',
        rate_limit=100,
        expires_in_days=30
    )

    print(f"Generated API Key: {api_key}")
    print(f"Key Object: {key_obj}")

    # キー検証テスト
    verified_key = api_key_manager.verify_api_key(api_key, 'ml.predict')
    print(f"Verification Result: {verified_key is not None}")