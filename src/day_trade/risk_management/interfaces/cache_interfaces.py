#!/usr/bin/env python3
"""
Cache Management Interfaces
キャッシュ管理インターフェース

Redis、Memory、File等の各種キャッシュ実装に対応した抽象インターフェース
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

class CacheStrategy(Enum):
    """キャッシュ戦略"""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In First Out
    RANDOM = "random"        # ランダム削除
    TTL_BASED = "ttl_based"  # TTL ベース

class CacheEntryStatus(Enum):
    """キャッシュエントリーステータス"""
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    INVALID = "invalid"

@dataclass
class CacheEntry:
    """キャッシュエントリー"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """期限切れ判定"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).seconds > self.ttl_seconds

@dataclass
class CacheStats:
    """キャッシュ統計"""
    total_entries: int
    memory_usage_bytes: int
    hit_rate: float
    miss_rate: float
    avg_access_time_ms: float
    total_hits: int
    total_misses: int
    evictions: int
    errors: int

class ICacheProvider(ABC):
    """キャッシュプロバイダーインターフェース"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """エントリー取得"""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """エントリー設定"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """エントリー削除"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """エントリー存在確認"""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """エントリークリア"""
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """キー一覧取得"""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """統計取得"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        pass

class ICacheSerializer(ABC):
    """キャッシュシリアライザーインターフェース"""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """オブジェクトシリアライズ"""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """データデシリアライズ"""
        pass

    @abstractmethod
    def get_content_type(self) -> str:
        """コンテンツタイプ取得"""
        pass

class ICacheEvictionPolicy(ABC):
    """キャッシュ削除ポリシーインターフェース"""

    @abstractmethod
    async def should_evict(
        self,
        entries: List[CacheEntry],
        new_entry_size: int,
        max_capacity: int
    ) -> List[str]:
        """削除対象キー決定"""
        pass

    @abstractmethod
    def get_policy_name(self) -> str:
        """ポリシー名取得"""
        pass

class IDistributedLock(ABC):
    """分散ロックインターフェース"""

    @abstractmethod
    async def acquire(
        self,
        key: str,
        timeout_seconds: int = 10,
        ttl_seconds: int = 60
    ) -> bool:
        """ロック取得"""
        pass

    @abstractmethod
    async def release(self, key: str) -> bool:
        """ロック解除"""
        pass

    @abstractmethod
    async def is_locked(self, key: str) -> bool:
        """ロック状態確認"""
        pass

    @abstractmethod
    async def extend_lock(self, key: str, ttl_seconds: int) -> bool:
        """ロック延長"""
        pass

class ICacheCluster(ABC):
    """キャッシュクラスターインターフェース"""

    @abstractmethod
    async def add_node(self, node_config: Dict[str, Any]) -> bool:
        """ノード追加"""
        pass

    @abstractmethod
    async def remove_node(self, node_id: str) -> bool:
        """ノード削除"""
        pass

    @abstractmethod
    async def get_cluster_status(self) -> Dict[str, Any]:
        """クラスター状態取得"""
        pass

    @abstractmethod
    async def rebalance(self) -> bool:
        """クラスター再バランス"""
        pass

class ICacheMiddleware(ABC):
    """キャッシュミドルウェアインターフェース"""

    @abstractmethod
    async def before_get(self, key: str) -> Optional[str]:
        """取得前処理"""
        pass

    @abstractmethod
    async def after_get(
        self,
        key: str,
        value: Optional[Any],
        cache_status: CacheEntryStatus
    ) -> Optional[Any]:
        """取得後処理"""
        pass

    @abstractmethod
    async def before_set(
        self,
        key: str,
        value: Any
    ) -> Optional[tuple]:  # (key, value)
        """設定前処理"""
        pass

    @abstractmethod
    async def after_set(
        self,
        key: str,
        value: Any,
        success: bool
    ) -> None:
        """設定後処理"""
        pass

# ヘルパー関数

def create_cache_key(
    namespace: str,
    identifier: str,
    version: Optional[str] = None
) -> str:
    """キャッシュキー作成"""
    parts = [namespace, identifier]
    if version:
        parts.append(version)
    return ":".join(parts)

def calculate_ttl_seconds(
    base_ttl: int,
    jitter_ratio: float = 0.1
) -> int:
    """TTL計算（ジッター付き）"""
    import random
    jitter = int(base_ttl * jitter_ratio * (random.random() - 0.5))
    return max(1, base_ttl + jitter)

def estimate_object_size(obj: Any) -> int:
    """オブジェクトサイズ推定"""
    try:
        import pickle
        return len(pickle.dumps(obj))
    except Exception:
        return len(str(obj).encode('utf-8'))

def is_cache_key_valid(key: str) -> bool:
    """キャッシュキー検証"""
    if not key or len(key) > 250:
        return False

    # 無効な文字チェック
    invalid_chars = {' ', '\n', '\r', '\t', '\0'}
    return not any(char in key for char in invalid_chars)
