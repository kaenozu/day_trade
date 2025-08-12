#!/usr/bin/env python3
"""
Redis Cache Provider
Redisキャッシュプロバイダー

分散キャッシュとして使用可能な高性能Redis実装
"""

import time
from typing import Any, Dict, List, Optional

from ..exceptions.risk_exceptions import CacheError, ExternalServiceError
from ..interfaces.cache_interfaces import ICacheProvider

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    aioredis = None
    REDIS_AVAILABLE = False


class RedisCacheProvider(ICacheProvider):
    """Redisキャッシュプロバイダー"""

    def __init__(self, config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise CacheError(
                "Redis is not available. Please install redis-py: pip install redis",
                cache_provider="redis",
                operation="init",
            )

        # Redis設定
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.password = config.get("password")
        self.socket_timeout = config.get("socket_timeout", 2.0)
        self.socket_connect_timeout = config.get("socket_connect_timeout", 2.0)
        self.socket_keepalive = config.get("socket_keepalive", True)
        self.socket_keepalive_options = config.get("socket_keepalive_options", {})
        self.connection_pool_size = config.get("connection_pool_size", 10)
        self.retry_on_timeout = config.get("retry_on_timeout", True)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 0.1)

        # キー設定
        self.key_prefix = config.get("key_prefix", "risk_cache:")
        self.default_ttl_seconds = config.get("ttl_seconds", 3600)

        # シリアライゼーション
        self._serializer = config.get("serializer")
        if not self._serializer:
            from .serializers import JsonSerializer

            self._serializer = JsonSerializer()

        # 統計情報
        self.enable_stats = config.get("enable_stats", True)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "total_requests": 0,
        }

        # Redis接続初期化
        self._setup_redis_connections()

    def _setup_redis_connections(self):
        """Redis接続設定"""
        try:
            # 同期Redis接続
            self._redis_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                socket_keepalive=self.socket_keepalive,
                socket_keepalive_options=self.socket_keepalive_options,
                max_connections=self.connection_pool_size,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=False,  # バイナリデータ対応
            )

            self._redis = redis.Redis(connection_pool=self._redis_pool)

            # 非同期Redis接続
            self._async_redis = None

            # 接続テスト
            self._redis.ping()

        except Exception as e:
            raise CacheError(
                f"Failed to connect to Redis at {self.host}:{self.port}",
                cache_provider="redis",
                operation="connect",
                cause=e,
            )

    def get(self, key: str) -> Optional[Any]:
        """キー取得"""
        full_key = self._make_key(key)

        for attempt in range(self.max_retries + 1):
            try:
                if self.enable_stats:
                    self._stats["total_requests"] += 1

                # Redis から値取得
                raw_value = self._redis.get(full_key)

                if raw_value is None:
                    if self.enable_stats:
                        self._stats["misses"] += 1
                    return None

                # デシリアライゼーション
                value = self._serializer.deserialize(raw_value)

                if self.enable_stats:
                    self._stats["hits"] += 1

                return value

            except redis.ConnectionError as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

                if self.enable_stats:
                    self._stats["errors"] += 1

                raise ExternalServiceError(
                    f"Redis connection failed for key: {key}",
                    service_name="redis",
                    endpoint=f"{self.host}:{self.port}",
                    cause=e,
                )

            except Exception as e:
                if self.enable_stats:
                    self._stats["errors"] += 1

                raise CacheError(
                    f"Failed to get cache item for key: {key}",
                    cache_key=key,
                    operation="get",
                    cache_provider="redis",
                    cause=e,
                )

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """キー設定"""
        full_key = self._make_key(key)

        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        for attempt in range(self.max_retries + 1):
            try:
                if self.enable_stats:
                    self._stats["total_requests"] += 1

                # シリアライゼーション
                serialized_value = self._serializer.serialize(value)

                # Redis に設定
                result = self._redis.setex(full_key, ttl_seconds, serialized_value)

                if self.enable_stats:
                    self._stats["sets"] += 1

                return bool(result)

            except redis.ConnectionError as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

                if self.enable_stats:
                    self._stats["errors"] += 1

                raise ExternalServiceError(
                    f"Redis connection failed for key: {key}",
                    service_name="redis",
                    endpoint=f"{self.host}:{self.port}",
                    cause=e,
                )

            except Exception as e:
                if self.enable_stats:
                    self._stats["errors"] += 1

                raise CacheError(
                    f"Failed to set cache item for key: {key}",
                    cache_key=key,
                    operation="set",
                    cache_provider="redis",
                    cause=e,
                )

    def delete(self, key: str) -> bool:
        """キー削除"""
        full_key = self._make_key(key)

        for attempt in range(self.max_retries + 1):
            try:
                if self.enable_stats:
                    self._stats["total_requests"] += 1

                result = self._redis.delete(full_key)

                if self.enable_stats:
                    self._stats["deletes"] += 1

                return result > 0

            except redis.ConnectionError as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

                if self.enable_stats:
                    self._stats["errors"] += 1

                raise ExternalServiceError(
                    f"Redis connection failed for key: {key}",
                    service_name="redis",
                    endpoint=f"{self.host}:{self.port}",
                    cause=e,
                )

            except Exception as e:
                if self.enable_stats:
                    self._stats["errors"] += 1

                raise CacheError(
                    f"Failed to delete cache item for key: {key}",
                    cache_key=key,
                    operation="delete",
                    cache_provider="redis",
                    cause=e,
                )

    def exists(self, key: str) -> bool:
        """キー存在確認"""
        full_key = self._make_key(key)

        try:
            return bool(self._redis.exists(full_key))
        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                f"Failed to check existence for key: {key}",
                cache_key=key,
                operation="exists",
                cache_provider="redis",
                cause=e,
            )

    def clear(self) -> bool:
        """キー範囲クリア（プレフィックス付きのみ）"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self._redis.keys(pattern)

            if keys:
                self._redis.delete(*keys)

            return True

        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                "Failed to clear cache",
                operation="clear",
                cache_provider="redis",
                cause=e,
            )

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """キー一覧取得"""
        try:
            if pattern:
                search_pattern = f"{self.key_prefix}{pattern}"
            else:
                search_pattern = f"{self.key_prefix}*"

            redis_keys = self._redis.keys(search_pattern)

            # プレフィックスを除去
            prefix_len = len(self.key_prefix)
            return [
                key.decode("utf-8")[prefix_len:] if isinstance(key, bytes) else key[prefix_len:]
                for key in redis_keys
            ]

        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                f"Failed to get keys with pattern: {pattern}",
                operation="keys",
                cache_provider="redis",
                cause=e,
            )

    def size(self) -> int:
        """キャッシュサイズ取得（プレフィックス付きキーのみ）"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self._redis.keys(pattern)
            return len(keys)

        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                "Failed to get cache size",
                operation="size",
                cache_provider="redis",
                cause=e,
            )

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            # Redis統計情報
            redis_info = self._redis.info()

            stats = self._stats.copy()
            stats.update(
                {
                    "redis_version": redis_info.get("redis_version"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "used_memory": redis_info.get("used_memory"),
                    "used_memory_human": redis_info.get("used_memory_human"),
                    "keyspace_hits": redis_info.get("keyspace_hits"),
                    "keyspace_misses": redis_info.get("keyspace_misses"),
                    "instantaneous_ops_per_sec": redis_info.get("instantaneous_ops_per_sec"),
                    "cache_size": self.size(),
                    "hit_rate": self._calculate_hit_rate(),
                }
            )

            return stats

        except Exception as e:
            # 統計情報取得失敗時は基本情報のみ返す
            return {**self._stats, "error": f"Failed to get Redis stats: {str(e)}"}

    def get_ttl(self, key: str) -> Optional[int]:
        """キーのTTL取得"""
        full_key = self._make_key(key)

        try:
            ttl = self._redis.ttl(full_key)
            if ttl == -2:  # キーが存在しない
                return None
            elif ttl == -1:  # TTLなし（永続）
                return -1
            else:
                return ttl

        except Exception as e:
            raise CacheError(
                f"Failed to get TTL for key: {key}",
                cache_key=key,
                operation="ttl",
                cache_provider="redis",
                cause=e,
            )

    def expire(self, key: str, ttl_seconds: int) -> bool:
        """キーにTTL設定"""
        full_key = self._make_key(key)

        try:
            return bool(self._redis.expire(full_key, ttl_seconds))

        except Exception as e:
            raise CacheError(
                f"Failed to set expiration for key: {key}",
                cache_key=key,
                operation="expire",
                cache_provider="redis",
                cause=e,
            )

    def increment(self, key: str, amount: int = 1) -> int:
        """カウンター増加"""
        full_key = self._make_key(key)

        try:
            return self._redis.incrby(full_key, amount)

        except Exception as e:
            raise CacheError(
                f"Failed to increment key: {key}",
                cache_key=key,
                operation="increment",
                cache_provider="redis",
                cause=e,
            )

    def decrement(self, key: str, amount: int = 1) -> int:
        """カウンター減少"""
        full_key = self._make_key(key)

        try:
            return self._redis.decrby(full_key, amount)

        except Exception as e:
            raise CacheError(
                f"Failed to decrement key: {key}",
                cache_key=key,
                operation="decrement",
                cache_provider="redis",
                cause=e,
            )

    def multi_get(self, keys: List[str]) -> Dict[str, Any]:
        """複数キー一括取得"""
        full_keys = [self._make_key(key) for key in keys]

        try:
            raw_values = self._redis.mget(full_keys)
            result = {}

            for i, raw_value in enumerate(raw_values):
                if raw_value is not None:
                    result[keys[i]] = self._serializer.deserialize(raw_value)

            return result

        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                f"Failed to get multiple keys: {keys}",
                operation="multi_get",
                cache_provider="redis",
                cause=e,
            )

    def multi_set(self, mapping: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """複数キー一括設定"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        try:
            pipe = self._redis.pipeline()

            for key, value in mapping.items():
                full_key = self._make_key(key)
                serialized_value = self._serializer.serialize(value)
                pipe.setex(full_key, ttl_seconds, serialized_value)

            results = pipe.execute()
            return all(results)

        except Exception as e:
            if self.enable_stats:
                self._stats["errors"] += 1

            raise CacheError(
                f"Failed to set multiple keys: {list(mapping.keys())}",
                operation="multi_set",
                cache_provider="redis",
                cause=e,
            )

    def close(self):
        """接続クローズ"""
        try:
            if hasattr(self, "_redis"):
                self._redis.close()
            if hasattr(self, "_redis_pool"):
                self._redis_pool.disconnect()
        except Exception:
            # クローズエラーは無視
            pass

    def _make_key(self, key: str) -> str:
        """完全キー名生成"""
        return f"{self.key_prefix}{key}"

    def _calculate_hit_rate(self) -> float:
        """ヒット率計算"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        if total_requests == 0:
            return 0.0
        return self._stats["hits"] / total_requests

    # 非同期メソッド（オプション）

    async def get_async(self, key: str) -> Optional[Any]:
        """非同期キー取得"""
        if self._async_redis is None:
            self._async_redis = aioredis.Redis(
                host=self.host, port=self.port, db=self.db, password=self.password
            )

        try:
            full_key = self._make_key(key)
            raw_value = await self._async_redis.get(full_key)

            if raw_value is None:
                return None

            return self._serializer.deserialize(raw_value)

        except Exception as e:
            raise CacheError(
                f"Failed to async get cache item for key: {key}",
                cache_key=key,
                operation="get_async",
                cache_provider="redis",
                cause=e,
            )

    async def set_async(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """非同期キー設定"""
        if self._async_redis is None:
            self._async_redis = aioredis.Redis(
                host=self.host, port=self.port, db=self.db, password=self.password
            )

        try:
            full_key = self._make_key(key)
            serialized_value = self._serializer.serialize(value)

            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds

            result = await self._async_redis.setex(full_key, ttl_seconds, serialized_value)
            return bool(result)

        except Exception as e:
            raise CacheError(
                f"Failed to async set cache item for key: {key}",
                cache_key=key,
                operation="set_async",
                cache_provider="redis",
                cause=e,
            )
