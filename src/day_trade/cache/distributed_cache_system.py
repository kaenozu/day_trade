#!/usr/bin/env python3
"""
分散キャッシュシステム
Issue #377: 高度なキャッシング戦略の導入

Redis/Memcached対応分散キャッシュと複数インスタンス間でのキャッシュ共有機能を実装
"""

import hashlib
import json
import pickle
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from ..utils.cache_utils import generate_safe_cache_key, sanitize_cache_value
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    def generate_safe_cache_key(*args, **kwargs):
        return hashlib.sha256(str(args).encode() + str(kwargs).encode()).hexdigest()

    def sanitize_cache_value(value):
        return value


# オプショナル依存関係
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Redis distributed cache will be disabled.", stacklevel=2)

try:
    import memcache

    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    warnings.warn(
        "Memcached not available. Memcached distributed cache will be disabled.",
        stacklevel=2,
    )

logger = get_context_logger(__name__)


@dataclass
class DistributedCacheEntry:
    """分散キャッシュエントリ"""

    key: str
    value: Any
    created_at: float
    expires_at: float
    version: int = 1
    node_id: str = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class DistributedCacheBackend(ABC):
    """分散キャッシュバックエンド抽象基底クラス"""

    @abstractmethod
    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> bool:
        """データ設定"""
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """データ取得"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """データ削除"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """キー存在チェック"""
        pass

    @abstractmethod
    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """接続状態チェック"""
        pass


class RedisBackend(DistributedCacheBackend):
    """Redis分散キャッシュバックエンド"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = None,
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available. Install redis-py: pip install redis")

        self.host = host
        self.port = port
        self.db = db

        # コネクションプール作成
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=connection_pool_size,
            socket_timeout=socket_timeout,
            decode_responses=False,  # バイトデータを扱うため
        )

        self.client = redis.Redis(connection_pool=self.pool)

        # 接続テスト
        try:
            self.client.ping()
            logger.info(f"Redis接続成功: {host}:{port}, DB={db}")
        except Exception as e:
            logger.error(f"Redis接続失敗: {e}")
            raise

    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> bool:
        """データ設定"""
        try:
            result = self.client.setex(key, ttl_seconds, value)
            return result is True
        except Exception as e:
            logger.error(f"Redis設定失敗 {key}: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """データ取得"""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis取得失敗 {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """データ削除"""
        try:
            result = self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis削除失敗 {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キー存在チェック"""
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis存在チェック失敗 {key}: {e}")
            return False

    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        try:
            if pattern:
                keys = self.client.keys(pattern)
                if keys:
                    return self.client.delete(*keys)
                return 0
            else:
                # 全削除（危険な操作なので注意深く）
                return self.client.flushdb()
        except Exception as e:
            logger.error(f"Redisクリア失敗: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            info = self.client.info()
            return {
                "backend": "redis",
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.error(f"Redis統計取得失敗: {e}")
            return {"backend": "redis", "error": str(e)}

    def is_connected(self) -> bool:
        """接続状態チェック"""
        try:
            self.client.ping()
            return True
        except Exception:
            return False


class MemcachedBackend(DistributedCacheBackend):
    """Memcached分散キャッシュバックエンド"""

    def __init__(self, servers: List[str] = None):
        if not MEMCACHED_AVAILABLE:
            raise RuntimeError(
                "Memcached not available. Install python-memcached: pip install python-memcached"
            )

        self.servers = servers or ["127.0.0.1:11211"]
        self.client = memcache.Client(self.servers, debug=0)

        # 接続テスト
        try:
            stats = self.client.get_stats()
            if stats:
                logger.info(f"Memcached接続成功: {self.servers}")
            else:
                raise ConnectionError("Memcached stats unavailable")
        except Exception as e:
            logger.error(f"Memcached接続失敗: {e}")
            raise

    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> bool:
        """データ設定"""
        try:
            return self.client.set(key, value, time=ttl_seconds)
        except Exception as e:
            logger.error(f"Memcached設定失敗 {key}: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """データ取得"""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Memcached取得失敗 {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """データ削除"""
        try:
            return self.client.delete(key) != 0
        except Exception as e:
            logger.error(f"Memcached削除失敗 {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キー存在チェック"""
        try:
            return self.client.get(key) is not None
        except Exception as e:
            logger.error(f"Memcached存在チェック失敗 {key}: {e}")
            return False

    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        try:
            if pattern:
                # Memcachedはパターンマッチ削除をサポートしていないので、全削除のみ
                logger.warning(
                    "Memcachedはパターンマッチ削除をサポートしていません。全削除を実行します。"
                )

            self.client.flush_all()
            return 1  # 削除操作成功を示す
        except Exception as e:
            logger.error(f"Memcachedクリア失敗: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            stats = self.client.get_stats()
            if not stats:
                return {"backend": "memcached", "error": "Stats unavailable"}

            # 最初のサーバーの統計情報を使用
            server_stats = stats[0][1] if stats else {}

            return {
                "backend": "memcached",
                "servers": self.servers,
                "bytes_used": int(server_stats.get("bytes", 0)),
                "current_items": int(server_stats.get("curr_items", 0)),
                "total_items": int(server_stats.get("total_items", 0)),
                "get_hits": int(server_stats.get("get_hits", 0)),
                "get_misses": int(server_stats.get("get_misses", 0)),
                "cmd_get": int(server_stats.get("cmd_get", 0)),
                "cmd_set": int(server_stats.get("cmd_set", 0)),
                "uptime": int(server_stats.get("uptime", 0)),
            }
        except Exception as e:
            logger.error(f"Memcached統計取得失敗: {e}")
            return {"backend": "memcached", "error": str(e)}

    def is_connected(self) -> bool:
        """接続状態チェック"""
        try:
            stats = self.client.get_stats()
            return stats is not None and len(stats) > 0
        except Exception:
            return False


class InMemoryBackend(DistributedCacheBackend):
    """インメモリフォールバックバックエンド"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._data = {}
        self._access_order = []
        self._lock = threading.RLock()
        self._stats = {"sets": 0, "gets": 0, "hits": 0, "misses": 0, "deletes": 0}

    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> bool:
        """データ設定"""
        with self._lock:
            try:
                # 容量チェック
                if len(self._data) >= self.max_size and key not in self._data:
                    self._evict_lru()

                expires_at = time.time() + ttl_seconds
                self._data[key] = (value, expires_at)

                # アクセス順序更新
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                self._stats["sets"] += 1
                return True

            except Exception as e:
                logger.error(f"InMemory設定失敗 {key}: {e}")
                return False

    def get(self, key: str) -> Optional[bytes]:
        """データ取得"""
        with self._lock:
            self._stats["gets"] += 1

            if key not in self._data:
                self._stats["misses"] += 1
                return None

            value, expires_at = self._data[key]

            # 有効期限チェック
            if time.time() > expires_at:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["misses"] += 1
                return None

            # アクセス順序更新
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            return value

    def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["deletes"] += 1
                return True
            return False

    def exists(self, key: str) -> bool:
        """キー存在チェック"""
        with self._lock:
            if key not in self._data:
                return False

            _, expires_at = self._data[key]
            return time.time() <= expires_at

    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        with self._lock:
            if pattern:
                cleared_count = 0
                keys_to_delete = []

                for key in self._data.keys():
                    if pattern in key:  # 簡単なパターンマッチング
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    del self._data[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    cleared_count += 1

                return cleared_count
            else:
                cleared_count = len(self._data)
                self._data.clear()
                self._access_order.clear()
                return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._lock:
            return {
                "backend": "in_memory",
                "current_items": len(self._data),
                "max_size": self.max_size,
                **self._stats,
                "hit_rate": self._stats["hits"] / max(self._stats["gets"], 1),
                "memory_usage_approx_mb": sum(len(v[0]) for v in self._data.values())
                / (1024 * 1024),
            }

    def is_connected(self) -> bool:
        """接続状態チェック"""
        return True

    def _evict_lru(self):
        """LRU削除"""
        if self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._data:
                del self._data[oldest_key]


class DistributedCacheManager:
    """分散キャッシュ管理システム"""

    def __init__(
        self,
        backend_type: str = "redis",
        backend_config: Dict[str, Any] = None,
        enable_fallback: bool = True,
        serialization: str = "pickle",
    ):
        self.backend_type = backend_type
        self.backend_config = backend_config or {}
        self.enable_fallback = enable_fallback
        self.serialization = serialization

        # バックエンド初期化
        self.primary_backend = self._create_backend(backend_type, self.backend_config)

        # フォールバックバックエンド
        self.fallback_backend = InMemoryBackend() if enable_fallback else None

        self._stats = {
            "primary_hits": 0,
            "primary_misses": 0,
            "fallback_hits": 0,
            "fallback_misses": 0,
            "primary_errors": 0,
            "sets": 0,
            "deletes": 0,
        }
        self._lock = threading.RLock()

        logger.info(
            f"DistributedCacheManager初期化完了: {backend_type}, フォールバック={enable_fallback}"
        )

    def _create_backend(self, backend_type: str, config: Dict[str, Any]) -> DistributedCacheBackend:
        """バックエンド作成"""
        try:
            if backend_type == "redis":
                return RedisBackend(**config)
            elif backend_type == "memcached":
                return MemcachedBackend(**config)
            elif backend_type == "memory":
                return InMemoryBackend(**config)
            else:
                raise ValueError(f"未サポートのバックエンドタイプ: {backend_type}")
        except Exception as e:
            logger.error(f"バックエンド作成失敗 {backend_type}: {e}")
            if self.enable_fallback:
                logger.info("フォールバックとしてInMemoryBackendを使用")
                return InMemoryBackend()
            raise

    def set(self, key: str, value: Any, ttl_seconds: int = 3600, tags: List[str] = None) -> bool:
        """データ設定"""
        with self._lock:
            try:
                # データシリアライズ
                serialized_data = self._serialize(value)

                # プライマリバックエンドに設定
                success = False

                if self.primary_backend.is_connected():
                    success = self.primary_backend.set(key, serialized_data, ttl_seconds)
                    if success:
                        logger.debug(f"分散キャッシュ設定成功(primary): {key}")
                else:
                    self._stats["primary_errors"] += 1

                # フォールバックバックエンドにも設定
                if self.fallback_backend:
                    fallback_success = self.fallback_backend.set(key, serialized_data, ttl_seconds)
                    if not success:
                        success = fallback_success
                        if success:
                            logger.debug(f"分散キャッシュ設定成功(fallback): {key}")

                if success:
                    self._stats["sets"] += 1

                return success

            except Exception as e:
                logger.error(f"分散キャッシュ設定失敗 {key}: {e}")
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """データ取得"""
        with self._lock:
            try:
                # プライマリバックエンドから取得試行
                if self.primary_backend.is_connected():
                    serialized_data = self.primary_backend.get(key)
                    if serialized_data is not None:
                        self._stats["primary_hits"] += 1
                        value = self._deserialize(serialized_data)
                        logger.debug(f"分散キャッシュヒット(primary): {key}")
                        return value
                    else:
                        self._stats["primary_misses"] += 1
                else:
                    self._stats["primary_errors"] += 1

                # フォールバックバックエンドから取得試行
                if self.fallback_backend:
                    serialized_data = self.fallback_backend.get(key)
                    if serialized_data is not None:
                        self._stats["fallback_hits"] += 1
                        value = self._deserialize(serialized_data)
                        logger.debug(f"分散キャッシュヒット(fallback): {key}")
                        return value
                    else:
                        self._stats["fallback_misses"] += 1

                return default

            except Exception as e:
                logger.error(f"分散キャッシュ取得失敗 {key}: {e}")
                return default

    def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            success = False

            # プライマリバックエンドから削除
            if self.primary_backend.is_connected():
                success = self.primary_backend.delete(key)

            # フォールバックバックエンドからも削除
            if self.fallback_backend:
                fallback_success = self.fallback_backend.delete(key)
                if not success:
                    success = fallback_success

            if success:
                self._stats["deletes"] += 1
                logger.debug(f"分散キャッシュ削除完了: {key}")

            return success

    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        with self._lock:
            total_cleared = 0

            # プライマリバックエンドクリア
            if self.primary_backend.is_connected():
                primary_cleared = self.primary_backend.clear(pattern)
                total_cleared += primary_cleared

            # フォールバックバックエンドクリア
            if self.fallback_backend:
                fallback_cleared = self.fallback_backend.clear(pattern)
                total_cleared += fallback_cleared

            if total_cleared > 0:
                logger.info(f"分散キャッシュクリア完了: {total_cleared}件")

            return total_cleared

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._lock:
            stats = {
                **self._stats,
                "backend_type": self.backend_type,
                "primary_connected": self.primary_backend.is_connected(),
                "fallback_enabled": self.fallback_backend is not None,
            }

            # バックエンド固有統計情報
            try:
                primary_stats = self.primary_backend.get_stats()
                stats["primary_backend"] = primary_stats
            except Exception as e:
                stats["primary_backend_error"] = str(e)

            if self.fallback_backend:
                try:
                    fallback_stats = self.fallback_backend.get_stats()
                    stats["fallback_backend"] = fallback_stats
                except Exception as e:
                    stats["fallback_backend_error"] = str(e)

            # 全体統計
            total_hits = self._stats["primary_hits"] + self._stats["fallback_hits"]
            total_misses = self._stats["primary_misses"] + self._stats["fallback_misses"]
            total_requests = total_hits + total_misses

            stats.update(
                {
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "total_requests": total_requests,
                    "hit_rate": total_hits / max(total_requests, 1),
                    "primary_hit_rate": self._stats["primary_hits"]
                    / max(self._stats["primary_hits"] + self._stats["primary_misses"], 1),
                    "fallback_usage_rate": (
                        self._stats["fallback_hits"] / max(total_hits, 1) if total_hits > 0 else 0
                    ),
                }
            )

            return stats

    def _serialize(self, value: Any) -> bytes:
        """データシリアライゼーション"""
        if self.serialization == "pickle":
            return pickle.dumps(sanitize_cache_value(value))
        elif self.serialization == "json":
            return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        else:
            raise ValueError(f"未サポートのシリアライゼーション: {self.serialization}")

    def _deserialize(self, data: bytes) -> Any:
        """データデシリアライゼーション"""
        if self.serialization == "pickle":
            return pickle.loads(data)
        elif self.serialization == "json":
            return json.loads(data.decode("utf-8"))
        else:
            raise ValueError(f"未サポートのシリアライゼーション: {self.serialization}")


# グローバルインスタンス
_global_distributed_cache: Optional[DistributedCacheManager] = None
_cache_lock = threading.Lock()


def get_distributed_cache(
    backend_type: str = "memory", backend_config: Dict[str, Any] = None
) -> DistributedCacheManager:
    """グローバル分散キャッシュインスタンス取得"""
    global _global_distributed_cache

    if _global_distributed_cache is None:
        with _cache_lock:
            if _global_distributed_cache is None:
                _global_distributed_cache = DistributedCacheManager(
                    backend_type=backend_type, backend_config=backend_config or {}
                )

    return _global_distributed_cache


# 便利なデコレータ
def distributed_cache(
    ttl_seconds: int = 3600,
    backend_type: str = "memory",
    backend_config: Dict[str, Any] = None,
    tags: List[str] = None,
):
    """分散キャッシュデコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_distributed_cache(backend_type=backend_type, backend_config=backend_config)

            # キャッシュキー生成
            cache_key = f"{func.__module__}.{func.__name__}:" + generate_safe_cache_key(
                *args, **kwargs
            )

            # キャッシュ取得試行
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # 関数実行
            result = func(*args, **kwargs)

            # 結果をキャッシュ
            cache.set(cache_key, result, ttl_seconds=ttl_seconds, tags=tags)

            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # テスト実行
    print("=== Issue #377 分散キャッシュシステムテスト ===")

    # インメモリバックエンドテスト
    cache = DistributedCacheManager(backend_type="memory")

    print("\n1. データ保存・取得テスト")
    test_data = {
        "message": "Hello, Distributed Cache!",
        "value": 42,
        "timestamp": time.time(),
    }

    cache.set("test_key", test_data, ttl_seconds=60)
    retrieved_data = cache.get("test_key")

    print(f"保存データ: {test_data}")
    print(f"取得データ: {retrieved_data}")
    print(f"データ一致: {test_data == retrieved_data}")

    print("\n2. 統計情報")
    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("\n3. デコレータテスト")

    @distributed_cache(ttl_seconds=30, backend_type="memory", tags=["test"])
    def expensive_operation(x, y):
        time.sleep(0.05)  # 重い処理をシミュレート
        return x * y + sum(range(1000))

    start_time = time.perf_counter()
    result1 = expensive_operation(10, 20)
    first_time = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    result2 = expensive_operation(10, 20)  # キャッシュからの取得
    cached_time = (time.perf_counter() - start_time) * 1000

    print(f"初回実行: {result1}, 時間: {first_time:.1f}ms")
    print(f"キャッシュ取得: {result2}, 時間: {cached_time:.1f}ms")
    print(f"高速化率: {first_time/cached_time:.1f}x")

    print("\n=== 分散キャッシュシステムテスト完了 ===")
