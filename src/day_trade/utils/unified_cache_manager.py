#!/usr/bin/env python3
"""
Unified Cache Manager
Issue #324: Cache Strategy Optimization

統合的なキャッシュ管理システム - 階層化アーキテクチャとスマート退避戦略
"""

import gzip
import hashlib
import json
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class CacheEntry:
    """統一キャッシュエントリ"""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    priority: float = 1.0  # 重要度 (0.1-10.0)
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """キャッシュ統計情報"""

    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0
    disk_usage_mb: float = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0


class CacheLayer(ABC):
    """キャッシュレイヤーの抽象基底クラス"""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """データ取得"""
        pass

    @abstractmethod
    def put(self, key: str, entry: CacheEntry) -> bool:
        """データ保存"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """データ削除"""
        pass

    @abstractmethod
    def clear(self):
        """全データ削除"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        pass


class L1HotCache(CacheLayer):
    """L1: ホットキャッシュ (メモリ内、超高速)"""

    def __init__(self, max_memory_mb: int = 64, ttl_seconds: int = 30):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0,
            "max_size": max_memory_mb,
        }

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]
            current_time = time.time()

            # TTL チェック
            if current_time - entry.created_at > self.ttl_seconds:
                del self.cache[key]
                self.current_size -= entry.size_bytes
                self.stats["misses"] += 1
                return None

            # LRU更新
            self.cache.move_to_end(key)
            entry.last_accessed = current_time
            entry.access_count += 1

            self.stats["hits"] += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            # 既存エントリの削除
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size_bytes

            # 容量チェックと退避
            while (
                self.current_size + entry.size_bytes > self.max_memory_bytes
                and self.cache
            ):
                self._evict_lru()

            if self.current_size + entry.size_bytes <= self.max_memory_bytes:
                self.cache[key] = entry
                self.current_size += entry.size_bytes
                self.stats["current_size"] = self.current_size // 1024 // 1024
                return True

            return False

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_size -= entry.size_bytes
                return True
            return False

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            self.stats["current_size"] = 0

    def _evict_lru(self):
        """LRU退避"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.current_size -= entry.size_bytes
            self.stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "layer": "L1_Hot",
                "hit_rate": hit_rate,
                "entries": len(self.cache),
                "memory_usage_mb": self.current_size / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                **self.stats,
            }


class L2WarmCache(CacheLayer):
    """L2: ウォームキャッシュ (メモリ内、高速)"""

    def __init__(self, max_memory_mb: int = 256, ttl_seconds: int = 300):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = OrderedDict()
        self.frequency_counter = {}
        self.current_size = 0
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0,
            "max_size": max_memory_mb,
        }

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]
            current_time = time.time()

            # TTL チェック
            if current_time - entry.created_at > self.ttl_seconds:
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None

            # LFU更新
            self.frequency_counter[key] = self.frequency_counter.get(key, 0) + 1
            entry.last_accessed = current_time
            entry.access_count += 1
            # アクセス順序を更新
            self.access_order.move_to_end(key)

            self.stats["hits"] += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            # 既存エントリの削除
            if key in self.cache:
                self._remove_entry(key)

            # 容量チェックと退避
            while (
                self.current_size + entry.size_bytes > self.max_memory_bytes
                and self.cache
            ):
                self._evict_lfu()

            if self.current_size + entry.size_bytes <= self.max_memory_bytes:
                self.cache[key] = entry
                self.access_order[key] = None  # O(1)で追加
                self.frequency_counter[key] = 1
                self.current_size += entry.size_bytes
                self.stats["current_size"] = self.current_size // 1024 // 1024
                return True

            return False

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.current_size = 0
            self.stats["current_size"] = 0

    def _remove_entry(self, key: str):
        """エントリ削除（内部用）"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes

            if key in self.frequency_counter:
                del self.frequency_counter[key]

            if key in self.access_order:
                del self.access_order[key]  # O(1)で削除

    def _evict_lfu(self):
        """LFU退避 (Least Frequently Used)"""
        if not self.cache:
            return

        # 最も使用頻度の低いキーを選択
        min_freq = min(self.frequency_counter.values())
        candidates = {k for k, v in self.frequency_counter.items() if v == min_freq}

        # 同じ頻度なら古いものを選択
        key_to_evict = None
        for key in self.access_order:
            if key in candidates:
                key_to_evict = key
                break

        if key_to_evict:
            self._remove_entry(key_to_evict)
            self.stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "layer": "L2_Warm",
                "hit_rate": hit_rate,
                "entries": len(self.cache),
                "memory_usage_mb": self.current_size / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "avg_frequency": np.mean(list(self.frequency_counter.values()))
                if self.frequency_counter
                else 0,
                **self.stats,
            }


class L3ColdCache(CacheLayer):
    """L3: コールドキャッシュ (ディスク、永続ストレージ)"""

    def __init__(
        self,
        db_path: str = "data/unified_cache.db",
        max_size_mb: int = 1024,
        ttl_seconds: int = 86400,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "current_size": 0,
            "max_size": max_size_mb,
        }
        self._initialize_db()

    def _initialize_db(self):
        """データベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    priority REAL,
                    compressed INTEGER,
                    metadata TEXT
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_priority ON cache_entries(priority DESC)"
            )
            conn.commit()

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        self.stats["misses"] += 1
                        return None

                    current_time = time.time()
                    created_at = row[2]

                    # TTL チェック
                    if current_time - created_at > self.ttl_seconds:
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        conn.commit()
                        self.stats["misses"] += 1
                        return None

                    # エントリ復元
                    value_blob = row[1]
                    compressed = bool(row[7])

                    if compressed:
                        value = pickle.loads(gzip.decompress(value_blob))
                    else:
                        value = pickle.loads(value_blob)

                    metadata = json.loads(row[8]) if row[8] else {}

                    entry = CacheEntry(
                        key=row[0],
                        value=value,
                        created_at=created_at,
                        last_accessed=current_time,
                        access_count=row[4] + 1,
                        size_bytes=row[5],
                        priority=row[6],
                        compressed=compressed,
                        metadata=metadata,
                    )

                    # アクセス情報更新
                    conn.execute(
                        "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                        (current_time, entry.access_count, key),
                    )
                    conn.commit()

                    self.stats["hits"] += 1
                    return entry

            except Exception as e:
                logger.error(f"L3キャッシュ取得エラー: {e}")
                self.stats["misses"] += 1
                return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            try:
                # データ圧縮判定 (1KB以上なら圧縮)
                should_compress = entry.size_bytes > 1024

                if should_compress:
                    value_blob = gzip.compress(pickle.dumps(entry.value))
                    entry.compressed = True
                else:
                    value_blob = pickle.dumps(entry.value)
                    entry.compressed = False

                compressed_size = len(value_blob)

                with sqlite3.connect(str(self.db_path)) as conn:
                    # 容量チェックと退避
                    while (
                        self._get_total_size(conn) + compressed_size
                        > self.max_size_bytes
                    ):
                        if not self._evict_oldest(conn):
                            break

                    # エントリ保存
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries
                        (key, value, created_at, last_accessed, access_count, size_bytes, priority, compressed, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            value_blob,
                            entry.created_at,
                            entry.last_accessed,
                            entry.access_count,
                            compressed_size,
                            entry.priority,
                            int(entry.compressed),
                            json.dumps(entry.metadata),
                        ),
                    )
                    conn.commit()

                return True

            except Exception as e:
                logger.error(f"L3キャッシュ保存エラー: {e}")
                return False

    def delete(self, key: str) -> bool:
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE key = ?", (key,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"L3キャッシュ削除エラー: {e}")
                return False

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

    def _evict_lru(self, oldest_key: str = None):
        """LRU削除"""
        if self._access_order:
            if oldest_key is None:
                oldest_key = self._access_order.pop(0) # Oldest by access time
            else:
                self._access_order.remove(oldest_key)

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

    def _create_backend(
        self, backend_type: str, config: Dict[str, Any]
    ) -> DistributedCacheBackend:
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

    def set(
        self, key: str, value: Any, ttl_seconds: int = 3600, tags: List[str] = None
    ) -> bool:
        """
        データ設定"""
        with self._lock:
            try:
                # データシリアライズ
                serialized_data = self._serialize(value)

                # プライマリバックエンドに設定
                success = False

                if self.primary_backend.is_connected():
                    success = self.primary_backend.set(
                        key, serialized_data, ttl_seconds
                    )
                    if success:
                        logger.debug(f"分散キャッシュ設定成功(primary): {key}")
                else:
                    self._stats["primary_errors"] += 1

                # フォールバックバックエンドにも設定
                if self.fallback_backend:
                    fallback_success = self.fallback_backend.set(
                        key, serialized_data, ttl_seconds
                    )
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
        """
        データ取得"""
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
        """
        データ削除"""
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
        """
        キャッシュクリア"""
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
        """
        統計情報取得"""
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
            total_misses = (
                self._stats["primary_misses"] + self._stats["fallback_misses"]
            )
            total_requests = total_hits + total_misses

            stats.update(
                {
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "total_requests": total_requests,
                    "hit_rate": total_hits / max(total_requests, 1),
                    "primary_hit_rate": self._stats["primary_hits"]
                    / max(
                        self._stats["primary_hits"] + self._stats["primary_misses"], 1
                    ),
                    "fallback_usage_rate": self._stats["fallback_hits"]
                    / max(total_hits, 1)
                    if total_hits > 0
                    else 0,
                }
            )

            return stats

    def _serialize(self, value: Any) -> bytes:
        """
        データシリアライゼーション"""
        if self.serialization == "pickle":
            return pickle.dumps(sanitize_cache_value(value))
        elif self.serialization == "json":
            return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        else:
            raise ValueError(f"未サポートのシリアライゼーション: {self.serialization}")

    def _deserialize(self, data: bytes) -> Any:
        """
        データデシリアライゼーション"""
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
    """
    グローバル分散キャッシュインスタンス取得"""
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
    """
    分散キャッシュデコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_distributed_cache(
                backend_type=backend_type, backend_config=backend_config
            )

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

    print("=== 分散キャッシュシステムテスト完了 ===")
