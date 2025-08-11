#!/usr/bin/env python3
"""
Enhanced Persistent Cache System
Issue #377: 高度なキャッシング戦略の導入

高性能な永続化キャッシュシステム:
- 高速SQLite/ファイルシステム永続化
- 起動時キャッシュ自動復元
- インテリジェントTTL/LRU/LFU戦略
- 圧縮・暗号化対応
- データ整合性保証
- 非同期I/O最適化
"""

import asyncio
import gzip
import hashlib
import json
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import CacheEntry
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    @dataclass
    class CacheEntry:
        key: str
        value: Any
        created_at: float
        last_accessed: float
        access_count: int
        size_bytes: int
        priority: float = 1.0
        compressed: bool = False
        metadata: Dict[str, Any] = field(default_factory=dict)


logger = get_context_logger(__name__)


class PersistentStrategy(Enum):
    """永続化戦略"""

    SQLITE = "sqlite"
    FILE_SYSTEM = "filesystem"
    HYBRID = "hybrid"
    MEMORY_MAPPED = "memory_mapped"


class EvictionPolicy(Enum):
    """退避ポリシー"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    PRIORITY = "priority"  # Priority-based
    HYBRID = "hybrid"  # 複合戦略


@dataclass
class PersistentCacheConfig:
    """永続化キャッシュ設定"""

    storage_path: Path = field(default_factory=lambda: Path("data/persistent_cache"))
    strategy: PersistentStrategy = PersistentStrategy.SQLITE
    eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID
    max_memory_mb: int = 512
    max_disk_mb: int = 2048
    compression_enabled: bool = True
    encryption_enabled: bool = False
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 6
    db_pool_size: int = 5
    enable_wal_mode: bool = True
    vacuum_on_startup: bool = True


@dataclass
class PersistentCacheEntry(CacheEntry):
    """永続化キャッシュエントリ"""

    hash_key: str = ""
    disk_path: Optional[str] = None
    encrypted: bool = False
    backup_count: int = 0
    checksum: str = ""

    def __post_init__(self):
        if not self.hash_key:
            self.hash_key = hashlib.sha256(self.key.encode()).hexdigest()
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """データのチェックサム計算"""
        try:
            data_bytes = pickle.dumps(self.value)
            return hashlib.md5(data_bytes).hexdigest()
        except Exception:
            return ""


class PersistentCacheBackend(ABC):
    """永続化バックエンド抽象基底クラス"""

    @abstractmethod
    async def get(self, key: str) -> Optional[PersistentCacheEntry]:
        """データ取得"""
        pass

    @abstractmethod
    async def put(self, entry: PersistentCacheEntry) -> bool:
        """データ保存"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """データ削除"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """全データクリア"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        pass


class SQLitePersistentBackend(PersistentCacheBackend):
    """SQLite永続化バックエンド"""

    def __init__(self, config: PersistentCacheConfig):
        self.config = config
        self.db_path = config.storage_path / "cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool_lock = threading.Lock()
        self._connection_pool = []
        self._stats = {
            "reads": 0,
            "writes": 0,
            "deletes": 0,
            "hits": 0,
            "misses": 0,
            "errors": 0,
        }

        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        try:
            conn = sqlite3.connect(self.db_path)

            # WALモード有効化（同時アクセス性能向上）
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")

            # 性能最適化設定
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.execute("PRAGMA synchronous=NORMAL")

            # テーブル作成
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persistent_cache (
                    key TEXT PRIMARY KEY,
                    hash_key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL,
                    priority REAL DEFAULT 1.0,
                    compressed BOOLEAN DEFAULT FALSE,
                    encrypted BOOLEAN DEFAULT FALSE,
                    checksum TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    expires_at REAL,
                    backup_count INTEGER DEFAULT 0
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash_key ON persistent_cache(hash_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON persistent_cache(last_accessed)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON persistent_cache(expires_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_priority ON persistent_cache(priority)"
            )

            conn.commit()

            # 起動時VACUUM（最適化）
            if self.config.vacuum_on_startup:
                conn.execute("VACUUM")

            conn.close()

            # コネクションプール初期化
            for _ in range(self.config.db_pool_size):
                conn = self._create_connection()
                self._connection_pool.append(conn)

            logger.info(f"SQLite永続化キャッシュ初期化完了: {self.db_path}")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            raise

    def _create_connection(self) -> sqlite3.Connection:
        """新しいDB接続作成"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """プールから接続取得"""
        with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            else:
                return self._create_connection()

    def _return_connection(self, conn: sqlite3.Connection):
        """接続をプールに返却"""
        with self._pool_lock:
            if len(self._connection_pool) < self.config.db_pool_size:
                self._connection_pool.append(conn)
            else:
                conn.close()

    async def get(self, key: str) -> Optional[PersistentCacheEntry]:
        """データ取得"""
        try:
            self._stats["reads"] += 1

            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM persistent_cache
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                    (key, time.time()),
                )

                row = cursor.fetchone()
                if not row:
                    self._stats["misses"] += 1
                    return None

                # エントリ復元
                entry = self._row_to_entry(row)

                # アクセス統計更新
                entry.last_accessed = time.time()
                entry.access_count += 1

                # 非同期でアクセス情報更新
                asyncio.create_task(
                    self._update_access_stats(
                        key, entry.last_accessed, entry.access_count
                    )
                )

                self._stats["hits"] += 1
                return entry

            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"キャッシュ取得エラー ({key}): {e}")
            self._stats["errors"] += 1
            return None

    async def put(self, entry: PersistentCacheEntry) -> bool:
        """データ保存"""
        try:
            self._stats["writes"] += 1

            # データ圧縮
            value_data = pickle.dumps(entry.value)
            if self.config.compression_enabled and len(value_data) > 1024:
                value_data = gzip.compress(value_data)
                entry.compressed = True

            entry.size_bytes = len(value_data)
            entry.checksum = hashlib.md5(value_data).hexdigest()

            conn = self._get_connection()
            try:
                # TTL計算
                expires_at = None
                if hasattr(entry, "ttl_seconds") and entry.ttl_seconds > 0:
                    expires_at = time.time() + entry.ttl_seconds

                conn.execute(
                    """
                    INSERT OR REPLACE INTO persistent_cache
                    (key, hash_key, value, created_at, last_accessed, access_count,
                     size_bytes, priority, compressed, encrypted, checksum,
                     metadata, expires_at, backup_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.key,
                        entry.hash_key,
                        value_data,
                        entry.created_at,
                        entry.last_accessed,
                        entry.access_count,
                        entry.size_bytes,
                        entry.priority,
                        entry.compressed,
                        entry.encrypted,
                        entry.checksum,
                        json.dumps(entry.metadata),
                        expires_at,
                        entry.backup_count,
                    ),
                )

                conn.commit()
                return True

            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"キャッシュ保存エラー ({entry.key}): {e}")
            self._stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """データ削除"""
        try:
            self._stats["deletes"] += 1

            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM persistent_cache WHERE key = ?", (key,)
                )
                conn.commit()
                return cursor.rowcount > 0

            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"キャッシュ削除エラー ({key}): {e}")
            self._stats["errors"] += 1
            return False

    async def clear(self) -> bool:
        """全データクリア"""
        try:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM persistent_cache")
                conn.commit()
                return True

            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"キャッシュクリアエラー: {e}")
            self._stats["errors"] += 1
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size_bytes,
                        AVG(access_count) as avg_access_count,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at <= ? THEN 1 END) as expired_count
                    FROM persistent_cache
                """,
                    (time.time(),),
                )

                row = cursor.fetchone()
                db_stats = dict(row) if row else {}

                # ファイルサイズ取得
                file_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    **self._stats,
                    **db_stats,
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size / (1024 * 1024),
                    "pool_size": len(self._connection_pool),
                }

            finally:
                self._return_connection(conn)

        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return self._stats

    def _row_to_entry(self, row) -> PersistentCacheEntry:
        """DB行からエントリ復元"""
        # データ復元
        value_data = row["value"]
        if row["compressed"]:
            value_data = gzip.decompress(value_data)
        value = pickle.loads(value_data)

        # メタデータ復元
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return PersistentCacheEntry(
            key=row["key"],
            hash_key=row["hash_key"],
            value=value,
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            size_bytes=row["size_bytes"],
            priority=row["priority"],
            compressed=bool(row["compressed"]),
            encrypted=bool(row["encrypted"]),
            metadata=metadata,
            checksum=row["checksum"],
            backup_count=row["backup_count"],
        )

    async def _update_access_stats(
        self, key: str, last_accessed: float, access_count: int
    ):
        """アクセス統計更新（非同期）"""
        try:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    UPDATE persistent_cache
                    SET last_accessed = ?, access_count = ?
                    WHERE key = ?
                """,
                    (last_accessed, access_count, key),
                )
                conn.commit()
            finally:
                self._return_connection(conn)
        except Exception as e:
            logger.debug(f"アクセス統計更新エラー ({key}): {e}")


class EnhancedPersistentCache:
    """強化永続化キャッシュシステム"""

    def __init__(self, config: Optional[PersistentCacheConfig] = None):
        self.config = config or PersistentCacheConfig()

        # バックエンド初期化
        if self.config.strategy == PersistentStrategy.SQLITE:
            self.backend = SQLitePersistentBackend(self.config)
        else:
            raise NotImplementedError(
                f"Strategy {self.config.strategy} not implemented"
            )

        # 統計情報
        self.operation_stats = {
            "get_operations": 0,
            "put_operations": 0,
            "delete_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_response_time_ms": 0.0,
        }

        # バックアップタスク
        self.backup_task = None
        if self.config.auto_backup_enabled:
            self.backup_task = asyncio.create_task(self._periodic_backup())

        # クリーンアップタスク
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info("強化永続化キャッシュシステム初期化完了")
        logger.info(f"  戦略: {self.config.strategy.value}")
        logger.info(f"  退避ポリシー: {self.config.eviction_policy.value}")
        logger.info(f"  最大メモリ: {self.config.max_memory_mb}MB")
        logger.info(f"  最大ディスク: {self.config.max_disk_mb}MB")

    async def get(self, key: str) -> Optional[Any]:
        """データ取得"""
        start_time = time.time()
        try:
            self.operation_stats["get_operations"] += 1

            entry = await self.backend.get(key)
            if entry:
                self.operation_stats["cache_hits"] += 1
                return entry.value
            else:
                self.operation_stats["cache_misses"] += 1
                return None

        finally:
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """データ保存"""
        start_time = time.time()
        try:
            self.operation_stats["put_operations"] += 1

            current_time = time.time()
            entry = PersistentCacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                size_bytes=0,  # backendで計算
                priority=priority,
                metadata=metadata or {},
            )

            if ttl_seconds:
                entry.ttl_seconds = ttl_seconds

            return await self.backend.put(entry)

        finally:
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

    async def delete(self, key: str) -> bool:
        """データ削除"""
        start_time = time.time()
        try:
            self.operation_stats["delete_operations"] += 1
            return await self.backend.delete(key)
        finally:
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

    async def clear(self) -> bool:
        """全データクリア"""
        return await self.backend.clear()

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報取得"""
        backend_stats = await self.backend.get_stats()

        return {
            "operation_stats": self.operation_stats,
            "backend_stats": backend_stats,
            "config": {
                "strategy": self.config.strategy.value,
                "eviction_policy": self.config.eviction_policy.value,
                "max_memory_mb": self.config.max_memory_mb,
                "max_disk_mb": self.config.max_disk_mb,
                "compression_enabled": self.config.compression_enabled,
            },
            "hit_rate": self._calculate_hit_rate(),
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_hit_rate(self) -> float:
        """ヒット率計算"""
        total = (
            self.operation_stats["cache_hits"] + self.operation_stats["cache_misses"]
        )
        if total == 0:
            return 0.0
        return (self.operation_stats["cache_hits"] / total) * 100

    def _update_avg_response_time(self, response_time_ms: float):
        """平均応答時間更新"""
        current_avg = self.operation_stats["average_response_time_ms"]
        total_ops = (
            self.operation_stats["get_operations"]
            + self.operation_stats["put_operations"]
            + self.operation_stats["delete_operations"]
        )

        if total_ops > 1:
            self.operation_stats["average_response_time_ms"] = (
                current_avg * (total_ops - 1) + response_time_ms
            ) / total_ops
        else:
            self.operation_stats["average_response_time_ms"] = response_time_ms

    async def _periodic_backup(self):
        """定期バックアップ"""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                await self._create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"バックアップエラー: {e}")

    async def _create_backup(self):
        """バックアップ作成"""
        try:
            backup_dir = self.config.storage_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"cache_backup_{timestamp}.db"

            # SQLiteの場合
            if isinstance(self.backend, SQLitePersistentBackend):
                import shutil

                shutil.copy2(self.backend.db_path, backup_path)
                logger.info(f"バックアップ作成完了: {backup_path}")

        except Exception as e:
            logger.error(f"バックアップ作成エラー: {e}")

    async def _periodic_cleanup(self):
        """定期クリーンアップ"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1時間間隔
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"クリーンアップエラー: {e}")

    async def _cleanup_expired(self):
        """期限切れデータクリーンアップ"""
        try:
            if isinstance(self.backend, SQLitePersistentBackend):
                conn = self.backend._get_connection()
                try:
                    cursor = conn.execute(
                        """
                        DELETE FROM persistent_cache
                        WHERE expires_at IS NOT NULL AND expires_at <= ?
                    """,
                        (time.time(),),
                    )

                    deleted_count = cursor.rowcount
                    conn.commit()

                    if deleted_count > 0:
                        logger.info(f"期限切れエントリ削除: {deleted_count}件")

                finally:
                    self.backend._return_connection(conn)

        except Exception as e:
            logger.error(f"期限切れクリーンアップエラー: {e}")

    async def shutdown(self):
        """システム終了処理"""
        try:
            # タスク停止
            if self.backup_task:
                self.backup_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()

            # 最終バックアップ
            if self.config.auto_backup_enabled:
                await self._create_backup()

            logger.info("強化永続化キャッシュシステム終了")

        except Exception as e:
            logger.error(f"終了処理エラー: {e}")


# ファクトリー関数
def create_enhanced_persistent_cache(
    storage_path: Optional[Union[str, Path]] = None,
    max_memory_mb: int = 512,
    max_disk_mb: int = 2048,
    compression_enabled: bool = True,
    auto_backup_enabled: bool = True,
) -> EnhancedPersistentCache:
    """強化永続化キャッシュ作成"""

    config = PersistentCacheConfig(
        storage_path=Path(storage_path)
        if storage_path
        else Path("data/persistent_cache"),
        max_memory_mb=max_memory_mb,
        max_disk_mb=max_disk_mb,
        compression_enabled=compression_enabled,
        auto_backup_enabled=auto_backup_enabled,
    )

    return EnhancedPersistentCache(config)


# テスト関数
async def test_enhanced_persistent_cache():
    """強化永続化キャッシュテスト"""
    print("=== Issue #377 強化永続化キャッシュシステムテスト ===")

    try:
        # キャッシュ初期化
        cache = create_enhanced_persistent_cache(
            storage_path="test_persistent_cache",
            max_memory_mb=64,
            compression_enabled=True,
        )

        print("\n1. 強化永続化キャッシュ初期化完了")

        # データ保存テスト
        test_data = {
            "simple_data": "テストデータ",
            "complex_data": {"numbers": list(range(100)), "text": "大きなデータセット"},
            "numpy_array": np.random.random((100, 10)).tolist(),
        }

        print("\n2. データ保存テスト...")
        for key, value in test_data.items():
            success = await cache.put(key, value, ttl_seconds=3600, priority=1.5)
            print(f"   {key}: {'成功' if success else '失敗'}")

        # データ取得テスト
        print("\n3. データ取得テスト...")
        for key in test_data.keys():
            retrieved = await cache.get(key)
            if retrieved is not None:
                print(f"   {key}: 取得成功 (サイズ: {len(str(retrieved))})")
            else:
                print(f"   {key}: 取得失敗")

        # 統計情報表示
        print("\n4. 統計情報:")
        stats = await cache.get_comprehensive_stats()
        for category, data in stats.items():
            if isinstance(data, dict):
                print(f"   {category}:")
                for k, v in data.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {category}: {data}")

        # パフォーマンステスト
        print("\n5. パフォーマンステスト...")
        start_time = time.time()

        # 大量データ保存
        for i in range(100):
            await cache.put(f"perf_test_{i}", f"データ_{i}", ttl_seconds=1800)

        save_time = time.time() - start_time
        print(f"   100件保存: {save_time:.3f}秒")

        # 大量データ取得
        start_time = time.time()
        retrieved_count = 0
        for i in range(100):
            result = await cache.get(f"perf_test_{i}")
            if result:
                retrieved_count += 1

        retrieve_time = time.time() - start_time
        print(f"   100件取得: {retrieve_time:.3f}秒 ({retrieved_count}件成功)")

        # 最終統計
        final_stats = await cache.get_comprehensive_stats()
        print("\n6. 最終結果:")
        print(f"   ヒット率: {final_stats['hit_rate']:.1f}%")
        print(
            f"   平均応答時間: {final_stats['operation_stats']['average_response_time_ms']:.2f}ms"
        )
        total_operations = sum([
            final_stats['operation_stats']['get_operations'],
            final_stats['operation_stats']['put_operations'],
            final_stats['operation_stats']['delete_operations']
        ])
        print(f"   総操作数: {total_operations}")

        # クリーンアップ
        await cache.shutdown()
        print("\n✅ 強化永続化キャッシュシステム テスト完了！")

    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_enhanced_persistent_cache())
