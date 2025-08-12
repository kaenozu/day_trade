#!/usr/bin/env python3
"""
Advanced Cache Layers - L4 Archive Cache Implementation
Issue #377: Advanced Caching Strategy

L4アーカイブキャッシュと予測的キャッシュローダーの実装
"""

import gzip
import json
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..utils.logging_config import get_context_logger
    from .unified_cache_manager import CacheEntry, CacheLayer
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # フォールバック用の基本クラス定義
    from abc import ABC, abstractmethod
    from dataclasses import dataclass, field

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

    class CacheLayer(ABC):
        @abstractmethod
        def get(self, key: str) -> Optional[CacheEntry]:
            pass

        @abstractmethod
        def put(self, key: str, entry: CacheEntry) -> bool:
            pass

        @abstractmethod
        def delete(self, key: str) -> bool:
            pass

        @abstractmethod
        def clear(self):
            pass

        @abstractmethod
        def get_stats(self) -> Dict[str, Any]:
            pass


logger = get_context_logger(__name__)


@dataclass
class ArchiveMetadata:
    """アーカイブメタデータ"""

    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_algorithm: str
    archived_at: float
    last_verified: float
    access_frequency: float
    related_keys: List[str] = field(default_factory=list)


class CompressionEngine:
    """高性能圧縮エンジン"""

    ALGORITHMS = {
        "gzip": {"compress": gzip.compress, "decompress": gzip.decompress, "level": 6}
    }

    @classmethod
    def compress(cls, data: bytes, algorithm: str = "gzip") -> Tuple[bytes, float]:
        """データ圧縮"""
        start_time = time.time()

        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

        try:
            compress_func = cls.ALGORITHMS[algorithm]["compress"]

            if algorithm == "gzip":
                compressed = compress_func(
                    data, compresslevel=cls.ALGORITHMS[algorithm]["level"]
                )
            else:
                compressed = compress_func(
                    data, compression_level=cls.ALGORITHMS[algorithm]["level"]
                )

            compression_time = time.time() - start_time
            return compressed, compression_time

        except Exception as e:
            logger.error(f"圧縮エラー ({algorithm}): {e}")
            raise

    @classmethod
    def decompress(
        cls, compressed_data: bytes, algorithm: str = "gzip"
    ) -> Tuple[bytes, float]:
        """データ展開"""
        start_time = time.time()

        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")

        try:
            decompress_func = cls.ALGORITHMS[algorithm]["decompress"]
            decompressed = decompress_func(compressed_data)
            decompression_time = time.time() - start_time
            return decompressed, decompression_time

        except Exception as e:
            logger.error(f"展開エラー ({algorithm}): {e}")
            raise

    @classmethod
    def get_best_algorithm(cls, data: bytes) -> str:
        """データサイズに応じた最適圧縮アルゴリズム選択"""
        # lz4が利用できないため、gzipのみ使用
        return "gzip"


class L4ArchiveCache(CacheLayer):
    """L4: アーカイブキャッシュ (長期永続・高圧縮ストレージ)"""

    def __init__(
        self,
        db_path: str = "data/archive_cache.db",
        max_size_gb: int = 10,
        ttl_days: int = 365,
        compression_algorithm: str = "auto",
        index_memory_mb: int = 100,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.ttl_seconds = ttl_days * 24 * 3600
        self.compression_algorithm = compression_algorithm
        self.index_memory_mb = index_memory_mb

        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0,
            "current_size_gb": 0,
            "max_size_gb": max_size_gb,
            "avg_compression_ratio": 0.0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
        }

        # インメモリインデックス（高速アクセス用）
        self.memory_index = OrderedDict()  # key -> (last_accessed, size, algorithm)
        self.memory_index_size = 0
        self.max_index_size = index_memory_mb * 1024 * 1024

        self._initialize_db()
        self._load_memory_index()

        logger.info(f"L4アーカイブキャッシュ初期化: {max_size_gb}GB, TTL={ttl_days}日")

    def _initialize_db(self):
        """データベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    compression_algorithm TEXT,
                    compression_ratio REAL,
                    priority REAL,
                    metadata TEXT,
                    access_frequency REAL DEFAULT 0.0
                )
                """
            )

            # インデックス作成
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON archive_entries(last_accessed)",
                "CREATE INDEX IF NOT EXISTS idx_priority ON archive_entries(priority DESC)",
                "CREATE INDEX IF NOT EXISTS idx_access_frequency ON archive_entries(access_frequency DESC)",
                "CREATE INDEX IF NOT EXISTS idx_created_at ON archive_entries(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_compression_ratio ON archive_entries(compression_ratio)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

            conn.commit()

            # WALモード有効化（並行性向上）
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")  # 10MB SQLiteキャッシュ

    def _load_memory_index(self):
        """メモリインデックス初期化"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT key, last_accessed, compressed_size, compression_algorithm, access_frequency
                    FROM archive_entries
                    ORDER BY access_frequency DESC, last_accessed DESC
                    LIMIT 10000
                    """
                )

                for row in cursor.fetchall():
                    key, last_accessed, size, algorithm, frequency = row
                    self.memory_index[key] = (last_accessed, size, algorithm, frequency)
                    self.memory_index_size += len(key) + 32  # 概算

                    if self.memory_index_size >= self.max_index_size:
                        break

                logger.info(
                    f"メモリインデックス読み込み完了: {len(self.memory_index)}エントリ"
                )

        except Exception as e:
            logger.error(f"メモリインデックス読み込みエラー: {e}")

    def get(self, key: str) -> Optional[CacheEntry]:
        """キー取得（高速インデックス + 遅延展開）"""
        with self.lock:
            # メモリインデックスで高速チェック
            if key not in self.memory_index:
                self.stats["misses"] += 1
                return None

            last_accessed, size, algorithm, frequency = self.memory_index[key]
            current_time = time.time()

            # TTL チェック（インデックスレベル）
            if current_time - last_accessed > self.ttl_seconds:
                del self.memory_index[key]
                self.memory_index_size -= len(key) + 32
                self.stats["misses"] += 1
                # DBからも削除（バックグラウンドで実行可能）
                threading.Thread(
                    target=self._delete_from_db, args=(key,), daemon=True
                ).start()
                return None

            try:
                # データベースから実データ取得
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM archive_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        # インデックスとDB不整合の場合
                        del self.memory_index[key]
                        self.memory_index_size -= len(key) + 32
                        self.stats["misses"] += 1
                        return None

                    # データ展開
                    compressed_data = row[1]
                    compression_algorithm = row[7]

                    (
                        decompressed_data,
                        decompression_time,
                    ) = CompressionEngine.decompress(
                        compressed_data, compression_algorithm
                    )
                    value = pickle.loads(decompressed_data)

                    # 統計更新
                    self.stats["hits"] += 1
                    self.stats["decompressions"] += 1
                    self.stats["total_decompression_time"] += decompression_time

                    # アクセス頻度更新（指数平滑化）
                    new_frequency = frequency * 0.9 + 1.0

                    # エントリ作成
                    entry = CacheEntry(
                        key=key,
                        value=value,
                        created_at=row[2],
                        last_accessed=current_time,
                        access_count=row[4] + 1,
                        size_bytes=row[5],
                        priority=row[9],
                        compressed=True,
                        metadata=json.loads(row[10]) if row[10] else {},
                    )

                    # インデックス更新
                    self.memory_index[key] = (
                        current_time,
                        row[6],
                        compression_algorithm,
                        new_frequency,
                    )
                    # LRU更新
                    self.memory_index.move_to_end(key)

                    # DBアクセス情報更新（非同期）
                    threading.Thread(
                        target=self._update_access_info,
                        args=(key, current_time, entry.access_count, new_frequency),
                        daemon=True,
                    ).start()

                    return entry

            except Exception as e:
                logger.error(f"L4キャッシュ取得エラー: {e}")
                self.stats["misses"] += 1
                return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """高圧縮保存"""
        with self.lock:
            try:
                # 圧縮アルゴリズム選択
                data_blob = pickle.dumps(entry.value)
                if self.compression_algorithm == "auto":
                    algorithm = CompressionEngine.get_best_algorithm(data_blob)
                else:
                    algorithm = self.compression_algorithm

                # データ圧縮
                compressed_data, compression_time = CompressionEngine.compress(
                    data_blob, algorithm
                )

                original_size = len(data_blob)
                compressed_size = len(compressed_data)
                compression_ratio = (
                    compressed_size / original_size if original_size > 0 else 1.0
                )

                # 容量チェックと退避
                while self._get_total_size() + compressed_size > self.max_size_bytes:
                    if not self._evict_lowest_priority():
                        logger.warning("L4キャッシュ容量不足: 退避失敗")
                        break

                with sqlite3.connect(str(self.db_path)) as conn:
                    # エントリ保存
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO archive_entries
                        (key, value, created_at, last_accessed, access_count,
                         original_size, compressed_size, compression_algorithm,
                         compression_ratio, priority, metadata, access_frequency)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            compressed_data,
                            entry.created_at,
                            entry.last_accessed,
                            entry.access_count,
                            original_size,
                            compressed_size,
                            algorithm,
                            compression_ratio,
                            entry.priority,
                            json.dumps(entry.metadata),
                            1.0,  # 初期アクセス頻度
                        ),
                    )
                    conn.commit()

                # インデックス更新
                self.memory_index[key] = (
                    entry.last_accessed,
                    compressed_size,
                    algorithm,
                    1.0,
                )
                self.memory_index_size += len(key) + 32

                # インデックスサイズ制限
                while (
                    self.memory_index_size > self.max_index_size
                    and len(self.memory_index) > 1000
                ):
                    # 最も古いエントリを削除
                    old_key = next(iter(self.memory_index))
                    del self.memory_index[old_key]
                    self.memory_index_size -= len(old_key) + 32

                # 統計更新
                self.stats["compressions"] += 1
                self.stats["total_compression_time"] += compression_time

                # 平均圧縮率更新（指数平滑化）
                alpha = 0.1
                self.stats["avg_compression_ratio"] = (
                    self.stats["avg_compression_ratio"] * (1 - alpha)
                    + compression_ratio * alpha
                )

                logger.debug(
                    f"L4保存成功: {key}, 圧縮率={compression_ratio:.2f}, "
                    f"時間={compression_time:.3f}s, アルゴリズム={algorithm}"
                )

                return True

            except Exception as e:
                logger.error(f"L4キャッシュ保存エラー: {e}")
                return False

    def delete(self, key: str) -> bool:
        """キー削除"""
        with self.lock:
            # インデックスから削除
            if key in self.memory_index:
                del self.memory_index[key]
                self.memory_index_size -= len(key) + 32

            return self._delete_from_db(key)

    def clear(self):
        """全削除"""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM archive_entries")
                    conn.commit()

                self.memory_index.clear()
                self.memory_index_size = 0
                self.stats["current_size_gb"] = 0

                logger.info("L4キャッシュクリア完了")

            except Exception as e:
                logger.error(f"L4キャッシュクリアエラー: {e}")

    def _delete_from_db(self, key: str) -> bool:
        """DB削除（内部用）"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM archive_entries WHERE key = ?", (key,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"DB削除エラー ({key}): {e}")
            return False

    def _update_access_info(
        self, key: str, last_accessed: float, access_count: int, frequency: float
    ):
        """アクセス情報更新（非同期）"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE archive_entries SET last_accessed = ?, access_count = ?, access_frequency = ? WHERE key = ?",
                    (last_accessed, access_count, frequency, key),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"アクセス情報更新エラー ({key}): {e}")

    def _get_total_size(self) -> int:
        """総サイズ取得"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT SUM(compressed_size) FROM archive_entries"
                )
                result = cursor.fetchone()[0]
                size_bytes = result or 0
                self.stats["current_size_gb"] = size_bytes / (1024**3)
                return size_bytes
        except Exception:
            return 0

    def _evict_lowest_priority(self) -> bool:
        """最低優先度エントリ退避"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 複合スコアで退避対象選択（優先度低い + アクセス頻度低い + 古い）
                cursor = conn.execute(
                    """
                    SELECT key FROM archive_entries
                    ORDER BY
                        (priority * 0.4 + access_frequency * 0.4 + (? - last_accessed) / 86400 * 0.2) ASC
                    LIMIT 1
                    """,
                    (time.time(),),
                )
                row = cursor.fetchone()

                if row:
                    key = row[0]
                    conn.execute("DELETE FROM archive_entries WHERE key = ?", (key,))
                    conn.commit()

                    # インデックスからも削除
                    if key in self.memory_index:
                        del self.memory_index[key]
                        self.memory_index_size -= len(key) + 32

                    self.stats["evictions"] += 1
                    logger.debug(f"L4退避: {key}")
                    return True

        except Exception as e:
            logger.error(f"L4退避エラー: {e}")

        return False

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    # エントリ数とサイズ
                    cursor = conn.execute(
                        "SELECT COUNT(*), SUM(compressed_size), AVG(compression_ratio) FROM archive_entries"
                    )
                    count, total_size, avg_ratio = cursor.fetchone()

                    total_size = total_size or 0
                    avg_ratio = avg_ratio or 0

                    # ヒット率計算
                    total_requests = self.stats["hits"] + self.stats["misses"]
                    hit_rate = (
                        self.stats["hits"] / total_requests if total_requests > 0 else 0
                    )

                    return {
                        "layer": "L4_Archive",
                        "hit_rate": hit_rate,
                        "entries": count or 0,
                        "disk_usage_gb": total_size / (1024**3),
                        "max_disk_gb": self.max_size_bytes / (1024**3),
                        "avg_compression_ratio": avg_ratio,
                        "memory_index_entries": len(self.memory_index),
                        "memory_index_mb": self.memory_index_size / (1024 * 1024),
                        "compression_time_avg": (
                            self.stats["total_compression_time"]
                            / self.stats["compressions"]
                            if self.stats["compressions"] > 0
                            else 0
                        ),
                        "decompression_time_avg": (
                            self.stats["total_decompression_time"]
                            / self.stats["decompressions"]
                            if self.stats["decompressions"] > 0
                            else 0
                        ),
                        **self.stats,
                    }

            except Exception as e:
                logger.error(f"L4統計取得エラー: {e}")
                return {"layer": "L4_Archive", "error": str(e), **self.stats}

    def optimize_compression(self) -> Dict[str, Any]:
        """圧縮最適化分析"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        compression_algorithm,
                        COUNT(*) as count,
                        AVG(compression_ratio) as avg_ratio,
                        SUM(compressed_size) as total_size
                    FROM archive_entries
                    GROUP BY compression_algorithm
                    """
                )

                results = {}
                for row in cursor.fetchall():
                    algorithm, count, avg_ratio, total_size = row
                    results[algorithm] = {
                        "entries": count,
                        "avg_compression_ratio": avg_ratio,
                        "total_size_mb": (total_size or 0) / (1024 * 1024),
                    }

                # 推奨アルゴリズム決定
                if results:
                    best_algorithm = min(
                        results.keys(),
                        key=lambda k: results[k]["avg_compression_ratio"],
                    )

                    return {
                        "current_distribution": results,
                        "recommended_algorithm": best_algorithm,
                        "potential_savings_mb": self._calculate_potential_savings(
                            results, best_algorithm
                        ),
                    }
                else:
                    return {"current_distribution": {}, "recommended_algorithm": "lz4"}

        except Exception as e:
            logger.error(f"圧縮最適化分析エラー: {e}")
            return {"error": str(e)}

    def _calculate_potential_savings(
        self, distribution: Dict, best_algorithm: str
    ) -> float:
        """潜在的節約容量計算"""
        total_savings = 0.0
        best_ratio = distribution[best_algorithm]["avg_compression_ratio"]

        for algorithm, stats in distribution.items():
            if algorithm != best_algorithm:
                current_size = stats["total_size_mb"]
                potential_size = current_size * (
                    best_ratio / stats["avg_compression_ratio"]
                )
                savings = current_size - potential_size
                total_savings += max(0, savings)

        return total_savings


if __name__ == "__main__":
    # L4アーカイブキャッシュテスト
    print("=== L4アーカイブキャッシュテスト ===")

    # テスト用一時ファイル
    test_db = "test_archive_cache.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    try:
        l4_cache = L4ArchiveCache(
            db_path=test_db,
            max_size_gb=1,  # 1GB制限
            ttl_days=30,
            compression_algorithm="auto",
        )

        # テストデータ
        test_entries = []
        for i in range(10):
            key = f"test_key_{i}"
            value = f"Test data {i}: " + "x" * (1000 + i * 100)  # 可変サイズ
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=len(value.encode()),
                priority=float(i + 1),
            )
            test_entries.append(entry)

        # 保存テスト
        print("\n1. データ保存テスト...")
        for entry in test_entries:
            success = l4_cache.put(entry.key, entry)
            print(f"  {entry.key}: {'成功' if success else '失敗'}")

        # 取得テスト
        print("\n2. データ取得テスト...")
        for entry in test_entries:
            retrieved = l4_cache.get(entry.key)
            print(f"  {entry.key}: {'ヒット' if retrieved else 'ミス'}")

        # 統計情報
        print("\n3. 統計情報...")
        stats = l4_cache.get_stats()
        print(f"  エントリ数: {stats['entries']}")
        print(f"  ディスク使用量: {stats['disk_usage_gb']:.3f} GB")
        print(f"  平均圧縮率: {stats['avg_compression_ratio']:.3f}")
        print(f"  ヒット率: {stats['hit_rate']:.2%}")

        # 圧縮最適化分析
        print("\n4. 圧縮最適化分析...")
        optimization = l4_cache.optimize_compression()
        print(f"  推奨アルゴリズム: {optimization.get('recommended_algorithm')}")
        print(f"  潜在的節約: {optimization.get('potential_savings_mb', 0):.2f} MB")

        print("\n✅ L4アーカイブキャッシュテスト完了")

    finally:
        # テストファイル削除
        if os.path.exists(test_db):
            os.remove(test_db)
        if os.path.exists(test_db + "-wal"):
            os.remove(test_db + "-wal")
        if os.path.exists(test_db + "-shm"):
            os.remove(test_db + "-shm")
