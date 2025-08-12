#!/usr/bin/env python3
"""
永続化キャッシュシステム
Issue #377: 高度なキャッシング戦略の導入

再起動後もデータを保持する永続化キャッシュとディスク最適化機能を実装
"""

import fcntl
import gzip
import hashlib
import json
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lz4
import pandas as pd

# プロジェクト内モジュール
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


logger = get_context_logger(__name__)


@dataclass
class CacheMetadata:
    """キャッシュメタデータ"""

    key: str
    created_at: float
    expires_at: float
    last_accessed: float
    access_count: int
    data_size: int
    data_type: str
    compression_type: Optional[str] = None
    compressed_size: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = None
    priority: float = 1.0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @property
    def is_expired(self) -> bool:
        """有効期限チェック"""
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """経過時間（秒）"""
        return time.time() - self.created_at

    @property
    def compression_ratio(self) -> float:
        """圧縮率"""
        if self.compressed_size and self.data_size:
            return self.compressed_size / self.data_size
        return 1.0


class CompressionManager:
    """圧縮管理システム"""

    ALGORITHMS = {
        "none": {"compress": lambda x: x, "decompress": lambda x: x, "extension": ""},
        "gzip": {
            "compress": lambda x: gzip.compress(x, compresslevel=6),
            "decompress": gzip.decompress,
            "extension": ".gz",
        },
        "lz4": {
            "compress": lz4.frame.compress,
            "decompress": lz4.frame.decompress,
            "extension": ".lz4",
        },
    }

    @classmethod
    def compress(cls, data: bytes, algorithm: str = "gzip") -> Tuple[bytes, float]:
        """データ圧縮"""
        if algorithm not in cls.ALGORITHMS:
            algorithm = "gzip"

        start_time = time.perf_counter()

        try:
            compressed_data = cls.ALGORITHMS[algorithm]["compress"](data)
            compression_time = (time.perf_counter() - start_time) * 1000

            return compressed_data, compression_time
        except Exception as e:
            logger.error(f"圧縮失敗 {algorithm}: {e}")
            return data, 0.0

    @classmethod
    def decompress(cls, data: bytes, algorithm: str = "gzip") -> Tuple[bytes, float]:
        """データ展開"""
        if algorithm not in cls.ALGORITHMS:
            algorithm = "gzip"

        start_time = time.perf_counter()

        try:
            decompressed_data = cls.ALGORITHMS[algorithm]["decompress"](data)
            decompression_time = (time.perf_counter() - start_time) * 1000

            return decompressed_data, decompression_time
        except Exception as e:
            logger.error(f"展開失敗 {algorithm}: {e}")
            return data, 0.0

    @classmethod
    def get_best_algorithm(cls, data: bytes) -> str:
        """データに最適な圧縮アルゴリズム選択"""
        if len(data) < 1024:  # 1KB未満は圧縮しない
            return "none"

        # サンプル圧縮テスト
        sample_size = min(len(data), 10240)  # 10KB
        sample = data[:sample_size]

        best_algorithm = "gzip"
        best_ratio = float("inf")

        for algo in ["gzip", "lz4"]:
            try:
                compressed, _ = cls.compress(sample, algo)
                ratio = len(compressed) / len(sample)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_algorithm = algo
            except Exception:
                continue

        return best_algorithm


class PersistentCacheStorage(ABC):
    """永続化キャッシュストレージ抽象基底クラス"""

    @abstractmethod
    def store(self, key: str, data: bytes, metadata: CacheMetadata) -> bool:
        """データ保存"""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Tuple[bytes, CacheMetadata]]:
        """データ取得"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """データ削除"""
        pass

    @abstractmethod
    def list_keys(self, pattern: str = None) -> List[str]:
        """キー一覧取得"""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """期限切れデータクリーンアップ"""
        pass

    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージ統計情報"""
        pass


class SQLiteStorage(PersistentCacheStorage):
    """SQLite永続化ストレージ"""

    def __init__(self, db_path: str, max_size_mb: int = 1000):
        self.db_path = Path(db_path)
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()

        # データベース初期化
        self._init_database()

        logger.info(f"SQLiteStorage初期化完了: {db_path}, 最大サイズ: {max_size_mb}MB")

    def _init_database(self):
        """データベース初期化"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_data (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    data_size INTEGER NOT NULL,
                    compressed_size INTEGER,
                    priority REAL DEFAULT 1.0
                )
            """
            )

            # インデックス作成
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_data(expires_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_priority_size ON cache_data(priority DESC, data_size DESC)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_data(last_accessed)
            """
            )

            conn.commit()

    def store(self, key: str, data: bytes, metadata: CacheMetadata) -> bool:
        """データ保存"""
        with self._lock:
            try:
                # 容量チェック・クリーンアップ
                self._ensure_capacity(len(data))

                with sqlite3.connect(self.db_path) as conn:
                    metadata_json = json.dumps(asdict(metadata))

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_data
                        (key, data, metadata, created_at, expires_at, last_accessed,
                         access_count, data_size, compressed_size, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            key,
                            data,
                            metadata_json,
                            metadata.created_at,
                            metadata.expires_at,
                            metadata.last_accessed,
                            metadata.access_count,
                            metadata.data_size,
                            metadata.compressed_size,
                            metadata.priority,
                        ),
                    )

                    conn.commit()
                    logger.debug(f"データ保存完了: {key}, サイズ: {len(data)}B")
                    return True

            except Exception as e:
                logger.error(f"データ保存失敗 {key}: {e}")
                return False

    def retrieve(self, key: str) -> Optional[Tuple[bytes, CacheMetadata]]:
        """データ取得"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT data, metadata FROM cache_data WHERE key = ?
                    """,
                        (key,),
                    )

                    row = cursor.fetchone()
                    if not row:
                        return None

                    data, metadata_json = row
                    metadata_dict = json.loads(metadata_json)
                    metadata = CacheMetadata(**metadata_dict)

                    # 期限チェック
                    if metadata.is_expired:
                        logger.debug(f"期限切れキャッシュ: {key}")
                        return None

                    # アクセス統計更新
                    self._update_access_stats(key)

                    return data, metadata

            except Exception as e:
                logger.error(f"データ取得失敗 {key}: {e}")
                return None

    def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_data WHERE key = ?", (key,)
                    )
                    conn.commit()

                    deleted = cursor.rowcount > 0
                    if deleted:
                        logger.debug(f"データ削除完了: {key}")
                    return deleted

            except Exception as e:
                logger.error(f"データ削除失敗 {key}: {e}")
                return False

    def list_keys(self, pattern: str = None) -> List[str]:
        """キー一覧取得"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if pattern:
                        cursor = conn.execute(
                            """
                            SELECT key FROM cache_data WHERE key LIKE ? AND expires_at > ?
                        """,
                            (pattern, time.time()),
                        )
                    else:
                        cursor = conn.execute(
                            """
                            SELECT key FROM cache_data WHERE expires_at > ?
                        """,
                            (time.time(),),
                        )

                    return [row[0] for row in cursor.fetchall()]

            except Exception as e:
                logger.error(f"キー一覧取得失敗: {e}")
                return []

    def cleanup_expired(self) -> int:
        """期限切れデータクリーンアップ"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        DELETE FROM cache_data WHERE expires_at < ?
                    """,
                        (time.time(),),
                    )
                    conn.commit()

                    deleted_count = cursor.rowcount
                    if deleted_count > 0:
                        logger.info(
                            f"期限切れデータクリーンアップ完了: {deleted_count}件"
                        )

                        # VACUUM実行でディスク領域回収
                        conn.execute("VACUUM")

                    return deleted_count

            except Exception as e:
                logger.error(f"クリーンアップ失敗: {e}")
                return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージ統計情報"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 基本統計
                    cursor = conn.execute(
                        """
                        SELECT
                            COUNT(*) as total_entries,
                            SUM(data_size) as total_size,
                            SUM(CASE WHEN compressed_size IS NOT NULL THEN compressed_size ELSE data_size END) as storage_size,
                            AVG(access_count) as avg_access_count,
                            COUNT(CASE WHEN expires_at < ? THEN 1 END) as expired_entries
                        FROM cache_data
                    """,
                        (time.time(),),
                    )

                    stats = cursor.fetchone()

                    # ファイルサイズ
                    file_size = (
                        self.db_path.stat().st_size if self.db_path.exists() else 0
                    )

                    return {
                        "total_entries": stats[0] or 0,
                        "total_data_size_mb": (stats[1] or 0) / (1024 * 1024),
                        "storage_size_mb": (stats[2] or 0) / (1024 * 1024),
                        "file_size_mb": file_size / (1024 * 1024),
                        "avg_access_count": stats[3] or 0,
                        "expired_entries": stats[4] or 0,
                        "compression_ratio": (
                            (stats[2] or 1) / (stats[1] or 1) if stats[1] else 1.0
                        ),
                        "storage_path": str(self.db_path),
                    }

            except Exception as e:
                logger.error(f"統計情報取得失敗: {e}")
                return {}

    def _update_access_stats(self, key: str):
        """アクセス統計更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE cache_data
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE key = ?
                """,
                    (time.time(), key),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"アクセス統計更新失敗 {key}: {e}")

    def _ensure_capacity(self, new_data_size: int):
        """容量確保（必要に応じてLRU削除）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 現在のサイズ確認
                cursor = conn.execute(
                    """
                    SELECT SUM(CASE WHEN compressed_size IS NOT NULL THEN compressed_size ELSE data_size END)
                    FROM cache_data
                """
                )
                current_size = cursor.fetchone()[0] or 0
                max_size_bytes = self.max_size_mb * 1024 * 1024

                # 容量チェック
                if current_size + new_data_size <= max_size_bytes:
                    return

                # 削除が必要なサイズ計算
                need_to_free = (current_size + new_data_size) - int(
                    max_size_bytes * 0.8
                )  # 80%まで削減

                logger.info(
                    f"容量不足: {current_size/1024/1024:.1f}MB, {need_to_free/1024/1024:.1f}MB削除必要"
                )

                # LRU順で削除
                cursor = conn.execute(
                    """
                    SELECT key, CASE WHEN compressed_size IS NOT NULL THEN compressed_size ELSE data_size END as size
                    FROM cache_data
                    ORDER BY priority ASC, last_accessed ASC
                """
                )

                freed_size = 0
                deleted_keys = []

                for key, size in cursor.fetchall():
                    if freed_size >= need_to_free:
                        break

                    conn.execute("DELETE FROM cache_data WHERE key = ?", (key,))
                    freed_size += size
                    deleted_keys.append(key)

                conn.commit()

                if deleted_keys:
                    logger.info(
                        f"LRU削除完了: {len(deleted_keys)}件, {freed_size/1024/1024:.1f}MB解放"
                    )

        except Exception as e:
            logger.error(f"容量確保失敗: {e}")


class FileSystemStorage(PersistentCacheStorage):
    """ファイルシステム永続化ストレージ"""

    def __init__(self, base_path: str, max_files: int = 10000):
        self.base_path = Path(base_path)
        self.max_files = max_files
        self._lock = threading.RLock()

        # ディレクトリ作成
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_path / "data"
        self.metadata_dir = self.base_path / "metadata"

        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        logger.info(
            f"FileSystemStorage初期化完了: {base_path}, 最大ファイル数: {max_files}"
        )

    def store(self, key: str, data: bytes, metadata: CacheMetadata) -> bool:
        """データ保存"""
        with self._lock:
            try:
                safe_key = self._safe_filename(key)

                data_path = self.data_dir / f"{safe_key}.bin"
                metadata_path = self.metadata_dir / f"{safe_key}.json"

                # データファイル保存
                with open(data_path, "wb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # 排他ロック
                    f.write(data)

                # メタデータ保存
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(metadata), f, indent=2)

                logger.debug(f"ファイル保存完了: {key}")
                return True

            except Exception as e:
                logger.error(f"ファイル保存失敗 {key}: {e}")
                return False

    def retrieve(self, key: str) -> Optional[Tuple[bytes, CacheMetadata]]:
        """データ取得"""
        with self._lock:
            try:
                safe_key = self._safe_filename(key)

                data_path = self.data_dir / f"{safe_key}.bin"
                metadata_path = self.metadata_dir / f"{safe_key}.json"

                # ファイル存在チェック
                if not data_path.exists() or not metadata_path.exists():
                    return None

                # メタデータ読み込み
                with open(metadata_path, encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                    metadata = CacheMetadata(**metadata_dict)

                # 期限チェック
                if metadata.is_expired:
                    logger.debug(f"期限切れファイル: {key}")
                    return None

                # データ読み込み
                with open(data_path, "rb") as f:
                    data = f.read()

                return data, metadata

            except Exception as e:
                logger.error(f"ファイル取得失敗 {key}: {e}")
                return None

    def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            try:
                safe_key = self._safe_filename(key)

                data_path = self.data_dir / f"{safe_key}.bin"
                metadata_path = self.metadata_dir / f"{safe_key}.json"

                deleted = False

                if data_path.exists():
                    data_path.unlink()
                    deleted = True

                if metadata_path.exists():
                    metadata_path.unlink()
                    deleted = True

                if deleted:
                    logger.debug(f"ファイル削除完了: {key}")

                return deleted

            except Exception as e:
                logger.error(f"ファイル削除失敗 {key}: {e}")
                return False

    def list_keys(self, pattern: str = None) -> List[str]:
        """キー一覧取得"""
        keys = []
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata_dict = json.load(f)
                        metadata = CacheMetadata(**metadata_dict)

                        if not metadata.is_expired:
                            key = metadata.key
                            if pattern is None or pattern in key:
                                keys.append(key)

                except Exception as e:
                    logger.debug(f"メタデータ読み込み失敗 {metadata_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"キー一覧取得失敗: {e}")

        return keys

    def cleanup_expired(self) -> int:
        """期限切れデータクリーンアップ"""
        deleted_count = 0

        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata_dict = json.load(f)
                        metadata = CacheMetadata(**metadata_dict)

                        if metadata.is_expired:
                            if self.delete(metadata.key):
                                deleted_count += 1

                except Exception as e:
                    logger.debug(f"クリーンアップ処理失敗 {metadata_file}: {e}")
                    continue

            if deleted_count > 0:
                logger.info(f"期限切れファイルクリーンアップ完了: {deleted_count}件")

        except Exception as e:
            logger.error(f"クリーンアップ失敗: {e}")

        return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージ統計情報"""
        try:
            total_files = 0
            total_size = 0
            expired_files = 0

            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    total_files += 1

                    with open(metadata_file, encoding="utf-8") as f:
                        metadata_dict = json.load(f)
                        metadata = CacheMetadata(**metadata_dict)

                        total_size += metadata.data_size

                        if metadata.is_expired:
                            expired_files += 1

                except Exception:
                    continue

            return {
                "total_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "expired_files": expired_files,
                "storage_path": str(self.base_path),
                "data_directory": str(self.data_dir),
                "metadata_directory": str(self.metadata_dir),
            }

        except Exception as e:
            logger.error(f"統計情報取得失敗: {e}")
            return {}

    def _safe_filename(self, key: str) -> str:
        """安全なファイル名生成"""
        return hashlib.sha256(key.encode()).hexdigest()


class PersistentCacheManager:
    """永続化キャッシュ管理システム"""

    def __init__(
        self,
        storage_type: str = "sqlite",
        storage_path: str = "data/persistent_cache",
        compression: bool = True,
        auto_cleanup: bool = True,
        cleanup_interval_seconds: int = 3600,  # 1時間
    ):
        self.compression = compression
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # ストレージ初期化
        if storage_type == "sqlite":
            db_path = Path(storage_path) / "cache.db"
            self.storage = SQLiteStorage(str(db_path))
        elif storage_type == "filesystem":
            self.storage = FileSystemStorage(storage_path)
        else:
            raise ValueError(f"未サポートのストレージタイプ: {storage_type}")

        self.compression_manager = CompressionManager()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "compression_saves_mb": 0.0,
            "last_cleanup": time.time(),
        }
        self._lock = threading.RLock()

        # 自動クリーンアップ開始
        if auto_cleanup:
            self._start_cleanup_timer()

        logger.info(
            f"PersistentCacheManager初期化完了: {storage_type}, 圧縮={compression}"
        )

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        tags: List[str] = None,
        priority: float = 1.0,
    ) -> bool:
        """データ設定"""
        with self._lock:
            try:
                # データシリアライズ
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    # Pandas特別処理
                    data = value.to_pickle()
                    data_type = "pandas"
                else:
                    data = pickle.dumps(sanitize_cache_value(value))
                    data_type = "pickle"

                original_size = len(data)

                # 圧縮処理
                compression_type = None
                compressed_size = None

                if self.compression and original_size > 1024:  # 1KB以上で圧縮
                    compression_type = self.compression_manager.get_best_algorithm(data)

                    if compression_type != "none":
                        (
                            compressed_data,
                            compression_time,
                        ) = self.compression_manager.compress(data, compression_type)

                        # 圧縮効率チェック
                        if (
                            len(compressed_data) < original_size * 0.9
                        ):  # 10%以上圧縮できた場合
                            data = compressed_data
                            compressed_size = len(compressed_data)
                            self._stats["compression_saves_mb"] += (
                                original_size - compressed_size
                            ) / (1024 * 1024)
                        else:
                            compression_type = None

                # メタデータ作成
                now = time.time()
                metadata = CacheMetadata(
                    key=key,
                    created_at=now,
                    expires_at=now + ttl_seconds,
                    last_accessed=now,
                    access_count=0,
                    data_size=original_size,
                    data_type=data_type,
                    compression_type=compression_type,
                    compressed_size=compressed_size,
                    checksum=hashlib.sha256(data).hexdigest(),
                    tags=tags or [],
                    priority=priority,
                )

                # ストレージに保存
                if self.storage.store(key, data, metadata):
                    self._stats["sets"] += 1
                    logger.debug(f"永続キャッシュ設定完了: {key}, TTL={ttl_seconds}s")
                    return True
                else:
                    return False

            except Exception as e:
                logger.error(f"永続キャッシュ設定失敗 {key}: {e}")
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """データ取得"""
        with self._lock:
            try:
                result = self.storage.retrieve(key)
                if not result:
                    self._stats["misses"] += 1
                    return default

                data, metadata = result

                # データ検証
                checksum = hashlib.sha256(data).hexdigest()
                if metadata.checksum and checksum != metadata.checksum:
                    logger.warning(f"データ整合性エラー {key}: チェックサム不一致")
                    self._stats["misses"] += 1
                    return default

                # 展開処理
                if metadata.compression_type:
                    data, decompression_time = self.compression_manager.decompress(
                        data, metadata.compression_type
                    )

                # デシリアライズ
                if metadata.data_type == "pandas":
                    value = pd.read_pickle(data)
                else:
                    value = pickle.loads(data)

                self._stats["hits"] += 1
                logger.debug(f"永続キャッシュヒット: {key}")
                return value

            except Exception as e:
                logger.error(f"永続キャッシュ取得失敗 {key}: {e}")
                self._stats["misses"] += 1
                return default

    def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            try:
                if self.storage.delete(key):
                    self._stats["deletes"] += 1
                    logger.debug(f"永続キャッシュ削除完了: {key}")
                    return True
                return False

            except Exception as e:
                logger.error(f"永続キャッシュ削除失敗 {key}: {e}")
                return False

    def clear(self, pattern: str = None) -> int:
        """キャッシュクリア"""
        with self._lock:
            try:
                keys = self.storage.list_keys(pattern)
                cleared_count = 0

                for key in keys:
                    if self.storage.delete(key):
                        cleared_count += 1

                logger.info(f"永続キャッシュクリア完了: {cleared_count}件")
                return cleared_count

            except Exception as e:
                logger.error(f"永続キャッシュクリア失敗: {e}")
                return 0

    def cleanup_expired(self) -> int:
        """期限切れデータクリーンアップ"""
        with self._lock:
            try:
                deleted_count = self.storage.cleanup_expired()
                self._stats["last_cleanup"] = time.time()
                return deleted_count

            except Exception as e:
                logger.error(f"期限切れクリーンアップ失敗: {e}")
                return 0

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._lock:
            storage_stats = self.storage.get_storage_stats()

            # 統合統計情報
            stats = {
                **self._stats,
                **storage_stats,
                "hit_rate": self._stats["hits"]
                / max(self._stats["hits"] + self._stats["misses"], 1),
                "total_operations": self._stats["hits"]
                + self._stats["misses"]
                + self._stats["sets"],
                "last_cleanup_age_seconds": time.time() - self._stats["last_cleanup"],
            }

            return stats

    def _start_cleanup_timer(self):
        """自動クリーンアップタイマー開始"""

        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval_seconds)
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"自動クリーンアップエラー: {e}")

        timer_thread = threading.Thread(target=cleanup_worker, daemon=True)
        timer_thread.start()
        logger.info(
            f"自動クリーンアップタイマー開始: {self.cleanup_interval_seconds}秒間隔"
        )


# グローバルインスタンス
_global_persistent_cache: Optional[PersistentCacheManager] = None
_cache_lock = threading.Lock()


def get_persistent_cache(
    storage_type: str = "sqlite", storage_path: str = "data/persistent_cache"
) -> PersistentCacheManager:
    """グローバル永続キャッシュインスタンス取得"""
    global _global_persistent_cache

    if _global_persistent_cache is None:
        with _cache_lock:
            if _global_persistent_cache is None:
                _global_persistent_cache = PersistentCacheManager(
                    storage_type=storage_type, storage_path=storage_path
                )

    return _global_persistent_cache


# 便利なデコレータ
def persistent_cache(
    ttl_seconds: int = 3600,
    storage_type: str = "sqlite",
    compression: bool = True,
    tags: List[str] = None,
):
    """永続キャッシュデコレータ"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_persistent_cache(storage_type=storage_type)

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
    print("=== Issue #377 永続化キャッシュシステムテスト ===")

    # SQLiteストレージテスト
    cache = PersistentCacheManager(storage_type="sqlite", storage_path="test_cache")

    print("\n1. データ保存・取得テスト")
    test_data = {
        "message": "Hello, World!",
        "numbers": [1, 2, 3, 4, 5],
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
        print(f"  {key}: {value}")

    print("\n3. デコレータテスト")

    @persistent_cache(ttl_seconds=30, tags=["test"])
    def expensive_calculation(n):
        time.sleep(0.1)  # 重い処理をシミュレート
        return sum(i * i for i in range(n))

    start_time = time.perf_counter()
    result1 = expensive_calculation(1000)
    first_time = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    result2 = expensive_calculation(1000)  # キャッシュからの取得
    cached_time = (time.perf_counter() - start_time) * 1000

    print(f"初回実行: {result1}, 時間: {first_time:.1f}ms")
    print(f"キャッシュ取得: {result2}, 時間: {cached_time:.1f}ms")
    print(f"高速化率: {first_time/cached_time:.1f}x")

    print("\n=== 永続化キャッシュシステムテスト完了 ===")
