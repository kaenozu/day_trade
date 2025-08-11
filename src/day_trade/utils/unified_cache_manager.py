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
        self.access_order = OrderedDict()  # O(1)での削除を可能にするためにOrderedDictを使用
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
                self.access_order[key] = None # O(1)で追加
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
                del self.access_order[key] # O(1)で削除

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

    def clear(self):
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("DELETE FROM cache_entries")
                    conn.commit()
                    self.stats["current_size"] = 0
            except Exception as e:
                logger.error(f"L3キャッシュクリアエラー: {e}")

    def _get_total_size(self, conn) -> int:
        """総サイズ取得"""
        cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
        result = cursor.fetchone()[0]
        return result or 0

    def _evict_oldest(self, conn) -> bool:
        """最古エントリ退避"""
        cursor = conn.execute(
            "SELECT key FROM cache_entries ORDER BY last_accessed ASC LIMIT 1"
        )
        row = cursor.fetchone()

        if row:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (row[0],))
            self.stats["evictions"] += 1
            return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    # エントリ数
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    entry_count = cursor.fetchone()[0]

                    # 総サイズ
                    total_size = self._get_total_size(conn)

                    # ヒット率
                    total_requests = self.stats["hits"] + self.stats["misses"]
                    hit_rate = (
                        self.stats["hits"] / total_requests if total_requests > 0 else 0
                    )

                    return {
                        "layer": "L3_Cold",
                        "hit_rate": hit_rate,
                        "entries": entry_count,
                        "disk_usage_mb": total_size / 1024 / 1024,
                        "max_disk_mb": self.max_size_bytes / 1024 / 1024,
                        **self.stats,
                    }
            except Exception as e:
                logger.error(f"L3統計取得エラー: {e}")
                return {"layer": "L3_Cold", "error": str(e)}


class UnifiedCacheManager:
    """統合キャッシュマネージャー"""

    def __init__(
        self,
        l1_memory_mb: int = 64,
        l2_memory_mb: int = 256,
        l3_disk_mb: int = 1024,
        l1_ttl: int = 30,
        l2_ttl: int = 300,
        l3_ttl: int = 86400,
        cache_db_path: str = "data/unified_cache.db",
    ):
        """統合キャッシュマネージャー初期化"""
        self.l1_cache = L1HotCache(l1_memory_mb, l1_ttl)
        self.l2_cache = L2WarmCache(l2_memory_mb, l2_ttl)
        self.l3_cache = L3ColdCache(cache_db_path, l3_disk_mb, l3_ttl)

        self.global_stats = CacheStats()
        self.access_times = deque(maxlen=1000)  # 最近1000回のアクセス時間

        logger.info(
            f"統合キャッシュマネージャー初期化完了 "
            f"(L1:{l1_memory_mb}MB, L2:{l2_memory_mb}MB, L3:{l3_disk_mb}MB)"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """階層化キャッシュからデータ取得"""
        start_time = time.time()

        try:
            # L1 (Hot) キャッシュから検索
            entry = self.l1_cache.get(key)
            if entry:
                self.global_stats.l1_hits += 1
                self._record_access_time(start_time)
                return entry.value

            # L2 (Warm) キャッシュから検索
            entry = self.l2_cache.get(key)
            if entry:
                self.global_stats.l2_hits += 1
                # L1に昇格
                self.l1_cache.put(key, entry)
                self._record_access_time(start_time)
                return entry.value

            # L3 (Cold) キャッシュから検索
            entry = self.l3_cache.get(key)
            if entry:
                self.global_stats.l3_hits += 1
                # L2に昇格 (高頻度アクセスならL1にも)
                self.l2_cache.put(key, entry)
                if entry.access_count > 5:  # 閾値は調整可能
                    self.l1_cache.put(key, entry)
                self._record_access_time(start_time)
                return entry.value

            # キャッシュミス
            self.global_stats.misses += 1
            self._record_access_time(start_time)
            return default

        except Exception as e:
            logger.error(f"キャッシュ取得エラー: {e}")
            self.global_stats.misses += 1
            self._record_access_time(start_time)
            return default

    def put(
        self, key: str, value: Any, priority: float = 1.0, target_layer: str = "auto"
    ) -> bool:
        """階層化キャッシュにデータ保存"""
        try:
            # エントリ作成
            current_time = time.time()
            value_size = len(pickle.dumps(value))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                size_bytes=value_size,
                priority=priority,
            )

            # レイヤー選択
            if target_layer == "auto":
                target_layer = self._select_optimal_layer(entry)

            success = False

            if target_layer in ["l1", "L1", "hot"]:
                success = self.l1_cache.put(key, entry)
            elif target_layer in ["l2", "L2", "warm"]:
                success = self.l2_cache.put(key, entry)
                # 小さくて重要なデータはL1にも
                if value_size < 10 * 1024 and priority > 5.0:
                    self.l1_cache.put(key, entry)
            else:  # L3 or default
                success = self.l3_cache.put(key, entry)
                # 中程度の重要度ならL2にも
                if priority > 2.0:
                    self.l2_cache.put(key, entry)

            return success

        except Exception as e:
            logger.error(f"キャッシュ保存エラー: {e}")
            return False

    def delete(self, key: str) -> bool:
        """全レイヤーからデータ削除"""
        success = False
        success |= self.l1_cache.delete(key)
        success |= self.l2_cache.delete(key)
        success |= self.l3_cache.delete(key)
        return success

    def clear_all(self):
        """全キャッシュクリア"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        self.global_stats = CacheStats()
        logger.info("全キャッシュクリア完了")

    def _select_optimal_layer(self, entry: CacheEntry) -> str:
        """最適レイヤー自動選択"""
        size_kb = entry.size_bytes / 1024

        # 超小サイズ・高優先度 → L1
        if size_kb < 10 and entry.priority > 7.0:
            return "l1"

        # 小〜中サイズ・中高優先度 → L2
        elif size_kb < 100 and entry.priority > 3.0:
            return "l2"

        # その他 → L3
        else:
            return "l3"

    def _record_access_time(self, start_time: float):
        """アクセス時間記録"""
        access_time_ms = (time.time() - start_time) * 1000
        self.access_times.append(access_time_ms)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()

        total_hits = (
            self.global_stats.l1_hits
            + self.global_stats.l2_hits
            + self.global_stats.l3_hits
        )
        total_requests = total_hits + self.global_stats.misses

        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        avg_access_time = np.mean(self.access_times) if self.access_times else 0

        return {
            "overall": {
                "hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "avg_access_time_ms": avg_access_time,
                "l1_hit_ratio": self.global_stats.l1_hits / total_requests
                if total_requests > 0
                else 0,
                "l2_hit_ratio": self.global_stats.l2_hits / total_requests
                if total_requests > 0
                else 0,
                "l3_hit_ratio": self.global_stats.l3_hits / total_requests
                if total_requests > 0
                else 0,
            },
            "layers": {"L1": l1_stats, "L2": l2_stats, "L3": l3_stats},
            "memory_usage_total_mb": (
                l1_stats.get("memory_usage_mb", 0) + l2_stats.get("memory_usage_mb", 0)
            ),
            "disk_usage_mb": l3_stats.get("disk_usage_mb", 0),
        }

    def optimize_memory(self):
        """メモリ最適化"""
        try:
            # システムメモリ使用率チェック
            memory_percent = psutil.virtual_memory().percent

            if memory_percent > 85:  # 85%以上ならアグレッシブにクリア
                logger.warning(
                    f"高メモリ使用率検出: {memory_percent}% - L1キャッシュクリア"
                )
                self.l1_cache.clear()

                if memory_percent > 90:  # 90%以上ならL2も部分クリア
                    logger.warning("極度の高メモリ使用率 - L2キャッシュ部分クリア")
                    # L2の半分をクリア
                    l2_entries = len(self.l2_cache.cache)
                    clear_count = l2_entries // 2
                    keys_to_clear = list(self.l2_cache.cache.keys())[:clear_count]
                    for key in keys_to_clear:
                        self.l2_cache.delete(key)

        except Exception as e:
            logger.error(f"メモリ最適化エラー: {e}")


# 統合キー生成ユーティリティ
def generate_unified_cache_key(
    component: str,
    operation: str,
    symbol: str = "",
    params: Dict[str, Any] = None,
    time_bucket_minutes: int = 1,
) -> str:
    """統一キャッシュキー生成"""

    # 時間バケット生成 (時間ベースキャッシュ無効化用)
    current_time = datetime.now()
    time_bucket = current_time.replace(
        minute=(current_time.minute // time_bucket_minutes) * time_bucket_minutes,
        second=0,
        microsecond=0,
    ).isoformat()

    # パラメータハッシュ
    params_hash = ""
    if params:
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    # キー組み立て
    key_parts = [component, operation]
    if symbol:
        key_parts.append(symbol)
    if params_hash:
        key_parts.append(params_hash)
    key_parts.append(time_bucket)

    return ":".join(key_parts)


if __name__ == "__main__":
    # テスト実行
    print("=== 統合キャッシュマネージャーテスト ===")

    cache_manager = UnifiedCacheManager()

    # テストデータ
    test_data = {
        "small_hot": "A" * 100,  # 小サイズ・ホット
        "medium_warm": "B" * 10000,  # 中サイズ・ウォーム
        "large_cold": "C" * 100000,  # 大サイズ・コールド
    }

    # データ保存
    print("\n1. データ保存テスト...")
    for key, value in test_data.items():
        priority = 8.0 if "hot" in key else 4.0 if "warm" in key else 1.0
        success = cache_manager.put(key, value, priority=priority)
        print(f"  {key}: {'成功' if success else '失敗'}")

    # データ取得
    print("\n2. データ取得テスト...")
    for key in test_data:
        value = cache_manager.get(key)
        print(f"  {key}: {'ヒット' if value else 'ミス'}")

    # 統計情報
    print("\n3. 統計情報...")
    stats = cache_manager.get_comprehensive_stats()
    overall = stats["overall"]
    print(f"  全体ヒット率: {overall['hit_rate']:.2%}")
    print(f"  平均アクセス時間: {overall['avg_access_time_ms']:.2f}ms")
    print(f"  メモリ使用量: {stats['memory_usage_total_mb']:.1f}MB")

    print("\n✅ 統合キャッシュマネージャーテスト完了")
