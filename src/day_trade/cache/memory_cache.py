"""
インメモリキャッシュ実装

統合キャッシュシステムの高性能インメモリキャッシュを提供します。
元のcache_utils.pyからキャッシュ実装を分離・改良。
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from .config import CacheConstants, get_cache_config
from .core import BaseCacheManager, CacheEntry

logger = logging.getLogger(__name__)


class MemoryCache(BaseCacheManager):
    """
    基本的なメモリキャッシュ実装

    LRU（Least Recently Used）による自動削除機能付き
    """

    def __init__(
        self, max_size: Optional[int] = None, enable_stats: bool = True, config=None
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            enable_stats: 統計収集を有効にするか
            config: キャッシュ設定（Noneの場合はグローバル設定を使用）
        """
        super().__init__(enable_stats)
        self._config = config or get_cache_config()
        self._cache = OrderedDict()
        self._max_size = max_size or self._config.default_ttl_cache_size
        self._lock = threading.RLock()

        logger.debug(f"MemoryCache initialized with max_size={self._max_size}")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """キャッシュからの値取得"""
        try:
            with self._lock:
                if key not in self._cache:
                    self._record_miss()
                    return default

                # LRU更新
                self._cache.move_to_end(key)
                self._record_hit()
                return self._cache[key]

        except Exception as e:
            self._record_error()
            logger.error(f"Error getting key '{key}' from memory cache: {e}")
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュへの値設定"""
        try:
            with self._lock:
                # 既存エントリの更新
                if key in self._cache:
                    self._cache[key] = value
                    self._cache.move_to_end(key)
                else:
                    # 新規エントリ
                    if len(self._cache) >= self._max_size:
                        self._evict_lru()
                    self._cache[key] = value

                self._record_set()
                return True

        except Exception as e:
            self._record_error()
            logger.error(f"Error setting key '{key}' in memory cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """キャッシュからの削除"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    self._record_delete()
                    return True
                return False

        except Exception as e:
            self._record_error()
            logger.error(f"Error deleting key '{key}' from memory cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        try:
            with self._lock:
                return key in self._cache
        except Exception as e:
            self._record_error()
            logger.error(f"Error checking existence of key '{key}': {e}")
            return False

    def clear(self) -> None:
        """全てのキャッシュエントリを削除"""
        try:
            with self._lock:
                self._cache.clear()
                logger.debug("Memory cache cleared")
        except Exception as e:
            self._record_error()
            logger.error(f"Error clearing memory cache: {e}")

    def size(self) -> int:
        """キャッシュに保存されているアイテム数を返す"""
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            self._record_error()
            logger.error(f"Error getting cache size: {e}")
            return 0

    def _evict_lru(self) -> None:
        """LRUアルゴリズムで最も使用されていないエントリを削除"""
        if self._cache:
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted LRU key: {oldest_key}")


class TTLCache(BaseCacheManager):
    """
    高性能TTL（Time To Live）キャッシュ実装

    スレッドセーフで効率的な期限管理を提供
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        default_ttl: Optional[int] = None,
        enable_stats: bool = True,
        config=None,
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            default_ttl: デフォルトTTL（秒、Noneの場合は設定から取得）
            enable_stats: 統計収集を有効にするか
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        super().__init__(enable_stats)
        self._config = config or get_cache_config()
        self._cache = OrderedDict()
        self._entries: Dict[str, CacheEntry] = {}
        self._max_size = max_size or self._config.default_ttl_cache_size
        self._default_ttl = default_ttl or self._config.default_ttl_seconds
        self._lock = threading.RLock()

        # パフォーマンス最適化
        self._time = time.time  # 関数参照をキャッシュ
        self._cleanup_counter = 0
        self._cleanup_frequency = CacheConstants.DEFAULT_CLEANUP_FREQUENCY

        logger.debug(
            f"TTLCache initialized: max_size={self._max_size}, default_ttl={self._default_ttl}"
        )

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """キャッシュからの値取得（TTLチェック付き）"""
        try:
            with self._lock:
                if key not in self._cache:
                    self._record_miss()
                    return default

                # TTLチェック
                entry = self._entries[key]
                current_time = self._time()

                if entry.is_expired(current_time):
                    # 期限切れ
                    self._remove_expired_key(key)
                    self._record_miss()
                    return default

                # LRU更新とアクセス記録
                self._cache.move_to_end(key)
                entry.touch(current_time)
                self._record_hit()
                return entry.value

        except Exception as e:
            self._record_error()
            logger.error(f"Error getting key '{key}' from TTL cache: {e}")
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュへの値設定"""
        try:
            if ttl is None:
                ttl = self._default_ttl

            current_time = self._time()
            entry = CacheEntry(value, ttl, current_time)

            with self._lock:
                # 既存エントリの更新
                if key in self._cache:
                    self._cache[key] = value
                    self._entries[key] = entry
                    self._cache.move_to_end(key)
                else:
                    # 新規エントリ
                    if len(self._cache) >= self._max_size:
                        self._evict_lru()
                    self._cache[key] = value
                    self._entries[key] = entry

                self._record_set()

                # 定期的なクリーンアップ（パフォーマンス最適化）
                self._cleanup_counter += 1
                if self._cleanup_counter >= self._cleanup_frequency:
                    self._cleanup_expired()
                    self._cleanup_counter = 0

                return True

        except Exception as e:
            self._record_error()
            logger.error(f"Error setting key '{key}' in TTL cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """キャッシュからの削除"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    del self._entries[key]
                    self._record_delete()
                    return True
                return False

        except Exception as e:
            self._record_error()
            logger.error(f"Error deleting key '{key}' from TTL cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック（TTL考慮）"""
        try:
            with self._lock:
                if key not in self._cache:
                    return False

                entry = self._entries[key]
                if entry.is_expired():
                    self._remove_expired_key(key)
                    return False

                return True

        except Exception as e:
            self._record_error()
            logger.error(f"Error checking existence of key '{key}': {e}")
            return False

    def clear(self) -> None:
        """全てのキャッシュエントリを削除"""
        try:
            with self._lock:
                self._cache.clear()
                self._entries.clear()
                self._cleanup_counter = 0
                logger.debug("TTL cache cleared")
        except Exception as e:
            self._record_error()
            logger.error(f"Error clearing TTL cache: {e}")

    def size(self) -> int:
        """キャッシュに保存されているアイテム数を返す"""
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            self._record_error()
            logger.error(f"Error getting TTL cache size: {e}")
            return 0

    def get_ttl(self, key: str) -> Optional[float]:
        """キーの残りTTLを取得"""
        try:
            with self._lock:
                if key not in self._entries:
                    return None

                entry = self._entries[key]
                return entry.get_remaining_ttl()

        except Exception as e:
            logger.error(f"Error getting TTL for key '{key}': {e}")
            return None

    def _evict_lru(self) -> None:
        """LRUアルゴリズムで最も使用されていないエントリを削除"""
        if self._cache:
            oldest_key, _ = self._cache.popitem(last=False)
            if oldest_key in self._entries:
                del self._entries[oldest_key]
            logger.debug(f"Evicted LRU key from TTL cache: {oldest_key}")

    def _remove_expired_key(self, key: str) -> None:
        """期限切れキーを削除"""
        try:
            if key in self._cache:
                del self._cache[key]
            if key in self._entries:
                del self._entries[key]
        except KeyError:
            pass  # 既に削除済み

    def _cleanup_expired(self) -> None:
        """期限切れエントリのクリーンアップ"""
        try:
            current_time = self._time()
            expired_keys = []

            # 期限切れキーを特定（リストコピーでdeadlock回避）
            for key, entry in list(self._entries.items()):
                if entry.is_expired(current_time):
                    expired_keys.append(key)

            # 期限切れキーを削除
            for key in expired_keys:
                self._remove_expired_key(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired keys")

        except Exception as e:
            logger.error(f"Error during TTL cache cleanup: {e}")


class HighPerformanceCache(BaseCacheManager):
    """
    超高性能キャッシュ実装

    最小限のロックと最適化されたデータ構造を使用
    read-heavy workloadに最適化
    """

    def __init__(
        self, max_size: Optional[int] = None, enable_stats: bool = True, config=None
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            enable_stats: 統計収集を有効にするか
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        super().__init__(enable_stats)
        self._config = config or get_cache_config()
        self._cache = {}
        self._access_times = {}
        self._max_size = max_size or self._config.high_perf_cache_size
        self._lock = threading.Lock()  # RLockより高速
        self._time = time.time
        self._cleanup_threshold = (
            self._max_size * CacheConstants.DEFAULT_CLEANUP_THRESHOLD_RATIO
        )

        logger.debug(f"HighPerformanceCache initialized with max_size={self._max_size}")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """超高速get操作"""
        try:
            # ロックなしの高速パス（read-heavy workload最適化）
            if key in self._cache:
                current_time = self._time()
                with self._lock:
                    if key in self._cache:  # double-checked locking
                        self._access_times[key] = current_time
                        self._record_hit()
                        return self._cache[key]

            self._record_miss()
            return default

        except Exception as e:
            self._record_error()
            logger.error(f"Error getting key '{key}' from high-performance cache: {e}")
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """高速set操作"""
        try:
            current_time = self._time()

            with self._lock:
                self._cache[key] = value
                self._access_times[key] = current_time

                # 自動サイズ管理
                if len(self._cache) > self._cleanup_threshold:
                    self._auto_cleanup()

                self._record_set()
                return True

        except Exception as e:
            self._record_error()
            logger.error(f"Error setting key '{key}' in high-performance cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """キャッシュからの削除"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]
                    self._record_delete()
                    return True
                return False

        except Exception as e:
            self._record_error()
            logger.error(f"Error deleting key '{key}' from high-performance cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        try:
            with self._lock:
                return key in self._cache
        except Exception as e:
            self._record_error()
            logger.error(f"Error checking existence of key '{key}': {e}")
            return False

    def clear(self) -> None:
        """全てのキャッシュエントリを削除"""
        try:
            with self._lock:
                self._cache.clear()
                self._access_times.clear()
                logger.debug("High-performance cache cleared")
        except Exception as e:
            self._record_error()
            logger.error(f"Error clearing high-performance cache: {e}")

    def size(self) -> int:
        """キャッシュに保存されているアイテム数を返す"""
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            self._record_error()
            logger.error(f"Error getting high-performance cache size: {e}")
            return 0

    def _auto_cleanup(self) -> None:
        """自動クリーンアップ（最も使用されていないエントリを削除）"""
        try:
            if len(self._cache) <= self._max_size * 0.5:
                return

            # アクセス時間順でソートして古いものを削除
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            remove_count = len(self._cache) - int(
                self._max_size * CacheConstants.DEFAULT_HIGH_PERF_CLEANUP_RATIO
            )

            removed_count = 0
            for key, _ in sorted_items[:remove_count]:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]
                    removed_count += 1

            logger.debug(
                f"Auto cleanup removed {removed_count} entries from high-performance cache"
            )

        except Exception as e:
            logger.error(f"Error during high-performance cache auto cleanup: {e}")


# デフォルトキャッシュインスタンス（設定ベース）
_default_memory_cache = None
_default_ttl_cache = None
_default_high_perf_cache = None
_cache_lock = threading.RLock()


def get_default_memory_cache() -> MemoryCache:
    """デフォルトメモリキャッシュインスタンスを取得"""
    global _default_memory_cache

    if _default_memory_cache is None:
        with _cache_lock:
            if _default_memory_cache is None:
                _default_memory_cache = MemoryCache()

    return _default_memory_cache


def get_default_ttl_cache() -> TTLCache:
    """デフォルトTTLキャッシュインスタンスを取得"""
    global _default_ttl_cache

    if _default_ttl_cache is None:
        with _cache_lock:
            if _default_ttl_cache is None:
                _default_ttl_cache = TTLCache()

    return _default_ttl_cache


def get_default_high_perf_cache() -> HighPerformanceCache:
    """デフォルト高性能キャッシュインスタンスを取得"""
    global _default_high_perf_cache

    if _default_high_perf_cache is None:
        with _cache_lock:
            if _default_high_perf_cache is None:
                _default_high_perf_cache = HighPerformanceCache()

    return _default_high_perf_cache


# 後方互換性のためのエイリアス
default_cache = get_default_ttl_cache  # TTLキャッシュをデフォルトとして使用
high_perf_cache = get_default_high_perf_cache
