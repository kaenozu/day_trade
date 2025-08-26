"""
TTL（Time To Live）キャッシュモジュール

高性能TTLキャッシュ実装を提供します。
スレッドセーフで効率的な期限管理とLRU（Least Recently Used）エビクションをサポートします。
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants
from .statistics import CacheStats
from .validators import validate_cache_key, validate_ttl

logger = get_logger(__name__)


class TTLCache:
    """
    高性能TTL（Time To Live）キャッシュ実装
    スレッドセーフで効率的な期限管理を提供
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        default_ttl: Optional[int] = None,
        config: Optional["CacheConfig"] = None,
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            default_ttl: デフォルトTTL（秒、Noneの場合は設定から取得）
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        self._config = config or get_cache_config()
        self._cache = OrderedDict()
        self._timestamps = {}
        self._ttls = {}
        self._max_size = max_size or self._config.default_ttl_cache_size
        self._default_ttl = default_ttl or self._config.default_ttl_seconds
        self._lock = threading.RLock()
        self._stats = CacheStats(self._config)

        # パフォーマンス最適化
        self._time = time.time  # 関数参照をキャッシュ
        self._cleanup_counter = 0
        self._cleanup_frequency = CacheConstants.DEFAULT_CLEANUP_FREQUENCY

        logger.debug(
            f"TTLCache initialized: max_size={self._max_size}, "
            f"default_ttl={self._default_ttl}, cleanup_frequency={self._cleanup_frequency}"
        )

    def get(self, key: str, default=None) -> Any:
        """
        キャッシュからの値取得（TTLチェック付き）

        Args:
            key: キャッシュキー
            default: キーが見つからない場合のデフォルト値

        Returns:
            キャッシュされた値またはデフォルト値
        """
        try:
            # キーの妥当性チェック（高速化のため例外を使わない版）
            if not key or not isinstance(key, str):
                self._stats.record_miss()
                return default

            with self._lock:
                if key not in self._cache:
                    self._stats.record_miss()
                    return default

                # TTLチェック（高速化）
                current_time = self._time()
                if current_time > self._timestamps[key] + self._ttls[key]:
                    # 期限切れ
                    self._remove_expired_key(key)
                    self._stats.record_miss()
                    return default

                # LRU更新
                self._cache.move_to_end(key)
                self._stats.record_hit()
                
                value = self._cache[key]
                logger.debug(f"TTLCache hit for key: {key[:50]}...")
                return value

        except Exception as e:
            logger.error(f"Error getting value from TTLCache: {e}", exc_info=True)
            self._stats.record_error()
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        キャッシュへの値設定

        Args:
            key: キャッシュキー
            value: 設定する値
            ttl: TTL（秒、Noneの場合はデフォルトTTLを使用）

        Returns:
            設定に成功したかどうか
        """
        try:
            # 入力検証
            validate_cache_key(key, self._config)
            if ttl is not None:
                validate_ttl(ttl)

            if ttl is None:
                ttl = self._default_ttl

            current_time = self._time()

            with self._lock:
                # 既存エントリの更新
                if key in self._cache:
                    self._cache[key] = value
                    self._timestamps[key] = current_time
                    self._ttls[key] = ttl
                    self._cache.move_to_end(key)
                    logger.debug(f"TTLCache updated existing key: {key[:50]}...")
                else:
                    # 新規エントリ
                    if len(self._cache) >= self._max_size:
                        self._evict_lru()

                    self._cache[key] = value
                    self._timestamps[key] = current_time
                    self._ttls[key] = ttl
                    logger.debug(f"TTLCache added new key: {key[:50]}...")

                self._stats.record_set()

                # 定期的なクリーンアップ（パフォーマンス最適化）
                self._cleanup_counter += 1
                if self._cleanup_counter >= self._cleanup_frequency:
                    self._cleanup_expired()
                    self._cleanup_counter = 0

            return True

        except Exception as e:
            logger.error(f"Error setting value in TTLCache: {e}", exc_info=True)
            self._stats.record_error()
            return False

    def delete(self, key: str) -> bool:
        """
        キャッシュからの削除

        Args:
            key: 削除するキー

        Returns:
            削除に成功したかどうか
        """
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    del self._timestamps[key]
                    del self._ttls[key]
                    logger.debug(f"TTLCache deleted key: {key[:50]}...")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting key from TTLCache: {e}", exc_info=True)
            self._stats.record_error()
            return False

    def clear(self) -> None:
        """キャッシュのクリア"""
        try:
            with self._lock:
                cleared_count = len(self._cache)
                self._cache.clear()
                self._timestamps.clear()
                self._ttls.clear()
                self._cleanup_counter = 0
                
                logger.info(f"TTLCache cleared {cleared_count} entries")

        except Exception as e:
            logger.error(f"Error clearing TTLCache: {e}", exc_info=True)
            self._stats.record_error()

    def _remove_expired_key(self, key: str) -> None:
        """期限切れキーの削除（内部メソッド）"""
        try:
            del self._cache[key]
            del self._timestamps[key]
            del self._ttls[key]
            self._stats.record_eviction()
            logger.debug(f"TTLCache expired key: {key[:50]}...")

        except KeyError:
            # キーが既に存在しない場合（競合状態）
            pass
        except Exception as e:
            logger.error(f"Error removing expired key: {e}", exc_info=True)

    def _evict_lru(self) -> None:
        """LRU方式での古いエントリ削除"""
        try:
            if self._cache:
                oldest_key = next(iter(self._cache))
                self._remove_expired_key(oldest_key)
                logger.debug(f"TTLCache evicted LRU key: {oldest_key[:50]}...")

        except Exception as e:
            logger.error(f"Error during LRU eviction: {e}", exc_info=True)

    def _cleanup_expired(self) -> None:
        """期限切れエントリの一括削除"""
        try:
            current_time = self._time()
            expired_keys = []

            # 期限切れキーを特定
            for key, timestamp in self._timestamps.items():
                if current_time > timestamp + self._ttls[key]:
                    expired_keys.append(key)

            # 期限切れキーを削除
            for key in expired_keys:
                self._remove_expired_key(key)

            if expired_keys:
                logger.debug(f"TTLCache cleanup removed {len(expired_keys)} expired entries")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}", exc_info=True)

    def size(self) -> int:
        """現在のキャッシュサイズ"""
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            logger.error(f"Error getting cache size: {e}", exc_info=True)
            return 0

    def is_empty(self) -> bool:
        """キャッシュが空かどうか"""
        return self.size() == 0

    def contains(self, key: str) -> bool:
        """キーがキャッシュに存在するかどうか（期限切れチェック付き）"""
        try:
            with self._lock:
                if key not in self._cache:
                    return False

                # TTLチェック
                current_time = self._time()
                if current_time > self._timestamps[key] + self._ttls[key]:
                    # 期限切れの場合は削除して False を返す
                    self._remove_expired_key(key)
                    return False

                return True

        except Exception as e:
            logger.error(f"Error checking key existence: {e}", exc_info=True)
            return False

    def get_ttl(self, key: str) -> Optional[float]:
        """
        キーの残りTTLを取得

        Args:
            key: 確認するキー

        Returns:
            残りTTL（秒）、キーが存在しない場合はNone
        """
        try:
            with self._lock:
                if key not in self._cache:
                    return None

                current_time = self._time()
                expiry_time = self._timestamps[key] + self._ttls[key]
                
                if current_time >= expiry_time:
                    # 期限切れ
                    self._remove_expired_key(key)
                    return None

                return expiry_time - current_time

        except Exception as e:
            logger.error(f"Error getting TTL for key: {e}", exc_info=True)
            return None

    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """
        キーのTTLを延長

        Args:
            key: 延長するキー
            additional_seconds: 追加する秒数

        Returns:
            延長に成功したかどうか
        """
        try:
            with self._lock:
                if key not in self._cache:
                    return False

                # 期限切れチェック
                current_time = self._time()
                if current_time > self._timestamps[key] + self._ttls[key]:
                    self._remove_expired_key(key)
                    return False

                # TTLを延長
                self._ttls[key] += additional_seconds
                logger.debug(f"TTLCache extended TTL for key: {key[:50]}... by {additional_seconds}s")
                return True

        except Exception as e:
            logger.error(f"Error extending TTL: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """キャッシュ統計の取得"""
        stats = self._stats.to_dict()
        stats.update({
            "cache_size": self.size(),
            "max_size": self._max_size,
            "default_ttl": self._default_ttl,
            "cleanup_frequency": self._cleanup_frequency,
            "cleanup_counter": self._cleanup_counter,
        })
        return stats

    def get_keys(self) -> list:
        """現在のキーリストを取得（デバッグ用）"""
        try:
            with self._lock:
                # 期限切れキーをクリーンアップしてからキーを返す
                self._cleanup_expired()
                return list(self._cache.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}", exc_info=True)
            return []