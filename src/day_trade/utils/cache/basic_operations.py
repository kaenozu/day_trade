"""
キャッシュ基本操作モジュール

高性能キャッシュの基本的なCRUD操作を提供します。
"""

import threading
import time
from typing import Any, Optional

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants
from .validators import validate_cache_key, validate_cache_size

logger = get_logger(__name__)


class BasicCacheOperations:
    """
    キャッシュの基本操作を提供するクラス
    """

    def __init__(
        self, 
        max_size: Optional[int] = None, 
        config: Optional["CacheConfig"] = None
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        self._config = config or get_cache_config()
        self._cache = {}
        self._access_times = {}
        self._max_size = max_size or self._config.high_perf_cache_size
        
        # 入力検証
        validate_cache_size(self._max_size)
        
        self._lock = threading.Lock()  # RLockより高速
        self._time = time.time  # 関数参照をキャッシュ
        self._cleanup_threshold = (
            self._max_size * CacheConstants.DEFAULT_CLEANUP_THRESHOLD_RATIO
        )

        logger.debug(
            f"BasicCacheOperations initialized: max_size={self._max_size}, "
            f"cleanup_threshold={self._cleanup_threshold}"
        )

    def get(self, key: str) -> Any:
        """
        超高速get操作
        
        read-heavy workload用にdouble-checked lockingパターンを使用
        
        Args:
            key: 取得するキー
            
        Returns:
            キャッシュされた値、存在しない場合はNone
        """
        try:
            # 基本的な検証（高速化のため最小限）
            if not key:
                return None

            # ロックなしの高速パス（read-heavy workload最適化）
            if key in self._cache:
                current_time = self._time()
                with self._lock:
                    if key in self._cache:  # double-checked locking
                        self._access_times[key] = current_time
                        value = self._cache[key]
                        logger.debug(f"BasicCacheOperations hit for key: {key[:50]}...")
                        return value

            # キャッシュミス
            return None

        except Exception as e:
            logger.error(f"Error getting value from BasicCacheOperations: {e}", exc_info=True)
            return None

    def set(self, key: str, value: Any) -> bool:
        """
        高速set操作
        
        Args:
            key: 設定するキー
            value: 設定する値
            
        Returns:
            設定に成功したかどうか
        """
        try:
            # 入力検証
            if not key or not isinstance(key, str):
                return False

            current_time = self._time()

            with self._lock:
                self._cache[key] = value
                self._access_times[key] = current_time

                # 自動サイズ管理
                if len(self._cache) > self._cleanup_threshold:
                    self._auto_cleanup()

                logger.debug(f"BasicCacheOperations set key: {key[:50]}...")

            return True

        except Exception as e:
            logger.error(f"Error setting value in BasicCacheOperations: {e}", exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        """
        キーの削除
        
        Args:
            key: 削除するキー
            
        Returns:
            削除に成功したかどうか
        """
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]
                    logger.debug(f"BasicCacheOperations deleted key: {key[:50]}...")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting key from BasicCacheOperations: {e}", exc_info=True)
            return False

    def clear(self) -> None:
        """キャッシュのクリア"""
        try:
            with self._lock:
                cleared_count = len(self._cache)
                self._cache.clear()
                self._access_times.clear()
                logger.info(f"BasicCacheOperations cleared {cleared_count} entries")

        except Exception as e:
            logger.error(f"Error clearing BasicCacheOperations: {e}", exc_info=True)

    def _auto_cleanup(self) -> int:
        """
        自動クリーンアップ（最も使用されていないエントリを削除）
        ロック内で呼び出される前提
        
        Returns:
            削除されたエントリの数
        """
        try:
            # 既に最適サイズ以下の場合は何もしない
            if len(self._cache) <= self._max_size * 0.5:
                return 0

            # アクセス時間順でソートして古いものを削除
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            remove_count = len(self._cache) - int(
                self._max_size * CacheConstants.DEFAULT_HIGH_PERF_CLEANUP_RATIO
            )

            removed_keys = []
            for key, _ in sorted_items[:remove_count]:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]
                    removed_keys.append(key)

            if removed_keys:
                logger.debug(
                    f"BasicCacheOperations auto-cleanup removed {len(removed_keys)} entries"
                )

            return len(removed_keys)

        except Exception as e:
            logger.error(f"Error during auto-cleanup: {e}", exc_info=True)
            return 0

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
        """キーが存在するかどうか"""
        try:
            return key in self._cache
        except Exception:
            return False

    def get_keys(self) -> list:
        """現在のキーリストを取得（デバッグ用）"""
        try:
            with self._lock:
                return list(self._cache.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}", exc_info=True)
            return []

    def trim_to_size(self, target_size: int) -> int:
        """
        指定サイズまでキャッシュを縮小
        
        Args:
            target_size: 目標サイズ
            
        Returns:
            削除されたエントリの数
        """
        try:
            validate_cache_size(target_size)
            
            with self._lock:
                current_size = len(self._cache)
                if current_size <= target_size:
                    return 0

                # 古いエントリから削除
                sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
                remove_count = current_size - target_size
                
                removed_keys = []
                for key, _ in sorted_items[:remove_count]:
                    if key in self._cache:
                        del self._cache[key]
                        del self._access_times[key]
                        removed_keys.append(key)

                logger.info(f"BasicCacheOperations trimmed {len(removed_keys)} entries to target size {target_size}")
                return len(removed_keys)

        except Exception as e:
            logger.error(f"Error trimming cache to size: {e}", exc_info=True)
            return 0