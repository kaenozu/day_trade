"""
高性能キャッシュモジュール

超高性能キャッシュ実装を提供します。
最小限のロックと最適化されたデータ構造を使用し、
read-heavy workloadに最適化されています。
"""

import threading
import time
from typing import Any, Dict, Optional

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants
from .validators import validate_cache_key, validate_cache_size

logger = get_logger(__name__)


class HighPerformanceCache:
    """
    超高性能キャッシュ実装
    最小限のロックと最適化されたデータ構造を使用
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
        
        # 統計情報（軽量版）
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._start_time = self._time()

        logger.debug(
            f"HighPerformanceCache initialized: max_size={self._max_size}, "
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
                self._misses += 1
                return None

            # ロックなしの高速パス（read-heavy workload最適化）
            if key in self._cache:
                current_time = self._time()
                with self._lock:
                    if key in self._cache:  # double-checked locking
                        self._access_times[key] = current_time
                        self._hits += 1
                        value = self._cache[key]
                        logger.debug(f"HighPerformanceCache hit for key: {key[:50]}...")
                        return value

            # キャッシュミス
            self._misses += 1
            return None

        except Exception as e:
            logger.error(f"Error getting value from HighPerformanceCache: {e}", exc_info=True)
            self._misses += 1
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
                self._sets += 1

                # 自動サイズ管理
                if len(self._cache) > self._cleanup_threshold:
                    self._auto_cleanup()

                logger.debug(f"HighPerformanceCache set key: {key[:50]}...")

            return True

        except Exception as e:
            logger.error(f"Error setting value in HighPerformanceCache: {e}", exc_info=True)
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
                    logger.debug(f"HighPerformanceCache deleted key: {key[:50]}...")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting key from HighPerformanceCache: {e}", exc_info=True)
            return False

    def clear(self) -> None:
        """キャッシュのクリア"""
        try:
            with self._lock:
                cleared_count = len(self._cache)
                self._cache.clear()
                self._access_times.clear()
                logger.info(f"HighPerformanceCache cleared {cleared_count} entries")

        except Exception as e:
            logger.error(f"Error clearing HighPerformanceCache: {e}", exc_info=True)

    def _auto_cleanup(self) -> None:
        """
        自動クリーンアップ（最も使用されていないエントリを削除）
        ロック内で呼び出される前提
        """
        try:
            # 既に最適サイズ以下の場合は何もしない
            if len(self._cache) <= self._max_size * 0.5:
                return

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

            self._evictions += len(removed_keys)
            
            if removed_keys:
                logger.debug(
                    f"HighPerformanceCache auto-cleanup removed {len(removed_keys)} entries"
                )

        except Exception as e:
            logger.error(f"Error during auto-cleanup: {e}", exc_info=True)

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

    def get_hit_rate(self) -> float:
        """ヒット率を取得"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        current_time = self._time()
        uptime = current_time - self._start_time
        total_requests = self._hits + self._misses
        
        return {
            # 基本統計
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "evictions": self._evictions,
            "total_requests": total_requests,
            
            # 効率指標
            "hit_rate": self.get_hit_rate(),
            "miss_rate": 1.0 - self.get_hit_rate() if total_requests > 0 else 0.0,
            
            # キャッシュ状態
            "cache_size": self.size(),
            "max_size": self._max_size,
            "cleanup_threshold": self._cleanup_threshold,
            "fill_ratio": self.size() / self._max_size if self._max_size > 0 else 0.0,
            
            # パフォーマンス指標
            "operations_per_second": total_requests / uptime if uptime > 0 else 0.0,
            "uptime_seconds": uptime,
            "start_time": self._start_time,
        }

    def get_keys(self) -> list:
        """現在のキーリストを取得（デバッグ用）"""
        try:
            with self._lock:
                return list(self._cache.keys())
        except Exception as e:
            logger.error(f"Error getting keys: {e}", exc_info=True)
            return []

    def get_most_accessed_keys(self, limit: int = 10) -> list:
        """
        最もアクセスされたキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（新しい順）
        """
        try:
            with self._lock:
                sorted_items = sorted(
                    self._access_times.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                return [key for key, _ in sorted_items[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting most accessed keys: {e}", exc_info=True)
            return []

    def get_least_accessed_keys(self, limit: int = 10) -> list:
        """
        最もアクセスされていないキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（古い順）
        """
        try:
            with self._lock:
                sorted_items = sorted(
                    self._access_times.items(), 
                    key=lambda x: x[1]
                )
                return [key for key, _ in sorted_items[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting least accessed keys: {e}", exc_info=True)
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

                self._evictions += len(removed_keys)
                
                logger.info(f"HighPerformanceCache trimmed {len(removed_keys)} entries to target size {target_size}")
                return len(removed_keys)

        except Exception as e:
            logger.error(f"Error trimming cache to size: {e}", exc_info=True)
            return 0

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        try:
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._start_time = self._time()
            logger.debug("HighPerformanceCache stats reset")
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}", exc_info=True)