#!/usr/bin/env python3
"""
Memory Cache Provider
メモリキャッシュプロバイダー

高速アクセス用のインメモリキャッシュ実装
"""

import threading
import time
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import weakref

from ..interfaces.cache_interfaces import ICacheProvider, ICacheSerializer, ICacheEvictionPolicy
from ..exceptions.risk_exceptions import CacheError

class CacheItem:
    """キャッシュアイテム"""

    def __init__(self, value: Any, ttl_seconds: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 1
        self.ttl_seconds = ttl_seconds
        self.expires_at = self.created_at + ttl_seconds if ttl_seconds else None

    def is_expired(self) -> bool:
        """期限切れ判定"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self):
        """アクセス時刻更新"""
        self.last_accessed = time.time()
        self.access_count += 1

    def age_seconds(self) -> float:
        """経過時間（秒）"""
        return time.time() - self.created_at

    def time_since_last_access(self) -> float:
        """最終アクセスからの経過時間（秒）"""
        return time.time() - self.last_accessed

class MemoryCacheProvider(ICacheProvider):
    """メモリキャッシュプロバイダー"""

    def __init__(self, config: Dict[str, Any]):
        self.max_size = config.get('max_size', 1000)
        self.default_ttl_seconds = config.get('ttl_seconds', 3600)
        self.thread_safe = config.get('thread_safe', True)
        self.cleanup_interval_seconds = config.get('cleanup_interval_seconds', 300)
        self.enable_stats = config.get('enable_stats', True)

        # キャッシュストレージ
        self._cache: Dict[str, CacheItem] = OrderedDict()

        # 統計情報
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'cleanups': 0,
            'memory_usage_bytes': 0
        }

        # スレッドセーフティ
        self._lock = threading.RLock() if self.thread_safe else None

        # 立ち退きポリシー
        self._eviction_policy = config.get('eviction_policy')

        # シリアライザー
        self._serializer = config.get('serializer')

        # バックグラウンドクリーンアップ
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        if self.cleanup_interval_seconds > 0:
            self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        """キー取得"""
        with self._get_lock():
            try:
                if key not in self._cache:
                    if self.enable_stats:
                        self._stats['misses'] += 1
                    return None

                item = self._cache[key]

                # 期限切れチェック
                if item.is_expired():
                    del self._cache[key]
                    if self.enable_stats:
                        self._stats['misses'] += 1
                    return None

                # アクセス情報更新
                item.touch()

                # LRU対応でアイテムを末尾に移動
                self._cache.move_to_end(key)

                if self.enable_stats:
                    self._stats['hits'] += 1

                return item.value

            except Exception as e:
                raise CacheError(
                    f"Failed to get cache item for key: {key}",
                    cache_key=key,
                    operation="get",
                    cache_provider="memory",
                    cause=e
                )

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """キー設定"""
        with self._get_lock():
            try:
                # TTLデフォルト値適用
                if ttl_seconds is None:
                    ttl_seconds = self.default_ttl_seconds

                # キャッシュサイズ制限チェック
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._evict_items(1)

                # アイテム作成・設定
                item = CacheItem(value, ttl_seconds)
                self._cache[key] = item

                # LRU対応でアイテムを末尾に移動
                self._cache.move_to_end(key)

                if self.enable_stats:
                    self._stats['sets'] += 1
                    self._update_memory_usage()

                return True

            except Exception as e:
                raise CacheError(
                    f"Failed to set cache item for key: {key}",
                    cache_key=key,
                    operation="set",
                    cache_provider="memory",
                    cause=e
                )

    def delete(self, key: str) -> bool:
        """キー削除"""
        with self._get_lock():
            try:
                if key in self._cache:
                    del self._cache[key]
                    if self.enable_stats:
                        self._stats['deletes'] += 1
                        self._update_memory_usage()
                    return True
                return False

            except Exception as e:
                raise CacheError(
                    f"Failed to delete cache item for key: {key}",
                    cache_key=key,
                    operation="delete",
                    cache_provider="memory",
                    cause=e
                )

    def exists(self, key: str) -> bool:
        """キー存在確認"""
        with self._get_lock():
            if key not in self._cache:
                return False

            item = self._cache[key]
            if item.is_expired():
                del self._cache[key]
                return False

            return True

    def clear(self) -> bool:
        """全キャッシュクリア"""
        with self._get_lock():
            try:
                self._cache.clear()
                if self.enable_stats:
                    self._stats['memory_usage_bytes'] = 0
                return True

            except Exception as e:
                raise CacheError(
                    "Failed to clear cache",
                    operation="clear",
                    cache_provider="memory",
                    cause=e
                )

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """キー一覧取得"""
        with self._get_lock():
            all_keys = list(self._cache.keys())

            if pattern is None:
                return all_keys

            # 簡易パターンマッチング
            import fnmatch
            return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]

    def size(self) -> int:
        """キャッシュサイズ取得"""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._get_lock():
            stats = self._stats.copy()
            stats.update({
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': self._calculate_hit_rate(),
                'expired_items': self._count_expired_items(),
                'average_age_seconds': self._calculate_average_age()
            })
            return stats

    def cleanup_expired(self) -> int:
        """期限切れアイテムクリーンアップ"""
        with self._get_lock():
            expired_keys = []

            for key, item in self._cache.items():
                if item.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            if self.enable_stats:
                self._stats['cleanups'] += 1
                self._update_memory_usage()

            return len(expired_keys)

    def get_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量情報取得"""
        import sys

        with self._get_lock():
            total_size = 0
            item_count = len(self._cache)

            # 概算メモリサイズ計算
            for key, item in self._cache.items():
                total_size += sys.getsizeof(key)
                total_size += sys.getsizeof(item)
                total_size += sys.getsizeof(item.value)

            return {
                'total_bytes': total_size,
                'item_count': item_count,
                'average_item_size': total_size // item_count if item_count > 0 else 0,
                'overhead_bytes': sys.getsizeof(self._cache)
            }

    def close(self):
        """リソース解放"""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
        self.clear()

    def _get_lock(self):
        """ロック取得（スレッドセーフな場合）"""
        if self._lock:
            return self._lock
        else:
            # ダミーコンテキストマネージャー
            from contextlib import nullcontext
            return nullcontext()

    def _evict_items(self, count: int):
        """アイテム立ち退き"""
        if self._eviction_policy and hasattr(self._eviction_policy, 'evict'):
            # カスタム立ち退きポリシー使用
            keys_to_evict = self._eviction_policy.evict(self._cache, count)
        else:
            # デフォルトLRU立ち退き
            keys_to_evict = list(self._cache.keys())[:count]

        for key in keys_to_evict:
            if key in self._cache:
                del self._cache[key]
                if self.enable_stats:
                    self._stats['evictions'] += 1

    def _calculate_hit_rate(self) -> float:
        """ヒット率計算"""
        total_requests = self._stats['hits'] + self._stats['misses']
        if total_requests == 0:
            return 0.0
        return self._stats['hits'] / total_requests

    def _count_expired_items(self) -> int:
        """期限切れアイテム数カウント"""
        expired_count = 0
        current_time = time.time()

        for item in self._cache.values():
            if item.expires_at and current_time > item.expires_at:
                expired_count += 1

        return expired_count

    def _calculate_average_age(self) -> float:
        """平均経過時間計算"""
        if not self._cache:
            return 0.0

        total_age = sum(item.age_seconds() for item in self._cache.values())
        return total_age / len(self._cache)

    def _update_memory_usage(self):
        """メモリ使用量更新"""
        memory_info = self.get_memory_usage()
        self._stats['memory_usage_bytes'] = memory_info['total_bytes']

    def _start_cleanup_thread(self):
        """バックグラウンドクリーンアップ開始"""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    self.cleanup_expired()
                    self._stop_cleanup.wait(self.cleanup_interval_seconds)
                except Exception:
                    # クリーンアップエラーは無視
                    pass

        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            daemon=True,
            name="MemoryCacheCleanup"
        )
        self._cleanup_thread.start()

    def __len__(self) -> int:
        """キャッシュサイズ"""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """キー存在確認"""
        return self.exists(key)

    def __getitem__(self, key: str) -> Any:
        """キー取得"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any):
        """キー設定"""
        self.set(key, value)

    def __delitem__(self, key: str):
        """キー削除"""
        if not self.delete(key):
            raise KeyError(key)
