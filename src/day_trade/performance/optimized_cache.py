#!/usr/bin/env python3
"""
高速キャッシュシステム

メモリ効率とアクセス速度を最適化したキャッシュ実装
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple
from functools import wraps
import hashlib
import pickle


class OptimizedCache:
    """最適化キャッシュクラス"""

    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0

    def _generate_key(self, args: Tuple, kwargs: Dict) -> str:
        """キー生成（高速化）"""
        # より高速なキー生成
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """期限切れチェック"""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl

    def _cleanup_expired(self):
        """期限切れエントリのクリーンアップ"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl
        ]

        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        """値取得"""
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                # LRU更新
                self._cache.move_to_end(key)
                self._hit_count += 1
                return self._cache[key]

            self._miss_count += 1
            return None

    def set(self, key: str, value: Any):
        """値設定"""
        with self._lock:
            # 容量制限チェック
            if len(self._cache) >= self.max_size:
                # 最も古いエントリを削除
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
                self._timestamps.pop(oldest_key, None)

            self._cache[key] = value
            self._timestamps[key] = time.time()

            # 定期的なクリーンアップ
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()

    def cache_decorator(self):
        """キャッシュデコレータ"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # キー生成
                cache_key = f"{func.__name__}_{self._generate_key(args, kwargs)}"

                # キャッシュから取得試行
                result = self.get(cache_key)
                if result is not None:
                    return result

                # 関数実行してキャッシュ
                result = func(*args, **kwargs)
                self.set(cache_key, result)
                return result

            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': hit_rate,
            'memory_usage': len(pickle.dumps(self._cache))
        }


class CacheManager:
    """キャッシュマネージャー"""

    def __init__(self):
        self.caches: Dict[str, OptimizedCache] = {}

    def get_cache(self, name: str, max_size: int = 1000, ttl: float = 3600) -> OptimizedCache:
        """キャッシュ取得（なければ作成）"""
        if name not in self.caches:
            self.caches[name] = OptimizedCache(max_size, ttl)
        return self.caches[name]

    def clear_all(self):
        """全キャッシュクリア"""
        for cache in self.caches.values():
            cache._cache.clear()
            cache._timestamps.clear()

    def get_global_stats(self) -> Dict[str, Any]:
        """全体統計"""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats


# グローバルキャッシュマネージャー
cache_manager = CacheManager()

# 便利な関数
def cached(cache_name: str = 'default', max_size: int = 1000, ttl: float = 3600):
    """キャッシュデコレータ"""
    cache = cache_manager.get_cache(cache_name, max_size, ttl)
    return cache.cache_decorator()

# 使用例:
# @cached('stock_data', max_size=500, ttl=1800)  # 30分キャッシュ
# def get_stock_price(symbol):
#     # 重い処理
#     return fetch_stock_price(symbol)
