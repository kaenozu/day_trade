"""
最適化されたキャッシュシステム v2

LRU、TTL、階層化キャッシュを統合したハイパフォーマンスキャッシュ
"""

import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union, Callable
from collections import OrderedDict
from functools import wraps
import pickle
import hashlib

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry(Generic[T]):
    """キャッシュエントリ"""
    value: T
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """期限切れチェック"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """アクセス時刻更新"""
        self.last_access = time.time()
        self.access_count += 1


class CacheStrategy(ABC, Generic[K, V]):
    """キャッシュ戦略の抽象基底クラス"""
    
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """値取得"""
        pass
    
    @abstractmethod
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """値設定"""
        pass
    
    @abstractmethod
    def evict(self, key: K) -> bool:
        """削除"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """全削除"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """サイズ取得"""
        pass


class LRUCache(CacheStrategy[K, V]):
    """LRU（Least Recently Used）キャッシュ"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # LRU更新（最近使用として末尾に移動）
            self._cache.move_to_end(key)
            entry.touch()
            
            return entry.value
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            # 既存エントリの更新
            if key in self._cache:
                self._cache[key] = CacheEntry(value, time.time(), ttl=ttl)
                self._cache.move_to_end(key)
                return
            
            # 容量チェック
            if len(self._cache) >= self.max_size:
                # 最古のエントリを削除
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(f"LRU eviction: {oldest_key}")
            
            # 新規エントリ追加
            self._cache[key] = CacheEntry(value, time.time(), ttl=ttl)
    
    def evict(self, key: K) -> bool:
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)


class TTLCache(CacheStrategy[K, V]):
    """TTL（Time To Live）キャッシュ"""
    
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._lock = threading.RLock()
        
        # バックグラウンドでの期限切れエントリ削除
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            self._cache[key] = CacheEntry(value, time.time(), ttl=effective_ttl)
    
    def evict(self, key: K) -> bool:
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)
    
    def _cleanup_expired(self) -> None:
        """期限切れエントリの定期削除"""
        while True:
            try:
                time.sleep(60)  # 1分間隔
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items() 
                        if entry.is_expired()
                    ]
                    for key in expired_keys:
                        del self._cache[key]
                    
                    if expired_keys:
                        logger.debug(f"TTL cleanup: removed {len(expired_keys)} expired entries")
            except Exception as e:
                logger.error(f"TTL cleanup error: {e}")


class HierarchicalCache(CacheStrategy[K, V]):
    """階層化キャッシュ（L1: LRU, L2: TTL）"""
    
    def __init__(self, l1_size: int = 100, l2_ttl: float = 300.0):
        self.l1_cache = LRUCache[K, V](max_size=l1_size)
        self.l2_cache = TTLCache[K, V](default_ttl=l2_ttl)
        self._lock = threading.RLock()
        
        # 統計情報
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get(self, key: K) -> Optional[V]:
        with self._lock:
            # L1キャッシュチェック
            value = self.l1_cache.get(key)
            if value is not None:
                self.l1_hits += 1
                return value
            
            # L2キャッシュチェック
            value = self.l2_cache.get(key)
            if value is not None:
                self.l2_hits += 1
                # L1に昇格
                self.l1_cache.put(key, value)
                return value
            
            self.misses += 1
            return None
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            # 両方のキャッシュに保存
            self.l1_cache.put(key, value, ttl)
            self.l2_cache.put(key, value, ttl)
    
    def evict(self, key: K) -> bool:
        with self._lock:
            evicted_l1 = self.l1_cache.evict(key)
            evicted_l2 = self.l2_cache.evict(key)
            return evicted_l1 or evicted_l2
    
    def clear(self) -> None:
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l1_hits = 0
            self.l2_hits = 0
            self.misses = 0
    
    def size(self) -> int:
        return self.l1_cache.size() + self.l2_cache.size()
    
    def get_hit_rate(self) -> float:
        """ヒット率取得"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        if total_requests == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits) / total_requests
    
    def get_stats(self) -> Dict[str, any]:
        """統計情報取得"""
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "l1_size": self.l1_cache.size(),
            "l2_size": self.l2_cache.size()
        }


class OptimizedCacheManager:
    """最適化されたキャッシュマネージャー"""
    
    def __init__(self):
        self._caches: Dict[str, CacheStrategy] = {}
        self._lock = threading.RLock()
    
    def create_cache(
        self, 
        name: str, 
        cache_type: str = "hierarchical",
        **kwargs
    ) -> CacheStrategy:
        """
        キャッシュインスタンス作成
        
        Args:
            name: キャッシュ名
            cache_type: キャッシュタイプ（lru, ttl, hierarchical）
            **kwargs: キャッシュ固有のパラメータ
            
        Returns:
            CacheStrategy: キャッシュインスタンス
        """
        with self._lock:
            if name in self._caches:
                logger.warning(f"Cache '{name}' already exists")
                return self._caches[name]
            
            if cache_type == "lru":
                cache = LRUCache(max_size=kwargs.get("max_size", 1000))
            elif cache_type == "ttl":
                cache = TTLCache(default_ttl=kwargs.get("default_ttl", 300.0))
            elif cache_type == "hierarchical":
                cache = HierarchicalCache(
                    l1_size=kwargs.get("l1_size", 100),
                    l2_ttl=kwargs.get("l2_ttl", 300.0)
                )
            else:
                raise ValueError(f"Unknown cache type: {cache_type}")
            
            self._caches[name] = cache
            logger.info(f"Created {cache_type} cache: {name}")
            return cache
    
    def get_cache(self, name: str) -> Optional[CacheStrategy]:
        """キャッシュ取得"""
        return self._caches.get(name)
    
    def clear_all(self) -> None:
        """全キャッシュクリア"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("All caches cleared")
    
    def get_global_stats(self) -> Dict[str, any]:
        """全キャッシュの統計情報"""
        stats = {}
        for name, cache in self._caches.items():
            if hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats()
            else:
                stats[name] = {"size": cache.size()}
        return stats


# グローバルキャッシュマネージャー
cache_manager_v2 = OptimizedCacheManager()


def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """
    関数結果キャッシングデコレーター
    
    Args:
        cache_name: キャッシュ名
        ttl: TTL（秒）
        key_func: キーアクセサ関数
    """
    def decorator(func):
        # キャッシュを作成（存在しない場合）
        cache = cache_manager_v2.get_cache(cache_name)
        if cache is None:
            cache = cache_manager_v2.create_cache(cache_name, "hierarchical")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キー生成
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # デフォルト：引数のハッシュ
                key_data = pickle.dumps((args, sorted(kwargs.items())))
                cache_key = hashlib.md5(key_data).hexdigest()
            
            # キャッシュから取得試行
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 関数実行してキャッシュに保存
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# 使用例とテスト用の関数
@cached(cache_name="technical_indicators", ttl=60.0)
def calculate_indicator_cached(symbol: str, indicator_type: str, period: int):
    """キャッシュ付きテクニカル指標計算のサンプル"""
    logger.info(f"Computing {indicator_type} for {symbol} (period={period})")
    # 実際の計算はここに入る
    time.sleep(0.1)  # 計算時間のシミュレーション
    return f"result_{symbol}_{indicator_type}_{period}"