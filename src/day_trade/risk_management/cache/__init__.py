"""
Cache Layer Abstraction
キャッシュレイヤー抽象化

統一されたキャッシュインターフェースと複数プロバイダー実装
"""

from .memory_cache import MemoryCacheProvider
from .redis_cache import RedisCacheProvider
from .file_cache import FileCacheProvider
from .distributed_cache import DistributedCacheProvider
from .hybrid_cache import HybridCacheProvider
from .cache_manager import CacheManager
from .cache_decorators import cache_result, invalidate_cache, cache_aside
from .serializers import (
    PickleSerializer,
    JsonSerializer,
    MsgPackSerializer,
    CompressionSerializer
)
from .eviction_policies import (
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    FIFOEvictionPolicy,
    TTLEvictionPolicy,
    RandomEvictionPolicy
)

__all__ = [
    # キャッシュプロバイダー
    "MemoryCacheProvider",
    "RedisCacheProvider",
    "FileCacheProvider",
    "DistributedCacheProvider",
    "HybridCacheProvider",

    # キャッシュ管理
    "CacheManager",

    # デコレーター
    "cache_result",
    "invalidate_cache",
    "cache_aside",

    # シリアライザー
    "PickleSerializer",
    "JsonSerializer",
    "MsgPackSerializer",
    "CompressionSerializer",

    # 立ち退きポリシー
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "FIFOEvictionPolicy",
    "TTLEvictionPolicy",
    "RandomEvictionPolicy"
]
