"""
Cache Layer Abstraction
キャッシュレイヤー抽象化

統一されたキャッシュインターフェースと複数プロバイダー実装
"""

from .cache_decorators import cache_aside, cache_result, invalidate_cache
from .cache_manager import CacheManager
from .distributed_cache import DistributedCacheProvider
from .eviction_policies import (
    FIFOEvictionPolicy,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    RandomEvictionPolicy,
    TTLEvictionPolicy,
)
from .file_cache import FileCacheProvider
from .hybrid_cache import HybridCacheProvider
from .memory_cache import MemoryCacheProvider
from .redis_cache import RedisCacheProvider
from .serializers import (
    CompressionSerializer,
    JsonSerializer,
    MsgPackSerializer,
    PickleSerializer,
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
    "RandomEvictionPolicy",
]
