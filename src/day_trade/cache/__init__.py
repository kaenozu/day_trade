"""
統合キャッシュシステム

Phase 5: パフォーマンス最適化の一環として、
巨大なcache_utils.py（64KB, 1760行）を機能ごとに分割し、
保守性とパフォーマンスを向上させます。

分割構造:
- core: 基本キャッシュ機能とインターフェース
- memory_cache: インメモリキャッシュ実装
- key_generator: キャッシュキー生成
- stats: 統計・メトリクス収集
- config: 設定管理
- errors: エラーハンドリング

使用例:
    from day_trade.cache import MemoryCache, TTLCache

    cache = MemoryCache(max_size=1000)
    ttl_cache = TTLCache(max_size=500, default_ttl=300)
"""

# 主要クラスとインターフェースをエクスポート
from .config import CacheConfig, CacheConstants, get_cache_config
from .core import BaseCacheManager, CacheEntry, CacheInterface
from .core import CacheStats as CoreCacheStats
from .errors import (
    CacheCircuitBreaker,
    CacheCircuitBreakerError,
    CacheError,
    CacheTimeoutError,
    get_cache_circuit_breaker,
)
from .key_generator import CacheKeyGenerator, generate_safe_cache_key
from .memory_cache import (
    HighPerformanceCache,
    MemoryCache,
    TTLCache,
    get_default_high_perf_cache,
    get_default_memory_cache,
    get_default_ttl_cache,
)
from .stats import (
    CacheMetrics,
    CacheStats,
    CacheStatsCollector,
    get_global_stats_collector,
)

# 後方互換性のためのエイリアス
generate_cache_key = generate_safe_cache_key

__version__ = "1.0.0"

__all__ = [
    # 主要インターフェース
    "CacheInterface",
    "BaseCacheManager",
    "CacheEntry",
    # キャッシュ実装
    "MemoryCache",
    "TTLCache",
    "HighPerformanceCache",
    # デフォルトインスタンス
    "get_default_memory_cache",
    "get_default_ttl_cache",
    "get_default_high_perf_cache",
    # ユーティリティ
    "generate_safe_cache_key",
    "generate_cache_key",  # 後方互換性
    "CacheKeyGenerator",
    # 統計・メトリクス
    "CacheStats",
    "CoreCacheStats",
    "CacheMetrics",
    "CacheStatsCollector",
    "get_global_stats_collector",
    # 設定
    "CacheConfig",
    "CacheConstants",
    "get_cache_config",
    # エラーハンドリング
    "CacheError",
    "CacheCircuitBreakerError",
    "CacheTimeoutError",
    "CacheCircuitBreaker",
    "get_cache_circuit_breaker",
]
