"""
キャッシュモジュールパッケージ

cache_utils.pyの分割されたモジュール群をまとめ、
後方互換性を提供するパッケージ初期化ファイル。

このパッケージは以下のモジュールから構成されます：
- constants: 定数と例外定義
- config: 設定管理
- circuit_breaker: サーキットブレーカー機能
- key_generator: キー生成とシリアライゼーション
- statistics: 統計管理
- validators: バリデーションとサニタイゼーション
- ttl_cache: TTLキャッシュ実装
- high_performance_cache: 高性能キャッシュ実装
"""

# 基本的なクラスと関数をインポート（後方互換性）
from .constants import (
    CacheConstants,
    CacheError,
    CacheCircuitBreakerError,
    CacheTimeoutError,
    CacheKeyError,
    CacheSerializationError,
    CacheValidationError,
    CacheConfigurationError,
)

from .config import (
    CacheConfig,
    get_cache_config,
    set_cache_config,
    reset_cache_config,
    cache_config,  # 後方互換性
)

from .circuit_breaker import (
    CacheCircuitBreaker,
    get_cache_circuit_breaker,
    set_cache_circuit_breaker,
    reset_cache_circuit_breaker,
)

from .key_generator import (
    generate_safe_cache_key,
    _normalize_arguments,
    _json_serializer,
    _generate_fallback_cache_key,
    _generate_emergency_cache_key,
    _estimate_data_complexity,
)

from .statistics import (
    CacheStats,
)

from .validators import (
    validate_cache_key,
    is_valid_cache_key,
    sanitize_cache_key,
    validate_cache_value,
    is_valid_cache_value,
    sanitize_cache_value,
    validate_ttl,
    sanitize_ttl,
    validate_cache_size,
    get_safe_cache_key_preview,
)

from .ttl_cache import (
    TTLCache,
)

from .high_performance_cache import (
    HighPerformanceCache,
)

# デフォルトキャッシュインスタンス（設定ベース・後方互換性）
default_cache = TTLCache()
high_perf_cache = HighPerformanceCache()

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Modular cache utilities package split from cache_utils.py"

# 公開API（明示的にエクスポートする要素）
__all__ = [
    # 定数と例外
    "CacheConstants",
    "CacheError",
    "CacheCircuitBreakerError", 
    "CacheTimeoutError",
    "CacheKeyError",
    "CacheSerializationError",
    "CacheValidationError",
    "CacheConfigurationError",
    
    # 設定管理
    "CacheConfig",
    "get_cache_config",
    "set_cache_config",
    "reset_cache_config",
    "cache_config",
    
    # サーキットブレーカー
    "CacheCircuitBreaker",
    "get_cache_circuit_breaker",
    "set_cache_circuit_breaker",
    "reset_cache_circuit_breaker",
    
    # キー生成
    "generate_safe_cache_key",
    
    # 統計
    "CacheStats",
    
    # バリデーション
    "validate_cache_key",
    "is_valid_cache_key",
    "sanitize_cache_key",
    "validate_cache_value",
    "is_valid_cache_value", 
    "sanitize_cache_value",
    "validate_ttl",
    "sanitize_ttl",
    "validate_cache_size",
    "get_safe_cache_key_preview",
    
    # キャッシュ実装
    "TTLCache",
    "HighPerformanceCache",
    
    # デフォルトインスタンス
    "default_cache",
    "high_perf_cache",
]

# 後方互換性のためのエイリアス
validate_cache_key_safe = is_valid_cache_key
validate_cache_value_safe = is_valid_cache_value

def get_package_info():
    """パッケージ情報を取得"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": [
            "constants",
            "config", 
            "circuit_breaker",
            "key_generator",
            "statistics",
            "validators",
            "ttl_cache",
            "high_performance_cache",
        ],
        "public_api_count": len(__all__),
    }

def create_ttl_cache(max_size=None, default_ttl=None, config=None):
    """
    TTLキャッシュインスタンスを作成
    
    Args:
        max_size: 最大サイズ
        default_ttl: デフォルトTTL
        config: キャッシュ設定
        
    Returns:
        TTLCacheインスタンス
    """
    return TTLCache(max_size=max_size, default_ttl=default_ttl, config=config)

def create_high_performance_cache(max_size=None, config=None):
    """
    高性能キャッシュインスタンスを作成
    
    Args:
        max_size: 最大サイズ
        config: キャッシュ設定
        
    Returns:
        HighPerformanceCacheインスタンス  
    """
    return HighPerformanceCache(max_size=max_size, config=config)

def reset_all_caches():
    """全てのデフォルトキャッシュをリセット"""
    global default_cache, high_perf_cache
    
    try:
        if hasattr(default_cache, 'clear'):
            default_cache.clear()
        if hasattr(high_perf_cache, 'clear'):
            high_perf_cache.clear()
            
        # サーキットブレーカーもリセット
        reset_cache_circuit_breaker()
        
    except Exception as e:
        from ..logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Error resetting caches: {e}", exc_info=True)

# モジュールレベルの初期化
def _initialize_package():
    """パッケージの初期化処理"""
    try:
        from ..logging_config import get_logger
        logger = get_logger(__name__)
        logger.debug(f"Cache package initialized: {len(__all__)} public APIs available")
        
    except Exception as e:
        # ロガー取得に失敗した場合は警告のみ
        print(f"Warning: Failed to initialize cache package logging: {e}")

# パッケージ読み込み時に初期化
_initialize_package()