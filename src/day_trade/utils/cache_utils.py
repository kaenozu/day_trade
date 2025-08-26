"""
キャッシュユーティリティ（後方互換性モジュール）

このモジュールは後方互換性を保つために、
新しいモジュラー構造（cache/パッケージ）からインポートします。

非推奨: 新しいコードでは直接 'from day_trade.utils.cache import ...' を使用してください。
"""

# 後方互換性のために、全ての公開APIをインポート
from .cache import *  # noqa: F401, F403

# 特定のアリアスと後方互換性関数
from .cache import (
    # 主要なクラスと関数
    CacheConstants,
    CacheError,
    CacheCircuitBreakerError,
    CacheTimeoutError,
    CacheKeyError,
    CacheSerializationError,
    CacheValidationError,
    CacheConfigurationError,
    CacheConfig,
    get_cache_config,
    set_cache_config,
    cache_config,  # グローバル設定インスタンス
    CacheCircuitBreaker,
    get_cache_circuit_breaker,
    generate_safe_cache_key,
    CacheStats,
    validate_cache_key,
    sanitize_cache_value,
    TTLCache,
    HighPerformanceCache,
    default_cache,
    high_perf_cache,
    
    # 内部関数（後方互換性のため）
    _normalize_arguments,
    _json_serializer,
    _generate_fallback_cache_key,
    _generate_emergency_cache_key,
    _estimate_data_complexity,
)

# モジュールレベルのメタデータ
__version__ = "2.0.0"  # モジュラー化されたバージョン
__author__ = "Day Trade System"
__description__ = "Backward compatibility module for cache utilities (now modular)"

# 警告メッセージ
def _show_deprecation_warning():
    """cache_utils.pyの使用に関する非推奨警告を表示"""
    import warnings
    warnings.warn(
        "cache_utils.py は非推奨です。新しいコードでは "
        "'from day_trade.utils.cache import ...' を使用してください。",
        DeprecationWarning,
        stacklevel=2
    )

# モジュールインポート時に警告を表示
_show_deprecation_warning()