"""
統合ユーティリティパッケージ

Phase 3: 設定管理統合とユーティリティ抽出の一環として、
分散していたユーティリティ機能を統合し、保守性を向上させます。
"""

# 新しい統合ユーティリティ（推奨）
# 既存のユーティリティ（後方互換性）
from .cache_utils import (
    CacheCircuitBreakerError,
    CacheError,
    CacheStats,
    CacheTimeoutError,
    HighPerformanceCache,
    TTLCache,
    default_cache,
    generate_safe_cache_key,
    high_perf_cache,
    sanitize_cache_value,
    validate_cache_key,
)
from .unified_formatters import (
    create_error_panel,
    create_status_table,
    create_stock_info_table,
    create_summary_panel,
    create_unified_table,
    format_currency,
    format_datetime,
    format_decimal_safe,
    format_large_number,
    format_percentage,
    format_volume,
    get_change_color,
)
from .unified_utils import (
    CircuitBreaker,
    ThreadSafeCounter,
    batch_process,
    calculate_time_difference,
    clean_string,
    convert_to_safe_dict,
    ensure_directory,
    generate_unique_id,
    merge_dictionaries,
    retry_on_failure,
    safe_execute,
    timing_decorator,
    validate_data_structure,
)

__all__ = [
    # 新しい統合フォーマッタ（推奨）
    "format_currency",
    "format_percentage",
    "format_volume",
    "format_large_number",
    "format_datetime",
    "format_decimal_safe",
    "get_change_color",
    "create_unified_table",
    "create_summary_panel",
    "create_status_table",
    "create_error_panel",
    "create_stock_info_table",
    # 新しい統合ユーティリティ（推奨）
    "safe_execute",
    "retry_on_failure",
    "timing_decorator",
    "batch_process",
    "generate_unique_id",
    "validate_data_structure",
    "convert_to_safe_dict",
    "clean_string",
    "merge_dictionaries",
    "calculate_time_difference",
    "ThreadSafeCounter",
    "CircuitBreaker",
    "ensure_directory",
    # キャッシュユーティリティ
    "generate_safe_cache_key",
    "validate_cache_key",
    "sanitize_cache_value",
    "CacheStats",
    "TTLCache",
    "HighPerformanceCache",
    "default_cache",
    "high_perf_cache",
    "CacheError",
    "CacheCircuitBreakerError",
    "CacheTimeoutError",
]
