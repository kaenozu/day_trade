"""
キャッシュ設定管理

統合キャッシュシステムの設定クラスと定数を提供します。
元のcache_utils.pyから設定関連機能を分離。
"""

import os
import threading
from typing import Any, Dict, Optional

from src.day_trade.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheConstants:
    """キャッシュ関連の定数定義"""

    # デフォルト値
    DEFAULT_MAX_KEY_LENGTH = 1000
    DEFAULT_MAX_VALUE_SIZE_MB = 10
    DEFAULT_MAX_RECURSION_DEPTH = 10
    DEFAULT_MIN_RECURSION_DEPTH = 3
    DEFAULT_MAX_ADAPTIVE_DEPTH = 20
    DEFAULT_SERIALIZATION_TIMEOUT = 5.0
    DEFAULT_LOCK_TIMEOUT = 1.0

    # キャッシュ統計
    DEFAULT_MAX_OPERATION_HISTORY = 1000
    DEFAULT_HIT_RATE_WINDOW_SIZE = 100
    DEFAULT_STATS_HISTORY_CLEANUP_SECONDS = 300  # 5分

    # TTLキャッシュ
    DEFAULT_TTL_CACHE_SIZE = 5000
    DEFAULT_TTL_SECONDS = 600  # 10分
    DEFAULT_CLEANUP_FREQUENCY = 100
    DEFAULT_CLEANUP_THRESHOLD_RATIO = 0.8

    # 高性能キャッシュ
    DEFAULT_HIGH_PERF_CACHE_SIZE = 10000
    DEFAULT_HIGH_PERF_CLEANUP_RATIO = 0.7

    # サーキットブレーカー設定
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RECOVERY_TIMEOUT = 60.0
    DEFAULT_HALF_OPEN_MAX_CALLS = 3

    # エラー処理
    MAX_COUNTER_VALUE = 2**63 - 1  # 64bit符号付き整数の最大値
    ERROR_PENALTY_MULTIPLIER = 50

    # シリアライゼーション
    MAX_SET_SORT_ATTEMPTS = 3
    CHARSET_DETECTION_CONFIDENCE_THRESHOLD = 0.7


class CacheConfig:
    """キャッシュ設定の管理クラス（config_manager統合対応・アダプティブ再帰制限付き）"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: ConfigManagerインスタンス（依存性注入）
        """
        self._config_manager = config_manager
        self._load_config()

    def _load_config(self):
        """設定を読み込み（config_manager優先、環境変数フォールバック）"""

        # config_managerから取得を試行
        cache_settings = {}
        if self._config_manager:
            try:
                cache_settings = getattr(self._config_manager, "cache_settings", {})
                if hasattr(self._config_manager, "get"):
                    # より一般的なget方式も試行
                    cache_settings = self._config_manager.get("cache", {}) or cache_settings
            except Exception as e:
                logger.warning(f"Failed to load cache settings from config_manager: {e}")

        # 設定値の決定（優先度: config_manager > 環境変数 > デフォルト）
        self.max_key_length = self._get_config_value(
            cache_settings,
            "max_key_length",
            "CACHE_MAX_KEY_LENGTH",
            CacheConstants.DEFAULT_MAX_KEY_LENGTH,
            int,
        )
        self.max_value_size_mb = self._get_config_value(
            cache_settings,
            "max_value_size_mb",
            "CACHE_MAX_VALUE_SIZE_MB",
            CacheConstants.DEFAULT_MAX_VALUE_SIZE_MB,
            int,
        )
        self.max_recursion_depth = self._get_config_value(
            cache_settings,
            "max_recursion_depth",
            "CACHE_MAX_RECURSION_DEPTH",
            CacheConstants.DEFAULT_MAX_RECURSION_DEPTH,
            int,
        )
        self.enable_size_warnings = self._get_config_value(
            cache_settings,
            "enable_size_warnings",
            "CACHE_ENABLE_SIZE_WARNINGS",
            True,
            bool,
        )

        # アダプティブ再帰制限の設定
        self.adaptive_recursion = self._get_config_value(
            cache_settings, "adaptive_recursion", "CACHE_ADAPTIVE_RECURSION", True, bool
        )
        self.min_recursion_depth = self._get_config_value(
            cache_settings,
            "min_recursion_depth",
            "CACHE_MIN_RECURSION_DEPTH",
            CacheConstants.DEFAULT_MIN_RECURSION_DEPTH,
            int,
        )
        self.max_adaptive_depth = self._get_config_value(
            cache_settings,
            "max_adaptive_depth",
            "CACHE_MAX_ADAPTIVE_DEPTH",
            CacheConstants.DEFAULT_MAX_ADAPTIVE_DEPTH,
            int,
        )

        # パフォーマンス設定
        self.enable_performance_logging = self._get_config_value(
            cache_settings,
            "enable_performance_logging",
            "CACHE_ENABLE_PERFORMANCE_LOGGING",
            False,
            bool,
        )
        self.serialization_timeout = self._get_config_value(
            cache_settings,
            "serialization_timeout",
            "CACHE_SERIALIZATION_TIMEOUT",
            CacheConstants.DEFAULT_SERIALIZATION_TIMEOUT,
            float,
        )

        # 統計設定
        self.max_operation_history = self._get_config_value(
            cache_settings,
            "max_operation_history",
            "CACHE_MAX_OPERATION_HISTORY",
            CacheConstants.DEFAULT_MAX_OPERATION_HISTORY,
            int,
        )
        self.hit_rate_window_size = self._get_config_value(
            cache_settings,
            "hit_rate_window_size",
            "CACHE_HIT_RATE_WINDOW_SIZE",
            CacheConstants.DEFAULT_HIT_RATE_WINDOW_SIZE,
            int,
        )
        self.lock_timeout = self._get_config_value(
            cache_settings,
            "lock_timeout",
            "CACHE_LOCK_TIMEOUT",
            CacheConstants.DEFAULT_LOCK_TIMEOUT,
            float,
        )

        # TTLキャッシュ設定
        self.default_ttl_cache_size = self._get_config_value(
            cache_settings,
            "default_ttl_cache_size",
            "CACHE_DEFAULT_TTL_SIZE",
            CacheConstants.DEFAULT_TTL_CACHE_SIZE,
            int,
        )
        self.default_ttl_seconds = self._get_config_value(
            cache_settings,
            "default_ttl_seconds",
            "CACHE_DEFAULT_TTL_SECONDS",
            CacheConstants.DEFAULT_TTL_SECONDS,
            int,
        )

        # 高性能キャッシュ設定
        self.high_perf_cache_size = self._get_config_value(
            cache_settings,
            "high_perf_cache_size",
            "CACHE_HIGH_PERF_SIZE",
            CacheConstants.DEFAULT_HIGH_PERF_CACHE_SIZE,
            int,
        )

        # サーキットブレーカー設定
        self.enable_circuit_breaker = self._get_config_value(
            cache_settings,
            "enable_circuit_breaker",
            "CACHE_ENABLE_CIRCUIT_BREAKER",
            True,
            bool,
        )
        self.failure_threshold = self._get_config_value(
            cache_settings,
            "failure_threshold",
            "CACHE_FAILURE_THRESHOLD",
            CacheConstants.DEFAULT_FAILURE_THRESHOLD,
            int,
        )
        self.recovery_timeout = self._get_config_value(
            cache_settings,
            "recovery_timeout",
            "CACHE_RECOVERY_TIMEOUT",
            CacheConstants.DEFAULT_RECOVERY_TIMEOUT,
            float,
        )
        self.half_open_max_calls = self._get_config_value(
            cache_settings,
            "half_open_max_calls",
            "CACHE_HALF_OPEN_MAX_CALLS",
            CacheConstants.DEFAULT_HALF_OPEN_MAX_CALLS,
            int,
        )

    def _get_config_value(
        self,
        config_dict: dict,
        key: str,
        env_key: str,
        default_value,
        type_converter=str,
    ):
        """設定値を優先順位に従って取得"""

        # config_managerから取得
        if key in config_dict:
            try:
                return type_converter(config_dict[key])
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid config value for '{key}': {config_dict[key]}, using fallback. Error: {e}"
                )

        # 環境変数から取得
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if isinstance(type_converter, type) and issubclass(type_converter, bool):
                    # 真偽値の特別な処理
                    return env_value.lower() in ("true", "1", "yes", "on")
                return type_converter(env_value)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid env value for '{env_key}': {env_value}, using default. Error: {e}"
                )

        # デフォルト値を返す
        return default_value

    def to_dict(self) -> Dict[str, Any]:
        """設定をdict形式で返す"""
        return {
            "max_key_length": self.max_key_length,
            "max_value_size_mb": self.max_value_size_mb,
            "max_recursion_depth": self.max_recursion_depth,
            "enable_size_warnings": self.enable_size_warnings,
            "adaptive_recursion": self.adaptive_recursion,
            "min_recursion_depth": self.min_recursion_depth,
            "max_adaptive_depth": self.max_adaptive_depth,
            "enable_performance_logging": self.enable_performance_logging,
            "serialization_timeout": self.serialization_timeout,
            "max_operation_history": self.max_operation_history,
            "hit_rate_window_size": self.hit_rate_window_size,
            "lock_timeout": self.lock_timeout,
            "default_ttl_cache_size": self.default_ttl_cache_size,
            "default_ttl_seconds": self.default_ttl_seconds,
            "high_perf_cache_size": self.high_perf_cache_size,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_max_calls": self.half_open_max_calls,
        }

    def __repr__(self) -> str:
        return f"CacheConfig({self.to_dict()})"


# グローバル設定管理
_global_cache_config: Optional[CacheConfig] = None
_config_lock = threading.RLock()


def get_cache_config(config_manager=None) -> CacheConfig:
    """
    グローバルキャッシュ設定を取得

    Args:
        config_manager: ConfigManagerインスタンス（初回のみ使用）

    Returns:
        CacheConfig: キャッシュ設定インスタンス
    """
    global _global_cache_config

    if _global_cache_config is None:
        with _config_lock:
            if _global_cache_config is None:
                _global_cache_config = CacheConfig(config_manager)

    return _global_cache_config


def set_cache_config(config: CacheConfig) -> None:
    """グローバルキャッシュ設定を設定（テスト用）"""
    global _global_cache_config

    with _config_lock:
        _global_cache_config = config


def reset_cache_config() -> None:
    """グローバルキャッシュ設定をリセット（テスト用）"""
    global _global_cache_config

    with _config_lock:
        _global_cache_config = None
