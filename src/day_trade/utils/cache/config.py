"""
キャッシュ設定管理モジュール

キャッシュ設定の読み込み、管理、依存性注入対応を提供します。
環境変数やconfig_managerからの設定読み込みをサポートし、
アダプティブな設定調整機能も含みます。
"""

import os
from typing import Any, Optional

from ..logging_config import get_logger
from .constants import CacheConstants

logger = get_logger(__name__)


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
                    cache_settings = (
                        self._config_manager.get("cache", {}) or cache_settings
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load cache settings from config_manager: {e}"
                )

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
                if type_converter is bool:
                    return env_value.lower() in ("true", "1", "yes", "on")
                return type_converter(env_value)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid environment value for '{env_key}': {env_value}, using default. Error: {e}"
                )

        return default_value

    def reload(self):
        """設定を再読み込み"""
        self._load_config()
        logger.info("Cache configuration reloaded")

    @property
    def max_value_size_bytes(self) -> int:
        """最大値サイズをバイト単位で取得"""
        return self.max_value_size_mb * 1024 * 1024

    def get_adaptive_depth_limit(self, data_complexity: int = 0) -> int:
        """
        データの複雑さに基づいてアダプティブな再帰制限を取得

        Args:
            data_complexity: データの複雑度（オブジェクト数など）

        Returns:
            適応された再帰制限値
        """
        if not self.adaptive_recursion:
            return self.max_recursion_depth

        # データの複雑さに基づいて制限を調整
        if data_complexity <= 10:
            # シンプルなデータ: 深く探索
            return min(self.max_adaptive_depth, self.max_recursion_depth * 2)
        elif data_complexity <= 100:
            # 中程度の複雑さ: 標準制限
            return self.max_recursion_depth
        else:
            # 複雑なデータ: 制限を厳しく
            return max(self.min_recursion_depth, self.max_recursion_depth // 2)

    def to_dict(self) -> dict:
        """設定を辞書として返す"""
        return {
            "max_key_length": self.max_key_length,
            "max_value_size_mb": self.max_value_size_mb,
            "max_value_size_bytes": self.max_value_size_bytes,
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
        }


# グローバル設定インスタンス（遅延初期化対応）
_cache_config = None


def get_cache_config(config_manager=None) -> CacheConfig:
    """
    キャッシュ設定を取得（シングルトン・依存性注入対応）

    Args:
        config_manager: ConfigManagerインスタンス（オプション）

    Returns:
        CacheConfigインスタンス
    """
    global _cache_config
    if _cache_config is None:
        _cache_config = CacheConfig(config_manager)
    return _cache_config


def set_cache_config(config: CacheConfig) -> None:
    """
    キャッシュ設定を設定（テスト用・依存性注入用）

    Args:
        config: 新しいCacheConfigインスタンス
    """
    global _cache_config
    _cache_config = config


def reset_cache_config() -> None:
    """キャッシュ設定をリセット（テスト用）"""
    global _cache_config
    _cache_config = None


# 後方互換性のためのプロパティ
cache_config = get_cache_config()