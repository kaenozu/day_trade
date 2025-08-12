"""
キャッシュ設定管理

統合キャッシュシステムの設定クラスと定数定義を提供します。
元のcache_utils.pyから設定関連機能を分離。
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


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
    DEFAULT_HIGH_PERF_SIZE = 10000
    DEFAULT_EVICTION_BATCH_SIZE = 100
    DEFAULT_MAINTENANCE_INTERVAL = 60  # 1分

    # サーキットブレーカー
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RECOVERY_TIMEOUT = 30.0
    DEFAULT_HALF_OPEN_MAX_CALLS = 3


class CacheConfig:
    """キャッシュシステムの設定クラス"""

    def __init__(
        self,
        max_key_length: int = CacheConstants.DEFAULT_MAX_KEY_LENGTH,
        max_value_size_mb: float = CacheConstants.DEFAULT_MAX_VALUE_SIZE_MB,
        max_recursion_depth: int = CacheConstants.DEFAULT_MAX_RECURSION_DEPTH,
        serialization_timeout: float = CacheConstants.DEFAULT_SERIALIZATION_TIMEOUT,
        lock_timeout: float = CacheConstants.DEFAULT_LOCK_TIMEOUT,
        enable_stats: bool = True,
        enable_circuit_breaker: bool = True,
        debug_mode: bool = False,
        # TTLキャッシュ設定
        default_ttl_seconds: int = CacheConstants.DEFAULT_TTL_SECONDS,
        ttl_cleanup_frequency: int = CacheConstants.DEFAULT_CLEANUP_FREQUENCY,
        # 統計設定
        max_operation_history: int = CacheConstants.DEFAULT_MAX_OPERATION_HISTORY,
        hit_rate_window_size: int = CacheConstants.DEFAULT_HIT_RATE_WINDOW_SIZE,
        # サーキットブレーカー設定
        failure_threshold: int = CacheConstants.DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: float = CacheConstants.DEFAULT_RECOVERY_TIMEOUT,
        half_open_max_calls: int = CacheConstants.DEFAULT_HALF_OPEN_MAX_CALLS,
    ):
        """
        キャッシュ設定の初期化

        Args:
            max_key_length: キャッシュキーの最大長
            max_value_size_mb: キャッシュ値の最大サイズ（MB）
            max_recursion_depth: シリアライゼーションの最大再帰深度
            serialization_timeout: シリアライゼーションのタイムアウト（秒）
            lock_timeout: ロック取得のタイムアウト（秒）
            enable_stats: 統計収集を有効にするか
            enable_circuit_breaker: サーキットブレーカーを有効にするか
            debug_mode: デバッグモードを有効にするか
            default_ttl_seconds: デフォルトTTL（秒）
            ttl_cleanup_frequency: TTLクリーンアップ頻度
            max_operation_history: 操作履歴の最大保持数
            hit_rate_window_size: ヒット率計算のウィンドウサイズ
            failure_threshold: サーキットブレーカーの失敗しきい値
            recovery_timeout: サーキットブレーカーの回復タイムアウト（秒）
            half_open_max_calls: HALF_OPEN状態での最大コール数
        """
        # 基本設定
        self.max_key_length = max_key_length
        self.max_value_size_mb = max_value_size_mb
        self.max_recursion_depth = max_recursion_depth
        self.serialization_timeout = serialization_timeout
        self.lock_timeout = lock_timeout

        # 機能制御
        self.enable_stats = enable_stats
        self.enable_circuit_breaker = enable_circuit_breaker
        self.debug_mode = debug_mode

        # TTL設定
        self.default_ttl_seconds = default_ttl_seconds
        self.ttl_cleanup_frequency = ttl_cleanup_frequency

        # 統計設定
        self.max_operation_history = max_operation_history
        self.hit_rate_window_size = hit_rate_window_size

        # サーキットブレーカー設定
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        # 検証
        self._validate_config()

    def _validate_config(self) -> None:
        """設定値の妥当性チェック"""
        if self.max_key_length <= 0:
            raise ValueError("max_key_length must be positive")

        if self.max_value_size_mb <= 0:
            raise ValueError("max_value_size_mb must be positive")

        if self.max_recursion_depth <= 0:
            raise ValueError("max_recursion_depth must be positive")

        if self.serialization_timeout <= 0:
            raise ValueError("serialization_timeout must be positive")

        if self.lock_timeout <= 0:
            raise ValueError("lock_timeout must be positive")

        if self.default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")

        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")

        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")

    def to_dict(self) -> dict:
        """設定を辞書形式で返す"""
        return {
            "max_key_length": self.max_key_length,
            "max_value_size_mb": self.max_value_size_mb,
            "max_recursion_depth": self.max_recursion_depth,
            "serialization_timeout": self.serialization_timeout,
            "lock_timeout": self.lock_timeout,
            "enable_stats": self.enable_stats,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "debug_mode": self.debug_mode,
            "default_ttl_seconds": self.default_ttl_seconds,
            "ttl_cleanup_frequency": self.ttl_cleanup_frequency,
            "max_operation_history": self.max_operation_history,
            "hit_rate_window_size": self.hit_rate_window_size,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_max_calls": self.half_open_max_calls,
        }

    def copy(self) -> "CacheConfig":
        """設定のコピーを作成"""
        return CacheConfig(**self.to_dict())

    def update(self, **kwargs) -> "CacheConfig":
        """設定を更新した新しいインスタンスを返す"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return CacheConfig(**config_dict)

    def __repr__(self) -> str:
        return f"CacheConfig({', '.join(f'{k}={v}' for k, v in self.to_dict().items())})"


# グローバル設定管理
_global_cache_config: Optional[CacheConfig] = None
_config_lock = threading.RLock()


def get_cache_config(config_manager=None) -> CacheConfig:
    """
    キャッシュ設定を取得

    Args:
        config_manager: 外部設定マネージャー（オプション）

    Returns:
        CacheConfig: キャッシュ設定インスタンス
    """
    global _global_cache_config

    if _global_cache_config is None:
        with _config_lock:
            if _global_cache_config is None:
                if config_manager:
                    try:
                        # 外部設定マネージャーから設定を取得
                        cache_settings = getattr(config_manager, "cache_settings", {})
                        _global_cache_config = CacheConfig(**cache_settings)
                    except Exception as e:
                        logger.warning(f"Failed to load cache config from manager: {e}")
                        _global_cache_config = CacheConfig()
                else:
                    _global_cache_config = CacheConfig()

    return _global_cache_config


def set_cache_config(config: CacheConfig) -> None:
    """
    グローバルキャッシュ設定を設定

    Args:
        config: 新しいキャッシュ設定
    """
    global _global_cache_config

    if not isinstance(config, CacheConfig):
        raise TypeError("config must be a CacheConfig instance")

    with _config_lock:
        _global_cache_config = config

    logger.info("Global cache configuration updated")


def reset_cache_config() -> None:
    """グローバルキャッシュ設定をリセット（テスト用）"""
    global _global_cache_config

    with _config_lock:
        _global_cache_config = None

    logger.debug("Global cache configuration reset")


def get_default_cache_config() -> CacheConfig:
    """デフォルトのキャッシュ設定を取得"""
    return CacheConfig()


def create_cache_config_from_dict(config_dict: dict) -> CacheConfig:
    """
    辞書からキャッシュ設定を作成

    Args:
        config_dict: 設定値を含む辞書

    Returns:
        CacheConfig: 作成された設定インスタンス
    """
    # デフォルト値を設定
    default_config = get_default_cache_config()
    config_values = default_config.to_dict()

    # 辞書の値で上書き
    config_values.update(config_dict)

    return CacheConfig(**config_values)
