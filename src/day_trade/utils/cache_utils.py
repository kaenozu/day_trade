"""
キャッシュユーティリティ
安全で効率的なキャッシュキー生成とキャッシュ操作を提供
堅牢なシリアライズ機能とスレッドセーフティを含む
"""

import hashlib
import json
import logging
import threading
import time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

logger = logging.getLogger(__name__)

# キャッシュ設定の定数
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
                cache_settings = getattr(self._config_manager, 'cache_settings', {})
                if hasattr(self._config_manager, 'get'):
                    # より一般的なget方式も試行
                    cache_settings = self._config_manager.get('cache', {}) or cache_settings
            except Exception as e:
                logger.warning(f"Failed to load cache settings from config_manager: {e}")

        # 設定値の決定（優先度: config_manager > 環境変数 > デフォルト）
        self.max_key_length = self._get_config_value(
            cache_settings, "max_key_length", "CACHE_MAX_KEY_LENGTH", CacheConstants.DEFAULT_MAX_KEY_LENGTH, int
        )
        self.max_value_size_mb = self._get_config_value(
            cache_settings, "max_value_size_mb", "CACHE_MAX_VALUE_SIZE_MB", CacheConstants.DEFAULT_MAX_VALUE_SIZE_MB, int
        )
        self.max_recursion_depth = self._get_config_value(
            cache_settings, "max_recursion_depth", "CACHE_MAX_RECURSION_DEPTH", CacheConstants.DEFAULT_MAX_RECURSION_DEPTH, int
        )
        self.enable_size_warnings = self._get_config_value(
            cache_settings, "enable_size_warnings", "CACHE_ENABLE_SIZE_WARNINGS", True, bool
        )

        # アダプティブ再帰制限の設定
        self.adaptive_recursion = self._get_config_value(
            cache_settings, "adaptive_recursion", "CACHE_ADAPTIVE_RECURSION", True, bool
        )
        self.min_recursion_depth = self._get_config_value(
            cache_settings, "min_recursion_depth", "CACHE_MIN_RECURSION_DEPTH", CacheConstants.DEFAULT_MIN_RECURSION_DEPTH, int
        )
        self.max_adaptive_depth = self._get_config_value(
            cache_settings, "max_adaptive_depth", "CACHE_MAX_ADAPTIVE_DEPTH", CacheConstants.DEFAULT_MAX_ADAPTIVE_DEPTH, int
        )

        # パフォーマンス設定
        self.enable_performance_logging = self._get_config_value(
            cache_settings, "enable_performance_logging", "CACHE_ENABLE_PERFORMANCE_LOGGING", False, bool
        )
        self.serialization_timeout = self._get_config_value(
            cache_settings, "serialization_timeout", "CACHE_SERIALIZATION_TIMEOUT", CacheConstants.DEFAULT_SERIALIZATION_TIMEOUT, float
        )

        # 統計設定
        self.max_operation_history = self._get_config_value(
            cache_settings, "max_operation_history", "CACHE_MAX_OPERATION_HISTORY", CacheConstants.DEFAULT_MAX_OPERATION_HISTORY, int
        )
        self.hit_rate_window_size = self._get_config_value(
            cache_settings, "hit_rate_window_size", "CACHE_HIT_RATE_WINDOW_SIZE", CacheConstants.DEFAULT_HIT_RATE_WINDOW_SIZE, int
        )
        self.lock_timeout = self._get_config_value(
            cache_settings, "lock_timeout", "CACHE_LOCK_TIMEOUT", CacheConstants.DEFAULT_LOCK_TIMEOUT, float
        )

        # TTLキャッシュ設定
        self.default_ttl_cache_size = self._get_config_value(
            cache_settings, "default_ttl_cache_size", "CACHE_DEFAULT_TTL_SIZE", CacheConstants.DEFAULT_TTL_CACHE_SIZE, int
        )
        self.default_ttl_seconds = self._get_config_value(
            cache_settings, "default_ttl_seconds", "CACHE_DEFAULT_TTL_SECONDS", CacheConstants.DEFAULT_TTL_SECONDS, int
        )

        # 高性能キャッシュ設定
        self.high_perf_cache_size = self._get_config_value(
            cache_settings, "high_perf_cache_size", "CACHE_HIGH_PERF_SIZE", CacheConstants.DEFAULT_HIGH_PERF_CACHE_SIZE, int
        )

    def _get_config_value(self, config_dict: dict, key: str, env_key: str, default_value, type_converter=str):
        """設定値を優先順位に従って取得"""
        import os

        # config_managerから取得
        if key in config_dict:
            try:
                return type_converter(config_dict[key])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid config value for '{key}': {config_dict[key]}, using fallback. Error: {e}")

        # 環境変数から取得
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if type_converter == bool:
                    return env_value.lower() in ('true', '1', 'yes', 'on')
                return type_converter(env_value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid environment value for '{env_key}': {env_value}, using default. Error: {e}")

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

# 後方互換性のためのプロパティ
cache_config = get_cache_config()

# グローバルサーキットブレーカーインスタンス（遅延初期化）
_cache_circuit_breaker = None

def get_cache_circuit_breaker() -> 'CacheCircuitBreaker':
    """
    キャッシュサーキットブレーカーを取得（遅延初期化・シングルトン）

    Returns:
        CacheCircuitBreakerインスタンス
    """
    global _cache_circuit_breaker
    if _cache_circuit_breaker is None:
        _cache_circuit_breaker = CacheCircuitBreaker()
    return _cache_circuit_breaker


class CacheError(Exception):
    """キャッシュ操作エラー"""
    def __init__(self, message: str, error_code: str = "CACHE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class CacheCircuitBreakerError(CacheError):
    """キャッシュサーキットブレーカーエラー"""
    def __init__(self, message: str, circuit_state: str, failure_count: int):
        super().__init__(message, "CACHE_CIRCUIT_BREAKER", {
            "circuit_state": circuit_state,
            "failure_count": failure_count
        })
        self.circuit_state = circuit_state
        self.failure_count = failure_count


class CacheTimeoutError(CacheError):
    """キャッシュタイムアウトエラー"""
    def __init__(self, message: str, timeout_seconds: float, operation: str):
        super().__init__(message, "CACHE_TIMEOUT", {
            "timeout_seconds": timeout_seconds,
            "operation": operation
        })
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class CacheCircuitBreaker:
    """キャッシュ操作用のサーキットブレーカー"""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """サーキットブレーカー付きでfunction呼び出し"""
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise CacheCircuitBreakerError(
                        "Circuit breaker is OPEN",
                        self._state,
                        self._failure_count
                    )
                else:
                    self._state = "HALF_OPEN"
                    self._half_open_calls = 0

            elif self._state == "HALF_OPEN":
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CacheCircuitBreakerError(
                        "Circuit breaker HALF_OPEN limit exceeded",
                        self._state,
                        self._failure_count
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """成功時の処理"""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = "CLOSED"
                    self._failure_count = 0
            elif self._state == "CLOSED":
                self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self):
        """失敗時の処理"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold or self._state == "HALF_OPEN":
                self._state = "OPEN"

    @property
    def state(self) -> str:
        """現在のサーキットブレーカー状態を取得"""
        return self._state

    @property
    def failure_count(self) -> int:
        """現在の失敗カウントを取得"""
        return self._failure_count


def _estimate_data_complexity(args: Union[Tuple, Dict, Any]) -> int:
    """
    データの複雑度を推定（再帰制限の動的調整用）

    Args:
        args: 複雑度を推定するデータ

    Returns:
        推定複雑度（オブジェクト数）
    """
    try:
        complexity = 0

        if isinstance(args, (dict, list, tuple)):
            complexity += len(args)

            # ネストした構造の場合は追加でカウント
            for item in (args.values() if isinstance(args, dict) else args):
                if isinstance(item, (dict, list, tuple)):
                    complexity += len(item)

        elif hasattr(args, '__dict__'):
            complexity += len(args.__dict__)

        return complexity

    except Exception:
        # エラーの場合は中程度の複雑度を返す
        return 50


def generate_safe_cache_key(func_name: str, *args, **kwargs) -> str:
    """
    安全なキャッシュキー生成（サーキットブレーカー・高度エラーハンドリング付き）

    Args:
        func_name: 関数名
        *args: 位置引数
        **kwargs: キーワード引数

    Returns:
        生成されたキャッシュキー

    Raises:
        CacheError: キャッシュキー生成に失敗した場合
        CacheCircuitBreakerError: サーキットブレーカーが作動した場合
        CacheTimeoutError: タイムアウトが発生した場合
    """
    def _generate_key_internal():
        """内部キー生成処理"""
        start_time = time.time()

        try:
            # データ複雑度を推定してアダプティブ再帰制限を設定
            args_complexity = _estimate_data_complexity(args)
            kwargs_complexity = _estimate_data_complexity(kwargs)
            total_complexity = args_complexity + kwargs_complexity

            adaptive_depth = cache_config.get_adaptive_depth_limit(total_complexity)

            # 引数を正规化してシリアライズ可能な形式に変換
            normalized_args = _normalize_arguments(args, max_depth=adaptive_depth)
            normalized_kwargs = _normalize_arguments(kwargs, max_depth=adaptive_depth)

            # シリアライズしてハッシュ化
            serializable_data = {
                "function": func_name,
                "args": normalized_args,
                "kwargs": (
                    sorted(normalized_kwargs.items())
                    if isinstance(normalized_kwargs, dict)
                    else normalized_kwargs
                ),
                "complexity": total_complexity,  # 複雑度も含めてキーの一意性を向上
                "depth": adaptive_depth
            }

            serialized = json.dumps(
                serializable_data, sort_keys=True, default=_json_serializer
            )
            cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            # 処理時間の監視
            elapsed_time = time.time() - start_time
            if elapsed_time > cache_config.serialization_timeout * 0.8:  # 80%を超えたら警告
                logger.warning(f"Cache key generation took {elapsed_time:.3f}s for {func_name}")

            # パフォーマンス最適化: デバッグログの条件付き出力
            if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generated cache key for {func_name}: {cache_key} (complexity: {total_complexity}, depth: {adaptive_depth})")

            return f"{func_name}:{cache_key}"

        except (TypeError, ValueError, OverflowError) as e:
            # 予期される例外: シリアライゼーションエラー
            logger.warning(f"JSON serialization failed for cache key, using fallback: {e}")
            try:
                return _generate_fallback_cache_key(func_name, args, kwargs)
            except Exception as fallback_error:
                raise CacheError(
                    f"キャッシュキー生成に失敗しました: {e}",
                    "CACHE_KEY_SERIALIZATION_FAILED",
                    {
                        "original_error": str(e),
                        "original_error_type": type(e).__name__,
                        "fallback_error": str(fallback_error),
                        "fallback_error_type": type(fallback_error).__name__,
                        "func_name": func_name,
                        "args_count": len(args) if args else 0,
                        "kwargs_keys": list(kwargs.keys()) if kwargs else [],
                        "args_complexity": args_complexity,
                        "kwargs_complexity": kwargs_complexity,
                        "processing_time": time.time() - start_time
                    }
                ) from e
        except Exception as e:
            # 予期しない例外
            if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
                logger.error(f"Unexpected error in cache key generation: {e}", exc_info=True)
            raise CacheError(
                f"キャッシュキー生成で予期しないエラーが発生しました: {e}",
                "CACHE_KEY_UNEXPECTED_ERROR",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "func_name": func_name,
                    "processing_time": time.time() - start_time,
                    "circuit_breaker_state": get_cache_circuit_breaker().state
                }
            ) from e

    # サーキットブレーカーを使用してキー生成を実行
    try:
        circuit_breaker = get_cache_circuit_breaker()
        return circuit_breaker.call(_generate_key_internal)
    except CacheCircuitBreakerError:
        # サーキットブレーカーが開いている場合は緊急フォールバック
        logger.error(f"Circuit breaker is open, using emergency fallback for {func_name}")
        return _generate_emergency_cache_key(func_name, args, kwargs)
    except Exception as e:
        # その他のエラーもサーキットブレーカーに記録される
        raise e


def _generate_emergency_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    緊急時のキャッシュキー生成（最も基本的な方法）

    Args:
        func_name: 関数名
        args: 位置引数
        kwargs: キーワード引数

    Returns:
        緊急時用のキャッシュキー
    """
    try:
        # 最小限の安全な方法でキーを生成
        import uuid
        timestamp = int(time.time() * 1000000)  # マイクロ秒
        unique_suffix = str(uuid.uuid4()).replace('-', '')[:8]
        args_hash = str(hash(str(args)))[-8:] if args else "0"
        kwargs_hash = str(hash(str(sorted(kwargs.items()))))[-8:] if kwargs else "0"

        return f"{func_name}:emergency:{timestamp}:{unique_suffix}:{args_hash}:{kwargs_hash}"
    except Exception as e:
        # 最終フォールバック
        logger.error(f"Emergency cache key generation failed: {e}")
        return f"{func_name}:final_fallback:{int(time.time())}"


def _normalize_arguments(args: Union[Tuple, Dict, Any], max_depth: int = None, current_depth: int = 0, seen_objects: Optional[set] = None) -> Any:
    """
    引数を正規化してシリアライズ可能にする（再帰深度制限付き・循環参照検出・イテラティブ最適化）

    Args:
        args: 正規化する引数
        max_depth: 最大再帰深度（Noneの場合は設定から取得）
        current_depth: 現在の再帰深度
        seen_objects: 循環参照検出用のオブジェクトセット

    Returns:
        正規化された引数

    Raises:
        CacheError: 再帰深度制限を超えた場合や循環参照が深すぎる場合
    """
    # max_depthが指定されていない場合は設定から取得
    if max_depth is None:
        max_depth = cache_config.max_recursion_depth

    # 循環参照検出用のセットを初期化
    if seen_objects is None:
        seen_objects = set()

    # 深度制限チェック（エラーハンドリング強化）
    if current_depth >= max_depth:
        # パフォーマンス最適化: 警告ログの条件付き出力
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
            logger.warning(f"Recursion depth limit reached ({max_depth}), truncating object of type {type(args).__name__}")

        # 深い構造の場合はより詳細な情報を返す
        if isinstance(args, (dict, list, tuple)):
            return f"<truncated {type(args).__name__} with {len(args)} items at depth {max_depth}>"
        else:
            return f"<truncated {type(args).__name__} at depth {max_depth}>"

    # 循環参照の検出（より堅牢なエラーハンドリング）
    try:
        obj_id = id(args)
        if obj_id in seen_objects:
            # 循環参照のより詳細な情報を提供
            return f"<circular reference to {type(args).__name__} at depth {current_depth}>"

        # 複雑なオブジェクトの場合のみseenに追加（メモリ効率も考慮）
        if isinstance(args, (dict, list, tuple)) and args:
            # 循環参照セットのサイズ制限（メモリリーク防止）
            if len(seen_objects) < CacheConstants.DEFAULT_MAX_RECURSION_DEPTH * 2:
                seen_objects.add(obj_id)
            else:
                # セットが大きくなりすぎた場合は警告
                if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Circular reference detection set is too large ({len(seen_objects)}), skipping detection for {type(args).__name__}")

    except Exception as e:
        # id()の取得に失敗した場合はデバッグログに記録
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to get object id for circular reference detection: {e}")
        # 循環参照チェックをスキップして処理を続行

    # Pydanticモデル
    if PYDANTIC_AVAILABLE and isinstance(args, BaseModel):
        try:
            model_data = args.model_dump() if hasattr(args, 'model_dump') else args.dict()
            return _normalize_arguments(model_data, max_depth, current_depth + 1, seen_objects)
        except Exception:
            return f"<pydantic model: {args.__class__.__name__}>"

    # Decimal型
    elif isinstance(args, Decimal):
        return str(args)

    # Enum型
    elif isinstance(args, Enum):
        return f"<enum {args.__class__.__name__}: {args.value}>"

    # コレクション型（エラーハンドリング強化）
    elif isinstance(args, (tuple, list)):
        try:
            normalized_items = []
            for i, arg in enumerate(args):
                try:
                    normalized_items.append(_normalize_arguments(arg, max_depth, current_depth + 1, seen_objects))
                except CacheError:
                    # CacheErrorは再発生
                    raise
                except Exception as e:
                    # その他のエラーは個別要素のエラーとして処理
                    normalized_items.append(f"<item_{i}_error: {e})")

                # 巨大なコレクションの処理を制限
                if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:  # 制限値を再利用
                    normalized_items.append(f"<truncated: {len(args) - i - 1} more items>")
                    break

            return normalized_items
        except CacheError:
            raise
        except Exception as e:
            return f"<list/tuple of {len(args)} items, error: {e}>"

    elif isinstance(args, dict):
        try:
            normalized_dict = {}
            processed_count = 0
            for k, v in args.items():
                try:
                    key_str = str(k)
                    normalized_dict[key_str] = _normalize_arguments(v, max_depth, current_depth + 1, seen_objects)
                    processed_count += 1

                    # 巨大な辞書の処理を制限
                    if processed_count >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
                        remaining = len(args) - processed_count
                        if remaining > 0:
                            normalized_dict["<truncated>"] = f"{remaining} more keys"
                        break

                except CacheError:
                    raise
                except Exception as e:
                    normalized_dict[f"<key_error_{k}>"] = f"<error: {e}>"

            return normalized_dict
        except CacheError:
            raise
        except Exception as e:
            return f"<dict with {len(args)} keys, error: {e}>"

    elif isinstance(args, set):
        try:
            normalized_items = []
            for i, item in enumerate(args):
                try:
                    normalized_items.append(_normalize_arguments(item, max_depth, current_depth + 1, seen_objects))
                except CacheError:
                    raise
                except Exception as e:
                    normalized_items.append(f"<set_item_error: {e}>")

                # セットサイズの制限
                if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
                    normalized_items.append(f"<truncated: {len(args) - i - 1} more items>")
                    break

            return {"__set__": sorted(normalized_items, key=str)}
        except CacheError:
            raise
        except Exception as e:
            return f"<set of {len(args)} items, error: {e}>"

    # datetime類
    elif hasattr(args, "isoformat"):
        return args.isoformat()

    # オブジェクト（エラーハンドリング強化）
    elif hasattr(args, "__dict__"):
        try:
            # __dict__の存在と内容をチェック
            obj_dict = getattr(args, "__dict__", {})
            if obj_dict:
                return _normalize_arguments(obj_dict, max_depth, current_depth + 1, seen_objects)
            else:
                return f"<empty_object: {args.__class__.__name__}>"
        except CacheError:
            raise
        except Exception as e:
            return f"<object_error: {args.__class__.__name__}, {e}>"

    # 関数（より安全な処理）
    elif callable(args):
        try:
            name = getattr(args, "__name__", None)
            if name:
                return f"<callable: {name}>"
            else:
                return f"<callable: {type(args).__name__}>"
        except Exception as e:
            return f"<callable_error: {e}>"

    # プリミティブ型（循環参照クリーンアップ付き）
    else:
        # 処理完了後、循環参照検出セットからIDを削除（メモリ効率）
        try:
            if isinstance(args, (dict, list, tuple)) and args:
                obj_id = id(args)
                seen_objects.discard(obj_id)
        except Exception as e:
            # クリーンアップに失敗してもエラーにはしない
            if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Failed to cleanup circular reference detection: {e}")

        # プリミティブ型の値を返す
        try:
            # 特殊な値のチェック
            if args is None:
                return None
            # 文字列の場合は長さをチェック
            elif isinstance(args, str) and len(args) > CacheConstants.DEFAULT_MAX_KEY_LENGTH:
                return f"<long_string: {len(args)} chars>"
            else:
                return args
        except Exception as e:
            return f"<primitive_error: {type(args).__name__}, {e}>"


def _json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザーのカスタムハンドラー（堅牢版・Pydantic v2対応・シンプルタイムアウト機能付き）

    Pydantic、Decimal、Enum、datetime等の主要な型に対応

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズ可能な値

    Raises:
        CacheError: シリアライゼーションが失敗またはタイムアウトした場合
    """
    start_time = time.time()
    timeout_seconds = cache_config.serialization_timeout

    def check_timeout():
        """タイムアウトチェック"""
        if time.time() - start_time > timeout_seconds:
            raise CacheTimeoutError(
                f"Serialization timeout after {timeout_seconds} seconds",
                timeout_seconds,
                "json_serialization"
            )

    try:
        # 型安全性を向上させた処理

        # Pydanticモデル（v2対応強化）
        if PYDANTIC_AVAILABLE and isinstance(obj, BaseModel):
            check_timeout()
            try:
                # Pydantic v2の場合
                if hasattr(obj, 'model_dump'):
                    return obj.model_dump(mode='json', exclude_unset=False)
                # Pydantic v1の場合
                elif hasattr(obj, 'dict'):
                    return obj.dict()
                else:
                    # フォールバック
                    return {"__pydantic_model__": obj.__class__.__name__, "data": str(obj)}
            except Exception as e:
                logger.warning(f"Pydantic model serialization failed: {e}")
                return {"__pydantic_model__": obj.__class__.__name__, "error": str(e)}

        # Decimal型（精度保持）
        elif isinstance(obj, Decimal):
            check_timeout()
            # 特殊値のチェック
            if obj.is_nan():
                return {"__decimal__": "NaN"}
            elif obj.is_infinite():
                return {"__decimal__": "Infinity" if obj > 0 else "-Infinity"}
            else:
                return {"__decimal__": str(obj)}

        # Enum型（より詳細な情報を保持）
        elif isinstance(obj, Enum):
            check_timeout()
            return {
                "__enum__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, '__module__', 'unknown'),
                "name": obj.name,
                "value": obj.value
            }

        # datetime/date/time オブジェクト（タイムゾーン情報保持）
        elif hasattr(obj, "isoformat"):
            check_timeout()
            try:
                iso_string = obj.isoformat()
                return {
                    "__datetime__": iso_string,
                    "__type__": obj.__class__.__name__,
                    "__timezone__": str(getattr(obj, 'tzinfo', None)) if hasattr(obj, 'tzinfo') else None
                }
            except Exception:
                return {"__datetime__": str(obj), "__type__": obj.__class__.__name__}

        # set型（ソート対応強化）
        elif isinstance(obj, set):
            check_timeout()
            try:
                # ソート可能な要素のみソート
                sorted_items = []
                unsorted_items = []

                for item in obj:
                    check_timeout()  # セット要素の処理でもタイムアウトチェック
                    try:
                        # シリアライズ可能かテスト（再帰呼び出しを避ける）
                        json.dumps(item, default=str)
                        sorted_items.append(item)
                    except Exception:
                        unsorted_items.append(str(item))

                try:
                    sorted_items.sort()
                except TypeError:
                    # ソートできない場合は文字列変換してソート
                    sorted_items = sorted([str(item) for item in sorted_items])

                return {"__set__": sorted_items + sorted(unsorted_items)}
            except Exception:
                return {"__set__": [str(item) for item in obj]}

        # frozenset型
        elif isinstance(obj, frozenset):
            check_timeout()
            try:
                return {"__frozenset__": _json_serializer(set(obj))["__set__"]}
            except Exception:
                return {"__frozenset__": [str(item) for item in obj]}

        # bytes型（エンコーディング検出強化）
        elif isinstance(obj, bytes):
            check_timeout()
            try:
                # UTF-8を最初に試行
                return {"__bytes__": obj.decode('utf-8'), "encoding": "utf-8"}
            except UnicodeDecodeError:
                try:
                    # 他のエンコーディングを試行
                    try:
                        import chardet
                        detected = chardet.detect(obj)
                        if detected and detected['encoding'] and detected.get('confidence', 0) > CacheConstants.CHARSET_DETECTION_CONFIDENCE_THRESHOLD:
                            return {
                                "__bytes__": obj.decode(detected['encoding']),
                                "encoding": detected['encoding'],
                                "confidence": detected['confidence']
                            }
                    except (ImportError, UnicodeDecodeError):
                        pass
                    # フォールバック: hexエンコーディング
                    return {"__bytes__": obj.hex(), "encoding": "hex"}
                except Exception:
                    # 最終フォールバック
                    return {"__bytes__": "<binary_data>", "encoding": "unknown", "size": len(obj)}

        # 関数またはメソッド（より詳細な情報）
        elif callable(obj):
            check_timeout()
            return {
                "__callable__": getattr(obj, "__name__", repr(obj)),
                "__module__": getattr(obj, "__module__", "unknown"),
                "__type__": "function" if hasattr(obj, "__name__") else "callable"
            }

        # オブジェクトに辞書がある場合（再帰深度チェック）
        elif hasattr(obj, "__dict__"):
            check_timeout()
            try:
                # __dict__の内容を安全にシリアライズ
                safe_dict = {}
                for key, value in obj.__dict__.items():
                    check_timeout()  # 各要素でもタイムアウトチェック
                    try:
                        safe_dict[str(key)] = _json_serializer(value)
                    except Exception as e:
                        safe_dict[str(key)] = f"<serialization_error: {e}>"

                return {
                    "__object__": obj.__class__.__name__,
                    "__module__": getattr(obj.__class__, '__module__', 'unknown'),
                    "data": safe_dict
                }
            except Exception as e:
                return {
                    "__object__": obj.__class__.__name__,
                    "__module__": getattr(obj.__class__, '__module__', 'unknown'),
                    "error": str(e)
                }

        # NumPy配列（オプション対応）
        elif hasattr(obj, 'tolist') and hasattr(obj, 'dtype'):
            check_timeout()
            try:
                return {
                    "__numpy_array__": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape
                }
            except Exception:
                return {"__numpy_array__": str(obj)}

        # その他の場合は文字列表現（型情報付き）
        else:
            check_timeout()
            return {
                "__unknown_type__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, '__module__', 'unknown'),
                "value": str(obj)
            }

    except CacheTimeoutError:
        # 既にCacheTimeoutErrorの場合はそのまま再発生
        raise
    except Exception as e:
        # 予期しないエラーのフォールバック
        logger.warning(f"Serialization fallback for {type(obj).__name__}: {e}")
        return {
            "__serialization_fallback__": True,
            "__type__": obj.__class__.__name__,
            "__error__": str(e),
            "value": str(obj)
        }


def _generate_fallback_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    フォールバック用のキャッシュキー生成（hashlibのみ使用）

    Args:
        func_name: 関数名
        args: 位置引数
        kwargs: キーワード引数

    Returns:
        生成されたフォールバックキー
    """
    try:
        # プロセス間で一貫性のあるハッシュ生成（hashlibのみ使用）
        args_str = str(args) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""

        # SHA256を使用してより堅牢なハッシュを生成
        combined_data = f"{func_name}:{args_str}:{kwargs_str}".encode()
        combined_hash = hashlib.sha256(combined_data).hexdigest()

        return f"{func_name}:fallback:{combined_hash}"

    except Exception as e:
        # 最終フォールバック
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
            logger.error(f"Fallback cache key generation failed: {e}")
        import time
        import uuid

        # より安全なユニークキー生成
        timestamp = int(time.time() * 1000000)  # マイクロ秒タイムスタンプ
        unique_id = str(uuid.uuid4()).replace('-', '')[:8]
        return f"{func_name}:emergency:{timestamp}:{unique_id}"


class CacheStats:
    """キャッシュ統計情報（スレッドセーフ版・デッドロック対策強化・パフォーマンス最適化）"""

    def __init__(self, config: Optional['CacheConfig'] = None):
        self._config = config or get_cache_config()
        self._lock = threading.RLock()  # 再帰可能ロック
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._errors = 0
        self._lock_timeout = self._config.lock_timeout

        # パフォーマンス最適化: アトミック操作用の追加統計
        self._start_time = time.time()
        self._last_reset_time = self._start_time

        # 高精度統計用（時間加重平均など）
        self._operation_times = []
        self._max_operation_history = self._config.max_operation_history

        # キャッシュ効率統計
        self._hit_rate_window = []
        self._window_size = self._config.hit_rate_window_size

    def _safe_lock_operation(self, operation_func, default_value=0):
        """安全なロック操作（タイムアウト付き・詳細エラーハンドリング）"""
        try:
            if self._lock.acquire(timeout=self._lock_timeout):
                try:
                    return operation_func()
                except Exception as e:
                    # 操作実行中のエラー
                    if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
                        logger.error(f"CacheStats operation failed: {e}", exc_info=True)
                    return default_value
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
                            logger.error(f"Failed to release CacheStats lock: {release_error}")
            else:
                # ロック取得タイムアウト
                if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"CacheStats lock timeout ({self._lock_timeout}s) - returning default value")
                return default_value
        except Exception as e:
            # ロック取得自体のエラー
            if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
                logger.error(f"CacheStats lock acquisition failed: {e}", exc_info=True)
            return default_value

    @property
    def hits(self) -> int:
        """ヒット数を取得"""
        return self._safe_lock_operation(lambda: self._hits)

    @property
    def misses(self) -> int:
        """ミス数を取得"""
        return self._safe_lock_operation(lambda: self._misses)

    @property
    def sets(self) -> int:
        """セット数を取得"""
        return self._safe_lock_operation(lambda: self._sets)

    @property
    def evictions(self) -> int:
        """エビクション数を取得"""
        return self._safe_lock_operation(lambda: self._evictions)

    @property
    def errors(self) -> int:
        """エラー数を取得"""
        return self._safe_lock_operation(lambda: self._errors)

    @property
    def hit_rate(self) -> float:
        """ヒット率を計算"""
        def calculate_hit_rate():
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

        result = self._safe_lock_operation(calculate_hit_rate)
        return result if isinstance(result, float) else 0.0

    def record_hit(self, count: int = 1) -> None:
        """キャッシュヒットを記録（移動平均更新付き）"""
        def record_with_moving_avg():
            self._increment_counter('_hits', count)
            # 移動平均ヒット率の更新
            total_requests = self._hits + self._misses
            if total_requests > 0:
                current_hit_rate = self._hits / total_requests
                self._update_hit_rate_window(current_hit_rate)
            # 操作時刻を記録
            self._record_operation_time()

        self._safe_lock_operation(record_with_moving_avg)

    def record_miss(self, count: int = 1) -> None:
        """キャッシュミスを記録（移動平均更新付き）"""
        def record_with_moving_avg():
            self._increment_counter('_misses', count)
            # 移動平均ヒット率の更新
            total_requests = self._hits + self._misses
            if total_requests > 0:
                current_hit_rate = self._hits / total_requests
                self._update_hit_rate_window(current_hit_rate)
            # 操作時刻を記録
            self._record_operation_time()

        self._safe_lock_operation(record_with_moving_avg)

    def record_set(self, count: int = 1) -> None:
        """キャッシュセットを記録"""
        self._safe_lock_operation(lambda: self._increment_counter('_sets', count))

    def record_eviction(self, count: int = 1) -> None:
        """キャッシュエビクションを記録"""
        self._safe_lock_operation(lambda: self._increment_counter('_evictions', count))

    def record_error(self, count: int = 1) -> None:
        """キャッシュエラーを記録"""
        self._safe_lock_operation(lambda: self._increment_counter('_errors', count))

    def _increment_counter(self, counter_name: str, count: int) -> None:
        """カउンターをインクリメント（ロック内で呼び出し・エラーハンドリング強化）"""
        try:
            if not hasattr(self, counter_name):
                raise AttributeError(f"Counter '{counter_name}' does not exist")

            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Count must be a non-negative integer, got: {count}")

            current_value = getattr(self, counter_name)
            new_value = current_value + count

            # オーバーフロー検査（Python の int は無限精度だが、異常に大きな値をチェック）
            if new_value > CacheConstants.MAX_COUNTER_VALUE:
                if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Counter '{counter_name}' approaching overflow, resetting to 0")
                setattr(self, counter_name, 0)
            else:
                setattr(self, counter_name, new_value)

        except (AttributeError, ValueError, TypeError) as e:
            if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.ERROR):
                logger.error(f"Failed to increment counter '{counter_name}': {e}")
            # エラーが発生した場合はカウントを無視

    def record_fallback(self, count: int = 1) -> None:
        """フォールバック使用を記録（エラーとして扱う）"""
        self.record_error(count)

    def reset(self) -> None:
        """統計をリセット（履歴情報も含む）"""
        def reset_counters():
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._errors = 0

            # 履歴情報もリセット
            self._operation_times.clear()
            self._hit_rate_window.clear()
            self._last_reset_time = time.time()

        self._safe_lock_operation(reset_counters)

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """統計情報を辞書として返す（スレッドセーフ・詳細統計付き）"""
        def create_stats_dict():
            current_time = time.time()
            uptime = current_time - self._start_time
            time_since_reset = current_time - self._last_reset_time

            # 基本統計
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            # 効率指標
            operations_per_second = total_requests / uptime if uptime > 0 else 0.0
            recent_ops_per_second = total_requests / time_since_reset if time_since_reset > 0 else 0.0

            # 移動平均ヒット率
            moving_avg_hit_rate = (
                sum(self._hit_rate_window) / len(self._hit_rate_window)
                if self._hit_rate_window else hit_rate
            )

            return {
                # 基本統計
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "errors": self._errors,
                "total_requests": total_requests,

                # 効率指標
                "hit_rate": hit_rate,
                "moving_avg_hit_rate": moving_avg_hit_rate,
                "miss_rate": 1.0 - hit_rate if total_requests > 0 else 0.0,

                # パフォーマンス指標
                "operations_per_second": operations_per_second,
                "recent_operations_per_second": recent_ops_per_second,
                "uptime_seconds": uptime,
                "time_since_reset_seconds": time_since_reset,

                # 高度な指標
                "error_rate": self._errors / total_requests if total_requests > 0 else 0.0,
                "eviction_rate": self._evictions / self._sets if self._sets > 0 else 0.0,
                "efficiency_score": self._calculate_efficiency_score(hit_rate, self._errors / total_requests if total_requests > 0 else 0.0),

                # メタデータ
                "last_reset_time": self._last_reset_time,
                "start_time": self._start_time,
                "operation_history_size": len(self._operation_times),
                "hit_rate_window_size": len(self._hit_rate_window)
            }

        result = self._safe_lock_operation(create_stats_dict)
        return result if isinstance(result, dict) else {}

    def _calculate_efficiency_score(self, hit_rate: float, error_rate: float) -> float:
        """キャッシュ効率スコアを計算（0-100の範囲）"""
        # ヒット率を基本スコアとし、エラー率でペナルティを課す
        base_score = hit_rate * 100
        error_penalty = error_rate * CacheConstants.ERROR_PENALTY_MULTIPLIER
        return max(0.0, base_score - error_penalty)

    def _update_hit_rate_window(self, hit_rate: float) -> None:
        """移動平均ヒット率ウィンドウを更新（ロック内で呼び出し前提）"""
        self._hit_rate_window.append(hit_rate)
        # ウィンドウサイズを超えた場合は古いデータを削除
        if len(self._hit_rate_window) > self._window_size:
            self._hit_rate_window.pop(0)

    def _record_operation_time(self) -> None:
        """操作時刻を記録（ロック内で呼び出し前提）"""
        current_time = time.time()
        self._operation_times.append(current_time)
        # 履歴サイズを制限
        if len(self._operation_times) > self._max_operation_history:
            self._operation_times.pop(0)

    def get_recent_operations_per_second(self, window_seconds: float = 60.0) -> float:
        """指定時間内の操作数/秒を取得（スレッドセーフ）"""
        def calculate_recent_ops():
            if not self._operation_times:
                return 0.0

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # 指定時間内の操作数をカウント
            recent_ops = sum(1 for op_time in self._operation_times if op_time >= cutoff_time)

            return recent_ops / window_seconds if window_seconds > 0 else 0.0

        result = self._safe_lock_operation(calculate_recent_ops)
        return result if isinstance(result, (int, float)) else 0.0

    def get_peak_operations_per_second(self, window_seconds: float = 10.0) -> float:
        """ピーク時の操作数/秒を取得（スレッドセーフ）"""
        def calculate_peak_ops():
            if len(self._operation_times) < 2:
                return 0.0

            max_ops_per_second = 0.0
            current_time = time.time()

            # 時間窓をスライドさせながらピークを探す
            for i in range(len(self._operation_times)):
                window_start = self._operation_times[i]
                if current_time - window_start > CacheConstants.DEFAULT_STATS_HISTORY_CLEANUP_SECONDS:
                    continue

                window_end = window_start + window_seconds
                ops_in_window = sum(
                    1 for op_time in self._operation_times
                    if window_start <= op_time <= window_end
                )

                ops_per_second = ops_in_window / window_seconds
                max_ops_per_second = max(max_ops_per_second, ops_per_second)

            return max_ops_per_second

        result = self._safe_lock_operation(calculate_peak_ops)
        return result if isinstance(result, (int, float)) else 0.0

    def add_stats(self, other_stats: 'CacheStats') -> None:
        """他の統計情報を追加（デッドロック回避）"""
        # デッドロックを避けるため、まず他の統計を安全に取得
        other_dict = other_stats.to_dict()

        def add_other_stats():
            self._hits += other_dict.get("hits", 0)
            self._misses += other_dict.get("misses", 0)
            self._sets += other_dict.get("sets", 0)
            self._evictions += other_dict.get("evictions", 0)
            self._errors += other_dict.get("errors", 0)

        self._safe_lock_operation(add_other_stats)


def validate_cache_key(key: str, config: Optional['CacheConfig'] = None) -> bool:
    """
    キャッシュキーの妥当性を検証

    Args:
        key: 検証するキー
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        キーが有効かどうか
    """
    if not key or not isinstance(key, str):
        return False

    # 設定を取得
    config = config or cache_config

    # 長すぎるキーを拒否（設定から取得）
    if len(key) > config.max_key_length:
        return False

    # 制御文字を含むキーを拒否
    return not any(ord(c) < 32 or ord(c) == 127 for c in key)


def sanitize_cache_value(value: Any, config: Optional['CacheConfig'] = None) -> Any:
    """
    キャッシュ値のサニタイズ

    Args:
        value: サニタイズする値
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        サニタイズされた値

    Raises:
        CacheError: 値が不正な場合
    """
    # None値はそのまま許可
    if value is None:
        return value

    # 設定を取得
    config = config or cache_config

    # 大きすぎるオブジェクトの警告と制限
    try:
        import sys

        size = sys.getsizeof(value)
        if size > config.max_value_size_bytes:
            if config.enable_size_warnings and hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Large cache value detected: {size} bytes (limit: {config.max_value_size_bytes})")

            # 設定によっては大きすぎる値をエラーとして扱う
            # 現在は警告のみだが、将来的にはCacheErrorを発生させることも可能

    except Exception as e:
        # サイズ取得に失敗した場合はデバッグログに記録
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Failed to get cache value size: {e}")

    return value


class TTLCache:
    """
    高性能TTL（Time To Live）キャッシュ実装
    スレッドセーフで効率的な期限管理を提供
    """

    def __init__(self, max_size: Optional[int] = None, default_ttl: Optional[int] = None, config: Optional['CacheConfig'] = None):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            default_ttl: デフォルトTTL（秒、Noneの場合は設定から取得）
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        import threading
        import time
        from collections import OrderedDict

        self._config = config or get_cache_config()
        self._cache = OrderedDict()
        self._timestamps = {}
        self._ttls = {}
        self._max_size = max_size or self._config.default_ttl_cache_size
        self._default_ttl = default_ttl or self._config.default_ttl_seconds
        self._lock = threading.RLock()
        self._stats = CacheStats(self._config)

        # パフォーマンス最適化
        self._time = time.time  # 関数参照をキャッシュ
        self._cleanup_counter = 0
        self._cleanup_frequency = CacheConstants.DEFAULT_CLEANUP_FREQUENCY

    def get(self, key: str, default=None):
        """キャッシュからの値取得（TTLチェック付き）"""
        with self._lock:
            if key not in self._cache:
                self._stats.record_miss()
                return default

            # TTLチェック（高速化）
            current_time = self._time()
            if current_time > self._timestamps[key] + self._ttls[key]:
                # 期限切れ
                self._remove_expired_key(key)
                self._stats.record_miss()
                return default

            # LRU更新
            self._cache.move_to_end(key)
            self._stats.record_hit()
            return self._cache[key]

    def set(self, key: str, value, ttl: Optional[int] = None):
        """キャッシュへの値設定"""
        if ttl is None:
            ttl = self._default_ttl

        current_time = self._time()

        with self._lock:
            # 既存エントリの更新
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = current_time
                self._ttls[key] = ttl
                self._cache.move_to_end(key)
            else:
                # 新規エントリ
                if len(self._cache) >= self._max_size:
                    self._evict_lru()

                self._cache[key] = value
                self._timestamps[key] = current_time
                self._ttls[key] = ttl

            self._stats.record_set()

            # 定期的なクリーンアップ（パフォーマンス最適化）
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_frequency:
                self._cleanup_expired()
                self._cleanup_counter = 0

    def delete(self, key: str) -> bool:
        """キャッシュからの削除"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                del self._ttls[key]
                return True
            return False

    def clear(self):
        """キャッシュのクリア"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._cleanup_counter = 0

    def _remove_expired_key(self, key: str):
        """期限切れキーの削除（内部メソッド）"""
        del self._cache[key]
        del self._timestamps[key]
        del self._ttls[key]
        self._stats.record_eviction()

    def _evict_lru(self):
        """LRU方式での古いエントリ削除"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_expired_key(oldest_key)

    def _cleanup_expired(self):
        """期限切れエントリの一括削除"""
        current_time = self._time()
        expired_keys = []

        for key, timestamp in self._timestamps.items():
            if current_time > timestamp + self._ttls[key]:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_expired_key(key)

    def size(self) -> int:
        """現在のキャッシュサイズ"""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """キャッシュ統計の取得"""
        return self._stats.to_dict()


class HighPerformanceCache:
    """
    超高性能キャッシュ実装
    最小限のロックと最適化されたデータ構造を使用
    """

    def __init__(self, max_size: Optional[int] = None, config: Optional['CacheConfig'] = None):
        import threading
        import time

        self._config = config or get_cache_config()
        self._cache = {}
        self._access_times = {}
        self._max_size = max_size or self._config.high_perf_cache_size
        self._lock = threading.Lock()  # RLockより高速
        self._time = time.time
        self._cleanup_threshold = self._max_size * CacheConstants.DEFAULT_CLEANUP_THRESHOLD_RATIO

    def get(self, key: str):
        """超高速get操作"""
        # ロックなしの高速パス（read-heavy workload最適化）
        if key in self._cache:
            current_time = self._time()
            with self._lock:
                if key in self._cache:  # double-checked locking
                    self._access_times[key] = current_time
                    return self._cache[key]
        return None

    def set(self, key: str, value):
        """高速set操作"""
        current_time = self._time()

        with self._lock:
            self._cache[key] = value
            self._access_times[key] = current_time

            # 自動サイズ管理
            if len(self._cache) > self._cleanup_threshold:
                self._auto_cleanup()

    def _auto_cleanup(self):
        """自動クリーンアップ（最も使用されていないエントリを削除）"""
        if len(self._cache) <= self._max_size * 0.5:
            return

        # アクセス時間順でソートして古いものを削除
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        remove_count = len(self._cache) - int(self._max_size * CacheConstants.DEFAULT_HIGH_PERF_CLEANUP_RATIO)

        for key, _ in sorted_items[:remove_count]:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]


# デフォルトキャッシュインスタンス（設定ベース）
default_cache = TTLCache()  # 設定からデフォルト値を取得
high_perf_cache = HighPerformanceCache()  # 設定からデフォルト値を取得
