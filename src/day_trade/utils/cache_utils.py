"""
キャッシュユーティリティ
安全で効率的なキャッシュキー生成とキャッシュ操作を提供
堅牢なシリアライズ機能とスレッドセーフティを含む
"""

import hashlib
import json
import logging
import threading
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

logger = logging.getLogger(__name__)


class CacheConfig:
    """キャッシュ設定の管理クラス（アダプティブ再帰制限付き）"""

    def __init__(self):
        # デフォルト設定値（環境変数で上書き可能）
        import os

        self.max_key_length = int(os.getenv("CACHE_MAX_KEY_LENGTH", "1000"))
        self.max_value_size_mb = int(os.getenv("CACHE_MAX_VALUE_SIZE_MB", "10"))
        self.max_recursion_depth = int(os.getenv("CACHE_MAX_RECURSION_DEPTH", "10"))
        self.enable_size_warnings = os.getenv("CACHE_ENABLE_SIZE_WARNINGS", "true").lower() == "true"

        # アダプティブ再帰制限の設定
        self.adaptive_recursion = os.getenv("CACHE_ADAPTIVE_RECURSION", "true").lower() == "true"
        self.min_recursion_depth = int(os.getenv("CACHE_MIN_RECURSION_DEPTH", "3"))
        self.max_adaptive_depth = int(os.getenv("CACHE_MAX_ADAPTIVE_DEPTH", "20"))

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


# グローバル設定インスタンス
cache_config = CacheConfig()


class CacheError(Exception):
    """キャッシュ操作エラー"""
    def __init__(self, message: str, error_code: str = "CACHE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


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
    安全なキャッシュキー生成

    Args:
        func_name: 関数名
        *args: 位置引数
        **kwargs: キーワード引数

    Returns:
        生成されたキャッシュキー

    Raises:
        CacheError: キャッシュキー生成に失敗した場合
    """
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
        }

        serialized = json.dumps(
            serializable_data, sort_keys=True, default=_json_serializer
        )
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        # パフォーマンス最適化: デバッグログの条件付き出力
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated cache key for {func_name}: {cache_key}")
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
                    "kwargs_keys": list(kwargs.keys()) if kwargs else []
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
                "func_name": func_name
            }
        ) from e


def _normalize_arguments(args: Union[Tuple, Dict, Any], max_depth: int = None, current_depth: int = 0, seen_objects: Optional[set] = None) -> Any:
    """
    引数を正規化してシリアライズ可能にする（再帰深度制限付き・循環参照検出）

    Args:
        args: 正規化する引数
        max_depth: 最大再帰深度（Noneの場合は設定から取得）
        current_depth: 現在の再帰深度
        seen_objects: 循環参照検出用のオブジェクトセット

    Returns:
        正規化された引数

    Raises:
        CacheError: 再帰深度制限を超えた場合
    """
    # max_depthが指定されていない場合は設定から取得
    if max_depth is None:
        max_depth = cache_config.max_recursion_depth

    # 循環参照検出用のセットを初期化
    if seen_objects is None:
        seen_objects = set()

    if current_depth >= max_depth:
        # パフォーマンス最適化: 警告ログの条件付き出力
        if hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
            logger.warning(f"Recursion depth limit reached ({max_depth}), truncating object")
        return f"<truncated at depth {max_depth}>"

    # 循環参照の検出（ハッシュ可能なオブジェクトのみ）
    try:
        obj_id = id(args)
        if obj_id in seen_objects:
            return f"<circular reference to {type(args).__name__}>"

        # 複雑なオブジェクトの場合のみseenに追加
        if isinstance(args, (dict, list, tuple)) and args:
            seen_objects.add(obj_id)

    except Exception:
        # id()の取得に失敗した場合は循環参照チェックをスキップ
        pass

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

    # コレクション型
    elif isinstance(args, (tuple, list)):
        try:
            return [_normalize_arguments(arg, max_depth, current_depth + 1, seen_objects) for arg in args]
        except Exception:
            return f"<list/tuple of {len(args)} items>"

    elif isinstance(args, dict):
        try:
            return {str(k): _normalize_arguments(v, max_depth, current_depth + 1, seen_objects) for k, v in args.items()}
        except Exception:
            return f"<dict with {len(args)} keys>"

    elif isinstance(args, set):
        try:
            return {"__set__": sorted([_normalize_arguments(item, max_depth, current_depth + 1, seen_objects) for item in args])}
        except Exception:
            return f"<set of {len(args)} items>"

    # datetime類
    elif hasattr(args, "isoformat"):
        return args.isoformat()

    # オブジェクト
    elif hasattr(args, "__dict__"):
        try:
            return _normalize_arguments(args.__dict__, max_depth, current_depth + 1, seen_objects)
        except Exception:
            return f"<object: {args.__class__.__name__}>"

    # 関数
    elif callable(args):
        return getattr(args, "__name__", str(type(args).__name__))

    # プリミティブ型
    else:
        # 処理完了後、循環参照検出セットからIDを削除
        try:
            if isinstance(args, (dict, list, tuple)) and args:
                seen_objects.discard(id(args))
        except Exception:
            pass
        return args


def _json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザーのカスタムハンドラー（堅牢版）

    Pydantic、Decimal、Enum、datetime等の主要な型に対応

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズ可能な値
    """
    # Pydanticモデル
    if PYDANTIC_AVAILABLE and isinstance(obj, BaseModel):
        return obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()

    # Decimal型
    elif isinstance(obj, Decimal):
        return str(obj)

    # Enum型
    elif isinstance(obj, Enum):
        return {"__enum__": obj.__class__.__name__, "value": obj.value}

    # datetime/date/time オブジェクト
    elif hasattr(obj, "isoformat"):
        return {"__datetime__": obj.isoformat()}

    # set型
    elif isinstance(obj, set):
        return {"__set__": sorted(list(obj))}

    # frozenset型
    elif isinstance(obj, frozenset):
        return {"__frozenset__": sorted(list(obj))}

    # bytes型
    elif isinstance(obj, bytes):
        try:
            return {"__bytes__": obj.decode('utf-8')}
        except UnicodeDecodeError:
            return {"__bytes__": obj.hex()}

    # 関数またはメソッド
    elif callable(obj):
        return {"__callable__": getattr(obj, "__name__", repr(obj))}

    # オブジェクトに辞書がある場合
    elif hasattr(obj, "__dict__"):
        try:
            return {"__object__": obj.__class__.__name__, "data": obj.__dict__}
        except Exception:
            return {"__object__": obj.__class__.__name__, "data": str(obj)}

    # その他の場合は文字列表現
    else:
        return str(obj)


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
        combined_data = f"{func_name}:{args_str}:{kwargs_str}".encode('utf-8')
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
    """キャッシュ統計情報（スレッドセーフ版・デッドロック対策強化）"""

    def __init__(self):
        self._lock = threading.RLock()  # 再帰可能ロック
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._errors = 0
        self._lock_timeout = 1.0  # ロック取得タイムアウト（秒）

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
        """キャッシュヒットを記録"""
        self._safe_lock_operation(lambda: self._increment_counter('_hits', count))

    def record_miss(self, count: int = 1) -> None:
        """キャッシュミスを記録"""
        self._safe_lock_operation(lambda: self._increment_counter('_misses', count))

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
            if new_value > 2**63 - 1:  # 64bit 符号付き整数の最大値
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
        """統計をリセット"""
        def reset_counters():
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._errors = 0

        self._safe_lock_operation(reset_counters)

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """統計情報を辞書として返す（スレッドセーフ）"""
        def create_stats_dict():
            return {
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "errors": self._errors,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
                "total_requests": self._hits + self._misses,
            }

        result = self._safe_lock_operation(create_stats_dict)
        return result if isinstance(result, dict) else {}

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


def validate_cache_key(key: str) -> bool:
    """
    キャッシュキーの妥当性を検証

    Args:
        key: 検証するキー

    Returns:
        キーが有効かどうか
    """
    if not key or not isinstance(key, str):
        return False

    # 長すぎるキーを拒否（設定から取得）
    if len(key) > cache_config.max_key_length:
        return False

    # 制御文字を含むキーを拒否
    return not any(ord(c) < 32 or ord(c) == 127 for c in key)


def sanitize_cache_value(value: Any) -> Any:
    """
    キャッシュ値のサニタイズ

    Args:
        value: サニタイズする値

    Returns:
        サニタイズされた値
    """
    # None値はそのまま許可
    if value is None:
        return value

    # 大きすぎるオブジェクトの警告
    try:
        import sys

        size = sys.getsizeof(value)
        if size > cache_config.max_value_size_bytes:
            if cache_config.enable_size_warnings and hasattr(logger, 'isEnabledFor') and logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Large cache value detected: {size} bytes")
    except Exception:
        pass  # サイズ取得に失敗した場合は無視

    return value


class TTLCache:
    """
    高性能TTL（Time To Live）キャッシュ実装
    スレッドセーフで効率的な期限管理を提供
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Args:
            max_size: 最大キャッシュサイズ
            default_ttl: デフォルトTTL（秒）
        """
        import time
        import threading
        from collections import OrderedDict

        self._cache = OrderedDict()
        self._timestamps = {}
        self._ttls = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = CacheStats()

        # パフォーマンス最適化
        self._time = time.time  # 関数参照をキャッシュ
        self._cleanup_counter = 0
        self._cleanup_frequency = 100  # 100回に1回クリーンアップ実行

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

    def __init__(self, max_size: int = 10000):
        import threading
        import time

        self._cache = {}
        self._access_times = {}
        self._max_size = max_size
        self._lock = threading.Lock()  # RLockより高速
        self._time = time.time
        self._cleanup_threshold = max_size * 0.8  # 80%で自動クリーンアップ

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
        remove_count = len(self._cache) - int(self._max_size * 0.7)

        for key, _ in sorted_items[:remove_count]:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]


# デフォルトキャッシュインスタンス
default_cache = TTLCache(max_size=5000, default_ttl=600)  # 10分TTL
high_perf_cache = HighPerformanceCache(max_size=10000)
