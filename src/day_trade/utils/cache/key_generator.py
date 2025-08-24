"""
キャッシュキー生成・シリアライゼーションモジュール

安全で効率的なキャッシュキー生成とオブジェクトのシリアライゼーション機能を提供します。
Pydantic、Decimal、Enum等の主要な型に対応し、循環参照や深い再帰にも対処します。
"""

import hashlib
import json
import logging
import time
import uuid
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from ..logging_config import get_logger
from .circuit_breaker import get_cache_circuit_breaker
from .config import get_cache_config
from .constants import (
    CacheConstants,
    CacheError,
    CacheCircuitBreakerError,
    CacheSerializationError,
    CacheTimeoutError,
)

logger = get_logger(__name__)

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


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
            for item in args.values() if isinstance(args, dict) else args:
                if isinstance(item, (dict, list, tuple)):
                    complexity += len(item)

        elif hasattr(args, "__dict__"):
            complexity += len(args.__dict__)

        return complexity

    except Exception:
        # エラーの場合は中程度の複雑度を返す
        return 50


def _normalize_arguments(
    args: Union[Tuple, Dict, Any],
    max_depth: int = None,
    current_depth: int = 0,
    seen_objects: Optional[set] = None,
) -> Any:
    """
    引数を正規化してシリアライズ可能な形式に変換

    Args:
        args: 正規化する引数
        max_depth: 最大再帰深度
        current_depth: 現在の再帰深度
        seen_objects: 既に処理したオブジェクトのセット（循環参照検出用）

    Returns:
        正規化された引数
    """
    cache_config = get_cache_config()
    
    if max_depth is None:
        max_depth = cache_config.max_recursion_depth
    if seen_objects is None:
        seen_objects = set()

    if current_depth >= max_depth:
        if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"Recursion depth limit reached ({max_depth}), truncating object of type {type(args).__name__}"
            )
        if isinstance(args, (dict, list, tuple)):
            return f"<truncated {type(args).__name__} with {len(args)} items at depth {max_depth}>"
        else:
            return f"<truncated {type(args).__name__} at depth {max_depth}>"

    try:
        obj_id = id(args)
    except TypeError:
        obj_id = None

    if obj_id is not None:
        if obj_id in seen_objects:
            return f"<circular reference to {type(args).__name__}>"
        seen_objects.add(obj_id)

    try:
        if PYDANTIC_AVAILABLE and isinstance(args, BaseModel):
            try:
                model_data = args.model_dump() if hasattr(args, "model_dump") else args.dict()
                return _normalize_arguments(model_data, max_depth, current_depth + 1, seen_objects)
            except Exception:
                return f"<pydantic model: {args.__class__.__name__}>"
        elif isinstance(args, Decimal):
            return str(args)
        elif isinstance(args, Enum):
            return f"<enum {args.__class__.__name__}: {args.value}>"
        elif isinstance(args, (tuple, list)):
            normalized_items = []
            for i, arg in enumerate(args):
                normalized_items.append(_normalize_arguments(arg, max_depth, current_depth + 1, seen_objects))
                if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
                    normalized_items.append(f"<truncated: {len(args) - i - 1} more items>")
                    break
            return normalized_items
        elif isinstance(args, dict):
            normalized_dict = {}
            processed_count = 0
            for k, v in args.items():
                key_str = str(k)
                normalized_dict[key_str] = _normalize_arguments(v, max_depth, current_depth + 1, seen_objects)
                processed_count += 1
                if processed_count >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
                    if len(args) - processed_count > 0:
                        normalized_dict["<truncated>"] = f"{len(args) - processed_count} more keys"
                    break
            return normalized_dict
        elif isinstance(args, set):
            normalized_items = []
            for i, item in enumerate(args):
                normalized_items.append(_normalize_arguments(item, max_depth, current_depth + 1, seen_objects))
                if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
                    normalized_items.append(f"<truncated: {len(args) - i - 1} more items>")
                    break
            return {"__set__": sorted(normalized_items, key=str)}
        elif hasattr(args, "isoformat"):
            return args.isoformat()
        elif hasattr(args, "__dict__"):
            obj_dict = getattr(args, "__dict__", {})
            if obj_dict:
                return _normalize_arguments(obj_dict, max_depth, current_depth + 1, seen_objects)
            else:
                return f"<empty_object: {args.__class__.__name__}>"
        elif callable(args):
            name = getattr(args, "__name__", repr(args))
            return f"<callable: {name}>"
        else:
            if isinstance(args, str) and len(args) > CacheConstants.DEFAULT_MAX_KEY_LENGTH:
                return f"<long_string: {len(args)} chars>"
            return args
    finally:
        if obj_id is not None:
            seen_objects.discard(obj_id)


def _json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザーのカスタムハンドラー（堅牢版・Pydantic v2対応・シンプルタイムアウト機能付き）

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズ可能な値

    Raises:
        CacheTimeoutError: シリアライゼーションがタイムアウトした場合
        CacheSerializationError: シリアライゼーションが失敗した場合
    """
    cache_config = get_cache_config()
    start_time = time.time()
    timeout_seconds = cache_config.serialization_timeout

    def check_timeout():
        """タイムアウトチェック"""
        if time.time() - start_time > timeout_seconds:
            raise CacheTimeoutError(
                f"Serialization timeout after {timeout_seconds} seconds",
                timeout_seconds,
                "json_serialization",
            )

    try:
        # Pydanticモデル（v2対応強化）
        if PYDANTIC_AVAILABLE and isinstance(obj, BaseModel):
            check_timeout()
            try:
                # Pydantic v2の場合
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(mode="json", exclude_unset=False)
                # Pydantic v1の場合
                elif hasattr(obj, "dict"):
                    return obj.dict()
                else:
                    # フォールバック
                    return {
                        "__pydantic_model__": obj.__class__.__name__,
                        "data": str(obj),
                    }
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

        # Enum型
        elif isinstance(obj, Enum):
            check_timeout()
            return {
                "__enum__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, "__module__", "unknown"),
                "name": obj.name,
                "value": obj.value,
            }

        # datetime/date/time オブジェクト
        elif hasattr(obj, "isoformat"):
            check_timeout()
            try:
                iso_string = obj.isoformat()
                return {
                    "__datetime__": iso_string,
                    "__type__": obj.__class__.__name__,
                    "__timezone__": (
                        str(getattr(obj, "tzinfo", None))
                        if hasattr(obj, "tzinfo")
                        else None
                    ),
                }
            except Exception:
                return {"__datetime__": str(obj), "__type__": obj.__class__.__name__}

        # set型
        elif isinstance(obj, set):
            check_timeout()
            try:
                sorted_items = []
                for item in obj:
                    check_timeout()
                    try:
                        json.dumps(item, default=str)
                        sorted_items.append(item)
                    except Exception:
                        sorted_items.append(str(item))

                try:
                    sorted_items.sort()
                except TypeError:
                    sorted_items = sorted([str(item) for item in sorted_items])

                return {"__set__": sorted_items}
            except Exception:
                return {"__set__": [str(item) for item in obj]}

        # frozenset型
        elif isinstance(obj, frozenset):
            check_timeout()
            try:
                return {"__frozenset__": _json_serializer(set(obj))["__set__"]}
            except Exception:
                return {"__frozenset__": [str(item) for item in obj]}

        # bytes型
        elif isinstance(obj, bytes):
            check_timeout()
            try:
                return {"__bytes__": obj.decode("utf-8"), "encoding": "utf-8"}
            except UnicodeDecodeError:
                try:
                    return {"__bytes__": obj.hex(), "encoding": "hex"}
                except Exception:
                    return {
                        "__bytes__": "<binary_data>",
                        "encoding": "unknown",
                        "size": len(obj),
                    }

        # 関数またはメソッド
        elif callable(obj):
            check_timeout()
            return {
                "__callable__": getattr(obj, "__name__", repr(obj)),
                "__module__": getattr(obj, "__module__", "unknown"),
                "__type__": "function" if hasattr(obj, "__name__") else "callable",
            }

        # オブジェクトに辞書がある場合
        elif hasattr(obj, "__dict__"):
            check_timeout()
            try:
                safe_dict = {}
                for key, value in obj.__dict__.items():
                    check_timeout()
                    try:
                        safe_dict[str(key)] = _json_serializer(value)
                    except Exception as e:
                        safe_dict[str(key)] = f"<serialization_error: {e}>"

                return {
                    "__object__": obj.__class__.__name__,
                    "__module__": getattr(obj.__class__, "__module__", "unknown"),
                    "data": safe_dict,
                }
            except Exception as e:
                return {
                    "__object__": obj.__class__.__name__,
                    "__module__": getattr(obj.__class__, "__module__", "unknown"),
                    "error": str(e),
                }

        # NumPy配列（オプション対応）
        elif hasattr(obj, "tolist") and hasattr(obj, "dtype"):
            check_timeout()
            try:
                return {
                    "__numpy_array__": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
            except Exception:
                return {"__numpy_array__": str(obj)}

        # その他の場合は文字列表現
        else:
            check_timeout()
            return {
                "__unknown_type__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, "__module__", "unknown"),
                "value": str(obj),
            }

    except CacheTimeoutError:
        raise
    except Exception as e:
        logger.warning(f"Serialization fallback for {type(obj).__name__}: {e}")
        return {
            "__serialization_fallback__": True,
            "__type__": obj.__class__.__name__,
            "__error__": str(e),
            "value": str(obj),
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
        if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logging.ERROR):
            logger.error(f"Fallback cache key generation failed: {e}")
        
        # より安全なユニークキー生成
        timestamp = int(time.time() * 1000000)  # マイクロ秒タイムスタンプ
        unique_id = str(uuid.uuid4()).replace("-", "")[:8]
        return f"{func_name}:emergency:{timestamp}:{unique_id}"


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
        timestamp = int(time.time() * 1000000)  # マイクロ秒
        unique_suffix = str(uuid.uuid4()).replace("-", "")[:8]
        args_hash = str(hash(str(args)))[-8:] if args else "0"
        kwargs_hash = str(hash(str(sorted(kwargs.items()))))[-8:] if kwargs else "0"

        return f"{func_name}:emergency:{timestamp}:{unique_suffix}:{args_hash}:{kwargs_hash}"
    except Exception as e:
        # 最終フォールバック
        logger.error(f"Emergency cache key generation failed: {e}")
        return f"{func_name}:final_fallback:{int(time.time())}"


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
        cache_config = get_cache_config()
        start_time = time.time()

        try:
            # データ複雑度を推定してアダプティブ再帰制限を設定
            args_complexity = _estimate_data_complexity(args)
            kwargs_complexity = _estimate_data_complexity(kwargs)
            total_complexity = args_complexity + kwargs_complexity

            adaptive_depth = cache_config.get_adaptive_depth_limit(total_complexity)

            # 引数を正規化してシリアライズ可能な形式に変換
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
                "complexity": total_complexity,
                "depth": adaptive_depth,
            }

            serialized = json.dumps(
                serializable_data, sort_keys=True, default=_json_serializer
            )
            cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            # 処理時間の監視
            elapsed_time = time.time() - start_time
            if elapsed_time > cache_config.serialization_timeout * 0.8:
                logger.warning(
                    f"Cache key generation took {elapsed_time:.3f}s for {func_name}"
                )

            if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Generated cache key for {func_name}: {cache_key} "
                    f"(complexity: {total_complexity}, depth: {adaptive_depth})"
                )

            return f"{func_name}:{cache_key}"

        except (TypeError, ValueError, OverflowError) as e:
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
                        "processing_time": time.time() - start_time,
                    },
                ) from e
        except Exception as e:
            if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logging.ERROR):
                logger.error(f"Unexpected error in cache key generation: {e}", exc_info=True)
            raise CacheError(
                f"キャッシュキー生成で予期しないエラーが発生しました: {e}",
                "CACHE_KEY_UNEXPECTED_ERROR",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "func_name": func_name,
                    "processing_time": time.time() - start_time,
                },
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
        raise e