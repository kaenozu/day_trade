"""
キャッシュキー生成機能

統合キャッシュシステムの安全で効率的なキーシュキー生成を提供します。
元のcache_utils.pyからキー生成関連機能を分離。
"""

import hashlib
import json
import logging
import time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple, Union

from src.day_trade.utils.logging_config import get_logger

from .config import CacheConstants, get_cache_config
from .errors import CacheError, CacheTimeoutError, get_cache_circuit_breaker

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

logger = get_logger(__name__)


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
        config = get_cache_config()

        try:
            # データ複雑度を推定してアダプティブ再帰制限を設定
            args_complexity = _estimate_data_complexity(args)
            kwargs_complexity = _estimate_data_complexity(kwargs)
            total_complexity = args_complexity + kwargs_complexity

            # 複雑度に基づいてアダプティブ深度を決定
            if config.adaptive_recursion:
                if total_complexity < 10:
                    adaptive_depth = config.min_recursion_depth
                elif total_complexity < 50:
                    adaptive_depth = config.max_recursion_depth
                else:
                    adaptive_depth = min(
                        config.max_adaptive_depth,
                        config.max_recursion_depth + (total_complexity // 20),
                    )
            else:
                adaptive_depth = config.max_recursion_depth

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
                "complexity": total_complexity,  # 複雑度も含めてキーの一意性を向上
                "depth": adaptive_depth,
            }

            serialized = json.dumps(
                serializable_data, sort_keys=True, default=_json_serializer
            )
            cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            # 処理時間の監視
            elapsed_time = time.time() - start_time
            if elapsed_time > config.serialization_timeout * 0.8:  # 80%を超えたら警告
                logger.warning(
                    f"Cache key generation took {elapsed_time:.3f}s for {func_name}"
                )

            # キーの長さチェック
            final_key = f"{func_name}:{cache_key}"
            if len(final_key) > config.max_key_length:
                logger.warning(
                    f"Generated key length ({len(final_key)}) exceeds maximum ({config.max_key_length})"
                )

            # パフォーマンス最適化: デバッグログの条件付き出力
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Generated cache key for {func_name}: {cache_key} "
                    f"(complexity: {total_complexity}, depth: {adaptive_depth})"
                )

            return final_key

        except (TypeError, ValueError, OverflowError) as e:
            # 予期される例外: シリアライゼーションエラー
            logger.warning(
                f"JSON serialization failed for cache key, using fallback: {e}"
            )
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
                        "processing_time": time.time() - start_time,
                    },
                ) from e
        except Exception as e:
            # 予期しない例外
            if logger.isEnabledFor(logging.ERROR):
                logger.error(
                    f"Unexpected error in cache key generation: {e}", exc_info=True
                )
            raise CacheError(
                f"キャッシュキー生成で予期しないエラーが発生しました: {e}",
                "CACHE_KEY_GENERATION_UNEXPECTED",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "func_name": func_name,
                    "processing_time": time.time() - start_time,
                },
            ) from e

    # タイムアウト処理
    start_time = time.time()
    config = get_cache_config()

    # サーキットブレーカーを使用してキー生成を実行
    try:
        circuit_breaker = get_cache_circuit_breaker()
        result = circuit_breaker.call(_generate_key_internal)

        # タイムアウトチェック
        elapsed_time = time.time() - start_time
        if elapsed_time > config.serialization_timeout:
            raise CacheTimeoutError(
                f"Cache key generation timed out after {elapsed_time:.3f}s",
                config.serialization_timeout,
                "key_generation",
            )

        return result

    except Exception as e:
        if not isinstance(e, (CacheError, CacheTimeoutError)):
            # 予期しない例外をCacheErrorにラップ
            raise CacheError(
                f"Cache key generation failed: {e}",
                "CACHE_KEY_GENERATION_FAILED",
                {"original_error": str(e), "func_name": func_name},
            ) from e
        raise


def _normalize_arguments(
    args: Union[Tuple, Dict, Any],
    max_depth: Optional[int] = None,
    current_depth: int = 0,
    seen_objects: Optional[Set] = None,
) -> Any:
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
    config = get_cache_config()

    # max_depthが指定されていない場合は設定から取得
    if max_depth is None:
        max_depth = config.max_recursion_depth

    # 循環参照検出用のセットを初期化
    if seen_objects is None:
        seen_objects = set()

    # 深度制限チェック（エラーハンドリング強化）
    if current_depth >= max_depth:
        # パフォーマンス最適化: 警告ログの条件付き出力
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"Recursion depth limit reached ({max_depth}), "
                f"truncating object of type {type(args).__name__}"
            )
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
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"Circular reference detection set is too large ({len(seen_objects)}), "
                        f"skipping detection for {type(args).__name__}"
                    )
    except Exception as e:
        # id()の取得に失敗した場合はデバッグログに記録
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Failed to get object id for circular reference detection: {e}"
            )
        # 循環参照チェックをスキップして処理を続行

    # Pydanticモデル
    if PYDANTIC_AVAILABLE and isinstance(args, BaseModel):
        try:
            model_data = (
                args.model_dump() if hasattr(args, "model_dump") else args.dict()
            )
            return _normalize_arguments(
                model_data, max_depth, current_depth + 1, seen_objects
            )
        except Exception as e:
            logger.warning(f"Failed to serialize Pydantic model: {e}")
            return f"<PydanticModel:{type(args).__name__}>"

    # 基本的なPythonオブジェクトの正規化
    if args is None:
        return None
    elif isinstance(args, (str, int, float, bool)):
        return args
    elif isinstance(args, Decimal):
        return float(args)
    elif isinstance(args, Enum):
        return args.value
    elif isinstance(args, (list, tuple)):
        try:
            return [
                _normalize_arguments(item, max_depth, current_depth + 1, seen_objects)
                for item in args
            ]
        except Exception as e:
            logger.warning(f"Failed to normalize list/tuple: {e}")
            return f"<{type(args).__name__} with {len(args)} items>"
    elif isinstance(args, dict):
        try:
            return {
                str(key): _normalize_arguments(
                    value, max_depth, current_depth + 1, seen_objects
                )
                for key, value in args.items()
            }
        except Exception as e:
            logger.warning(f"Failed to normalize dict: {e}")
            return f"<dict with {len(args)} items>"
    elif isinstance(args, set):
        try:
            # セットを安全にソート
            return sorted(
                [
                    _normalize_arguments(
                        item, max_depth, current_depth + 1, seen_objects
                    )
                    for item in args
                ],
                key=str,
            )
        except Exception as e:
            logger.warning(f"Failed to normalize set: {e}")
            return f"<set with {len(args)} items>"
    else:
        # その他のオブジェクトは型名と文字列表現を使用
        try:
            return f"<{type(args).__name__}:{str(args)[:100]}>"
        except Exception:
            return f"<{type(args).__name__}:repr_failed>"


def _estimate_data_complexity(
    data: Any, current_depth: int = 0, max_check_depth: int = 5
) -> int:
    """
    データ構造の複雑度を推定（パフォーマンス考慮）

    Args:
        data: 複雑度を推定するデータ
        current_depth: 現在のチェック深度
        max_check_depth: 最大チェック深度

    Returns:
        推定された複雑度スコア
    """
    if current_depth > max_check_depth:
        return 1  # 深すぎる場合は最小値

    try:
        if data is None:
            return 0
        elif isinstance(data, (str, int, float, bool)):
            return 1
        elif isinstance(data, (list, tuple)):
            if not data:
                return 1
            # サンプリングで効率化（大きなリストの場合）
            sample_size = min(len(data), 10)
            sample_complexity = sum(
                _estimate_data_complexity(data[i], current_depth + 1, max_check_depth)
                for i in range(0, len(data), max(1, len(data) // sample_size))
            )
            return len(data) + sample_complexity
        elif isinstance(data, dict):
            if not data:
                return 1
            # サンプリングで効率化
            sample_items = list(data.items())[:10]
            return len(data) + sum(
                _estimate_data_complexity(key, current_depth + 1, max_check_depth)
                + _estimate_data_complexity(value, current_depth + 1, max_check_depth)
                for key, value in sample_items
            )
        elif isinstance(data, set):
            return len(data) + 1
        else:
            return 2  # その他のオブジェクト
    except Exception:
        return 1  # エラーの場合は最小値


def _generate_fallback_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    フォールバック用のキャッシュキー生成

    Args:
        func_name: 関数名
        args: 位置引数
        kwargs: キーワード引数

    Returns:
        フォールバックキー
    """
    try:
        # 基本的な情報のみを使用
        fallback_data = {
            "function": func_name,
            "args_count": len(args) if args else 0,
            "kwargs_keys": sorted(kwargs.keys()) if kwargs else [],
            "timestamp": int(time.time() * 1000),  # ミリ秒タイムスタンプ
        }

        # 引数の型情報を追加
        if args:
            fallback_data["args_types"] = [type(arg).__name__ for arg in args]
        if kwargs:
            fallback_data["kwargs_types"] = {
                key: type(value).__name__ for key, value in kwargs.items()
            }

        serialized = json.dumps(fallback_data, sort_keys=True)
        cache_key = hashlib.md5(serialized.encode("utf-8")).hexdigest()

        logger.info(f"Generated fallback cache key for {func_name}: {cache_key}")
        return f"{func_name}:fallback:{cache_key}"

    except Exception as e:
        # 最後の手段：関数名 + タイムスタンプ
        timestamp = int(time.time() * 1000)
        fallback_key = f"{func_name}:emergency:{timestamp}"
        logger.warning(
            f"Emergency fallback cache key generated for {func_name}: {fallback_key}, error: {e}"
        )
        return fallback_key


def _json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザー（カスタムオブジェクト対応）

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズ可能な値
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "__dict__"):
        return str(obj)
    else:
        return str(obj)


class CacheKeyGenerator:
    """
    キャッシュキー生成クラス

    設定可能なキー生成戦略を提供します。
    """

    def __init__(self, config=None):
        """
        Args:
            config: キャッシュ設定（Noneの場合はグローバル設定を使用）
        """
        self.config = config or get_cache_config()

    def generate_key(self, func_name: str, *args, **kwargs) -> str:
        """
        キャッシュキーを生成

        Args:
            func_name: 関数名
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            生成されたキャッシュキー
        """
        return generate_safe_cache_key(func_name, *args, **kwargs)

    def generate_simple_key(self, *components: str) -> str:
        """
        シンプルなキーを生成

        Args:
            *components: キーの構成要素

        Returns:
            生成されたキー
        """
        key_data = ":".join(str(c) for c in components)
        if len(key_data) <= self.config.max_key_length:
            return key_data

        # 長すぎる場合はハッシュ化
        hash_value = hashlib.sha256(key_data.encode("utf-8")).hexdigest()
        return f"hashed:{hash_value}"

    def generate_time_based_key(self, base_key: str, ttl_seconds: int) -> str:
        """
        時間ベースのキーを生成（TTL考慮）

        Args:
            base_key: ベースキー
            ttl_seconds: TTL（秒）

        Returns:
            時間ベースのキー
        """
        time_bucket = int(time.time() // ttl_seconds)
        return f"{base_key}:t{time_bucket}"
