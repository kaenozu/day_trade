"""
キャッシュキー生成モジュール

安全で効率的なキャッシュキー生成機能を提供します。
サーキットブレーカー機能、高度なエラーハンドリング、
タイムアウト処理などを備えています。
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Dict, Tuple

from ..logging_config import get_logger
from .argument_normalizer import _estimate_data_complexity, _normalize_arguments
from .circuit_breaker import get_cache_circuit_breaker
from .config import get_cache_config
from .constants import (
    CacheError,
    CacheCircuitBreakerError,
)
from .json_serializer import _json_serializer

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
        # その他のエラーもサーキットブレーカーに記録される
        raise e


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