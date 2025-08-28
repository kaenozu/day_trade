"""
株価データ取得用キャッシュシステム
キャッシュデコレータとメイン機能の統合
"""

import logging
from functools import wraps

from ...utils.cache_utils import (
    CacheStats,
    generate_safe_cache_key,
    sanitize_cache_value,
    validate_cache_key,
)
from ...utils.exceptions import (
    APIError,
    DataError,
    NetworkError,
)
from ...utils.logging_config import get_context_logger

from .exceptions import DataNotFoundError, InvalidSymbolError, StockFetcherError
from .cache_core import DataCache
from .cache_tuning import TunableDataCache


def cache_with_ttl(
    ttl_seconds: int, max_size: int = 1000, stale_while_revalidate: int = None
):
    """TTL付きキャッシュデコレータ（フォールバック機能付き改善版）"""
    if stale_while_revalidate is None:
        stale_while_revalidate = ttl_seconds * 5  # デフォルトでTTLの5倍

    cache = TunableDataCache(ttl_seconds, max_size, stale_while_revalidate)
    stats = CacheStats()
    # キャッシュデコレータ用のロガーを取得
    cache_logger = get_context_logger(__name__)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 安全なキャッシュキーを生成
                cache_key = generate_safe_cache_key(func.__name__, *args, **kwargs)

                # キーの妥当性を検証
                if not validate_cache_key(cache_key):
                    cache_logger.warning(
                        f"Invalid cache key generated for {func.__name__}, skipping cache"
                    )
                    stats.record_error()
                    return func(*args, **kwargs)

                # フレッシュキャッシュを試行
                cached_result = cache.get(cache_key, allow_stale=False)
                if cached_result is not None:
                    # パフォーマンス最適化: デバッグログの条件付き出力
                    if hasattr(
                        cache_logger, "isEnabledFor"
                    ) and cache_logger.isEnabledFor(logging.DEBUG):
                        cache_logger.debug(
                            f"フレッシュキャッシュヒット: {func.__name__}"
                        )
                    stats.record_hit()
                    return cached_result

                # APIリクエスト実行を試行
                try:
                    stats.record_miss()
                    result = func(*args, **kwargs)

                    # 結果をサニタイズしてキャッシュに保存
                    if result is not None:
                        sanitized_result = sanitize_cache_value(result)
                        cache.set(cache_key, sanitized_result)
                        stats.record_set()
                        # パフォーマンス最適化: デバッグログの条件付き出力
                        if hasattr(
                            cache_logger, "isEnabledFor"
                        ) and cache_logger.isEnabledFor(logging.DEBUG):
                            cache_logger.debug(
                                f"新しいデータをキャッシュに保存: {func.__name__}"
                            )

                    return result

                except (APIError, NetworkError, DataError) as api_error:
                    # APIエラー時はstaleキャッシュをフォールバックとして使用
                    stale_result = cache.get(cache_key, allow_stale=True)
                    if stale_result is not None:
                        cache_logger.warning(
                            f"API失敗、staleキャッシュを使用: {func.__name__} - {api_error}"
                        )
                        stats.record_fallback()
                        return stale_result
                    else:
                        # staleキャッシュもない場合は例外を再発生
                        cache_logger.error(
                            f"API失敗かつキャッシュなし: {func.__name__} - {api_error}"
                        )
                        raise api_error

            except Exception as e:
                cache_logger.error(f"キャッシュ処理でエラーが発生: {e}")
                stats.record_error()
                # 重要でないエラーの場合は継続を試行
                if not isinstance(e, (APIError, NetworkError, DataError)):
                    return func(*args, **kwargs)
                raise e

        # キャッシュ管理メソッドを追加
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.reset_stats = stats.reset
        wrapper.get_cache_info = cache.get_cache_info
        wrapper.auto_tune_cache = cache.auto_tune_cache_settings

        return wrapper

    return decorator


# 後方互換性のためにDataCacheをエクスポート
__all__ = ["DataCache", "TunableDataCache", "cache_with_ttl"]