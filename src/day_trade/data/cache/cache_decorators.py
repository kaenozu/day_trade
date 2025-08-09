"""
キャッシュデコレータ

関数をキャッシュ化するためのデコレータ機能
"""

import logging
from functools import wraps
from typing import Callable, Any

from ...utils.cache_utils import (
    CacheStats,
    generate_safe_cache_key,
    sanitize_cache_value,
    validate_cache_key,
)
from ...utils.exceptions import APIError, NetworkError, DataError
from ...utils.logging_config import get_context_logger

from .data_cache import DataCache

logger = get_context_logger(__name__)


def cache_with_ttl(
    ttl_seconds: int, max_size: int = 1000, stale_while_revalidate: int = None
) -> Callable:
    """
    TTL付きキャッシュデコレータ（フォールバック機能付き改善版）

    Args:
        ttl_seconds: キャッシュの有効期限（秒）
        max_size: 最大キャッシュサイズ
        stale_while_revalidate: 期限切れ後のフォールバック期間（秒）

    Returns:
        デコレートされた関数
    """
    if stale_while_revalidate is None:
        stale_while_revalidate = ttl_seconds * 5  # デフォルトでTTLの5倍

    cache = DataCache(ttl_seconds, max_size, stale_while_revalidate)
    stats = CacheStats()
    # キャッシュデコレータ用のロガーを取得
    cache_logger = get_context_logger(__name__)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
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
        wrapper.get_cache_stats = cache.get_cache_stats
        wrapper.auto_tune_cache = cache.auto_tune_cache_settings

        return wrapper

    return decorator


def adaptive_cache(
    initial_ttl: int = 300,
    max_size: int = 1000,
    auto_tune_threshold: int = 100,
) -> Callable:
    """
    適応的キャッシュデコレータ
    
    使用パターンに基づいて自動的にキャッシュ設定を調整

    Args:
        initial_ttl: 初期TTL（秒）
        max_size: 最大キャッシュサイズ
        auto_tune_threshold: 自動調整を開始する最小リクエスト数

    Returns:
        デコレートされた関数
    """
    cache = DataCache(initial_ttl, max_size, initial_ttl * 2)
    stats = CacheStats()
    auto_tune_counter = 0

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal auto_tune_counter
            
            try:
                cache_key = generate_safe_cache_key(func.__name__, *args, **kwargs)
                
                if not validate_cache_key(cache_key):
                    logger.warning(f"無効なキャッシュキー: {func.__name__}")
                    return func(*args, **kwargs)

                # キャッシュ確認
                cached_result = cache.get(cache_key, allow_stale=False)
                if cached_result is not None:
                    stats.record_hit()
                    auto_tune_counter += 1
                    
                    # 定期的な自動調整
                    if auto_tune_counter % auto_tune_threshold == 0:
                        tune_result = cache.auto_tune_cache_settings()
                        if tune_result["adjusted"]:
                            logger.info(f"キャッシュ自動調整: {tune_result['adjustments']}")
                    
                    return cached_result

                # キャッシュミス - 関数実行
                stats.record_miss()
                result = func(*args, **kwargs)

                if result is not None:
                    sanitized_result = sanitize_cache_value(result)
                    cache.set(cache_key, sanitized_result)
                    stats.record_set()

                auto_tune_counter += 1
                return result

            except Exception as e:
                logger.error(f"適応キャッシュエラー: {e}")
                stats.record_error()
                return func(*args, **kwargs)

        # 管理メソッド追加
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.get_cache_info = cache.get_cache_info
        wrapper.force_tune = cache.auto_tune_cache_settings

        return wrapper

    return decorator


def smart_cache(
    ttl_seconds: int = 300,
    max_size: int = 1000,
    error_ttl: int = 60,
    enable_fallback: bool = True,
) -> Callable:
    """
    スマートキャッシュデコレータ
    
    エラー処理とフォールバック機能を強化

    Args:
        ttl_seconds: 通常のTTL（秒）
        max_size: 最大キャッシュサイズ
        error_ttl: エラー時の短縮TTL（秒）
        enable_fallback: staleフォールバック有効化

    Returns:
        デコレートされた関数
    """
    main_cache = DataCache(ttl_seconds, max_size, ttl_seconds * 3)
    error_cache = DataCache(error_ttl, max_size // 4, error_ttl * 2)
    stats = CacheStats()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                cache_key = generate_safe_cache_key(func.__name__, *args, **kwargs)
                
                if not validate_cache_key(cache_key):
                    return func(*args, **kwargs)

                # メインキャッシュ確認
                cached_result = main_cache.get(cache_key, allow_stale=False)
                if cached_result is not None:
                    stats.record_hit()
                    return cached_result

                # エラーキャッシュ確認（前回エラーだった場合の短縮チェック）
                error_result = error_cache.get(cache_key, allow_stale=False)
                if error_result is not None and isinstance(error_result, Exception):
                    logger.debug(f"エラーキャッシュヒット、スキップ: {func.__name__}")
                    raise error_result

                # 関数実行
                try:
                    stats.record_miss()
                    result = func(*args, **kwargs)
                    
                    if result is not None:
                        sanitized_result = sanitize_cache_value(result)
                        main_cache.set(cache_key, sanitized_result)
                        stats.record_set()
                        
                        # エラーキャッシュから削除（成功したため）
                        if error_cache.get(cache_key) is not None:
                            error_cache._remove_key(cache_key)
                    
                    return result

                except (APIError, NetworkError, DataError) as api_error:
                    # エラーをキャッシュして短期間スキップ
                    error_cache.set(cache_key, api_error)
                    
                    # フォールバック試行
                    if enable_fallback:
                        stale_result = main_cache.get(cache_key, allow_stale=True)
                        if stale_result is not None:
                            logger.warning(f"フォールバック使用: {func.__name__}")
                            stats.record_fallback()
                            return stale_result
                    
                    raise api_error

            except Exception as e:
                logger.error(f"スマートキャッシュエラー: {e}")
                stats.record_error()
                raise e

        # 管理メソッド
        wrapper.clear_cache = lambda: (main_cache.clear(), error_cache.clear())
        wrapper.cache_size = lambda: main_cache.size() + error_cache.size()
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.get_main_cache_info = main_cache.get_cache_info
        wrapper.get_error_cache_info = error_cache.get_cache_info

        return wrapper

    return decorator


def memory_efficient_cache(
    ttl_seconds: int = 300,
    max_memory_mb: int = 50,
    compression_threshold: int = 1024,
) -> Callable:
    """
    メモリ効率重視キャッシュデコレータ
    
    メモリ使用量を監視し、必要に応じて圧縮を実行

    Args:
        ttl_seconds: TTL（秒）
        max_memory_mb: 最大メモリ使用量（MB）
        compression_threshold: 圧縮閾値（バイト）

    Returns:
        デコレートされた関数
    """
    import sys
    import pickle
    import gzip
    
    # 動的サイズ計算（概算）
    estimated_entry_size = 512  # バイト
    max_size = (max_memory_mb * 1024 * 1024) // estimated_entry_size
    
    cache = DataCache(ttl_seconds, max_size, ttl_seconds)
    stats = CacheStats()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                cache_key = generate_safe_cache_key(func.__name__, *args, **kwargs)
                
                # キャッシュ取得と解凍
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    # 圧縮されているかチェック
                    if isinstance(cached_result, dict) and cached_result.get('compressed'):
                        try:
                            decompressed = gzip.decompress(cached_result['data'])
                            result = pickle.loads(decompressed)
                            stats.record_hit()
                            return result
                        except Exception as e:
                            logger.warning(f"解凍エラー: {e}")
                    else:
                        stats.record_hit()
                        return cached_result

                # 関数実行
                stats.record_miss()
                result = func(*args, **kwargs)

                if result is not None:
                    # メモリ使用量推定
                    try:
                        serialized = pickle.dumps(result)
                        data_size = len(serialized)
                        
                        # 圧縮判定
                        if data_size > compression_threshold:
                            compressed_data = gzip.compress(serialized)
                            if len(compressed_data) < data_size * 0.8:  # 20%以上圧縮できた場合
                                cached_data = {
                                    'compressed': True,
                                    'data': compressed_data,
                                    'original_size': data_size,
                                    'compressed_size': len(compressed_data)
                                }
                                logger.debug(f"データ圧縮: {data_size}→{len(compressed_data)}バイト")
                            else:
                                cached_data = result
                        else:
                            cached_data = result

                        cache.set(cache_key, cached_data)
                        stats.record_set()
                        
                    except Exception as e:
                        logger.warning(f"キャッシュ保存エラー: {e}")

                return result

            except Exception as e:
                logger.error(f"メモリ効率キャッシュエラー: {e}")
                stats.record_error()
                return func(*args, **kwargs)

        # 管理メソッド
        wrapper.clear_cache = cache.clear
        wrapper.cache_size = cache.size
        wrapper.get_stats = lambda: stats.to_dict()
        wrapper.get_cache_info = cache.get_cache_info
        wrapper.estimate_memory_usage = lambda: cache.size() * estimated_entry_size

        return wrapper

    return decorator