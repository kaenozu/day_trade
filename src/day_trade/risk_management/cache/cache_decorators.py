#!/usr/bin/env python3
"""
Cache Decorators
キャッシュデコレーター

関数レベルでのキャッシュ機能を提供
"""

import asyncio
import functools
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

from ..exceptions.risk_exceptions import CacheError
from ..interfaces.cache_interfaces import ICacheProvider

# グローバルキャッシュプロバイダー
_default_cache_provider: Optional[ICacheProvider] = None


def set_default_cache_provider(provider: ICacheProvider):
    """デフォルトキャッシュプロバイダー設定"""
    global _default_cache_provider
    _default_cache_provider = provider


def get_default_cache_provider() -> Optional[ICacheProvider]:
    """デフォルトキャッシュプロバイダー取得"""
    return _default_cache_provider


def cache_result(
    ttl_seconds: int = 3600,
    key_prefix: str = "",
    cache_provider: Optional[ICacheProvider] = None,
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[List[int]] = None,
    exclude_kwargs: Optional[List[str]] = None,
    key_func: Optional[Callable] = None,
    condition: Optional[Callable] = None,
    on_cache_hit: Optional[Callable] = None,
    on_cache_miss: Optional[Callable] = None,
):
    """
    関数結果キャッシュデコレーター

    Args:
        ttl_seconds: キャッシュ有効期間（秒）
        key_prefix: キーのプレフィックス
        cache_provider: 使用するキャッシュプロバイダー
        include_args: 位置引数をキーに含めるか
        include_kwargs: キーワード引数をキーに含めるか
        exclude_args: キーから除外する位置引数のインデックス
        exclude_kwargs: キーから除外するキーワード引数名
        key_func: カスタムキー生成関数
        condition: キャッシュ条件関数（結果を引数に取る）
        on_cache_hit: キャッシュヒット時のコールバック
        on_cache_miss: キャッシュミス時のコールバック
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return await func(*args, **kwargs)

            # キャッシュキー生成
            cache_key = _generate_cache_key(
                func,
                args,
                kwargs,
                key_prefix,
                include_args,
                include_kwargs,
                exclude_args,
                exclude_kwargs,
                key_func,
            )

            try:
                # キャッシュから取得試行
                cached_result = provider.get(cache_key)
                if cached_result is not None:
                    if on_cache_hit:
                        on_cache_hit(cache_key, cached_result)
                    return cached_result

                # キャッシュミス - 関数実行
                result = await func(*args, **kwargs)

                # キャッシュ条件チェック
                if condition is None or condition(result):
                    provider.set(cache_key, result, ttl_seconds)

                if on_cache_miss:
                    on_cache_miss(cache_key, result)

                return result

            except CacheError:
                # キャッシュエラー時は関数を直接実行
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return func(*args, **kwargs)

            # キャッシュキー生成
            cache_key = _generate_cache_key(
                func,
                args,
                kwargs,
                key_prefix,
                include_args,
                include_kwargs,
                exclude_args,
                exclude_kwargs,
                key_func,
            )

            try:
                # キャッシュから取得試行
                cached_result = provider.get(cache_key)
                if cached_result is not None:
                    if on_cache_hit:
                        on_cache_hit(cache_key, cached_result)
                    return cached_result

                # キャッシュミス - 関数実行
                result = func(*args, **kwargs)

                # キャッシュ条件チェック
                if condition is None or condition(result):
                    provider.set(cache_key, result, ttl_seconds)

                if on_cache_miss:
                    on_cache_miss(cache_key, result)

                return result

            except CacheError:
                # キャッシュエラー時は関数を直接実行
                return func(*args, **kwargs)

        # 非同期関数かどうかで分岐
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper

        # メタデータ追加
        wrapper._cached_function = True
        wrapper._cache_ttl = ttl_seconds
        wrapper._cache_key_prefix = key_prefix

        return wrapper

    return decorator


def invalidate_cache(key_pattern: str = "*", cache_provider: Optional[ICacheProvider] = None):
    """
    キャッシュ無効化デコレーター

    Args:
        key_pattern: 無効化するキーのパターン
        cache_provider: 使用するキャッシュプロバイダー
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # 関数実行後にキャッシュを無効化
            provider = cache_provider or get_default_cache_provider()
            if provider:
                try:
                    _invalidate_matching_keys(provider, key_pattern, func, args, kwargs)
                except CacheError:
                    # 無効化エラーは無視
                    pass

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # 関数実行後にキャッシュを無効化
            provider = cache_provider or get_default_cache_provider()
            if provider:
                try:
                    _invalidate_matching_keys(provider, key_pattern, func, args, kwargs)
                except CacheError:
                    # 無効化エラーは無視
                    pass

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_aside(
    get_func: Callable,
    set_func: Optional[Callable] = None,
    ttl_seconds: int = 3600,
    cache_provider: Optional[ICacheProvider] = None,
):
    """
    Cache-Aside パターンデコレーター

    Args:
        get_func: データ取得関数
        set_func: データ更新関数（オプション）
        ttl_seconds: キャッシュ有効期間
        cache_provider: キャッシュプロバイダー
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return await func(*args, **kwargs)

            # キー生成
            cache_key = _generate_cache_key(func, args, kwargs)

            try:
                # 1. キャッシュから取得試行
                cached_data = provider.get(cache_key)
                if cached_data is not None:
                    return cached_data

                # 2. キャッシュミス - データソースから取得
                data = await get_func(*args, **kwargs)

                # 3. キャッシュに保存
                if data is not None:
                    provider.set(cache_key, data, ttl_seconds)

                return data

            except CacheError:
                # キャッシュエラー時は直接データソースから取得
                return await get_func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return func(*args, **kwargs)

            cache_key = _generate_cache_key(func, args, kwargs)

            try:
                # 1. キャッシュから取得試行
                cached_data = provider.get(cache_key)
                if cached_data is not None:
                    return cached_data

                # 2. キャッシュミス - データソースから取得
                data = get_func(*args, **kwargs)

                # 3. キャッシュに保存
                if data is not None:
                    provider.set(cache_key, data, ttl_seconds)

                return data

            except CacheError:
                # キャッシュエラー時は直接データソースから取得
                return get_func(*args, **kwargs)

        if asyncio.iscoroutinefunction(get_func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cached_property(ttl_seconds: int = 3600, cache_provider: Optional[ICacheProvider] = None):
    """
    キャッシュ付きプロパティデコレーター

    Args:
        ttl_seconds: キャッシュ有効期間
        cache_provider: キャッシュプロバイダー
    """

    def decorator(func: Callable) -> property:
        @functools.wraps(func)
        def wrapper(self):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return func(self)

            # オブジェクトIDとプロパティ名でキー生成
            cache_key = f"{self.__class__.__name__}:{id(self)}:{func.__name__}"

            try:
                # キャッシュから取得試行
                cached_value = provider.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # キャッシュミス - プロパティ計算
                value = func(self)

                # キャッシュに保存
                provider.set(cache_key, value, ttl_seconds)

                return value

            except CacheError:
                # キャッシュエラー時は直接計算
                return func(self)

        return property(wrapper)

    return decorator


def cache_with_lock(
    ttl_seconds: int = 3600,
    lock_timeout_seconds: int = 30,
    cache_provider: Optional[ICacheProvider] = None,
):
    """
    分散ロック付きキャッシュデコレーター

    Args:
        ttl_seconds: キャッシュ有効期間
        lock_timeout_seconds: ロックタイムアウト
        cache_provider: キャッシュプロバイダー
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return await func(*args, **kwargs)

            cache_key = _generate_cache_key(func, args, kwargs)
            lock_key = f"lock:{cache_key}"

            try:
                # キャッシュから取得試行
                cached_result = provider.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # ロック取得試行
                lock_acquired = False
                start_time = time.time()

                while time.time() - start_time < lock_timeout_seconds:
                    if provider.set(lock_key, "locked", 5):  # 5秒のロック
                        lock_acquired = True
                        break
                    await asyncio.sleep(0.1)  # 100ms待機

                if not lock_acquired:
                    # ロック取得失敗 - 関数を直接実行
                    return await func(*args, **kwargs)

                try:
                    # ダブルチェック - ロック取得中に他のプロセスがキャッシュした可能性
                    cached_result = provider.get(cache_key)
                    if cached_result is not None:
                        return cached_result

                    # 関数実行してキャッシュに保存
                    result = await func(*args, **kwargs)
                    provider.set(cache_key, result, ttl_seconds)

                    return result

                finally:
                    # ロック解放
                    provider.delete(lock_key)

            except CacheError:
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None:
                return func(*args, **kwargs)

            cache_key = _generate_cache_key(func, args, kwargs)
            lock_key = f"lock:{cache_key}"

            try:
                # キャッシュから取得試行
                cached_result = provider.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # ロック取得試行（同期版は簡略化）
                if provider.set(lock_key, "locked", lock_timeout_seconds):
                    try:
                        # ダブルチェック
                        cached_result = provider.get(cache_key)
                        if cached_result is not None:
                            return cached_result

                        # 関数実行してキャッシュに保存
                        result = func(*args, **kwargs)
                        provider.set(cache_key, result, ttl_seconds)

                        return result

                    finally:
                        # ロック解放
                        provider.delete(lock_key)
                else:
                    # ロック取得失敗
                    return func(*args, **kwargs)

            except CacheError:
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def bulk_cache(
    batch_size: int = 100,
    ttl_seconds: int = 3600,
    cache_provider: Optional[ICacheProvider] = None,
):
    """
    一括キャッシュデコレーター

    Args:
        batch_size: バッチサイズ
        ttl_seconds: キャッシュ有効期間
        cache_provider: キャッシュプロバイダー
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            provider = cache_provider or get_default_cache_provider()
            if provider is None or not hasattr(provider, "multi_get"):
                return func(items, *args, **kwargs)

            try:
                # キャッシュキー生成
                cache_keys = []
                key_to_item = {}

                for item in items:
                    cache_key = _generate_cache_key(func, (item, *args), kwargs)
                    cache_keys.append(cache_key)
                    key_to_item[cache_key] = item

                # 一括キャッシュ取得
                cached_results = provider.multi_get(cache_keys)

                # キャッシュミスのアイテムを特定
                missing_items = []
                results = {}

                for cache_key, item in key_to_item.items():
                    if cache_key in cached_results:
                        results[item] = cached_results[cache_key]
                    else:
                        missing_items.append(item)

                # キャッシュミスのアイテムに対して関数実行
                if missing_items:
                    missing_results = func(missing_items, *args, **kwargs)

                    # 結果をキャッシュに保存
                    cache_data = {}
                    for item, result in zip(missing_items, missing_results):
                        cache_key = _generate_cache_key(func, (item, *args), kwargs)
                        cache_data[cache_key] = result
                        results[item] = result

                    if hasattr(provider, "multi_set"):
                        provider.multi_set(cache_data, ttl_seconds)
                    else:
                        # 個別設定
                        for key, value in cache_data.items():
                            provider.set(key, value, ttl_seconds)

                # 元の順序で結果を返す
                return [results[item] for item in items]

            except CacheError:
                return func(items, *args, **kwargs)

        return wrapper

    return decorator


# ヘルパー関数


def _generate_cache_key(
    func: Callable,
    args: tuple = (),
    kwargs: Dict[str, Any] = None,
    key_prefix: str = "",
    include_args: bool = True,
    include_kwargs: bool = True,
    exclude_args: Optional[List[int]] = None,
    exclude_kwargs: Optional[List[str]] = None,
    key_func: Optional[Callable] = None,
) -> str:
    """キャッシュキー生成"""

    if key_func:
        return key_func(*args, **(kwargs or {}))

    if kwargs is None:
        kwargs = {}

    # 基本キー情報
    key_parts = []

    # プレフィックス
    if key_prefix:
        key_parts.append(key_prefix)

    # 関数名
    key_parts.append(f"{func.__module__}.{func.__name__}")

    # 位置引数
    if include_args and args:
        filtered_args = []
        for i, arg in enumerate(args):
            if exclude_args is None or i not in exclude_args:
                filtered_args.append(_serialize_for_key(arg))
        if filtered_args:
            key_parts.append(f"args:{':'.join(filtered_args)}")

    # キーワード引数
    if include_kwargs and kwargs:
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if exclude_kwargs is None or key not in exclude_kwargs:
                filtered_kwargs[key] = _serialize_for_key(value)

        if filtered_kwargs:
            sorted_kwargs = sorted(filtered_kwargs.items())
            kwargs_str = ":".join(f"{k}={v}" for k, v in sorted_kwargs)
            key_parts.append(f"kwargs:{kwargs_str}")

    # キーを結合してハッシュ化
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()


def _serialize_for_key(value: Any) -> str:
    """キー用のシリアライゼーション"""
    if value is None:
        return "None"
    elif isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        return f"[{','.join(_serialize_for_key(item) for item in value)}]"
    elif isinstance(value, dict):
        items = sorted(value.items())
        return f"{{{','.join(f'{k}:{_serialize_for_key(v)}' for k, v in items)}}}"
    elif hasattr(value, "__dict__"):
        # オブジェクトの場合は属性をシリアライズ
        attrs = sorted(value.__dict__.items())
        return f"obj:{value.__class__.__name__}:{{{','.join(f'{k}:{_serialize_for_key(v)}' for k, v in attrs)}}}"
    else:
        # その他の場合は文字列表現
        return f"obj:{type(value).__name__}:{str(value)}"


def _invalidate_matching_keys(
    provider: ICacheProvider,
    pattern: str,
    func: Callable,
    args: tuple,
    kwargs: Dict[str, Any],
):
    """パターンに一致するキーを無効化"""

    try:
        # パターンに一致するキーを取得
        if hasattr(provider, "keys"):
            matching_keys = provider.keys(pattern)

            # キーを削除
            for key in matching_keys:
                provider.delete(key)
        else:
            # keys メソッドがない場合は個別キー削除のみ
            if pattern == "*":
                # 全削除はclearメソッド使用
                if hasattr(provider, "clear"):
                    provider.clear()

    except Exception:
        # 無効化エラーは無視
        pass


# キャッシュ統計情報取得


def get_cache_stats(func: Callable) -> Optional[Dict[str, Any]]:
    """キャッシュされた関数の統計情報取得"""

    if not hasattr(func, "_cached_function"):
        return None

    provider = get_default_cache_provider()
    if provider is None or not hasattr(provider, "get_stats"):
        return None

    try:
        return provider.get_stats()
    except CacheError:
        return None


def clear_function_cache(func: Callable, cache_provider: Optional[ICacheProvider] = None):
    """特定関数のキャッシュをクリア"""

    provider = cache_provider or get_default_cache_provider()
    if provider is None:
        return

    # 関数名を含むキーをすべて削除
    pattern = f"*{func.__module__}.{func.__name__}*"
    _invalidate_matching_keys(provider, pattern, func, (), {})
