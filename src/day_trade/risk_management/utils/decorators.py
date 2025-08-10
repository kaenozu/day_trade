#!/usr/bin/env python3
"""
Risk Management Decorators
リスク管理デコレーター

統一されたエラーハンドリング、メトリクス収集、キャッシュ等を提供
"""

import functools
import asyncio
import time
from typing import Any, Callable, Dict, Optional, Type, Union
from datetime import datetime, timedelta

from ..exceptions.risk_exceptions import (
    RiskManagementError,
    TimeoutError,
    RateLimitError,
    AnalysisError
)
from ..interfaces.risk_interfaces import IMetricsCollector, ICacheManager

def error_handler(
    fallback_value: Any = None,
    catch_exceptions: tuple = (Exception,),
    log_errors: bool = True,
    raise_on_error: bool = False,
    error_metrics_name: Optional[str] = None
):
    """統一エラーハンドリングデコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # 成功メトリクス記録
                if error_metrics_name:
                    # メトリクス収集器があれば記録
                    pass

                return result

            except catch_exceptions as e:
                processing_time = time.time() - start_time

                # エラーログ記録
                if log_errors:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'args': str(args)[:200],  # 長すぎる引数は切り詰め
                            'processing_time': processing_time,
                            'error_type': type(e).__name__
                        }
                    )

                # エラーメトリクス記録
                if error_metrics_name:
                    # メトリクス収集器があれば記録
                    pass

                # エラー再発生またはフォールバック値返却
                if raise_on_error:
                    # RiskManagementError でない場合はラップ
                    if not isinstance(e, RiskManagementError):
                        raise AnalysisError(
                            message=f"Error in {func.__name__}: {str(e)}",
                            analyzer_name=func.__name__,
                            analysis_stage="execution",
                            cause=e
                        ) from e
                    raise
                else:
                    return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result

            except catch_exceptions as e:
                processing_time = time.time() - start_time

                if log_errors:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'args': str(args)[:200],
                            'processing_time': processing_time,
                            'error_type': type(e).__name__
                        }
                    )

                if raise_on_error:
                    if not isinstance(e, RiskManagementError):
                        raise AnalysisError(
                            message=f"Error in {func.__name__}: {str(e)}",
                            analyzer_name=func.__name__,
                            analysis_stage="execution",
                            cause=e
                        ) from e
                    raise
                else:
                    return fallback_value

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def timeout_handler(
    timeout_seconds: float,
    error_message: Optional[str] = None
):
    """タイムアウトハンドリングデコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                message = error_message or f"Function {func.__name__} timed out after {timeout_seconds} seconds"
                raise TimeoutError(
                    message=message,
                    operation=func.__name__,
                    timeout_seconds=timeout_seconds
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同期関数のタイムアウトは別途実装が必要
            # ここでは簡易版として直接実行
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def metrics_collector(
    counter_name: Optional[str] = None,
    histogram_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    metrics_collector: Optional[IMetricsCollector] = None
):
    """メトリクス収集デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            # カウンターメトリクス（実行開始）
            if metrics_collector and counter_name:
                metrics_collector.record_counter(
                    counter_name + "_total",
                    labels={**(labels or {}), "function": func.__name__}
                )

            try:
                result = await func(*args, **kwargs)

                # 成功メトリクス
                processing_time = time.time() - start_time

                if metrics_collector:
                    if histogram_name:
                        metrics_collector.record_histogram(
                            histogram_name + "_duration_seconds",
                            processing_time,
                            labels={**(labels or {}), "function": func.__name__, "status": "success"}
                        )

                    if counter_name:
                        metrics_collector.record_counter(
                            counter_name + "_success_total",
                            labels={**(labels or {}), "function": func.__name__}
                        )

                return result

            except Exception as e:
                # エラーメトリクス
                processing_time = time.time() - start_time

                if metrics_collector:
                    if histogram_name:
                        metrics_collector.record_histogram(
                            histogram_name + "_duration_seconds",
                            processing_time,
                            labels={**(labels or {}), "function": func.__name__, "status": "error"}
                        )

                    if counter_name:
                        metrics_collector.record_counter(
                            counter_name + "_error_total",
                            labels={**(labels or {}), "function": func.__name__, "error_type": type(e).__name__}
                        )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            if metrics_collector and counter_name:
                metrics_collector.record_counter(
                    counter_name + "_total",
                    labels={**(labels or {}), "function": func.__name__}
                )

            try:
                result = func(*args, **kwargs)

                processing_time = time.time() - start_time

                if metrics_collector:
                    if histogram_name:
                        metrics_collector.record_histogram(
                            histogram_name + "_duration_seconds",
                            processing_time,
                            labels={**(labels or {}), "function": func.__name__, "status": "success"}
                        )

                    if counter_name:
                        metrics_collector.record_counter(
                            counter_name + "_success_total",
                            labels={**(labels or {}), "function": func.__name__}
                        )

                return result

            except Exception as e:
                processing_time = time.time() - start_time

                if metrics_collector:
                    if histogram_name:
                        metrics_collector.record_histogram(
                            histogram_name + "_duration_seconds",
                            processing_time,
                            labels={**(labels or {}), "function": func.__name__, "status": "error"}
                        )

                    if counter_name:
                        metrics_collector.record_counter(
                            counter_name + "_error_total",
                            labels={**(labels or {}), "function": func.__name__, "error_type": type(e).__name__}
                        )

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def cache_result(
    cache_key_template: str,
    ttl_seconds: int = 3600,
    cache_manager: Optional[ICacheManager] = None,
    use_args_in_key: bool = True,
    key_args_indices: Optional[list] = None
):
    """結果キャッシュデコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if cache_manager is None:
                # キャッシュが無効な場合は直接実行
                return await func(*args, **kwargs)

            # キャッシュキー生成
            cache_key = _generate_cache_key(
                cache_key_template,
                func.__name__,
                args,
                kwargs,
                use_args_in_key,
                key_args_indices
            )

            # キャッシュから取得を試行
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # キャッシュミスの場合は関数実行
            result = await func(*args, **kwargs)

            # 結果をキャッシュに保存
            await cache_manager.set(cache_key, result, ttl_seconds)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同期版キャッシュは簡易実装
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def rate_limit(
    max_calls: int,
    window_seconds: int,
    key_func: Optional[Callable] = None
):
    """レート制限デコレーター"""

    # レート制限状態を保持する辞書
    _rate_limit_state: Dict[str, Dict] = {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # レート制限キー生成
            if key_func:
                limit_key = key_func(*args, **kwargs)
            else:
                limit_key = f"{func.__module__}.{func.__name__}"

            current_time = time.time()

            # レート制限状態初期化
            if limit_key not in _rate_limit_state:
                _rate_limit_state[limit_key] = {
                    'calls': [],
                    'window_start': current_time
                }

            state = _rate_limit_state[limit_key]

            # 古い記録をクリーンアップ
            cutoff_time = current_time - window_seconds
            state['calls'] = [call_time for call_time in state['calls'] if call_time > cutoff_time]

            # レート制限チェック
            if len(state['calls']) >= max_calls:
                oldest_call = min(state['calls'])
                retry_after = int(oldest_call + window_seconds - current_time)
                raise RateLimitError(
                    message=f"Rate limit exceeded for {func.__name__}",
                    rate_limit=max_calls,
                    retry_after_seconds=max(retry_after, 1)
                )

            # 呼び出し記録
            state['calls'].append(current_time)

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同期版も同様の実装
            if key_func:
                limit_key = key_func(*args, **kwargs)
            else:
                limit_key = f"{func.__module__}.{func.__name__}"

            current_time = time.time()

            if limit_key not in _rate_limit_state:
                _rate_limit_state[limit_key] = {
                    'calls': [],
                    'window_start': current_time
                }

            state = _rate_limit_state[limit_key]
            cutoff_time = current_time - window_seconds
            state['calls'] = [call_time for call_time in state['calls'] if call_time > cutoff_time]

            if len(state['calls']) >= max_calls:
                oldest_call = min(state['calls'])
                retry_after = int(oldest_call + window_seconds - current_time)
                raise RateLimitError(
                    message=f"Rate limit exceeded for {func.__name__}",
                    rate_limit=max_calls,
                    retry_after_seconds=max(retry_after, 1)
                )

            state['calls'].append(current_time)

            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

def retry_on_failure(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    retry_exceptions: tuple = (Exception,)
):
    """失敗時リトライデコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except retry_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = delay_seconds * (backoff_multiplier ** attempt)
                        await asyncio.sleep(delay)

                        # ログ記録
                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}"
                        )
                    else:
                        break

            # すべてのリトライが失敗した場合
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retry_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = delay_seconds * (backoff_multiplier ** attempt)
                        time.sleep(delay)

                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}"
                        )
                    else:
                        break

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

# ヘルパー関数

def _generate_cache_key(
    template: str,
    func_name: str,
    args: tuple,
    kwargs: dict,
    use_args: bool,
    key_args_indices: Optional[list]
) -> str:
    """キャッシュキー生成"""
    import hashlib

    key_parts = [template, func_name]

    if use_args:
        if key_args_indices:
            # 指定されたインデックスの引数のみ使用
            selected_args = [args[i] for i in key_args_indices if i < len(args)]
            key_parts.extend([str(arg) for arg in selected_args])
        else:
            # すべての引数を使用
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

    # ハッシュ化してキー長を制限
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

# 組み合わせデコレーター

def risk_analysis_handler(
    analyzer_name: str,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    cache_ttl_seconds: int = 300
):
    """リスク分析用統合デコレーター"""

    def decorator(func: Callable) -> Callable:
        # 複数のデコレーターを組み合わせ
        decorated_func = func

        # エラーハンドリング
        decorated_func = error_handler(
            fallback_value=None,
            catch_exceptions=(Exception,),
            log_errors=True,
            raise_on_error=True,
            error_metrics_name=f"{analyzer_name}_analysis"
        )(decorated_func)

        # タイムアウト
        decorated_func = timeout_handler(
            timeout_seconds=timeout_seconds,
            error_message=f"{analyzer_name} analysis timed out"
        )(decorated_func)

        # リトライ
        decorated_func = retry_on_failure(
            max_retries=max_retries,
            delay_seconds=1.0,
            backoff_multiplier=1.5
        )(decorated_func)

        # メトリクス収集
        decorated_func = metrics_collector(
            counter_name=f"{analyzer_name}_analysis",
            histogram_name=f"{analyzer_name}_analysis"
        )(decorated_func)

        return decorated_func

    return decorator
