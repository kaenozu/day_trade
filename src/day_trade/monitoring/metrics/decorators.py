#!/usr/bin/env python3
"""
メトリクス収集デコレーター
関数・メソッドの実行時間・呼び出し回数・エラー率を自動収集
"""

import functools
import time
from typing import Any, Callable

from ...utils.logging_config import get_context_logger
from .prometheus_metrics import (
    get_ai_metrics,
    get_health_metrics,
    get_metrics_collector,
    get_risk_metrics,
    get_trading_metrics,
)

logger = get_context_logger(__name__)


def measure_execution_time(component: str = "general"):
    """実行時間測定デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # メトリクス記録
                collector = get_metrics_collector()
                collector.metrics_collection_duration.labels(collector_type=component).observe(
                    execution_time
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"関数実行エラー {func.__name__}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(error_type=type(e).__name__, component=component)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # メトリクス記録
                collector = get_metrics_collector()
                collector.metrics_collection_duration.labels(collector_type=component).observe(
                    execution_time
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"関数実行エラー {func.__name__}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(error_type=type(e).__name__, component=component)
                raise

        # 非同期関数の場合
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def count_method_calls(component: str = "general"):
    """メソッド呼び出し回数カウントデコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 呼び出し回数インクリメント
            collector = get_metrics_collector()
            collector.metrics_collection_total.labels(collector_type=component).inc()

            return func(*args, **kwargs)

        return wrapper

    return decorator


def track_errors(component: str = "general"):
    """エラー追跡デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # エラーメトリクス記録
                get_health_metrics().record_error(error_type=type(e).__name__, component=component)
                logger.error(f"追跡対象エラー in {func.__name__}: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # エラーメトリクス記録
                get_health_metrics().record_error(error_type=type(e).__name__, component=component)
                logger.error(f"追跡対象エラー in {func.__name__}: {e}")
                raise

        # 非同期関数判定
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def measure_risk_analysis_performance():
    """リスク分析パフォーマンス測定デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            analysis_type = func.__name__

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # リスクレベル判定
                risk_level = "unknown"
                if hasattr(result, "risk_score"):
                    score = result.risk_score
                    if score >= 0.8:
                        risk_level = "critical"
                    elif score >= 0.6:
                        risk_level = "high"
                    elif score >= 0.3:
                        risk_level = "medium"
                    else:
                        risk_level = "low"
                elif hasattr(result, "overall_risk_score"):
                    score = result.overall_risk_score
                    if score >= 0.8:
                        risk_level = "critical"
                    elif score >= 0.6:
                        risk_level = "high"
                    elif score >= 0.3:
                        risk_level = "medium"
                    else:
                        risk_level = "low"

                # リスク分析メトリクス記録
                get_risk_metrics().record_risk_analysis(
                    analysis_type=analysis_type,
                    risk_level=risk_level,
                    duration=duration,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"リスク分析エラー {analysis_type}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(
                    error_type=type(e).__name__, component="risk_analysis"
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            analysis_type = func.__name__

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # リスクレベル判定
                risk_level = "unknown"
                if hasattr(result, "risk_score"):
                    score = result.risk_score
                    if score >= 0.8:
                        risk_level = "critical"
                    elif score >= 0.6:
                        risk_level = "high"
                    elif score >= 0.3:
                        risk_level = "medium"
                    else:
                        risk_level = "low"

                # リスク分析メトリクス記録
                get_risk_metrics().record_risk_analysis(
                    analysis_type=analysis_type,
                    risk_level=risk_level,
                    duration=duration,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"リスク分析エラー {analysis_type}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(
                    error_type=type(e).__name__, component="risk_analysis"
                )
                raise

        # 非同期関数判定
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def measure_trading_performance():
    """取引パフォーマンス測定デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # 取引実行時間記録
                get_trading_metrics().trade_execution_duration.labels(
                    trade_type=func.__name__
                ).observe(duration)

                # 取引結果記録
                trade_result = "success" if result else "failed"
                get_trading_metrics().trades_total.labels(
                    trade_type=func.__name__, symbol="unknown", result=trade_result
                ).inc()

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"取引実行エラー {func.__name__}: {e}")

                # 失敗した取引記録
                get_trading_metrics().trades_total.labels(
                    trade_type=func.__name__, symbol="unknown", result="error"
                ).inc()

                # エラーメトリクス記録
                get_health_metrics().record_error(error_type=type(e).__name__, component="trading")
                raise

        return async_wrapper

    return decorator


def measure_ai_prediction_performance():
    """AI予測パフォーマンス測定デコレーター"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            model_type = func.__name__

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # AI予測実行記録
                get_ai_metrics().ai_predictions_total.labels(
                    model_type=model_type, symbol="unknown"
                ).inc()

                # AI予測時間記録
                get_ai_metrics().ai_prediction_duration.labels(model_type=model_type).observe(
                    duration
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"AI予測エラー {model_type}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(
                    error_type=type(e).__name__, component="ai_engine"
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            model_type = func.__name__

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # AI予測実行記録
                get_ai_metrics().ai_predictions_total.labels(
                    model_type=model_type, symbol="unknown"
                ).inc()

                # AI予測時間記録
                get_ai_metrics().ai_prediction_duration.labels(model_type=model_type).observe(
                    duration
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"AI予測エラー {model_type}: {e}")

                # エラーメトリクス記録
                get_health_metrics().record_error(
                    error_type=type(e).__name__, component="ai_engine"
                )
                raise

        # 非同期関数判定
        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
