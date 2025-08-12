"""
統一例外処理ハンドラ

例外処理の標準化とログ出力の統一を提供します。
"""

import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

from .exceptions import (
    AnalysisError,
    APIError,
    DataError,
    DayTradeError,
    handle_database_exception,
    handle_network_exception,
)
from .logging_config import get_context_logger


class ExceptionContext:
    """例外コンテキスト管理クラス"""

    def __init__(self, component: str, operation: str, logger_name: Optional[str] = None):
        self.component = component
        self.operation = operation
        self.logger = get_context_logger(logger_name or f"{component}.{operation}")

    def handle(self, exc: Exception) -> DayTradeError:
        """
        標準例外をDayTradeError例外に変換

        Args:
            exc: 変換対象の例外

        Returns:
            変換されたDayTradeError例外
        """
        # 既にDayTradeErrorの場合はそのまま返す
        if isinstance(exc, DayTradeError):
            return exc

        try:
            import requests.exceptions
            import sqlalchemy.exc

            # ネットワーク関連例外
            if isinstance(exc, requests.exceptions.RequestException):
                return handle_network_exception(exc)

            # データベース関連例外
            if isinstance(exc, sqlalchemy.exc.SQLAlchemyError):
                return handle_database_exception(exc)

        except ImportError:
            # オプショナル依存関係が利用できない場合は汎用処理
            pass

        # データ関連例外
        if isinstance(exc, (ValueError, TypeError)):
            return DataError(
                message=f"データ処理エラー: {exc}",
                error_code="DATA_PROCESSING_ERROR",
                details={
                    "component": self.component,
                    "operation": self.operation,
                    "original_error": str(exc),
                },
            )

        # ファイル関連例外
        if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
            from .exceptions import FileOperationError

            return FileOperationError(
                message=f"ファイル操作エラー: {exc}",
                error_code="FILE_OPERATION_ERROR",
                details={
                    "component": self.component,
                    "operation": self.operation,
                    "original_error": str(exc),
                },
            )

        # メモリ関連例外
        if isinstance(exc, MemoryError):
            return AnalysisError(
                message=f"メモリ不足エラー: {exc}",
                error_code="MEMORY_ERROR",
                details={
                    "component": self.component,
                    "operation": self.operation,
                    "original_error": str(exc),
                },
            )

        # その他の例外は汎用エラーとして処理
        return AnalysisError(
            message=f"{self.component}で予期しないエラーが発生: {exc}",
            error_code="UNEXPECTED_ERROR",
            details={
                "component": self.component,
                "operation": self.operation,
                "original_error": str(exc),
                "exception_type": exc.__class__.__name__,
            },
        )


def log_exception(
    logger,
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error",
) -> None:
    """
    統一例外ログ出力

    Args:
        logger: ログ出力用ロガー
        exc: ログ対象例外
        context: 追加コンテキスト情報
        level: ログレベル (error, warning, critical)
    """
    context = context or {}

    if isinstance(exc, DayTradeError):
        # ビジネス例外の場合
        log_data = {
            **exc.to_dict(),
            **context,
        }
        getattr(logger, level)("ビジネスエラー発生", extra=log_data)
    else:
        # システム例外の場合
        log_data = {
            "error_message": str(exc),
            "error_type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
            **context,
        }
        logger.critical("システムエラー発生", extra=log_data)


def with_exception_handling(component: str, operation: str, reraise: bool = True) -> Callable:
    """
    例外処理デコレータ

    Args:
        component: コンポーネント名
        operation: 操作名
        reraise: 例外を再発生するかどうか

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ExceptionContext(component, operation)
            try:
                return await func(*args, **kwargs)
            except DayTradeError:
                raise  # すでに変換済みは再スロー
            except Exception as e:
                converted_exc = context.handle(e)
                log_exception(
                    context.logger,
                    converted_exc,
                    {"component": component, "operation": operation},
                )
                if reraise:
                    raise converted_exc
                return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = ExceptionContext(component, operation)
            try:
                return func(*args, **kwargs)
            except DayTradeError:
                raise  # すでに変換済みは再スロー
            except Exception as e:
                converted_exc = context.handle(e)
                log_exception(
                    context.logger,
                    converted_exc,
                    {"component": component, "operation": operation},
                )
                if reraise:
                    raise converted_exc
                return None

        # 関数がasync関数かどうかを判定
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ErrorRecoveryStrategy:
    """エラー回復戦略クラス"""

    def __init__(self):
        self.recovery_strategies = {}

    def register_strategy(self, error_type: Type[Exception], strategy: Callable[[Exception], Any]):
        """
        エラータイプに対する回復戦略を登録

        Args:
            error_type: 対象エラータイプ
            strategy: 回復戦略関数
        """
        self.recovery_strategies[error_type] = strategy

    def recover(self, exc: Exception) -> Any:
        """
        例外からの回復を試行

        Args:
            exc: 回復対象例外

        Returns:
            回復結果（回復不可能な場合はNone）
        """
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(exc, error_type):
                try:
                    return strategy(exc)
                except Exception:
                    # 回復戦略自体が失敗した場合は継続
                    continue
        return None


# グローバル回復戦略インスタンス
global_recovery_strategy = ErrorRecoveryStrategy()


def with_recovery(component: str, operation: str) -> Callable:
    """
    エラー回復機能付きデコレータ

    Args:
        component: コンポーネント名
        operation: 操作名

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ExceptionContext(component, operation)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # まず回復を試行
                recovery_result = global_recovery_strategy.recover(e)
                if recovery_result is not None:
                    context.logger.info(
                        f"エラーからの回復に成功: {e}",
                        extra={"recovery_result": str(recovery_result)},
                    )
                    return recovery_result

                # 回復不可能な場合は例外変換・ログ出力
                converted_exc = context.handle(e)
                log_exception(
                    context.logger,
                    converted_exc,
                    {"component": component, "operation": operation},
                )
                raise converted_exc

        return wrapper

    return decorator


# デフォルト回復戦略の登録
def register_default_recovery_strategies():
    """デフォルトの回復戦略を登録"""

    def network_retry(exc):
        """ネットワークエラー時の再試行戦略"""
        import time

        time.sleep(1)  # 1秒待機
        return "RETRY_REQUIRED"

    def data_fallback(exc):
        """データエラー時のフォールバック戦略"""
        return "FALLBACK_DATA"

    global_recovery_strategy.register_strategy(APIError, network_retry)
    global_recovery_strategy.register_strategy(DataError, data_fallback)


# 初期化時に戦略を登録
register_default_recovery_strategies()
