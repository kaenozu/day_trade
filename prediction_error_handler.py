#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Error Handler - 予測エラーハンドリングシステム

統合予測システムの堅牢なエラーハンドリングとログ管理
Issue #855関連：エラーハンドリングとログ強化
"""

import asyncio
import traceback
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

# Windows環境対応
apply_windows_encoding_fix()

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """エラー重要度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    SYSTEM_ERROR = "system_error"
    PREDICTION_ERROR = "prediction_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    PERFORMANCE_ERROR = "performance_error"


@dataclass
class ErrorContext:
    """エラーコンテキスト"""
    symbol: str = ""
    system_id: str = ""
    operation: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: str = ""


@dataclass
class PredictionError:
    """予測エラー情報"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'context': {
                'symbol': self.context.symbol,
                'system_id': self.context.system_id,
                'operation': self.context.operation,
                'timestamp': self.context.timestamp.isoformat(),
                'stack_trace': self.context.stack_trace
            },
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'resolved': self.resolved
        }


class ErrorRecoveryStrategy:
    """エラー回復戦略"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute_with_retry(
            self, operation: Callable, *args, **kwargs
    ) -> Tuple[Any, Optional[PredictionError]]:
        """リトライ付きで操作を実行"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                return result, None

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    # 指数バックオフ
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"操作失敗、{delay}秒後にリトライ "
                        f"(試行 {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"最大リトライ回数に達しました: {e}")

        # 全てのリトライが失敗した場合
        operation_name = (
            operation.__name__ if hasattr(operation, '__name__')
            else str(operation)
        )
        error_context = ErrorContext(
            operation=operation_name,
            stack_trace=traceback.format_exc()
        )

        prediction_error = PredictionError(
            error_id=f"retry_failed_{datetime.now().timestamp()}",
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.ERROR,
            message=f"最大リトライ回数({self.max_retries})に達しました: {last_error}",
            context=error_context,
            retry_count=self.max_retries
        )

        return None, prediction_error


class CircuitBreaker:
    """サーキットブレーカーパターン実装"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """サーキットブレーカー付きで操作を実行"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("サーキットブレーカー: HALF_OPEN状態に移行")
            else:
                raise Exception("サーキットブレーカーが開いています")

        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """リセット試行の判定"""
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.timeout_seconds

    def _on_success(self):
        """成功時の処理"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("サーキットブレーカー: CLOSED状態に復帰")

    def _on_failure(self):
        """失敗時の処理"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"サーキットブレーカー: OPEN状態に移行 "
                f"(失敗回数: {self.failure_count})"
            )


class PredictionErrorHandler:
    """統合予測エラーハンドリングシステム"""

    def __init__(self, log_file_path: Optional[Path] = None):
        self.log_file_path = (
            log_file_path or Path("logs/prediction_errors.log")
        )
        self.errors: List[PredictionError] = []
        self.recovery_strategy = ErrorRecoveryStrategy()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # エラー統計
        self.error_stats: Dict[str, int] = {}
        self.system_health: Dict[str, Dict[str, Any]] = {}

        # ログファイルの初期化
        self._init_error_logging()

    def _init_error_logging(self):
        """エラーログの初期化"""
        try:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # ファイルハンドラーの設定
            file_handler = logging.FileHandler(
                self.log_file_path, encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)

            # エラー専用ロガーの設定
            error_logger = logging.getLogger('prediction_errors')
            error_logger.addHandler(file_handler)
            error_logger.setLevel(logging.WARNING)

        except Exception as e:
            logger.warning(f"エラーログファイルの初期化に失敗: {e}")

    def get_circuit_breaker(self, system_id: str) -> CircuitBreaker:
        """システム別サーキットブレーカーの取得"""
        if system_id not in self.circuit_breakers:
            self.circuit_breakers[system_id] = CircuitBreaker()
        return self.circuit_breakers[system_id]

    async def handle_prediction_error(
            self, error: Exception, context: ErrorContext
    ) -> PredictionError:
        """予測エラーの処理"""
        # エラーカテゴリの判定
        category = self._categorize_error(error)
        severity = self._determine_severity(error, category)

        # エラーIDの生成
        error_id = f"{category.value}_{datetime.now().timestamp()}"

        # エラー情報の作成
        prediction_error = PredictionError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            context=context
        )

        # エラーの記録
        self.errors.append(prediction_error)
        self._update_error_stats(category, severity)
        self._update_system_health(context.system_id, False)

        # ログ出力
        self._log_error(prediction_error)

        # 自動回復の試行
        if severity in [ErrorSeverity.WARNING, ErrorSeverity.ERROR]:
            await self._attempt_auto_recovery(prediction_error)

        return prediction_error

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """エラーカテゴリの判定"""
        error_message = str(error).lower()

        if 'timeout' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK_ERROR
        elif 'data' in error_message or 'dataframe' in error_message:
            return ErrorCategory.DATA_ERROR
        elif 'config' in error_message or 'setting' in error_message:
            return ErrorCategory.CONFIGURATION_ERROR
        elif 'performance' in error_message or 'slow' in error_message:
            return ErrorCategory.PERFORMANCE_ERROR
        elif 'prediction' in error_message or 'model' in error_message:
            return ErrorCategory.PREDICTION_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR

    def _determine_severity(
            self, error: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """エラー重要度の判定"""
        error_message = str(error).lower()

        # クリティカルエラーのキーワード
        critical_keywords = ['critical', 'fatal', 'corruption', 'security']
        if any(keyword in error_message for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL

        # 設定エラーは通常重要
        if category == ErrorCategory.CONFIGURATION_ERROR:
            return ErrorSeverity.ERROR

        # ネットワークエラーは警告レベル
        if category == ErrorCategory.NETWORK_ERROR:
            return ErrorSeverity.WARNING

        # パフォーマンスエラーは情報レベル
        if category == ErrorCategory.PERFORMANCE_ERROR:
            return ErrorSeverity.INFO

        # デフォルトはエラーレベル
        return ErrorSeverity.ERROR

    def _update_error_stats(self, category: ErrorCategory,
                            severity: ErrorSeverity):
        """エラー統計の更新"""
        key = f"{category.value}_{severity.value}"
        self.error_stats[key] = self.error_stats.get(key, 0) + 1

    def _update_system_health(self, system_id: str, success: bool):
        """システムヘルス状態の更新"""
        if system_id not in self.system_health:
            self.system_health[system_id] = {
                'total_calls': 0,
                'success_count': 0,
                'error_count': 0,
                'success_rate': 0.0,
                'last_update': datetime.now()
            }

        health = self.system_health[system_id]
        health['total_calls'] += 1
        health['last_update'] = datetime.now()

        if success:
            health['success_count'] += 1
        else:
            health['error_count'] += 1

        # 成功率の更新
        health['success_rate'] = (
            health['success_count'] / health['total_calls']
        )

    def _log_error(self, prediction_error: PredictionError):
        """エラーログの出力"""
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(prediction_error.severity, logging.ERROR)

        log_message = (
            f"[{prediction_error.category.value}] {prediction_error.message} "
            f"(System: {prediction_error.context.system_id}, "
            f"Symbol: {prediction_error.context.symbol})"
        )

        logger.log(log_level, log_message)

        # エラー専用ログファイルへの記録
        error_logger = logging.getLogger('prediction_errors')
        error_logger.log(log_level, json.dumps(
            prediction_error.to_dict(), ensure_ascii=False
        ))

    async def _attempt_auto_recovery(self, prediction_error: PredictionError):
        """自動回復の試行"""
        try:
            if prediction_error.category == ErrorCategory.NETWORK_ERROR:
                # ネットワークエラー: 短時間待機後に再試行フラグを設定
                await asyncio.sleep(1.0)
                logger.info(f"ネットワークエラー回復待機完了: {prediction_error.error_id}")

            elif prediction_error.category == ErrorCategory.DATA_ERROR:
                # データエラー: データ検証とクリーニングのフラグを設定
                logger.info(f"データエラー回復準備: {prediction_error.error_id}")

            elif prediction_error.category == ErrorCategory.PERFORMANCE_ERROR:
                # パフォーマンスエラー: キャッシュクリアのフラグを設定
                logger.info(f"パフォーマンスエラー回復準備: {prediction_error.error_id}")

        except Exception as e:
            logger.warning(f"自動回復の試行に失敗: {e}")

    def get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリーの取得"""
        recent_errors = [
            error for error in self.errors
            if (datetime.now() - error.timestamp).total_seconds() < 3600
        ]

        return {
            'total_errors': len(self.errors),
            'recent_errors': len(recent_errors),
            'error_stats': self.error_stats,
            'system_health': self.system_health,
            'circuit_breaker_status': {
                system_id: breaker.state
                for system_id, breaker in self.circuit_breakers.items()
            }
        }

    def clear_old_errors(self, retention_hours: int = 24):
        """古いエラーのクリア"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        self.errors = [
            error for error in self.errors if error.timestamp > cutoff_time
        ]
        logger.info(f"古いエラーをクリアしました: {retention_hours}時間以前")


# グローバルエラーハンドラーインスタンス
prediction_error_handler = PredictionErrorHandler()


# ユーティリティ関数
async def safe_predict(
        operation: Callable, context: ErrorContext, *args, **kwargs
) -> Tuple[Any, Optional[PredictionError]]:
    """安全な予測実行（エラーハンドリング付き）"""
    try:
        # サーキットブレーカーチェック
        circuit_breaker = prediction_error_handler.get_circuit_breaker(
            context.system_id
        )
        result = await circuit_breaker.call(operation, *args, **kwargs)

        # 成功時の健康状態更新
        prediction_error_handler._update_system_health(
            context.system_id, True
        )

        return result, None

    except Exception as e:
        # エラーハンドリング
        prediction_error = (
            await prediction_error_handler.handle_prediction_error(e, context)
        )
        return None, prediction_error


def create_error_context(symbol: str = "", system_id: str = "",
                         operation: str = "",
                         input_data: Dict[str, Any] = None) -> ErrorContext:
    """ErrorContextの作成ヘルパー"""
    return ErrorContext(
        symbol=symbol,
        system_id=system_id,
        operation=operation,
        input_data=input_data or {},
        stack_trace=traceback.format_exc()
    )


if __name__ == "__main__":
    # エラーハンドリングテスト
    async def test_error_handling():
        logger.info("エラーハンドリングシステムテスト開始")

        # テスト用エラー関数
        async def failing_operation():
            raise ValueError("テスト用エラー")

        # エラーコンテキスト作成
        context = create_error_context(
            symbol="TEST",
            system_id="test_system",
            operation="test_predict"
        )

        # 安全な実行テスト
        result, error = await safe_predict(failing_operation, context)

        if error:
            logger.info(f"エラーが正常にハンドリングされました: {error.error_id}")

        # エラーサマリーの表示
        summary = prediction_error_handler.get_error_summary()
        logger.info(f"エラーサマリー: {summary}")

    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s: %(message)s'
    )
    asyncio.run(test_error_handling())
