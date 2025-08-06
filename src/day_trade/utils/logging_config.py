"""
構造化ロギング設定モジュール

アプリケーション全体で使用する構造化ロギング機能を提供。
JSON形式での出力、フィルタリング、ログレベル管理を統一。
"""

import contextlib
import logging
import os
import sys
import time
from typing import Any, Dict, Generator

import structlog
from structlog.types import Processor

from .security_helpers import SecurityHelpers


class LoggingConfig:
    """ロギング設定管理クラス"""

    def __init__(self):
        self.is_configured = False
        self.log_level = self._get_log_level()
        self.log_format = self._get_log_format()

    def _get_log_level(self) -> str:
        """環境変数からログレベルを取得"""
        return os.getenv("LOG_LEVEL", "INFO").upper()

    def _get_log_format(self) -> str:
        """環境変数からログフォーマットを取得"""
        return os.getenv("LOG_FORMAT", "json").lower()

    def configure_logging(self) -> None:
        """構造化ロギングを設定"""
        if self.is_configured:
            return

        # 標準ログレベルの設定
        log_level = getattr(logging, self.log_level, logging.INFO)

        # structlogの設定
        processors = self._get_processors()

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # 標準ライブラリのloggingの設定
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
        )

        # サードパーティライブラリのログレベル調整
        self._configure_third_party_logging()

        self.is_configured = True

    def _get_processors(self) -> list[Processor]:
        """ログプロセッサーを取得"""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        # 開発環境か本番環境かでフォーマットを変更
        if self._is_development():
            processors.extend([structlog.dev.ConsoleRenderer(colors=True)])
        else:
            processors.extend([structlog.processors.JSONRenderer()])

        return processors

    def _is_development(self) -> bool:
        """開発環境かどうかを判定"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        return env in ("development", "dev", "local")

    def _configure_third_party_logging(self) -> None:
        """サードパーティライブラリのログレベルを調整"""
        # SQLAlchemyのログを制限
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)

        # パフォーマンスクリティカルなライブラリのログを制限
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("yfinance").setLevel(logging.WARNING)

        # urllib3のログを制限
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        # requests関連のログを制限
        logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)

        # パフォーマンス最適化: アプリケーション内の高頻度処理モジュールを制限
        logging.getLogger("day_trade.utils.cache_utils").setLevel(logging.WARNING)
        logging.getLogger("day_trade.data.stock_fetcher").setLevel(logging.INFO)
        logging.getLogger("day_trade.analysis.indicators").setLevel(logging.WARNING)

    def get_logger(self, name: str) -> Any:
        """構造化ロガーを取得"""
        if not self.is_configured:
            self.configure_logging()
        return structlog.get_logger(name)


# グローバルインスタンス
_logging_config = LoggingConfig()


def setup_logging() -> None:
    """ロギング設定を初期化"""
    _logging_config.configure_logging()


def get_logger(name: str = None) -> Any:
    """構造化ロガーを取得"""
    if name is None:
        # 呼び出し元のモジュール名を自動取得
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")

    return _logging_config.get_logger(name)


class ContextLogger:
    """コンテキスト情報を保持するロガーラッパー"""

    def __init__(self, logger: Any, context: Dict[str, Any] = None):
        self.logger = logger
        self.context = context or {}

    def bind(self, **kwargs) -> "ContextLogger":
        """新しいコンテキストでロガーを作成"""
        new_context = {**self.context, **kwargs}
        return ContextLogger(self.logger, new_context)

    def info(self, message: str, **kwargs) -> None:
        """情報ログ出力"""
        self.logger.info(message, **{**self.context, **kwargs})

    def warning(self, message: str, **kwargs) -> None:
        """警告ログ出力"""
        self.logger.warning(message, **{**self.context, **kwargs})

    def error(self, message: str, **kwargs) -> None:
        """エラーログ出力"""
        self.logger.error(message, **{**self.context, **kwargs})

    def debug(self, message: str, **kwargs) -> None:
        """デバッグログ出力"""
        self.logger.debug(message, **{**self.context, **kwargs})

    def critical(self, message: str, **kwargs) -> None:
        """クリティカルログ出力"""
        self.logger.critical(message, **{**self.context, **kwargs})


def get_context_logger(name: str = None, **context) -> ContextLogger:
    """コンテキスト付きロガーを取得"""
    logger = get_logger(name)
    return ContextLogger(logger, context)


# 便利な関数群
def log_function_call(func_name: str, **kwargs) -> None:
    """関数呼び出しをログ出力"""
    logger = get_logger()
    logger.info("Function called", function=func_name, **kwargs)


def log_error_with_context(error: Exception, context: Dict[str, Any] = None) -> None:
    """エラー情報をコンテキスト付きでログ出力"""
    logger = get_logger()
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {}),
    }
    logger.error("Error occurred", **error_context)


def log_performance_metric(
    metric_name: str, value: float, unit: str = "ms", **kwargs
) -> None:
    """パフォーマンスメトリクスをログ出力"""
    logger = get_logger()
    logger.info(
        "Performance metric", metric_name=metric_name, value=value, unit=unit, **kwargs
    )


def log_business_event(event_name: str, **kwargs) -> None:
    """ビジネスイベントをログ出力"""
    logger = get_logger()
    logger.info("Business event", event_name=event_name, **kwargs)


def log_database_operation(operation: str, table: str, **kwargs) -> None:
    """データベース操作をログ出力"""
    logger = get_logger()
    logger.info("Database operation", operation=operation, table=table, **kwargs)


def log_api_call(
    api_name: str, method: str, url: str, status_code: int = None, **kwargs
) -> None:
    """API呼び出しをログ出力"""
    logger = get_logger()
    logger.info(
        "API call",
        api_name=api_name,
        method=method,
        url=url,
        status_code=status_code,
        **kwargs,
    )


# パフォーマンス最適化関連
class PerformanceCriticalLogger:
    """パフォーマンスクリティカルなセクション用の最適化されたロガー"""

    def __init__(self, logger: Any, min_level: int = logging.WARNING):
        """
        Args:
            logger: ベースロガー
            min_level: 最小ログレベル（これ以下のレベルはスキップ）
        """
        self.logger = logger
        self.min_level = min_level
        self._is_enabled_cache = {}  # ログレベルチェックのキャッシュ
        self._log_buffer = []  # バッファードログ用
        self._buffer_size = 100  # バッファサイズ

    def _is_enabled(self, level: int) -> bool:
        """ログレベルが有効かどうかをキャッシュ付きでチェック"""
        if level not in self._is_enabled_cache:
            self._is_enabled_cache[level] = level >= self.min_level
        return self._is_enabled_cache[level]

    def debug_batch(self, messages: list, **common_context) -> None:
        """バッチデバッグログ（パフォーマンス最適化）"""
        if not self._is_enabled(logging.DEBUG):
            return

        # メッセージを集約してまとめて出力
        batch_message = f"Batch debug: {len(messages)} items"
        self.logger.debug(
            batch_message,
            messages=messages[:5],
            total_count=len(messages),
            **common_context,
        )

    def info_sampled(self, message: str, sample_rate: float = 0.1, **kwargs) -> None:
        """サンプリングされた情報ログ（高頻度処理用）"""
        if not self._is_enabled(logging.INFO):
            return

        # セキュリティ強化：安全な乱数生成を使用
        if SecurityHelpers.secure_random_float(0.0, 1.0) < sample_rate:
            self.logger.info(f"[SAMPLED] {message}", sample_rate=sample_rate, **kwargs)

    def performance_summary(
        self, operation: str, metrics: Dict[str, float], **kwargs
    ) -> None:
        """パフォーマンス概要ログ（詳細を集約）"""
        if not self._is_enabled(logging.INFO):
            return

        summary = {
            "avg": sum(metrics.values()) / len(metrics) if metrics else 0,
            "max": max(metrics.values()) if metrics else 0,
            "min": min(metrics.values()) if metrics else 0,
            "count": len(metrics),
        }

        self.logger.info(f"Performance summary: {operation}", summary=summary, **kwargs)

    def buffer_log(self, level: int, message: str, **kwargs) -> None:
        """ログをバッファに蓄積（高頻度処理用）"""
        if not self._is_enabled(level):
            return

        self._log_buffer.append(
            {
                "level": level,
                "message": message,
                "timestamp": time.time(),
                "kwargs": kwargs,
            }
        )

        # バッファが満杯になったら一括出力
        if len(self._log_buffer) >= self._buffer_size:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        """バッファ内のログを一括出力"""
        if not self._log_buffer:
            return

        # レベル別に集約
        level_counts = {}
        for log_entry in self._log_buffer:
            level = log_entry["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        # 集約結果をログ出力
        self.logger.info(
            "Buffered logs flushed",
            buffer_size=len(self._log_buffer),
            level_counts=level_counts,
            time_span=self._log_buffer[-1]["timestamp"]
            - self._log_buffer[0]["timestamp"],
        )

        # バッファをクリア
        self._log_buffer.clear()

    def __del__(self):
        """デストラクタでバッファの残りを出力"""
        if hasattr(self, "_log_buffer") and self._log_buffer:
            self.flush_buffer()


def get_performance_logger(
    name: str = None, min_level: int = logging.WARNING
) -> PerformanceCriticalLogger:
    """パフォーマンスクリティカルなロガーを取得"""
    base_logger = get_logger(name)
    return PerformanceCriticalLogger(base_logger, min_level)


# 遅延評価ログ機能
class LazyLogMessage:
    """遅延評価されるログメッセージ（重い処理を含む場合に使用）"""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.func(*self.args, **self.kwargs)


def lazy_log(func):
    """重い処理を含むログメッセージを遅延評価する"""
    return lambda *args, **kwargs: LazyLogMessage(func, *args, **kwargs)


# ログ無効化コンテキストマネージャー


@contextlib.contextmanager
def disable_logging(logger_names: list = None) -> Generator[None, None, None]:
    """指定されたロガーを一時的に無効化"""
    if logger_names is None:
        logger_names = ["day_trade"]  # デフォルトでメインロガーを無効化

    original_levels = {}
    try:
        # ログレベルを一時的に引き上げ
        for name in logger_names:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL + 1)  # 全てのログを無効化

        yield

    finally:
        # 元のログレベルに復元
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


# マクロレベルでのパフォーマンス最適化設定
class PerformanceOptimizedLogging:
    """パフォーマンス最適化されたロギング設定"""

    @staticmethod
    def configure_for_production():
        """本番環境向けの最適化設定"""
        # サードパーティライブラリのログを更に制限
        performance_critical_loggers = [
            "sqlalchemy",
            "urllib3",
            "requests",
            "yfinance",
            "pandas",
            "numpy",
        ]

        for logger_name in performance_critical_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    @staticmethod
    def configure_for_backtesting():
        """バックテスト実行時の最適化設定"""
        # バックテスト中は詳細ログを制限
        logging.getLogger("day_trade.analysis.backtest").setLevel(logging.WARNING)
        logging.getLogger("day_trade.data.stock_fetcher").setLevel(logging.ERROR)

    @staticmethod
    def configure_for_high_frequency():
        """高頻度取引時の最適化設定"""
        # リアルタイム処理では最小限のログのみ
        trading_loggers = [
            "day_trade.data",
            "day_trade.analysis.signals",
            "day_trade.analysis.indicators",
        ]

        for logger_name in trading_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)


# 条件付きロギング用デコレータ
def conditional_log(condition_func):
    """条件が満たされた場合のみログを出力するデコレータ"""

    def decorator(log_func):
        def wrapper(*args, **kwargs):
            if condition_func():
                return log_func(*args, **kwargs)

        return wrapper

    return decorator


# パフォーマンス測定とロギングの統合
class PerformanceTimer:
    """パフォーマンス測定とロギングを統合したクラス"""

    def __init__(self, logger: Any, operation: str, threshold_ms: float = 100.0):
        """
        Args:
            logger: ロガーインスタンス
            operation: 操作名
            threshold_ms: ログ出力の閾値（ミリ秒）
        """
        self.logger = logger
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000

            # 閾値を超えた場合のみログ出力
            if elapsed_ms > self.threshold_ms:
                self.logger.warning(
                    "Slow operation detected",
                    operation=self.operation,
                    elapsed_ms=elapsed_ms,
                    threshold_ms=self.threshold_ms,
                )


# 集約ロギング機能
class AggregatedLogger:
    """複数のログエントリを集約して効率的に出力"""

    def __init__(self, logger: Any, flush_interval: float = 1.0):
        """
        Args:
            logger: ベースロガー
            flush_interval: フラッシュ間隔（秒）
        """
        self.logger = logger
        self.flush_interval = flush_interval
        self.counters = {}
        self.metrics = {}
        self.last_flush = time.time()

    def increment_counter(self, key: str, value: int = 1):
        """カウンターをインクリメント"""
        self.counters[key] = self.counters.get(key, 0) + value
        self._maybe_flush()

    def record_metric(self, key: str, value: float):
        """メトリクスを記録"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
        self._maybe_flush()

    def _maybe_flush(self):
        """必要に応じてフラッシュ"""
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()

    def flush(self):
        """集約データを出力"""
        if self.counters or self.metrics:
            # メトリクスの統計計算
            metric_stats = {}
            for key, values in self.metrics.items():
                if values:
                    metric_stats[key] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            self.logger.info(
                "Aggregated metrics",
                counters=self.counters,
                metrics=metric_stats,
                flush_interval=self.flush_interval,
            )

            # データをクリア
            self.counters.clear()
            self.metrics.clear()
            self.last_flush = time.time()


# セキュリティ関連のロギング（最適化版）
def log_security_event(event_type: str, severity: str = "info", **kwargs) -> None:
    """セキュリティイベントをログ出力（パフォーマンス最適化版）"""
    # セキュリティイベントは必ず出力するが、詳細度を調整
    logger = get_logger()

    # 重要度に応じてログレベルを調整
    log_func = getattr(logger, severity.lower(), logger.info)

    # 最小限の情報のみを記録
    log_func(
        "Security event",
        event_type=event_type,
        severity=severity,
        timestamp=time.time(),
        **{k: v for k, v in kwargs.items() if k in ["user_id", "ip_address", "action"]},
    )
