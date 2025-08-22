#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Logging Module - 統一ログシステム
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_FILE_MAX_BYTES = 10 * 1024 * 1024 # 10 MB


class DayTradeLogger:
    """デイトレードシステム専用ログマネージャー"""

    def __init__(self, name: str = "daytrade", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # ロガー設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # ハンドラーがすでに設定済みかチェック
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """ログハンドラーの設定"""

        # コンソールハンドラー（INFO以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # ファイルハンドラー（DEBUG以上、日次ローテーション）
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_dir / f"{self.name}.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # エラーファイルハンドラー（ERROR以上）
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_dir / f"{self.name}_error.log",
            when='midnight',
            interval=1,
            backupCount=90,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # ハンドラーをロガーに追加
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """モジュール専用ロガーを取得"""
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger

    def info(self, message: str):
        """INFO レベルログ"""
        self.logger.info(message)

    def debug(self, message: str):
        """DEBUG レベルログ"""
        self.logger.debug(message)

    def warning(self, message: str):
        """WARNING レベルログ"""
        self.logger.warning(message)

    def error(self, message: str):
        """ERROR レベルログ"""
        self.logger.error(message)

    def critical(self, message: str):
        """CRITICAL レベルログ"""
        self.logger.critical(message)

    def exception(self, message: str):
        """例外情報付きERRORログ"""
        self.logger.exception(message)

    def trading_event(self, symbol: str, action: str, price: float, confidence: float):
        """トレーディングイベント専用ログ"""
        self.logger.info(
            f"TRADING_EVENT - Symbol: {symbol}, Action: {action}, "
            f"Price: {price}, Confidence: {confidence:.2%}"
        )

    def analysis_result(self, symbol: str, signal: str, accuracy: float, details: str = ""):
        """分析結果専用ログ"""
        self.logger.info(
            f"ANALYSIS_RESULT - Symbol: {symbol}, Signal: {signal}, "
            f"Accuracy: {accuracy:.2%}, Details: {details}"
        )

    def system_status(self, component: str, status: str, details: str = ""):
        """システム状態専用ログ"""
        self.logger.info(
            f"SYSTEM_STATUS - Component: {component}, Status: {status}, "
            f"Details: {details}"
        )

    def performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """パフォーマンスメトリック専用ログ"""
        self.logger.info(
            f"PERFORMANCE_METRIC - {metric_name}: {value} {unit}"
        )

    def data_quality_check(self, source: str, quality_score: float, issues: str = ""):
        """データ品質チェック専用ログ"""
        level = logging.WARNING if quality_score < 0.8 else logging.INFO
        self.logger.log(
            level,
            f"DATA_QUALITY - Source: {source}, Score: {quality_score:.2%}, "
            f"Issues: {issues}"
        )

    def api_request(self, endpoint: str, method: str, response_time: float, status_code: int):
        """API リクエスト専用ログ"""
        level = logging.WARNING if status_code >= 400 else logging.DEBUG
        self.logger.log(
            level,
            f"API_REQUEST - {method} {endpoint}, Time: {response_time:.3f}s, "
            f"Status: {status_code}"
        )

    @classmethod
    def setup_global_logging(cls, debug: bool = False):
        """グローバルログ設定"""
        level = logging.DEBUG if debug else logging.INFO

        # ルートロガーの設定
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # 外部ライブラリのログレベル調整
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('plotly').setLevel(logging.WARNING)

        return cls()


# グローバルロガーインスタンス
_global_logger = None


def get_logger(name: str = "daytrade") -> DayTradeLogger:
    """グローバルロガーインスタンスを取得"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DayTradeLogger(name)
    return _global_logger


def setup_logging(debug: bool = False) -> DayTradeLogger:
    """ログシステムの初期化"""
    return DayTradeLogger.setup_global_logging(debug)


# 便利関数
def log_info(message: str):
    """INFO ログ出力"""
    get_logger().info(message)


def log_error(message: str):
    """ERROR ログ出力"""
    get_logger().error(message)


def log_debug(message: str):
    """DEBUG ログ出力"""
    get_logger().debug(message)


def log_trading_event(symbol: str, action: str, price: float, confidence: float):
    """トレーディングイベントログ出力"""
    get_logger().trading_event(symbol, action, price, confidence)


def log_analysis_result(symbol: str, signal: str, accuracy: float, details: str = ""):
    """分析結果ログ出力"""
    get_logger().analysis_result(symbol, signal, accuracy, details)


if __name__ == "__main__":
    # テスト実行
    logger = setup_logging(debug=True)

    logger.info("ログシステムテスト開始")
    logger.debug("デバッグメッセージ")
    logger.warning("警告メッセージ")
    logger.error("エラーメッセージ")

    logger.trading_event("7203", "BUY", 1000.0, 0.95)
    logger.analysis_result("8306", "STRONG_BUY", 0.93, "高いボラティリティ検出")
    logger.system_status("ML_ENGINE", "OPERATIONAL", "93.2% accuracy")
    logger.performance_metric("response_time", 0.245, "seconds")
    logger.data_quality_check("yfinance", 0.95, "全データ正常")
    logger.api_request("/api/analysis", "GET", 0.123, 200)

    logger.info("ログシステムテスト完了")