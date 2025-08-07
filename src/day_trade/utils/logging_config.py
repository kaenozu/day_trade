"""
構造化ロギング設定モジュール

アプリケーション全体で使用する構造化ロギング機能を提供。
JSON形式での出力、フィルタリング、ログレベル管理を統一。
"""

import logging
import os
import sys
from typing import Any, Dict


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
        return os.getenv("LOG_FORMAT", "simple").lower()

    def configure_logging(self) -> None:
        """基本ロギングを設定"""
        if self.is_configured:
            return

        # 標準ログレベルの設定
        log_level = getattr(logging, self.log_level, logging.INFO)

        # 基本的なログ設定
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.is_configured = True


# グローバルロギング設定インスタンス
_logging_config = LoggingConfig()


def setup_logging():
    """ロギング設定を初期化"""
    _logging_config.configure_logging()


def get_context_logger(name: str, component: str = None) -> logging.Logger:
    """
    コンテキスト付きロガーを取得

    Args:
        name: ロガー名
        component: コンポーネント名

    Returns:
        設定済みロガー
    """
    logger_name = f"{name}.{component}" if component else name

    return logging.getLogger(logger_name)


def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """
    コンテキスト付きでエラーをログ出力

    Args:
        error: エラーオブジェクト
        context: コンテキスト情報
    """
    logger = logging.getLogger(__name__)
    logger.error(f"Error occurred: {error}. Context: {context}", exc_info=True)
