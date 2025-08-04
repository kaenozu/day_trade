"""
構造化ロギングのテスト
"""

import logging
from unittest.mock import patch

import pytest

from src.day_trade.utils.logging_config import (
    ContextLogger,
    get_context_logger,
    get_logger,
    log_api_call,
    log_business_event,
    log_error_with_context,
    log_performance_metric,
    setup_logging,
)


class TestLoggingConfig:
    """ロギング設定のテスト"""

    def test_setup_logging(self):
        """ロギング設定の初期化テスト"""
        setup_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger(self):
        """ロガー取得テスト"""
        logger = get_logger("test_module")
        assert logger is not None

    def test_get_context_logger(self):
        """コンテキストロガー取得テスト"""
        logger = get_context_logger("test_module", test_key="test_value")
        assert isinstance(logger, ContextLogger)
        assert logger.context["test_key"] == "test_value"

    def test_context_logger_bind(self):
        """コンテキストロガーのバインドテスト"""
        logger = get_context_logger("test_module", key1="value1")
        bound_logger = logger.bind(key2="value2")

        assert bound_logger.context["key1"] == "value1"
        assert bound_logger.context["key2"] == "value2"

    @patch.dict("os.environ", {"LOG_LEVEL": "DEBUG", "LOG_FORMAT": "json"})
    def test_environment_settings(self):
        """環境変数設定テスト"""
        from src.day_trade.utils.logging_config import LoggingConfig

        config = LoggingConfig()
        assert config.log_level == "DEBUG"
        assert config.log_format == "json"

    def test_log_functions(self):
        """ログ関数テスト"""
        # ログ関数が正常に実行されることを確認（出力内容は検証しない）
        setup_logging()

        # 例外が発生せずに実行されることを確認
        log_business_event("test_event", key="value")
        log_api_call("test_api", "POST", "http://example.com", 200)
        log_performance_metric("test_metric", 123.45, "ms")

        # エラーログのテスト
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_error_with_context(e, {"context_key": "context_value"})


class TestContextLogger:
    """コンテキストロガーのテスト"""

    def test_context_preservation(self):
        """コンテキスト保持テスト"""
        logger = get_context_logger("test", operation="test_op", request_id="123")

        # コンテキストが正しく設定されていることを確認
        assert logger.context["operation"] == "test_op"
        assert logger.context["request_id"] == "123"

    def test_logging_methods(self):
        """ロギングメソッドのテスト"""
        setup_logging()
        logger = get_context_logger("test", test_key="test_value")

        # 各レベルのログ出力が例外なく実行されることをテスト
        logger.info("Test info message", extra_key="extra_value")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")
        logger.critical("Test critical message")


class TestIntegration:
    """統合テスト"""

    def test_trade_operations_logging(self):
        """取引操作ロギングの統合テスト"""
        from unittest.mock import Mock

        setup_logging()

        # TradeOperationsのモック（ロギング部分のみテスト）
        from src.day_trade.core.trade_operations import TradeOperations

        # StockFetcherをモック化
        mock_fetcher = Mock()
        mock_fetcher.get_company_info.return_value = {
            "name": "Test Company",
            "sector": "Technology",
        }
        mock_fetcher.get_current_price.return_value = {"price": 1000.0}

        trade_ops = TradeOperations(mock_fetcher)

        # ロガーが正しく初期化されていることを確認
        assert hasattr(trade_ops, "logger")
        assert isinstance(trade_ops.logger, ContextLogger)

    def test_stock_fetcher_logging(self):
        """株価データ取得ロギングの統合テスト"""
        setup_logging()

        from src.day_trade.data.stock_fetcher import StockFetcher

        # StockFetcherの初期化でロギングが動作することを確認
        fetcher = StockFetcher()

        # ロガーが正しく初期化されていることを確認
        assert hasattr(fetcher, "logger")
        assert isinstance(fetcher.logger, ContextLogger)


class TestLoggingOutput:
    """ログ出力形式のテスト"""

    @patch.dict("os.environ", {"ENVIRONMENT": "production"})
    def test_json_output_format(self):
        """本番環境でのJSON出力テスト"""
        from src.day_trade.utils.logging_config import LoggingConfig

        config = LoggingConfig()

        # 本番環境ではJSONフォーマットが使用されることを確認
        processors = config._get_processors()

        # プロセッサーにJSONRendererが含まれることを確認
        from structlog.processors import JSONRenderer

        processor_types = [type(processor) for processor in processors]
        assert JSONRenderer in processor_types

    @patch.dict("os.environ", {"ENVIRONMENT": "development"})
    def test_console_output_format(self):
        """開発環境でのコンソール出力テスト"""
        from src.day_trade.utils.logging_config import LoggingConfig

        config = LoggingConfig()

        # 開発環境ではConsoleRendererが使用されることを確認
        processors = config._get_processors()

        # プロセッサーにConsoleRendererが含まれることを確認
        from structlog.dev import ConsoleRenderer

        processor_types = [type(processor) for processor in processors]
        assert ConsoleRenderer in processor_types

    def test_third_party_logging_levels(self):
        """サードパーティライブラリのログレベル設定テスト"""
        setup_logging()

        # SQLAlchemyのログレベルが適切に設定されていることを確認
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        assert sqlalchemy_logger.level == logging.WARNING

        # urllib3のログレベルが適切に設定されていることを確認
        urllib3_logger = logging.getLogger("urllib3.connectionpool")
        assert urllib3_logger.level == logging.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
