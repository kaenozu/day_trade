"""
統一的なエラーハンドリング統合テスト
EnhancedErrorHandlerとカスタム例外クラスの動作検証
"""

from io import StringIO
from unittest.mock import Mock, patch

import pytest
from rich.panel import Panel

from src.day_trade.utils.enhanced_error_handler import (
    EnhancedErrorHandler,
    create_error_handler,
    get_default_error_handler,
    handle_cli_error,
    reset_default_error_handler,
    set_default_error_handler,
)
from src.day_trade.utils.exceptions import (
    APIError,
    BacktestError,
    DatabaseError,
    DayTradeError,
    NetworkError,
    TradingError,
    ValidationError,
    handle_database_exception,
    handle_network_exception,
)
from src.day_trade.utils.i18n_messages import I18nMessageHandler, Language


class TestErrorHandlingIntegration:
    """統一的なエラーハンドリングの統合テスト"""

    def setup_method(self):
        """各テストメソッド前の初期化"""
        # デフォルトハンドラーをリセット
        reset_default_error_handler()

    def test_custom_exception_hierarchy(self):
        """カスタム例外階層のテスト"""
        # 基底例外
        base_error = DayTradeError(
            message="基底エラー", error_code="BASE_ERROR", details={"key": "value"}
        )
        assert base_error.message == "基底エラー"
        assert base_error.error_code == "BASE_ERROR"
        assert base_error.details == {"key": "value"}

        # 継承関係の確認
        api_error = APIError(message="API エラー", error_code="API_ERROR")
        assert isinstance(api_error, DayTradeError)
        assert isinstance(api_error, APIError)

        db_error = DatabaseError(message="DB エラー", error_code="DB_ERROR")
        assert isinstance(db_error, DayTradeError)
        assert isinstance(db_error, DatabaseError)

        trading_error = TradingError(message="取引エラー", error_code="TRADING_ERROR")
        assert isinstance(trading_error, DayTradeError)
        assert isinstance(trading_error, TradingError)

    def test_exception_to_dict_serialization(self):
        """例外のシリアライゼーションテスト"""
        error = ValidationError(
            message="検証エラー",
            error_code="VALIDATION_ERROR",
            details={"field": "symbol", "value": "invalid"},
        )

        error_dict = error.to_dict()
        expected = {
            "error_type": "ValidationError",
            "message": "検証エラー",
            "error_code": "VALIDATION_ERROR",
            "details": {"field": "symbol", "value": "invalid"},
        }

        assert error_dict == expected

    def test_enhanced_error_handler_initialization(self):
        """EnhancedErrorHandler初期化テスト"""
        handler = create_error_handler()
        assert isinstance(handler, EnhancedErrorHandler)
        assert handler.language == Language.JAPANESE
        assert handler.enable_sanitization is True

        # カスタム設定での初期化
        custom_handler = create_error_handler(
            language=Language.ENGLISH, debug_mode=True, enable_sanitization=False
        )
        assert custom_handler.language == Language.ENGLISH
        assert custom_handler.debug_mode is True
        assert custom_handler.enable_sanitization is False

    def test_error_panel_creation(self):
        """エラーパネル作成テスト"""
        handler = create_error_handler()
        error = ValidationError(
            message="データ形式が正しくありません",
            error_code="DATA_FORMAT_ERROR",
            details={"input": "invalid_data"},
        )

        context = {"user_action": "データ検証", "user_input": "test_input"}

        _panel = handler.handle_error(error, context=context, show_technical=False)
        assert isinstance(_panel, Panel)

    def test_error_sanitization(self):
        """エラー情報サニタイズテスト"""
        handler = create_error_handler(enable_sanitization=True)

        # 機密情報を含むコンテキスト
        sensitive_context = {
            "api_key": "secret_key_12345",
            "password": "my_password",
            "user_data": "normal_data",
            "database_password": "db_secret",
        }

        error = APIError(message="API エラー", error_code="API_ERROR")

        with patch.object(handler, "_enhanced_sanitize_context") as mock_sanitize:
            mock_sanitize.return_value = {
                "api_key": "[機密情報のため非表示]",
                "password": "[機密情報のため非表示]",
                "user_data": "normal_data",
                "database_password": "[機密情報のため非表示]",
            }

            _panel = handler.handle_error(error, context=sensitive_context)
            mock_sanitize.assert_called_once()

    def test_logging_integration(self):
        """ロギング統合テスト"""
        handler = create_error_handler()
        error = TradingError(
            message="取引処理エラー",
            error_code="TRADING_PROCESS_ERROR",
            details={"symbol": "7203", "quantity": 100},
        )

        context = {"user_action": "株式売買", "operation": "buy_order"}

        with patch("src.day_trade.utils.enhanced_error_handler.logger") as mock_logger:
            handler.log_error(error, context=context)
            mock_logger.error.assert_called_once()

            # ログ呼び出し引数の検証
            call_args = mock_logger.error.call_args
            assert "User-facing error occurred" in call_args[0]
            assert "extra" in call_args[1]

            extra_data = call_args[1]["extra"]
            assert extra_data["error_type"] == "TradingError"
            assert extra_data["error_code"] == "TRADING_PROCESS_ERROR"

    def test_cli_error_handling(self):
        """CLI エラーハンドリングテスト"""
        error = BacktestError(
            message="バックテスト実行エラー", error_code="BACKTEST_EXECUTION_ERROR"
        )

        # 出力をキャプチャ（使用されていないが将来的に必要になる可能性のため保持）
        _captured_output = StringIO()

        # エラーハンドラーのconsoleインスタンスをモック
        with patch.object(get_default_error_handler(), "console") as mock_console:
            mock_console.print = Mock()

            handle_cli_error(
                error=error,
                context={"user_input": "test_symbol"},
                user_action="バックテスト実行",
                show_technical=False,
            )

            # エラーハンドラーがconsole.printを呼び出すことを確認
            mock_console.print.assert_called()

    def test_exception_conversion_helpers(self):
        """例外変換ヘルパー関数のテスト"""
        # SQLAlchemy例外の変換テスト（実際の例外を使用）
        from sqlalchemy.exc import IntegrityError

        real_exc = IntegrityError("test", "test", "test")
        converted = handle_database_exception(real_exc)

        assert isinstance(converted, DatabaseError)
        assert "データ整合性エラー" in converted.message

        # requests例外の変換テスト（実際の例外を使用）
        from requests.exceptions import ConnectionError

        real_exc = ConnectionError("connection failed")
        converted = handle_network_exception(real_exc)

        assert isinstance(converted, NetworkError)
        assert "ネットワーク接続エラー" in converted.message

    def test_performance_stats_collection(self):
        """パフォーマンス統計収集テスト"""
        handler = create_error_handler()

        # 複数のエラー処理を実行
        errors = [
            ValidationError(message="エラー1", error_code="ERROR_1"),
            TradingError(message="エラー2", error_code="ERROR_2"),
            DatabaseError(message="エラー3", error_code="ERROR_3"),
        ]

        for error in errors:
            handler.handle_error(error)

        # 統計情報の確認
        stats = handler.get_performance_stats()
        assert isinstance(stats, dict)
        assert "errors_handled" in stats
        assert stats["errors_handled"] == len(errors)
        assert "config" in stats
        assert isinstance(stats["config"], dict)

    def test_default_handler_management(self):
        """デフォルトハンドラー管理テスト"""
        # デフォルトハンドラーの取得
        default1 = get_default_error_handler()
        default2 = get_default_error_handler()
        assert default1 is default2  # シングルトンパターン

        # カスタムハンドラーの設定
        custom_handler = create_error_handler(debug_mode=True)
        set_default_error_handler(custom_handler)

        current_default = get_default_error_handler()
        assert current_default is custom_handler
        assert current_default.debug_mode is True

        # 不正なハンドラー設定のテスト
        with pytest.raises(ValueError, match="EnhancedErrorHandlerインスタンス"):
            set_default_error_handler("invalid_handler")

    def test_multilevel_error_context(self):
        """多層エラーコンテキストテスト"""
        handler = create_error_handler()

        # 深いネストのコンテキスト
        deep_context = {
            "level1": {
                "level2": {
                    "level3": {"sensitive_key": "secret_value", "normal_key": "normal"}
                }
            },
            "api_key": "another_secret",
            "public_data": "public_info",
        }

        error = APIError(message="深層エラー", error_code="DEEP_ERROR")

        # サニタイズが機能することを確認
        panel = handler.handle_error(error, context=deep_context)
        assert isinstance(panel, Panel)

    def test_error_message_localization_fallback(self):
        """エラーメッセージローカライゼーションフォールバックテスト"""
        # メッセージハンドラーが失敗する場合のテスト
        mock_handler = Mock(spec=I18nMessageHandler)
        mock_handler.get_message.side_effect = Exception("message handler error")

        handler = EnhancedErrorHandler(message_handler=mock_handler)
        error = ValidationError(message="テストエラー", error_code="TEST_ERROR")

        # フォールバックメッセージが使用されることを確認
        panel = handler.handle_error(error)
        assert isinstance(panel, Panel)

    def test_context_size_limitation(self):
        """コンテキストサイズ制限テスト"""
        handler = create_error_handler()

        # 大量のコンテキストアイテムを作成
        large_context = {f"item_{i}": f"value_{i}" for i in range(50)}

        error = ValidationError(message="大量コンテキスト", error_code="LARGE_CONTEXT")

        panel = handler.handle_error(error, context=large_context)
        assert isinstance(panel, Panel)

    def test_concurrent_error_handling(self):
        """並行エラーハンドリングテスト"""
        import threading

        handler = create_error_handler()
        results = []
        errors = []

        def handle_error_worker(worker_id):
            try:
                error = ValidationError(
                    message=f"ワーカー{worker_id}エラー",
                    error_code=f"WORKER_{worker_id}_ERROR",
                )
                panel = handler.handle_error(error)
                results.append((worker_id, panel))
            except Exception as e:
                errors.append((worker_id, e))

        # 複数スレッドで並行実行
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=handle_error_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # 全スレッドの完了を待機
        for thread in threads:
            thread.join(timeout=10)

        # 結果の検証
        assert len(results) == 5
        assert len(errors) == 0
        for _worker_id, panel in results:
            assert isinstance(panel, Panel)


class TestSpecificErrorHandlers:
    """特定用途エラーハンドラーのテスト"""

    def test_stock_fetch_error_handler(self):
        """株価取得エラーハンドラーテスト"""
        from src.day_trade.utils.enhanced_error_handler import handle_stock_fetch_error

        with patch(
            "src.day_trade.utils.enhanced_error_handler.handle_cli_error"
        ) as mock:
            error = NetworkError(
                message="株価データ取得失敗", error_code="STOCK_FETCH_ERROR"
            )

            handle_stock_fetch_error(error, stock_code="7203", show_technical=False)

            mock.assert_called_once()
            call_kwargs = mock.call_args[1]
            assert call_kwargs["context"]["user_input"] == "7203"
            assert call_kwargs["user_action"] == "株価情報の取得"

    def test_database_error_handler(self):
        """データベースエラーハンドラーテスト"""
        from src.day_trade.utils.enhanced_error_handler import handle_database_error

        with patch(
            "src.day_trade.utils.enhanced_error_handler.handle_cli_error"
        ) as mock:
            error = DatabaseError(
                message="DB接続失敗", error_code="DB_CONNECTION_ERROR"
            )

            handle_database_error(error, operation="データ取得", show_technical=True)

            mock.assert_called_once()
            call_kwargs = mock.call_args[1]
            assert call_kwargs["context"]["operation"] == "データ取得"
            assert call_kwargs["user_action"] == "データベース操作"

    def test_config_error_handler(self):
        """設定エラーハンドラーテスト"""
        from src.day_trade.utils.enhanced_error_handler import handle_config_error

        with patch(
            "src.day_trade.utils.enhanced_error_handler.handle_cli_error"
        ) as mock:
            error = ValidationError(
                message="設定値不正", error_code="CONFIG_VALIDATION_ERROR"
            )

            handle_config_error(
                error,
                config_key="api_key",
                config_value="invalid",
                show_technical=False,
            )

            mock.assert_called_once()
            call_kwargs = mock.call_args[1]
            assert call_kwargs["context"]["config_key"] == "api_key"
            assert call_kwargs["context"]["user_input"] == "invalid"


class TestErrorHandlingValidation:
    """エラーハンドリング検証テスト"""

    def test_validation_integration_check(self):
        """統合検証チェック"""
        from src.day_trade.utils.enhanced_error_handler import (
            validate_error_handler_integration,
        )

        result = validate_error_handler_integration()
        assert isinstance(result, dict)
        assert result["validation_passed"] is True
        assert "config_integration" in result
        assert "i18n_integration" in result
        assert "sanitization_enabled" in result

    def test_performance_stats_retrieval(self):
        """パフォーマンス統計取得テスト"""
        from src.day_trade.utils.enhanced_error_handler import (
            get_error_handler_performance_stats,
        )

        stats = get_error_handler_performance_stats()
        assert isinstance(stats, dict)
        assert "errors_handled" in stats
        assert "fallback_rate" in stats
        assert "sanitization_rate" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
