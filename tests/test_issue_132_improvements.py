"""
Issue #132: コードレビューに基づくアプリケーション改善点のテスト
"""

import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.day_trade.core.alert_strategies import (
    AlertStrategyFactory,
)
from src.day_trade.core.alerts import (
    AlertCondition,
    AlertManager,
    AlertPriority,
    AlertTrigger,
)
from src.day_trade.core.config import ConfigManager
from src.day_trade.core.persistent_alerts import PersistentAlertManager
from src.day_trade.core.security_config import (
    EnvironmentConfigLoader,
    SecureConfigManager,
)
from src.day_trade.models.enums import AlertType


class TestAlertManagerBulkOptimization:
    """AlertManagerのバルクデータ取得最適化のテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.mock_stock_fetcher = Mock()
        self.alert_manager = AlertManager(stock_fetcher=self.mock_stock_fetcher)

    def test_fetch_bulk_market_data_with_bulk_support(self):
        """バルクデータ取得（StockFetcherがbulk対応）のテスト"""
        # StockFetcherがbulk対応の場合
        self.mock_stock_fetcher.get_bulk_current_prices = Mock(return_value={
            "7203": {"current_price": 2800, "volume": 1000000, "change_percent": 2.5},
            "9984": {"current_price": 9500, "volume": 500000, "change_percent": -1.2},
        })

        symbols = ["7203", "9984"]
        result = self.alert_manager._fetch_bulk_market_data(symbols)

        assert len(result) == 2
        assert "7203" in result
        assert "9984" in result
        assert result["7203"]["current_data"]["current_price"] == 2800

        # bulk メソッドが呼ばれていることを確認
        self.mock_stock_fetcher.get_bulk_current_prices.assert_called_once_with(symbols)

    def test_fetch_bulk_market_data_fallback_to_individual(self):
        """バルクデータ取得のフォールバック（個別取得）のテスト"""
        # StockFetcherがbulk非対応の場合
        def mock_get_current_price(symbol):
            if symbol == "7203":
                return {"current_price": 2800, "volume": 1000000, "change_percent": 2.5}
            elif symbol == "9984":
                return {"current_price": 9500, "volume": 500000, "change_percent": -1.2}
            return None

        # bulk対応メソッドが存在しないことを明示的に設定
        if hasattr(self.mock_stock_fetcher, 'get_bulk_current_prices'):
            delattr(self.mock_stock_fetcher, 'get_bulk_current_prices')

        self.mock_stock_fetcher.get_current_price = Mock(side_effect=mock_get_current_price)

        symbols = ["7203", "9984"]
        result = self.alert_manager._fetch_bulk_market_data(symbols)

        assert len(result) == 2
        assert "7203" in result
        assert "9984" in result

        # 個別メソッドが2回呼ばれていることを確認
        assert self.mock_stock_fetcher.get_current_price.call_count == 2

    def test_check_symbol_alerts_with_data_lazy_historical_loading(self):
        """履歴データの遅延読み込みのテスト"""
        # 履歴データが不要な条件のみ
        condition = AlertCondition(
            alert_id="test_price",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500")
        )

        market_data = {
            "current_data": {"current_price": 2800, "volume": 1000000, "change_percent": 2.5},
            "historical_data": None
        }

        with patch.object(self.alert_manager, '_should_check_condition', return_value=True):
            with patch.object(self.alert_manager, '_handle_alert_trigger') as mock_handle:
                self.alert_manager._check_symbol_alerts_with_data("7203", [condition], market_data)

                # 履歴データが取得されていないことを確認
                assert market_data["historical_data"] is None

                # アラートが発火していることを確認
                mock_handle.assert_called_once()

    def test_check_symbol_alerts_with_data_historical_loading_when_needed(self):
        """必要時の履歴データ読み込みのテスト"""
        # 履歴データが必要な条件
        condition = AlertCondition(
            alert_id="test_volume",
            symbol="7203",
            alert_type=AlertType.VOLUME_SPIKE,
            condition_value=2.0
        )

        market_data = {
            "current_data": {"current_price": 2800, "volume": 2000000, "change_percent": 2.5},
            "historical_data": None
        }

        # モック履歴データ
        mock_historical = Mock()
        mock_historical.empty = False
        mock_historical.__getitem__ = Mock(return_value=Mock())

        with patch.object(self.alert_manager, '_should_check_condition', return_value=True):
            with patch.object(self.alert_manager.stock_fetcher, 'get_historical_data', return_value=mock_historical):
                self.alert_manager._check_symbol_alerts_with_data("7203", [condition], market_data)

                # 履歴データがキャッシュされていることを確認
                assert market_data["historical_data"] is not None


class TestAlertStrategies:
    """アラート評価戦略のテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.mock_technical_indicators = Mock()
        self.strategy_factory = AlertStrategyFactory(self.mock_technical_indicators)

    def test_price_above_strategy(self):
        """価格上昇戦略のテスト"""
        strategy = self.strategy_factory.get_strategy(AlertType.PRICE_ABOVE)

        # 条件満足の場合
        is_triggered, message, current_value = strategy.evaluate(
            condition_value="2500",
            current_price=Decimal("2800"),
            volume=1000000,
            change_percent=2.5,
            historical_data=None
        )

        assert is_triggered is True
        assert "上回りました" in message
        assert current_value == Decimal("2800")

    def test_volume_spike_strategy(self):
        """出来高急増戦略のテスト"""
        strategy = self.strategy_factory.get_strategy(AlertType.VOLUME_SPIKE)

        # モック履歴データの設定を修正
        mock_historical = Mock()
        mock_historical.empty = False

        # 出来高データのモック（平均を1000000に設定）
        mock_volume_series = Mock()
        mock_mean_series = Mock()
        # iloc[-1]で正しい値を返すように設定
        mock_mean_series.iloc = Mock()
        mock_mean_series.iloc.__getitem__ = Mock(return_value=1000000)
        mock_rolling = Mock()
        mock_rolling.mean.return_value = mock_mean_series
        mock_volume_series.rolling.return_value = mock_rolling
        mock_historical.__getitem__ = Mock(return_value=mock_volume_series)

        # 条件値1.5、現在出来高2000000（平均1000000の2倍）なので発火するはず
        is_triggered, message, current_value = strategy.evaluate(
            condition_value=1.5,
            current_price=Decimal("2800"),
            volume=2000000,
            change_percent=2.5,
            historical_data=mock_historical
        )

        assert is_triggered is True
        assert "出来高急増" in message
        assert current_value == 2.0  # volume_ratio が返される

    def test_rsi_strategy_with_custom_period(self):
        """RSI戦略（カスタム期間）のテスト"""
        strategy = self.strategy_factory.get_strategy(AlertType.RSI_OVERBOUGHT)

        # モック履歴データとRSI計算
        mock_historical = Mock()
        mock_historical.empty = False
        mock_close_series = Mock()
        mock_historical.__getitem__ = Mock(return_value=mock_close_series)

        mock_rsi_series = Mock()
        mock_rsi_series.empty = False
        mock_rsi_series.iloc = [75.0]  # RSI 75
        self.mock_technical_indicators.calculate_rsi.return_value = mock_rsi_series

        is_triggered, message, current_value = strategy.evaluate(
            condition_value=70.0,
            current_price=Decimal("2800"),
            volume=1000000,
            change_percent=2.5,
            historical_data=mock_historical,
            custom_parameters={"rsi_period": 21}  # カスタム期間
        )

        assert is_triggered is True
        assert "買われすぎ" in message
        assert current_value == 75.0

        # カスタム期間でRSI計算が呼ばれていることを確認
        self.mock_technical_indicators.calculate_rsi.assert_called_with(
            mock_close_series, period=21
        )


class TestPersistentAlertManager:
    """永続化対応アラートマネージャーのテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.mock_db_manager = Mock()
        self.mock_stock_fetcher = Mock()
        self.persistent_manager = PersistentAlertManager(
            stock_fetcher=self.mock_stock_fetcher,
            db_manager_instance=self.mock_db_manager
        )

    def test_add_alert_persistence(self):
        """アラート追加の永続化テスト"""
        condition = AlertCondition(
            alert_id="test_alert",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("3000")
        )

        # モックセッション
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db_manager.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        self.mock_db_manager.session_scope.return_value.__exit__ = Mock(return_value=None)

        with patch.object(self.persistent_manager, '_validate_condition', return_value=True):
            result = self.persistent_manager.add_alert(condition)

            assert result is True
            # セッションにモデルが追加されていることを確認
            mock_session.add.assert_called_once()

    def test_get_alert_history_from_database(self):
        """データベースからのアラート履歴取得テスト"""
        # モックトリガーモデル
        mock_trigger_model = Mock()
        mock_trigger_model.to_alert_trigger.return_value = Mock(spec=AlertTrigger)

        mock_session = Mock()
        mock_query = mock_session.query.return_value
        mock_query.filter.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_trigger_model]

        self.mock_db_manager.session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        self.mock_db_manager.session_scope.return_value.__exit__ = Mock(return_value=None)

        history = self.persistent_manager.get_alert_history(symbol="7203", hours=24)

        assert len(history) == 1
        mock_trigger_model.to_alert_trigger.assert_called_once()


class TestConfigManagerErrorHandling:
    """ConfigManagerのエラーハンドリングテスト"""

    def test_save_config_permission_error(self):
        """設定保存時の権限エラーハンドリング"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_manager = ConfigManager(config_path)

            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                with pytest.raises(PermissionError):
                    config_manager.save_config()

    def test_import_config_file_not_found(self):
        """設定インポート時のファイル未存在エラー"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_manager = ConfigManager(config_path)

            non_existent_path = Path(temp_dir) / "non_existent.json"

            with pytest.raises(FileNotFoundError):
                config_manager.import_config(non_existent_path)

    def test_import_config_json_decode_error(self):
        """設定インポート時のJSON形式エラー"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_manager = ConfigManager(config_path)

            # 不正なJSONファイルを作成
            invalid_json_path = Path(temp_dir) / "invalid.json"
            with open(invalid_json_path, 'w') as f:
                f.write("invalid json content")

            with pytest.raises((ValueError, OSError)):
                config_manager.import_config(invalid_json_path)


class TestSecureConfigManager:
    """セキュリティ強化された設定管理のテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "secure_config.json"

    def test_sensitive_key_detection(self):
        """機密情報キーの検出テスト"""
        secure_config = SecureConfigManager(self.config_path)

        assert secure_config._is_sensitive_key("password") is True
        assert secure_config._is_sensitive_key("api_key") is True
        assert secure_config._is_sensitive_key("smtp_password") is True
        assert secure_config._is_sensitive_key("normal_setting") is False

    @patch.dict(os.environ, {'DAYTRADE_API_TIMEOUT': '60', 'DAYTRADE_LOG_LEVEL': 'DEBUG'})
    def test_environment_variable_priority(self):
        """環境変数の優先度テスト"""
        secure_config = SecureConfigManager(self.config_path)
        config_data = {"api_timeout": 30, "log_level": "INFO"}

        # 環境変数が優先されることを確認
        api_timeout = secure_config.get_from_env_or_config("api_timeout", config_data, 15)
        log_level = secure_config.get_from_env_or_config("log_level", config_data, "WARNING")

        assert api_timeout == 60  # 環境変数の値
        assert log_level == "DEBUG"  # 環境変数の値

    def test_config_security_validation(self):
        """設定のセキュリティ検証テスト"""
        secure_config = SecureConfigManager(self.config_path)

        # セキュリティ問題のある設定
        insecure_config = {
            "api_key": "plain_text_api_key",  # プレーンテキストの機密情報
            "password": "123",  # 短いパスワード
            "normal_setting": "value"
        }

        warnings = secure_config.validate_config_security(insecure_config)

        assert len(warnings) >= 2  # 少なくとも2つの警告
        assert any("プレーンテキスト" in warning for warning in warnings)
        assert any("短すぎます" in warning for warning in warnings)


class TestEnvironmentConfigLoader:
    """環境変数設定読み込みのテスト"""

    @patch.dict(os.environ, {
        'DAYTRADE_DATABASE_URL': 'postgresql://test',
        'DAYTRADE_DATABASE_POOL_SIZE': '15',
        'DAYTRADE_API_TIMEOUT': '45'
    })
    def test_load_database_config(self):
        """データベース設定の読み込みテスト"""
        config = EnvironmentConfigLoader.load_database_config()

        assert config['database_url'] == 'postgresql://test'
        assert config['database_pool_size'] == 15
        assert config['database_max_overflow'] == 20  # デフォルト値

    @patch.dict(os.environ, {
        'DAYTRADE_SMTP_SERVER': 'smtp.test.com',
        'DAYTRADE_SMTP_PORT': '465',
        'DAYTRADE_SMTP_TO_EMAILS': 'test1@example.com,test2@example.com'
    })
    def test_load_smtp_config(self):
        """SMTP設定の読み込みテスト"""
        config = EnvironmentConfigLoader.load_smtp_config()

        assert config['smtp_server'] == 'smtp.test.com'
        assert config['smtp_port'] == 465
        assert config['smtp_to_emails'] == ['test1@example.com', 'test2@example.com']


class TestIntegrationScenarios:
    """統合シナリオのテスト"""

    def test_alert_evaluation_with_strategies_end_to_end(self):
        """戦略パターンを使用したアラート評価の統合テスト"""
        mock_stock_fetcher = Mock()
        alert_manager = AlertManager(stock_fetcher=mock_stock_fetcher)

        # アラート条件を追加
        condition = AlertCondition(
            alert_id="integration_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
            priority=AlertPriority.HIGH
        )

        alert_manager.add_alert(condition)

        # 市場データを設定
        market_data = {
            "current_data": {"current_price": 2800, "volume": 1000000, "change_percent": 2.5},
            "historical_data": None
        }

        with patch.object(alert_manager, '_should_check_condition', return_value=True):
            with patch.object(alert_manager, '_handle_alert_trigger') as mock_handle:
                alert_manager._check_symbol_alerts_with_data("7203", [condition], market_data)

                # アラートが発火していることを確認
                mock_handle.assert_called_once()

                # 発火されたトリガーの内容を確認
                triggered_alert = mock_handle.call_args[0][0]
                assert triggered_alert.alert_id == "integration_test"
                assert triggered_alert.symbol == "7203"
                assert triggered_alert.alert_type == AlertType.PRICE_ABOVE
                assert "上回りました" in triggered_alert.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
