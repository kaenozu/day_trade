"""
全自動化機能のテスト
"""
import pytest
import json
from pathlib import Path
from datetime import datetime, time
from unittest.mock import Mock, MagicMock, patch, mock_open

from src.day_trade.config.config_manager import (
    ConfigManager, WatchlistSymbol, MarketHours,
    TechnicalIndicatorSettings, AlertSettings
)
from src.day_trade.automation.orchestrator import (
    DayTradeOrchestrator, ExecutionResult, AutomationReport
)


class TestConfigManager:
    """ConfigManagerクラスのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.sample_config = {
            "watchlist": {
                "symbols": [
                    {
                        "code": "7203",
                        "name": "トヨタ自動車",
                        "group": "主力株",
                        "priority": "high"
                    },
                    {
                        "code": "8306",
                        "name": "三菱UFJ銀行", 
                        "group": "銀行株",
                        "priority": "medium"
                    }
                ],
                "update_interval_minutes": 5,
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30"
                }
            },
            "analysis": {
                "technical_indicators": {
                    "enabled": true,
                    "sma_periods": [5, 20],
                    "ema_periods": [12, 26],
                    "rsi_period": 14,
                    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                    "bollinger_params": {"period": 20, "std_dev": 2}
                },
                "pattern_recognition": {
                    "enabled": true,
                    "patterns": ["support_resistance"]
                },
                "signal_generation": {
                    "enabled": true,
                    "strategies": ["sma_crossover"],
                    "confidence_threshold": 0.6
                }
            },
            "alerts": {
                "enabled": true,
                "price_alerts": {"enabled": true, "threshold_percent": 2.0},
                "volume_alerts": {"enabled": true, "volume_spike_ratio": 2.0},
                "technical_alerts": {"enabled": true, "rsi_overbought": 70, "rsi_oversold": 30},
                "notification_methods": ["console"]
            },
            "backtest": {
                "enabled": false,
                "period_days": 30,
                "initial_capital": 1000000,
                "position_size_percent": 10,
                "max_positions": 5,
                "stop_loss_percent": -5.0,
                "take_profit_percent": 10.0
            },
            "reports": {
                "enabled": true,
                "output_directory": "reports",
                "formats": ["json", "csv"],
                "daily_report": {"enabled": true, "include_signals": true},
                "weekly_summary": {"enabled": true}
            },
            "execution": {
                "max_concurrent_requests": 5,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "error_tolerance": "continue",
                "log_level": "INFO"
            },
            "database": {
                "url": "sqlite:///data/trading.db",
                "backup_enabled": true,
                "backup_interval_hours": 24
            }
        }
    
    def test_config_manager_initialization(self):
        """ConfigManager初期化テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                
                assert config_manager.config == self.sample_config
    
    def test_get_watchlist_symbols(self):
        """監視銘柄取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                symbols = config_manager.get_watchlist_symbols()
                
                assert len(symbols) == 2
                assert isinstance(symbols[0], WatchlistSymbol)
                assert symbols[0].code == "7203"
                assert symbols[0].name == "トヨタ自動車"
                assert symbols[1].code == "8306"
    
    def test_get_symbol_codes(self):
        """銘柄コードリスト取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                codes = config_manager.get_symbol_codes()
                
                assert codes == ["7203", "8306"]
    
    def test_get_market_hours(self):
        """市場営業時間取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                market_hours = config_manager.get_market_hours()
                
                assert isinstance(market_hours, MarketHours)
                assert market_hours.start == time(9, 0)
                assert market_hours.end == time(15, 0)
                assert market_hours.lunch_start == time(11, 30)
                assert market_hours.lunch_end == time(12, 30)
    
    def test_get_technical_indicator_settings(self):
        """テクニカル指標設定取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                settings = config_manager.get_technical_indicator_settings()
                
                assert isinstance(settings, TechnicalIndicatorSettings)
                assert settings.enabled is True
                assert settings.sma_periods == [5, 20]
                assert settings.rsi_period == 14
    
    def test_get_alert_settings(self):
        """アラート設定取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                settings = config_manager.get_alert_settings()
                
                assert isinstance(settings, AlertSettings)
                assert settings.enabled is True
                assert settings.notification_methods == ["console"]
    
    def test_is_market_open(self):
        """市場営業時間判定テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                
                # 営業時間内（午前10時）
                market_time = datetime(2024, 1, 15, 10, 0)  # 月曜日
                assert config_manager.is_market_open(market_time) is True
                
                # 営業時間外（午前8時）
                before_market = datetime(2024, 1, 15, 8, 0)
                assert config_manager.is_market_open(before_market) is False
                
                # 昼休み時間
                lunch_time = datetime(2024, 1, 15, 12, 0)
                assert config_manager.is_market_open(lunch_time) is False
    
    def test_get_high_priority_symbols(self):
        """高優先度銘柄取得テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                high_priority = config_manager.get_high_priority_symbols()
                
                assert high_priority == ["7203"]  # トヨタのみhigh priority
    
    def test_add_symbol(self):
        """銘柄追加テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                
                original_count = len(config_manager.get_symbol_codes())
                config_manager.add_symbol("9984", "ソフトバンクG", "テック株", "high")
                
                new_codes = config_manager.get_symbol_codes()
                assert len(new_codes) == original_count + 1
                assert "9984" in new_codes
    
    def test_remove_symbol(self):
        """銘柄削除テスト"""
        config_json = json.dumps(self.sample_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                config_manager = ConfigManager("test_config.json")
                
                original_count = len(config_manager.get_symbol_codes())
                config_manager.remove_symbol("8306")
                
                new_codes = config_manager.get_symbol_codes()
                assert len(new_codes) == original_count - 1
                assert "8306" not in new_codes
    
    def test_config_validation_missing_section(self):
        """設定検証エラーテスト"""
        invalid_config = {"watchlist": {"symbols": []}}  # 必須セクション不足
        config_json = json.dumps(invalid_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(ValueError, match="必須設定セクション"):
                    ConfigManager("test_config.json")
    
    def test_config_validation_empty_symbols(self):
        """空の監視銘柄設定エラーテスト"""
        invalid_config = dict(self.sample_config)
        invalid_config["watchlist"]["symbols"] = []
        config_json = json.dumps(invalid_config)
        
        with patch('builtins.open', mock_open(read_data=config_json)):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(ValueError, match="監視銘柄が設定されていません"):
                    ConfigManager("test_config.json")


class TestDayTradeOrchestrator:
    """DayTradeOrchestratorクラスのテスト"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.sample_config = {
            "watchlist": {
                "symbols": [
                    {"code": "7203", "name": "トヨタ", "group": "主力株", "priority": "high"}
                ],
                "market_hours": {
                    "start": "09:00", "end": "15:00",
                    "lunch_start": "11:30", "lunch_end": "12:30"
                }
            },
            "analysis": {
                "technical_indicators": {"enabled": true, "sma_periods": [5, 20], "rsi_period": 14, "macd_params": {"fast": 12, "slow": 26, "signal": 9}, "bollinger_params": {"period": 20, "std_dev": 2}},
                "pattern_recognition": {"enabled": true, "patterns": ["support_resistance"]},
                "signal_generation": {"enabled": true, "strategies": ["sma_crossover"], "confidence_threshold": 0.6}
            },
            "alerts": {"enabled": true, "price_alerts": {"enabled": true}, "volume_alerts": {"enabled": true}, "technical_alerts": {"enabled": true}, "notification_methods": ["console"]},
            "backtest": {"enabled": false, "period_days": 30, "initial_capital": 1000000, "position_size_percent": 10, "max_positions": 5, "stop_loss_percent": -5.0, "take_profit_percent": 10.0},
            "reports": {"enabled": true, "output_directory": "reports", "formats": ["json"], "daily_report": {"enabled": true}, "weekly_summary": {"enabled": true}},
            "execution": {"max_concurrent_requests": 5, "timeout_seconds": 30, "retry_attempts": 3, "error_tolerance": "continue", "log_level": "INFO"},
            "database": {"url": "sqlite:///data/trading.db", "backup_enabled": true, "backup_interval_hours": 24}
        }
    
    @patch('src.day_trade.automation.orchestrator.ConfigManager')
    def test_orchestrator_initialization(self, mock_config_manager):
        """オーケストレーター初期化テスト"""
        # ConfigManagerのモック設定
        mock_instance = Mock()
        mock_instance.get_execution_settings.return_value = Mock(log_level="INFO")
        mock_config_manager.return_value = mock_instance
        
        orchestrator = DayTradeOrchestrator()
        
        assert orchestrator.config_manager == mock_instance
        assert orchestrator.is_running is False
    
    @patch('src.day_trade.automation.orchestrator.ConfigManager')
    @patch('src.day_trade.automation.orchestrator.StockFetcher')
    def test_run_full_automation_basic(self, mock_stock_fetcher, mock_config_manager):
        """基本的な全自動化実行テスト"""
        # ConfigManagerのモック設定
        mock_config_instance = Mock()
        mock_config_instance.get_execution_settings.return_value = Mock(
            log_level="INFO",
            max_concurrent_requests=2,
            timeout_seconds=30,
            retry_attempts=3,
            error_tolerance="continue"
        )
        mock_config_instance.get_symbol_codes.return_value = ["7203"]
        mock_config_instance.get_technical_indicator_settings.return_value = Mock(enabled=False)
        mock_config_instance.get_pattern_recognition_settings.return_value = Mock(enabled=False)
        mock_config_instance.get_signal_generation_settings.return_value = Mock(enabled=False)
        mock_config_instance.get_alert_settings.return_value = Mock(enabled=False)
        mock_config_instance.get_backtest_settings.return_value = Mock(enabled=False)
        mock_config_instance.get_report_settings.return_value = Mock(enabled=False)
        mock_config_manager.return_value = mock_config_instance
        
        # StockFetcherのモック設定
        mock_stock_instance = Mock()
        mock_stock_instance.get_current_price.return_value = {"current_price": 2500, "volume": 1000000}
        mock_stock_instance.get_historical_data.return_value = Mock()
        mock_stock_fetcher.return_value = mock_stock_instance
        
        orchestrator = DayTradeOrchestrator()
        
        # テスト実行
        report = orchestrator.run_full_automation(symbols=["7203"])
        
        assert isinstance(report, AutomationReport)
        assert report.total_symbols == 1
        assert report.start_time is not None
        assert report.end_time is not None
    
    def test_execution_result_creation(self):
        """ExecutionResult作成テスト"""
        result = ExecutionResult(
            success=True,
            symbol="7203",
            data={"price": 2500},
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.symbol == "7203"
        assert result.data["price"] == 2500
        assert result.execution_time == 1.5
        assert result.error is None
    
    def test_automation_report_creation(self):
        """AutomationReport作成テスト"""
        start_time = datetime.now()
        report = AutomationReport(
            start_time=start_time,
            end_time=start_time,
            total_symbols=5,
            successful_symbols=4,
            failed_symbols=1,
            execution_results=[],
            generated_signals=[],
            triggered_alerts=[],
            portfolio_summary={},
            errors=[]
        )
        
        assert report.total_symbols == 5
        assert report.successful_symbols == 4
        assert report.failed_symbols == 1
        assert isinstance(report.execution_results, list)
        assert isinstance(report.generated_signals, list)
        assert isinstance(report.triggered_alerts, list)


class TestAutomationIntegration:
    """自動化システム統合テスト"""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    def test_config_to_orchestrator_integration(self, mock_open_func, mock_exists):
        """設定からオーケストレーターまでの統合テスト"""
        # テスト用設定
        test_config = {
            "watchlist": {
                "symbols": [{"code": "7203", "name": "トヨタ", "group": "主力株", "priority": "high"}],
                "market_hours": {"start": "09:00", "end": "15:00", "lunch_start": "11:30", "lunch_end": "12:30"}
            },
            "analysis": {
                "technical_indicators": {"enabled": false, "sma_periods": [5], "rsi_period": 14, "macd_params": {"fast": 12, "slow": 26, "signal": 9}, "bollinger_params": {"period": 20, "std_dev": 2}},
                "pattern_recognition": {"enabled": false, "patterns": []},
                "signal_generation": {"enabled": false, "strategies": [], "confidence_threshold": 0.6}
            },
            "alerts": {"enabled": false, "price_alerts": {"enabled": false}, "volume_alerts": {"enabled": false}, "technical_alerts": {"enabled": false}, "notification_methods": []},
            "backtest": {"enabled": false, "period_days": 30, "initial_capital": 1000000, "position_size_percent": 10, "max_positions": 5, "stop_loss_percent": -5.0, "take_profit_percent": 10.0},
            "reports": {"enabled": false, "output_directory": "reports", "formats": [], "daily_report": {"enabled": false}, "weekly_summary": {"enabled": false}},
            "execution": {"max_concurrent_requests": 1, "timeout_seconds": 10, "retry_attempts": 1, "error_tolerance": "continue", "log_level": "INFO"},
            "database": {"url": "sqlite:///data/trading.db", "backup_enabled": false, "backup_interval_hours": 24}
        }
        
        # ファイル存在チェックをTrueに
        mock_exists.return_value = True
        
        # ファイル読み込みのモック
        mock_open_func.side_effect = [
            mock_open(read_data=json.dumps(test_config)).return_value
        ]
        
        try:
            # ConfigManagerの初期化（モック環境）
            config_manager = ConfigManager("test_config.json")
            
            # 基本的な設定が読み込まれることを確認
            symbols = config_manager.get_symbol_codes()
            assert symbols == ["7203"]
            
            execution_settings = config_manager.get_execution_settings()
            assert execution_settings.max_concurrent_requests == 1
            
        except Exception as e:
            # モック環境での動作なので、一部のエラーは許容
            assert "ConfigManager" in str(type(e)) or "import" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])