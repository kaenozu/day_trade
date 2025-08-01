"""
アンサンブル設定のテストケース
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.day_trade.config.config_manager import ConfigManager, EnsembleSettings


class TestEnsembleConfig:
    """アンサンブル設定のテストクラス"""

    @pytest.fixture
    def sample_config(self):
        """サンプル設定データ"""
        return {
            "version": "1.0.0",
            "name": "Test Config",
            "watchlist": {
                "symbols": [
                    {
                        "code": "7203",
                        "name": "トヨタ",
                        "group": "主力株",
                        "priority": "high",
                    }
                ],
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30",
                },
            },
            "analysis": {
                "technical_indicators": {
                    "enabled": True,
                    "sma_periods": [5, 20],
                    "ema_periods": [12, 26],
                    "rsi_period": 14,
                    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                    "bollinger_params": {"period": 20, "std_dev": 2},
                },
                "pattern_recognition": {
                    "enabled": True,
                    "patterns": ["support_resistance"],
                },
                "signal_generation": {
                    "enabled": True,
                    "strategies": ["sma_crossover"],
                    "confidence_threshold": 0.6,
                },
                "ensemble": {
                    "enabled": True,
                    "strategy_type": "balanced",
                    "voting_type": "soft",
                    "performance_file_path": "test_performance.json",
                    "strategy_weights": {
                        "conservative_rsi": 0.3,
                        "aggressive_momentum": 0.2,
                        "trend_following": 0.2,
                        "mean_reversion": 0.2,
                        "default_integrated": 0.1,
                    },
                    "confidence_thresholds": {
                        "conservative": 65.0,
                        "aggressive": 25.0,
                        "balanced": 40.0,
                        "adaptive": 35.0,
                    },
                    "meta_learning_enabled": True,
                    "adaptive_weights_enabled": False,
                },
            },
            "alerts": {
                "enabled": True,
                "price_alerts": {"enabled": True, "threshold_percent": 2.0},
                "volume_alerts": {"enabled": True, "volume_spike_ratio": 2.0},
                "technical_alerts": {
                    "enabled": True,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                },
                "notification_methods": ["console"],
            },
            "backtest": {
                "enabled": False,
                "period_days": 30,
                "initial_capital": 1000000,
                "position_size_percent": 10,
                "max_positions": 5,
                "stop_loss_percent": -5.0,
                "take_profit_percent": 10.0,
            },
            "reports": {
                "enabled": True,
                "output_directory": "reports",
                "formats": ["json"],
                "daily_report": {
                    "enabled": True,
                    "include_charts": False,
                    "include_signals": True,
                    "include_portfolio": True,
                },
                "weekly_summary": {"enabled": True, "generate_on_friday": True},
            },
            "execution": {
                "max_concurrent_requests": 5,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "error_tolerance": "continue",
                "log_level": "INFO",
            },
            "database": {
                "url": "sqlite:///test.db",
                "backup_enabled": True,
                "backup_interval_hours": 24,
            },
        }

    @pytest.fixture
    def temp_config_file(self, sample_config):
        """一時設定ファイル"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_config, f, ensure_ascii=False, indent=2)
            temp_path = f.name

        yield temp_path

        # クリーンアップ
        Path(temp_path).unlink(missing_ok=True)

    def test_ensemble_settings_with_config(self, temp_config_file):
        """設定ファイルからのアンサンブル設定読み込みテスト"""
        config_manager = ConfigManager(temp_config_file)
        ensemble_settings = config_manager.get_ensemble_settings()

        assert isinstance(ensemble_settings, EnsembleSettings)
        assert ensemble_settings.enabled is True
        assert ensemble_settings.strategy_type == "balanced"
        assert ensemble_settings.voting_type == "soft"
        assert ensemble_settings.performance_file_path == "test_performance.json"
        assert ensemble_settings.meta_learning_enabled is True
        assert ensemble_settings.adaptive_weights_enabled is False

        # 戦略重みのテスト
        assert ensemble_settings.strategy_weights["conservative_rsi"] == 0.3
        assert ensemble_settings.strategy_weights["aggressive_momentum"] == 0.2

        # 信頼度閾値のテスト
        assert ensemble_settings.confidence_thresholds["conservative"] == 65.0
        assert ensemble_settings.confidence_thresholds["aggressive"] == 25.0

    def test_ensemble_settings_default_values(self):
        """デフォルト値でのアンサンブル設定テスト"""
        # ensembleセクションがない設定でテスト
        minimal_config = {
            "watchlist": {
                "symbols": [
                    {
                        "code": "7203",
                        "name": "Test",
                        "group": "Test",
                        "priority": "high",
                    }
                ],
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30",
                },
            },
            "analysis": {
                "technical_indicators": {
                    "enabled": True,
                    "sma_periods": [5],
                    "ema_periods": [12],
                    "rsi_period": 14,
                    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                    "bollinger_params": {"period": 20, "std_dev": 2},
                },
                "pattern_recognition": {"enabled": True, "patterns": ["test"]},
                "signal_generation": {
                    "enabled": True,
                    "strategies": ["test"],
                    "confidence_threshold": 0.6,
                },
            },
            "alerts": {
                "enabled": True,
                "price_alerts": {},
                "volume_alerts": {},
                "technical_alerts": {},
                "notification_methods": [],
            },
            "backtest": {
                "enabled": False,
                "period_days": 30,
                "initial_capital": 1000000,
                "position_size_percent": 10,
                "max_positions": 5,
                "stop_loss_percent": -5.0,
                "take_profit_percent": 10.0,
            },
            "reports": {
                "enabled": True,
                "output_directory": "reports",
                "formats": ["json"],
                "daily_report": {},
                "weekly_summary": {},
            },
            "execution": {
                "max_concurrent_requests": 5,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "error_tolerance": "continue",
                "log_level": "INFO",
            },
            "database": {
                "url": "sqlite:///test.db",
                "backup_enabled": True,
                "backup_interval_hours": 24,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(minimal_config, f, ensure_ascii=False, indent=2)
            temp_path = f.name

        try:
            config_manager = ConfigManager(temp_path)
            ensemble_settings = config_manager.get_ensemble_settings()

            # デフォルト値の確認
            assert ensemble_settings.enabled is True  # デフォルトでは有効
            assert ensemble_settings.strategy_type == "balanced"
            assert ensemble_settings.voting_type == "soft"
            assert (
                ensemble_settings.performance_file_path
                == "data/ensemble_performance.json"
            )
            assert ensemble_settings.meta_learning_enabled is True
            assert ensemble_settings.adaptive_weights_enabled is True

            # デフォルト重みの確認
            expected_weights = {
                "conservative_rsi": 0.2,
                "aggressive_momentum": 0.25,
                "trend_following": 0.25,
                "mean_reversion": 0.2,
                "default_integrated": 0.1,
            }
            assert ensemble_settings.strategy_weights == expected_weights

            # デフォルト閾値の確認
            expected_thresholds = {
                "conservative": 60.0,
                "aggressive": 30.0,
                "balanced": 45.0,
                "adaptive": 40.0,
            }
            assert ensemble_settings.confidence_thresholds == expected_thresholds

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_ensemble_settings_partial_config(self):
        """部分的な設定でのアンサンブル設定テスト"""
        partial_config = {
            "watchlist": {
                "symbols": [
                    {
                        "code": "7203",
                        "name": "Test",
                        "group": "Test",
                        "priority": "high",
                    }
                ],
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30",
                },
            },
            "analysis": {
                "technical_indicators": {
                    "enabled": True,
                    "sma_periods": [5],
                    "ema_periods": [12],
                    "rsi_period": 14,
                    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
                    "bollinger_params": {"period": 20, "std_dev": 2},
                },
                "pattern_recognition": {"enabled": True, "patterns": ["test"]},
                "signal_generation": {
                    "enabled": True,
                    "strategies": ["test"],
                    "confidence_threshold": 0.6,
                },
                "ensemble": {
                    "enabled": False,
                    "strategy_type": "conservative",
                    # 他の設定は省略
                },
            },
            "alerts": {
                "enabled": True,
                "price_alerts": {},
                "volume_alerts": {},
                "technical_alerts": {},
                "notification_methods": [],
            },
            "backtest": {
                "enabled": False,
                "period_days": 30,
                "initial_capital": 1000000,
                "position_size_percent": 10,
                "max_positions": 5,
                "stop_loss_percent": -5.0,
                "take_profit_percent": 10.0,
            },
            "reports": {
                "enabled": True,
                "output_directory": "reports",
                "formats": ["json"],
                "daily_report": {},
                "weekly_summary": {},
            },
            "execution": {
                "max_concurrent_requests": 5,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "error_tolerance": "continue",
                "log_level": "INFO",
            },
            "database": {
                "url": "sqlite:///test.db",
                "backup_enabled": True,
                "backup_interval_hours": 24,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(partial_config, f, ensure_ascii=False, indent=2)
            temp_path = f.name

        try:
            config_manager = ConfigManager(temp_path)
            ensemble_settings = config_manager.get_ensemble_settings()

            # 設定された値
            assert ensemble_settings.enabled is False
            assert ensemble_settings.strategy_type == "conservative"

            # デフォルト値
            assert ensemble_settings.voting_type == "soft"
            assert ensemble_settings.meta_learning_enabled is True

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_ensemble_settings_validation(self, temp_config_file):
        """アンサンブル設定の妥当性テスト"""
        config_manager = ConfigManager(temp_config_file)
        ensemble_settings = config_manager.get_ensemble_settings()

        # 重みの合計が1.0に近いことを確認
        total_weight = sum(ensemble_settings.strategy_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

        # すべての戦略に重みが設定されていることを確認
        expected_strategies = [
            "conservative_rsi",
            "aggressive_momentum",
            "trend_following",
            "mean_reversion",
            "default_integrated",
        ]
        for strategy in expected_strategies:
            assert strategy in ensemble_settings.strategy_weights
            assert ensemble_settings.strategy_weights[strategy] >= 0

        # 信頼度閾値が妥当な範囲にあることを確認
        for (
            threshold_type,
            threshold_value,
        ) in ensemble_settings.confidence_thresholds.items():
            assert 0 <= threshold_value <= 100

        # 戦略タイプが有効な値であることを確認
        valid_strategy_types = ["conservative", "aggressive", "balanced", "adaptive"]
        assert ensemble_settings.strategy_type in valid_strategy_types

        # 投票タイプが有効な値であることを確認
        valid_voting_types = ["soft", "hard", "weighted"]
        assert ensemble_settings.voting_type in valid_voting_types


if __name__ == "__main__":
    pytest.main([__file__])
