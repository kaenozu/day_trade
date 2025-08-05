"""
アンサンブル設定のテストケース
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.day_trade.analysis.ensemble import EnsembleTradingStrategy
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
                "adaptive": 70.0,  # ensemble.pyのデフォルト値に合わせて修正
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
            _threshold_type,
            threshold_value,
        ) in ensemble_settings.confidence_thresholds.items():
            assert 0 <= threshold_value <= 100

        # 戦略タイプが有効な値であることを確認
        valid_strategy_types = ["conservative", "aggressive", "balanced", "adaptive"]
        assert ensemble_settings.strategy_type in valid_strategy_types

        # 投票タイプが有効な値であることを確認
        valid_voting_types = ["soft", "hard", "weighted"]
        assert ensemble_settings.voting_type in valid_voting_types

    def test_ensemble_settings_validation_errors(self):
        """EnsembleSettingsのバリデーションエラーシナリオのテスト"""

        # 無効な戦略タイプ
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(strategy_type="invalid_strategy")
        assert "strategy_type must be one of" in str(exc_info.value)

        # 無効な投票タイプ
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(voting_type="invalid_voting")
        assert "voting_type must be one of" in str(exc_info.value)

        # 重みの合計が1.0から大きく外れる場合
        invalid_weights = {
            "conservative_rsi": 0.5,
            "aggressive_momentum": 0.5,
            "trend_following": 0.5,  # 合计1.5 > 1.05
            "mean_reversion": 0.0,
            "default_integrated": 0.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(strategy_weights=invalid_weights)
        assert "Sum of strategy weights must be close to 1.0" in str(exc_info.value)

        # 個別重みが範囲外
        invalid_weights_negative = {
            "conservative_rsi": -0.1,  # 負の値
            "aggressive_momentum": 0.4,
            "trend_following": 0.4,
            "mean_reversion": 0.2,
            "default_integrated": 0.1,
        }
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(strategy_weights=invalid_weights_negative)
        assert "must be between 0.0 and 1.0" in str(exc_info.value)

        # 信頼度閾値が範囲外
        invalid_thresholds = {
            "conservative": 150.0,  # 100を超える
            "aggressive": 30.0,
            "balanced": 45.0,
            "adaptive": 40.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(confidence_thresholds=invalid_thresholds)
        assert "must be between 0.0 and 100.0" in str(exc_info.value)

        # 負の信頼度閾値
        negative_thresholds = {
            "conservative": -10.0,  # 負の値
            "aggressive": 30.0,
            "balanced": 45.0,
            "adaptive": 40.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            EnsembleSettings(confidence_thresholds=negative_thresholds)
        assert "must be between 0.0 and 100.0" in str(exc_info.value)

    def test_ensemble_settings_sync_with_ensemble_py(self):
        """EnsembleSettingsのデフォルト値とensemble.pyの同期確認"""
        from src.day_trade.analysis.ensemble import EnsembleStrategy

        # EnsembleSettingsのデフォルト値を取得
        settings = EnsembleSettings()

        # EnsembleTradingStrategyのインスタンスを作成してデフォルト値を確認
        EnsembleTradingStrategy()

        # デフォルト重みの同期確認
        # ensemble.pyの_initialize_weightsで定義されている値と比較
        expected_balanced_weights = {
            "conservative_rsi": 0.2,
            "aggressive_momentum": 0.25,
            "trend_following": 0.25,
            "mean_reversion": 0.2,
            "default_integrated": 0.1,
        }
        assert settings.strategy_weights == expected_balanced_weights

        # 信頼度閾値の同期確認（正しいEnum型を使用）
        # balanced戦略の閾値をテスト
        balanced_strategy = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED
        )
        expected_balanced_threshold = balanced_strategy._get_confidence_threshold()
        assert settings.confidence_thresholds["balanced"] == expected_balanced_threshold

        # conservative戦略の閾値をテスト
        conservative_strategy = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.CONSERVATIVE
        )
        expected_conservative_threshold = (
            conservative_strategy._get_confidence_threshold()
        )
        assert (
            settings.confidence_thresholds["conservative"]
            == expected_conservative_threshold
        )

        # aggressive戦略の閾値をテスト
        aggressive_strategy = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.AGGRESSIVE
        )
        expected_aggressive_threshold = aggressive_strategy._get_confidence_threshold()
        assert (
            settings.confidence_thresholds["aggressive"]
            == expected_aggressive_threshold
        )

    def test_ensemble_settings_complete_validation(self, temp_config_file):
        """get_ensemble_settings()が返すEnsembleSettingsインスタンスの完全な検証"""
        config_manager = ConfigManager(temp_config_file)
        ensemble_settings = config_manager.get_ensemble_settings()

        # Pydanticのmodel_dump()を使用した網羅的な検証
        settings_dict = ensemble_settings.model_dump()

        # 必須フィールドの存在確認
        required_fields = [
            "enabled",
            "strategy_type",
            "voting_type",
            "performance_file_path",
            "strategy_weights",
            "confidence_thresholds",
            "meta_learning_enabled",
            "adaptive_weights_enabled",
        ]
        for field in required_fields:
            assert field in settings_dict, f"Required field {field} missing"

        # 各フィールドの型確認
        assert isinstance(settings_dict["enabled"], bool)
        assert isinstance(settings_dict["strategy_type"], str)
        assert isinstance(settings_dict["voting_type"], str)
        assert isinstance(settings_dict["performance_file_path"], str)
        assert isinstance(settings_dict["strategy_weights"], dict)
        assert isinstance(settings_dict["confidence_thresholds"], dict)
        assert isinstance(settings_dict["meta_learning_enabled"], bool)
        assert isinstance(settings_dict["adaptive_weights_enabled"], bool)

        # sample_configの値が正しくマッピングされているか確認
        assert settings_dict["enabled"] is True
        assert settings_dict["strategy_type"] == "balanced"
        assert settings_dict["voting_type"] == "soft"
        assert settings_dict["performance_file_path"] == "test_performance.json"
        assert settings_dict["meta_learning_enabled"] is True
        assert settings_dict["adaptive_weights_enabled"] is False

    def test_ensemble_settings_dynamic_strategy_names(self):
        """期待される戦略名の動的な取得テスト"""
        # EnsembleTradingStrategyから動的に戦略名を取得
        strategy = EnsembleTradingStrategy()
        actual_strategies = list(strategy.strategies.keys())

        # EnsembleSettingsのデフォルト重みと比較
        settings = EnsembleSettings()
        expected_strategies = list(settings.strategy_weights.keys())

        # 戦略名の一致確認
        assert (
            set(actual_strategies) == set(expected_strategies)
        ), f"Strategy names mismatch: expected {expected_strategies}, got {actual_strategies}"

        # 各戦略に重みが設定されていることを確認
        for strategy_name in actual_strategies:
            assert strategy_name in settings.strategy_weights
            assert 0.0 <= settings.strategy_weights[strategy_name] <= 1.0

    def test_ensemble_settings_edge_cases(self):
        """エッジケースのテスト"""
        # 重みの合計がぎりぎり許容範囲内
        borderline_weights = {
            "conservative_rsi": 0.19,
            "aggressive_momentum": 0.24,
            "trend_following": 0.24,
            "mean_reversion": 0.19,
            "default_integrated": 0.095,  # 合計: 0.95 (ぎりぎり許容)
        }
        settings = EnsembleSettings(strategy_weights=borderline_weights)
        assert abs(sum(settings.strategy_weights.values()) - 0.95) < 0.01

        # 重みが0の戦略を含む場合
        zero_weight_settings = {
            "conservative_rsi": 0.5,
            "aggressive_momentum": 0.5,
            "trend_following": 0.0,  # 0重み
            "mean_reversion": 0.0,  # 0重み
            "default_integrated": 0.0,  # 0重み
        }
        settings = EnsembleSettings(strategy_weights=zero_weight_settings)
        assert settings.strategy_weights["trend_following"] == 0.0

        # 閾値の境界値
        boundary_thresholds = {
            "conservative": 0.0,  # 最小値
            "aggressive": 100.0,  # 最大値
            "balanced": 50.0,  # 中間値
            "adaptive": 25.5,  # 小数点値
        }
        settings = EnsembleSettings(confidence_thresholds=boundary_thresholds)
        assert settings.confidence_thresholds["conservative"] == 0.0
        assert settings.confidence_thresholds["aggressive"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
