"""
拡張版スクリーニング機能のテスト
Issue #139の改善点をテスト
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.day_trade.analysis.screener import StockScreener
from src.day_trade.analysis.screener_enhanced import (
    EnhancedStockScreener,
    ScreenerCondition,
    ScreenerCriteria,
    ScreenerResult,
)
from src.day_trade.analysis.screening_config import ScreeningConfig
from src.day_trade.analysis.screening_strategies import (
    ScreeningStrategyFactory,
)


class TestScreeningStrategies:
    """スクリーニング戦略のテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.strategy_factory = ScreeningStrategyFactory()

    def test_rsi_oversold_strategy(self):
        """RSI過売り戦略のテスト"""
        strategy = self.strategy_factory.get_strategy("rsi_oversold")
        assert strategy is not None
        assert strategy.condition_name == "RSI_OVERSOLD"

        # モックデータの作成
        df = pd.DataFrame(
            {"Close": [100, 95, 90, 85, 80], "Volume": [1000, 1100, 1200, 1300, 1400]}
        )

        indicators = pd.DataFrame(
            {
                "RSI": [50, 40, 30, 25, 20]  # 最後がRSI 20（過売り）
            }
        )

        # RSI 20で閾値30のテスト
        meets_condition, score = strategy.evaluate(df, indicators, threshold=30.0)
        assert meets_condition is True
        assert score > 0

        # RSI 35で閾値30のテスト
        indicators_high = pd.DataFrame({"RSI": [50, 40, 35, 35, 35]})
        meets_condition, score = strategy.evaluate(df, indicators_high, threshold=30.0)
        assert meets_condition is False

    def test_golden_cross_strategy(self):
        """ゴールデンクロス戦略のテスト"""
        strategy = self.strategy_factory.get_strategy("golden_cross")
        assert strategy is not None

        # モックデータ（ゴールデンクロス発生）
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        indicators = pd.DataFrame(
            {
                "SMA_20": [
                    99,
                    100,
                    100.5,
                    100.8,
                    101.2,
                ],  # 短期線（前回≤長期線、今回>長期線）
                "SMA_50": [101, 101, 101, 101, 101],  # 長期線
            }
        )

        meets_condition, score = strategy.evaluate(df, indicators)
        assert meets_condition is True
        assert score > 0

    def test_volume_spike_strategy(self):
        """出来高急増戦略のテスト"""
        strategy = self.strategy_factory.get_strategy("volume_spike")
        assert strategy is not None

        # モックデータ（出来高急増）
        df = pd.DataFrame(
            {
                "Close": [100] * 21,
                "Volume": [1000] * 20 + [3000],  # 最後が3倍の出来高
            }
        )

        indicators = pd.DataFrame({"dummy": [0] * 21})

        meets_condition, score = strategy.evaluate(df, indicators, threshold=2.0)
        assert meets_condition is True
        assert score > 0

    def test_all_strategies_available(self):
        """すべての戦略が利用可能かテスト"""
        expected_strategies = [
            "rsi_oversold",
            "rsi_overbought",
            "macd_bullish",
            "macd_bearish",
            "golden_cross",
            "dead_cross",
            "volume_spike",
            "strong_momentum",
            "bollinger_breakout",
            "bollinger_squeeze",
            "price_near_support",
            "price_near_resistance",
            "reversal_pattern",
        ]

        for strategy_name in expected_strategies:
            strategy = self.strategy_factory.get_strategy(strategy_name)
            assert strategy is not None, f"戦略 {strategy_name} が見つかりません"


class TestScreeningConfig:
    """スクリーニング設定のテスト"""

    def test_config_loading(self):
        """設定読み込みのテスト"""
        # テンポラリファイルで設定をテスト
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {
                "default_thresholds": {"RSI_OVERSOLD": 25.0, "VOLUME_SPIKE": 1.8},
                "performance_settings": {"max_workers": 3},
            }
            json.dump(test_config, f)
            config_path = Path(f.name)

        try:
            config = ScreeningConfig(config_path)
            assert config.get_threshold("RSI_OVERSOLD") == 25.0
            assert config.get_threshold("VOLUME_SPIKE") == 1.8
            assert config.get_performance_setting("max_workers") == 3
        finally:
            config_path.unlink()

    def test_default_config(self):
        """デフォルト設定のテスト"""
        # 存在しないパスでデフォルト設定を使用
        config = ScreeningConfig(Path("/non/existent/path"))

        # デフォルト値が設定されていることを確認
        assert config.get_threshold("RSI_OVERSOLD", 30.0) == 30.0
        assert config.get_max_workers() >= 1
        assert config.get_data_period() == "3mo"

    def test_cpu_based_max_workers(self):
        """CPU数に基づく最大ワーカー数のテスト"""
        config = ScreeningConfig()
        max_workers = config.get_max_workers()

        # 1以上で、設定値以下であることを確認
        assert max_workers >= 1
        assert max_workers <= config.get_performance_setting("max_workers", 5)


class TestEnhancedStockScreener:
    """拡張版スクリーナーのテスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.mock_stock_fetcher = Mock()
        self.screener = EnhancedStockScreener(self.mock_stock_fetcher)

    def test_initialization(self):
        """初期化のテスト"""
        assert self.screener is not None
        assert self.screener.stock_fetcher is not None
        assert self.screener.strategy_factory is not None

    def test_get_default_criteria(self):
        """デフォルト基準取得のテスト"""
        criteria = self.screener.get_default_criteria()
        assert len(criteria) > 0
        assert all(isinstance(c, ScreenerCriteria) for c in criteria)

    def test_cache_functionality(self):
        """キャッシュ機能のテスト"""
        # 初期状態
        cache_info = self.screener.get_cache_info()
        assert cache_info["cache_size"] == 0

        # キャッシュクリア
        self.screener.clear_cache()
        cache_info = self.screener.get_cache_info()
        assert cache_info["cache_size"] == 0

    @patch("src.day_trade.analysis.screener_enhanced.TechnicalIndicators")
    def test_evaluate_symbol_enhanced(self, mock_indicators):
        """個別銘柄評価のテスト"""
        # モックデータの設定（十分なデータポイント数）
        mock_df = pd.DataFrame(
            {
                "Close": [100] * 50,  # 50日分のデータ
                "Volume": [1000] * 50,
                "High": [105] * 50,
                "Low": [95] * 50,
            }
        )

        mock_indicators_df = pd.DataFrame(
            {
                "RSI": [25] * 50,  # RSI過売り（最低データポイント数を満たす）
                "SMA_20": [100] * 50,
                "SMA_50": [100] * 50,
            }
        )

        # モックの正しい設定
        with patch.object(self.screener, "_get_data_with_cache", return_value=mock_df):
            mock_indicators.calculate_all.return_value = mock_indicators_df

            criteria = [
                ScreenerCriteria(
                    condition=ScreenerCondition.RSI_OVERSOLD, threshold=30.0, weight=1.0
                )
            ]

            result = self.screener._evaluate_symbol_enhanced(
                "TEST", criteria, "3mo", use_cache=False
            )

            assert result is not None
            assert result.symbol == "TEST"
            assert result.score > 0
            assert ScreenerCondition.RSI_OVERSOLD in result.matched_conditions

    @patch("src.day_trade.analysis.screener_enhanced.TechnicalIndicators")
    def test_screen_stocks_integration(self, mock_indicators):
        """統合スクリーニングのテスト"""
        # モックデータの設定
        mock_df = pd.DataFrame(
            {
                "Close": [100] * 50,
                "Volume": [1000] * 50,
                "High": [105] * 50,
                "Low": [95] * 50,
            }
        )

        mock_indicators_df = pd.DataFrame(
            {
                "RSI": [25] * 50,  # RSI過売り
                "SMA_20": [100] * 50,
                "SMA_50": [100] * 50,
            }
        )

        self.mock_stock_fetcher.get_historical_data.return_value = mock_df
        mock_indicators.calculate_all.return_value = mock_indicators_df

        symbols = ["TEST1", "TEST2"]
        results = self.screener.screen_stocks(
            symbols=symbols, min_score=0.0, max_results=10, use_cache=False
        )

        # 何らかの結果が返されることを確認
        assert isinstance(results, list)


class TestStockScreenerCompatibility:
    """既存StockScreenerの後方互換性テスト"""

    def setup_method(self):
        """テスト用のセットアップ"""
        self.mock_stock_fetcher = Mock()
        self.screener = StockScreener(self.mock_stock_fetcher)

    def test_initialization_compatibility(self):
        """初期化の後方互換性テスト"""
        assert hasattr(self.screener, "stock_fetcher")
        assert hasattr(self.screener, "signal_generator")
        assert hasattr(self.screener, "default_criteria")

    def test_screen_stocks_method_exists(self):
        """screen_stocksメソッドの存在確認"""
        assert hasattr(self.screener, "screen_stocks")
        assert callable(self.screener.screen_stocks)

    def test_existing_methods_compatibility(self):
        """既存メソッドの後方互換性確認"""
        # 既存のメソッドが存在することを確認
        methods_to_check = [
            "create_custom_screener",
            "get_predefined_screeners",
            "_evaluate_symbol",
            "_evaluate_condition",
            "_summarize_technical_data",
        ]

        for method_name in methods_to_check:
            assert hasattr(self.screener, method_name), (
                f"メソッド {method_name} が見つかりません"
            )

    def test_enhanced_features(self):
        """拡張機能の存在確認"""
        # 新機能が追加されていることを確認
        assert hasattr(self.screener, "clear_cache")
        assert hasattr(self.screener, "get_cache_info")


class TestReportGeneration:
    """レポート生成のテスト"""

    def test_create_screening_report(self):
        """スクリーニングレポート生成のテスト"""
        from src.day_trade.analysis.screener import create_screening_report

        # テスト用の結果を作成
        results = [
            ScreenerResult(
                symbol="TEST1",
                score=75.5,
                matched_conditions=[
                    ScreenerCondition.RSI_OVERSOLD,
                    ScreenerCondition.VOLUME_SPIKE,
                ],
                technical_data={
                    "rsi": 25.3,
                    "price_change_1d": 2.5,
                    "price_position": 15.8,
                },
                last_price=1250.0,
                volume=1500000,
            ),
            ScreenerResult(
                symbol="TEST2",
                score=60.0,
                matched_conditions=[ScreenerCondition.GOLDEN_CROSS],
                technical_data={"rsi": 45.0, "price_change_1d": -1.2},
                last_price=890.0,
                volume=800000,
            ),
        ]

        report = create_screening_report(results)
        assert isinstance(report, str)
        assert "TEST1" in report
        assert "TEST2" in report
        assert "75.5" in report  # スコア
        assert "¥1,250" in report  # 価格フォーマット
        assert "1.5M" in report or "1,500,000" in report  # 出来高フォーマット

    def test_empty_results_report(self):
        """空の結果のレポート生成テスト"""
        from src.day_trade.analysis.screener import create_screening_report

        report = create_screening_report([])
        assert "条件を満たす銘柄が見つかりませんでした" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
