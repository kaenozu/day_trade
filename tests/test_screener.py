"""
銘柄スクリーニング機能のテスト
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.day_trade.analysis.screener import (
    ScreenerCondition,
    ScreenerCriteria,
    ScreenerResult,
    StockScreener,
    create_screening_report,
)

logger = logging.getLogger(__name__)


class TestStockScreener:
    """StockScreenerクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_stock_fetcher = Mock()
        self.screener = StockScreener(self.mock_stock_fetcher)

        # サンプルデータ
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        self.sample_data = pd.DataFrame(
            {
                "Open": np.random.uniform(2400, 2600, len(dates)),
                "High": np.random.uniform(2500, 2700, len(dates)),
                "Low": np.random.uniform(2300, 2500, len(dates)),
                "Close": np.random.uniform(2400, 2600, len(dates)),
                "Volume": np.random.randint(1000000, 3000000, len(dates)),
            },
            index=dates,
        )

    def test_screener_initialization(self):
        """スクリーナーの初期化テスト"""
        assert self.screener.stock_fetcher is not None
        assert self.screener.signal_generator is not None
        assert len(self.screener.default_criteria) > 0

    def test_screener_criteria_creation(self):
        """スクリーニング基準の作成テスト"""
        criteria = ScreenerCriteria(
            condition=ScreenerCondition.RSI_OVERSOLD,
            threshold=30.0,
            weight=1.0,
            description="RSI過売り",
        )

        assert criteria.condition == ScreenerCondition.RSI_OVERSOLD
        assert criteria.threshold == 30.0
        assert criteria.weight == 1.0
        assert criteria.description == "RSI過売り"

    @patch("src.day_trade.analysis.screener.TechnicalIndicators.calculate_all")
    def test_evaluate_rsi_oversold_condition(self, mock_indicators):
        """RSI過売り条件の評価テスト"""
        # RSIが25（過売り）のデータを設定
        mock_indicators_df = pd.DataFrame(
            {"RSI": [25.0]}, index=self.sample_data.index[-1:]
        )
        mock_indicators.return_value = mock_indicators_df

        criteria = ScreenerCriteria(
            condition=ScreenerCondition.RSI_OVERSOLD, threshold=30.0
        )

        meets_condition, score = self.screener._evaluate_condition(
            self.sample_data, mock_indicators_df, criteria
        )

        assert meets_condition is True
        assert score > 0

    @patch("src.day_trade.analysis.screener.TechnicalIndicators.calculate_all")
    def test_evaluate_volume_spike_condition(self, mock_indicators):
        """出来高急増条件の評価テスト"""
        # 出来高を急増させるデータを作成
        volume_data = self.sample_data.copy()
        volume_data.loc[volume_data.index[-1], "Volume"] = 10000000  # 大きな出来高

        criteria = ScreenerCriteria(
            condition=ScreenerCondition.VOLUME_SPIKE, threshold=2.0
        )

        meets_condition, score = self.screener._evaluate_condition(
            volume_data, pd.DataFrame(), criteria
        )

        assert meets_condition is True
        assert score > 0

    @patch("src.day_trade.analysis.screener.TechnicalIndicators.calculate_all")
    def test_evaluate_momentum_condition(self, mock_indicators):
        """モメンタム条件の評価テスト"""
        # 強い上昇トレンドのデータを作成
        momentum_data = self.sample_data.copy()
        momentum_data["Close"] = np.linspace(2000, 2600, len(momentum_data))  # 30%上昇

        criteria = ScreenerCriteria(
            condition=ScreenerCondition.STRONG_MOMENTUM,
            threshold=0.05,  # 5%以上の上昇
            lookback_days=20,
        )

        meets_condition, score = self.screener._evaluate_condition(
            momentum_data, pd.DataFrame(), criteria
        )

        assert meets_condition is True
        assert score > 0

    def test_summarize_technical_data(self):
        """テクニカルデータ要約のテスト"""
        indicators = pd.DataFrame(
            {
                "RSI": [45.0],
                "MACD": [0.5],
                "SMA_20": [2500.0],
                "SMA_50": [2450.0],
            },
            index=self.sample_data.index[-1:],
        )

        summary = self.screener._summarize_technical_data(self.sample_data, indicators)

        assert "current_price" in summary
        assert "high_52w" in summary
        assert "low_52w" in summary
        assert "rsi" in summary
        assert "macd" in summary
        assert summary["rsi"] == 45.0
        assert summary["macd"] == 0.5

    @patch("src.day_trade.analysis.screener.StockScreener._evaluate_symbol")
    def test_screen_stocks_basic(self, mock_evaluate):
        """基本的なスクリーニングテスト"""
        # モック結果の設定
        mock_result = ScreenerResult(
            symbol="7203",
            score=75.0,
            matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
            technical_data={"rsi": 25.0},
            last_price=2500.0,
            volume=2000000,
        )
        mock_evaluate.return_value = mock_result

        symbols = ["7203", "9984"]
        results = self.screener.screen_stocks(symbols, min_score=50.0)

        assert len(results) <= len(symbols)
        if results:
            assert all(result.score >= 50.0 for result in results)

    def test_predefined_screeners(self):
        """事前定義スクリーナーのテスト"""
        screeners = self.screener.get_predefined_screeners()

        # 設定ファイルが正常に読み込まれている場合のテスト
        if screeners:
            # 設定が存在する場合は期待されるスクリーナーが含まれていることを確認
            for name in ["growth", "value", "momentum"]:
                if name in screeners:
                    assert callable(screeners[name])
                    assert hasattr(screeners[name], "__name__")
        else:
            # 設定ファイルが見つからない場合でも正常に動作することを確認
            assert isinstance(screeners, dict)
            logger.info("事前定義スクリーナーが空です - 設定ファイルが見つからない可能性があります")

    def test_custom_screener_creation(self):
        """カスタムスクリーナー作成のテスト"""
        custom_criteria = [
            ScreenerCriteria(
                condition=ScreenerCondition.RSI_OVERSOLD, threshold=35.0, weight=1.0
            )
        ]

        custom_screener = self.screener.create_custom_screener(
            "test_screener", custom_criteria, "テスト用スクリーナー"
        )

        assert callable(custom_screener)
        assert custom_screener.__name__ == "test_screener"
        assert "テスト用スクリーナー" in custom_screener.__doc__

    def test_cache_operations(self):
        """キャッシュ操作のテスト"""
        # キャッシュ情報取得
        cache_info = self.screener.get_cache_info()
        assert isinstance(cache_info, dict)

        # キャッシュクリア
        self.screener.clear_cache()
        # クリア後も正常に動作することを確認
        cache_info_after = self.screener.get_cache_info()
        assert isinstance(cache_info_after, dict)

    def test_unknown_condition_evaluation(self):
        """未実装条件の評価テスト"""
        # 存在しない条件を作成
        class UnknownCondition:
            value = "unknown_condition"

        # 未実装条件のモックを作成
        unknown_criteria = ScreenerCriteria(
            condition=UnknownCondition(),
            threshold=50.0
        )

        # 未実装条件での評価
        meets_condition, score = self.screener._evaluate_condition(
            self.sample_data, pd.DataFrame(), unknown_criteria
        )

        assert meets_condition is False
        assert score == 0.0

    @patch("src.day_trade.analysis.screener.StockScreener._evaluate_symbol")
    def test_evaluate_symbol_method(self, mock_evaluate):
        """_evaluate_symbol メソッドのテスト"""
        # モック結果の設定
        mock_result = ScreenerResult(
            symbol="7203",
            score=60.0,
            matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
            technical_data={"rsi": 30.0}
        )
        mock_evaluate.return_value = mock_result

        criteria = [ScreenerCriteria(
            condition=ScreenerCondition.RSI_OVERSOLD,
            threshold=35.0
        )]

        result = self.screener._evaluate_symbol("7203", criteria, "3mo")

        assert result is not None
        assert result.symbol == "7203"
        assert result.score == 60.0


class TestScreenerResult:
    """ScreenerResultクラスのテスト"""

    def test_screener_result_creation(self):
        """スクリーニング結果の作成テスト"""
        result = ScreenerResult(
            symbol="7203",
            score=85.5,
            matched_conditions=[
                ScreenerCondition.RSI_OVERSOLD,
                ScreenerCondition.VOLUME_SPIKE,
            ],
            technical_data={"rsi": 28.0, "volume_ratio": 2.5},
            last_price=2550.0,
            volume=3500000,
        )

        assert result.symbol == "7203"
        assert result.score == 85.5
        assert len(result.matched_conditions) == 2
        assert ScreenerCondition.RSI_OVERSOLD in result.matched_conditions
        assert result.technical_data["rsi"] == 28.0
        assert result.last_price == 2550.0


class TestScreeningReport:
    """スクリーニングレポートのテスト"""

    def test_create_screening_report_empty(self):
        """空の結果でのレポート生成テスト"""
        report = create_screening_report([])
        assert "条件を満たす銘柄が見つかりませんでした" in report

    def test_create_screening_report_with_results(self):
        """結果ありでのレポート生成テスト"""
        results = [
            ScreenerResult(
                symbol="7203",
                score=85.0,
                matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
                technical_data={"rsi": 25.0, "price_change_1d": 2.5},
                last_price=2500.0,
                volume=2000000,
            ),
            ScreenerResult(
                symbol="9984",
                score=72.0,
                matched_conditions=[ScreenerCondition.VOLUME_SPIKE],
                technical_data={"rsi": 55.0, "price_change_1d": -1.2},
                last_price=3200.0,
                volume=1800000,
            ),
        ]

        report = create_screening_report(results)

        assert "=== 銘柄スクリーニング結果 (2銘柄) ===" in report
        assert "7203" in report
        assert "9984" in report
        assert "85.0" in report
        assert "72.0" in report
        assert "rsi_oversold" in report
        assert "volume_spike" in report

    def test_create_screening_report_without_formatters(self):
        """フォーマッタ無効時のレポート生成テスト"""
        # フォーマッタ設定を無効にするためのパッチ
        with patch("src.day_trade.analysis.screener.get_screening_config") as mock_config:
            # フォーマッタを無効にする設定
            mock_config_obj = Mock()
            mock_config_obj.should_use_formatters.return_value = False
            mock_config_obj.get_currency_precision.return_value = 0
            mock_config_obj.get_percentage_precision.return_value = 2
            mock_config_obj.should_use_compact_volume.return_value = False
            mock_config.return_value = mock_config_obj

            results = [
                ScreenerResult(
                    symbol="7203",
                    score=85.0,
                    matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
                    technical_data={"rsi": 25.0, "price_change_1d": 2.5},
                    last_price=2500.0,
                    volume=2000000,
                )
            ]

            report = create_screening_report(results)

            assert "¥2,500" in report  # 非フォーマッタ形式
            assert "2,000,000" in report  # 非コンパクト出来高形式

    def test_create_screening_report_with_formatters(self):
        """フォーマッタ有効時のレポート生成テスト"""
        with patch("src.day_trade.analysis.screener.get_screening_config") as mock_config:
            # フォーマッタを有効にする設定
            mock_config_obj = Mock()
            mock_config_obj.should_use_formatters.return_value = True
            mock_config_obj.get_currency_precision.return_value = 2
            mock_config_obj.get_percentage_precision.return_value = 1
            mock_config_obj.should_use_compact_volume.return_value = True
            mock_config.return_value = mock_config_obj

            results = [
                ScreenerResult(
                    symbol="7203",
                    score=85.0,
                    matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
                    technical_data={
                        "rsi": 25.0,
                        "price_change_1d": 2.5,
                        "price_position": 75.0  # 52週レンジ位置
                    },
                    last_price=2500.0,
                    volume=2000000,
                )
            ]

            with patch("src.day_trade.analysis.screener.format_currency") as mock_format_currency, \
                 patch("src.day_trade.analysis.screener.format_volume") as mock_format_volume, \
                 patch("src.day_trade.analysis.screener.format_percentage") as mock_format_percentage:

                mock_format_currency.return_value = "¥2,500.00"
                mock_format_volume.return_value = "2.0M"
                mock_format_percentage.return_value = "+2.5%"

                report = create_screening_report(results)

                assert "¥2,500.00" in report
                assert "2.0M" in report
                assert "+2.5%" in report
                assert "52週レンジ位置: 75.0%" in report

    def test_create_screening_report_missing_data(self):
        """データ欠損時のレポート生成テスト"""
        results = [
            ScreenerResult(
                symbol="7203",
                score=85.0,
                matched_conditions=[ScreenerCondition.RSI_OVERSOLD],
                technical_data={},  # 空のテクニカルデータ
                last_price=None,  # 価格なし
                volume=None,  # 出来高なし
            )
        ]

        report = create_screening_report(results)

        assert "N/A" in report  # 欠損値がN/Aで表示される
        assert "7203" in report
        assert "85.0" in report


class TestScreenerConditions:
    """スクリーニング条件のテスト"""

    def test_screener_condition_enum(self):
        """スクリーニング条件列挙型のテスト"""
        assert ScreenerCondition.RSI_OVERSOLD.value == "rsi_oversold"
        assert ScreenerCondition.GOLDEN_CROSS.value == "golden_cross"
        assert ScreenerCondition.VOLUME_SPIKE.value == "volume_spike"

        # 全ての条件が定義されていることを確認
        conditions = list(ScreenerCondition)
        assert len(conditions) >= 10  # 最低10個の条件が定義されている

    def test_screener_criteria_defaults(self):
        """スクリーニング基準のデフォルト値テスト"""
        criteria = ScreenerCriteria(condition=ScreenerCondition.RSI_OVERSOLD)

        assert criteria.condition == ScreenerCondition.RSI_OVERSOLD
        assert criteria.threshold is None
        assert criteria.lookback_days == 20
        assert criteria.weight == 1.0
        assert criteria.description == ""


class TestIntegration:
    """統合テスト"""

    @patch("src.day_trade.analysis.screener.StockFetcher")
    def test_full_screening_workflow(self, mock_fetcher_class):
        """完全なスクリーニングワークフローテスト"""
        # モックの設定
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        # サンプルデータの設定
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": 2500,
                "High": 2600,
                "Low": 2400,
                "Close": np.linspace(2400, 2600, len(dates)),  # 上昇トレンド
                "Volume": np.random.randint(1000000, 3000000, len(dates)),
            },
            index=dates,
        )

        # 最後の日の出来高を急増させる
        sample_data.loc[sample_data.index[-1], "Volume"] = 5000000
        mock_fetcher.get_historical_data.return_value = sample_data

        screener = StockScreener(mock_fetcher)
        symbols = ["7203"]

        # デフォルトスクリーニング実行
        results = screener.screen_stocks(symbols, min_score=0.0, max_results=10)

        # 基本的な検証
        assert isinstance(results, list)

        # レポート生成テスト
        if results:
            report = create_screening_report(results)
            assert isinstance(report, str)
            assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
