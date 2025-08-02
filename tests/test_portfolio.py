"""
ポートフォリオ分析機能のテスト
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.day_trade.core.portfolio import (
    PerformanceReport,
    PortfolioAnalyzer,
    PortfolioMetrics,
    SectorAllocation,
)
from src.day_trade.core.trade_manager import TradeManager, TradeType


class TestPortfolioAnalyzer:
    """PortfolioAnalyzerクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.trade_manager = TradeManager(load_from_db=False)
        self.mock_stock_fetcher = Mock()
        self.analyzer = PortfolioAnalyzer(self.trade_manager, self.mock_stock_fetcher)

        # サンプル取引データを追加
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False)
        self.trade_manager.add_trade("8306", TradeType.BUY, 50, Decimal("800"), persist_to_db=False)
        self.trade_manager.add_trade("9984", TradeType.BUY, 10, Decimal("15000"), persist_to_db=False)

        # 現在価格を設定
        self.trade_manager.update_current_prices(
            {"7203": Decimal("2600"), "8306": Decimal("850"), "9984": Decimal("14500")}
        )

    def test_get_portfolio_metrics_basic(self):
        """基本的なポートフォリオメトリクス取得テスト"""
        metrics = self.analyzer.get_portfolio_metrics()

        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.total_value > 0
        assert metrics.total_cost > 0
        assert isinstance(metrics.total_pnl, Decimal)
        assert isinstance(metrics.total_pnl_percent, Decimal)

    def test_get_portfolio_metrics_with_profit(self):
        """利益が出ている場合のメトリクステスト"""
        # 価格上昇させる
        self.trade_manager.update_current_prices(
            {
                "7203": Decimal("3000"),  # +500円
                "8306": Decimal("1000"),  # +200円
                "9984": Decimal("16000"),  # +1000円
            }
        )

        metrics = self.analyzer.get_portfolio_metrics()

        assert metrics.total_pnl > 0
        assert metrics.total_pnl_percent > 0

    def test_get_portfolio_metrics_with_loss(self):
        """損失が出ている場合のメトリクステスト"""
        # 価格下落させる
        self.trade_manager.update_current_prices(
            {
                "7203": Decimal("2000"),  # -500円
                "8306": Decimal("600"),  # -200円
                "9984": Decimal("14000"),  # -1000円
            }
        )

        metrics = self.analyzer.get_portfolio_metrics()

        assert metrics.total_pnl < 0
        assert metrics.total_pnl_percent < 0

    def test_get_portfolio_metrics_empty_portfolio(self):
        """空のポートフォリオでのメトリクステスト"""
        empty_trade_manager = TradeManager(load_from_db=False)
        empty_analyzer = PortfolioAnalyzer(empty_trade_manager, self.mock_stock_fetcher)

        metrics = empty_analyzer.get_portfolio_metrics()

        assert metrics.total_value == 0
        assert metrics.total_cost == 0
        assert metrics.total_pnl == 0
        assert metrics.total_pnl_percent == 0

    def test_get_sector_allocation(self):
        """セクター配分取得テスト"""
        allocations = self.analyzer.get_sector_allocation()

        assert isinstance(allocations, list)
        assert len(allocations) > 0

        # セクター名の確認
        sectors = [alloc.sector for alloc in allocations]
        assert "Automotive" in sectors  # 7203: トヨタ
        assert "Financial" in sectors  # 8306: 三菱UFJ
        assert "Technology" in sectors  # 9984: ソフトバンク

        # パーセンテージの合計が100%に近いことを確認
        total_percentage = sum(alloc.percentage for alloc in allocations)
        assert abs(total_percentage - 100) < Decimal("0.01")

    def test_get_sector_allocation_single_sector(self):
        """単一セクターの場合の配分テスト"""
        # 単一セクター（Technology）のみの取引
        single_tm = TradeManager(load_from_db=False)
        single_tm.add_trade("9984", TradeType.BUY, 100, Decimal("15000"), persist_to_db=False)
        single_tm.update_current_prices({"9984": Decimal("15500")})

        single_analyzer = PortfolioAnalyzer(single_tm, self.mock_stock_fetcher)
        allocations = single_analyzer.get_sector_allocation()

        assert len(allocations) == 1
        assert allocations[0].sector == "Technology"
        assert allocations[0].percentage == 100

    def test_get_performance_rankings(self):
        """パフォーマンスランキング取得テスト"""
        top_performers, worst_performers = self.analyzer.get_performance_rankings(3)

        assert isinstance(top_performers, list)
        assert isinstance(worst_performers, list)
        assert len(top_performers) <= 3
        assert len(worst_performers) <= 3

        # 各要素が (symbol, pnl_percent) のタプルであることを確認
        for symbol, pnl_pct in top_performers:
            assert isinstance(symbol, str)
            assert isinstance(pnl_pct, Decimal)

        for symbol, pnl_pct in worst_performers:
            assert isinstance(symbol, str)
            assert isinstance(pnl_pct, Decimal)

    def test_generate_performance_report(self):
        """パフォーマンスレポート生成テスト"""
        report = self.analyzer.generate_performance_report()

        assert isinstance(report, PerformanceReport)
        assert isinstance(report.date, datetime)
        assert isinstance(report.metrics, PortfolioMetrics)
        assert isinstance(report.sector_allocations, list)
        assert isinstance(report.top_performers, list)
        assert isinstance(report.worst_performers, list)

    def test_performance_report_to_dict(self):
        """パフォーマンスレポートの辞書変換テスト"""
        report = self.analyzer.generate_performance_report()
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "date" in report_dict
        assert "metrics" in report_dict
        assert "sector_allocations" in report_dict
        assert "top_performers" in report_dict
        assert "worst_performers" in report_dict

    @patch("pandas.DataFrame.to_csv")
    def test_export_report_to_csv(self, mock_to_csv):
        """CSV エクスポート テスト"""
        self.analyzer.export_report_to_csv("test_report.csv")

        # to_csv が呼び出されることを確認
        assert mock_to_csv.called

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_export_report_to_json(self, mock_json_dump, mock_open):
        """JSON エクスポート テスト"""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        self.analyzer.export_report_to_json("test_report.json")

        # ファイルが開かれ、JSONが書き込まれることを確認
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()

    def test_calculate_volatility_with_mock_data(self):
        """ボラティリティ計算テスト（モックデータ使用）"""
        # モックの履歴データを作成
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        mock_data = pd.DataFrame(
            {"Close": [2500 + i * 10 + (i % 5) * 20 for i in range(30)]}, index=dates
        )

        self.mock_stock_fetcher.get_historical_data.return_value = mock_data

        positions = self.trade_manager.get_all_positions()
        volatility = self.analyzer._calculate_portfolio_volatility(positions)

        # ボラティリティが計算されることを確認
        assert volatility is not None
        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_calculate_sharpe_ratio(self):
        """シャープレシオ計算テスト"""
        positions = self.trade_manager.get_all_positions()
        volatility = 0.2  # 20%のボラティリティ

        sharpe = self.analyzer._calculate_sharpe_ratio(positions, volatility)

        # シャープレシオが計算されることを確認
        assert sharpe is not None
        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """ボラティリティがゼロの場合のシャープレシオ計算テスト"""
        positions = self.trade_manager.get_all_positions()
        volatility = 0

        sharpe = self.analyzer._calculate_sharpe_ratio(positions, volatility)

        # ボラティリティがゼロの場合はNoneが返される
        assert sharpe is None

    def test_calculate_max_drawdown(self):
        """最大ドローダウン計算テスト"""
        positions = self.trade_manager.get_all_positions()
        max_dd = self.analyzer._calculate_max_drawdown(positions)

        assert max_dd is not None
        assert isinstance(max_dd, float)
        assert max_dd >= 0

    def test_calculate_win_ratio_no_realized_pnl(self):
        """実現損益がない場合の勝率計算テスト"""
        win_ratio = self.analyzer._calculate_win_ratio()

        # 実現損益がない場合はNoneが返される
        assert win_ratio is None

    def test_calculate_concentration_risk(self):
        """集中リスク計算テスト"""
        positions = self.trade_manager.get_all_positions()
        concentration = self.analyzer._calculate_concentration_risk(positions)

        assert concentration is not None
        assert isinstance(concentration, float)
        assert 0 <= concentration <= 1  # ハーフィンダール指数は0-1の範囲

    def test_calculate_sector_diversity(self):
        """セクター多様性計算テスト"""
        positions = self.trade_manager.get_all_positions()
        diversity = self.analyzer._calculate_sector_diversity(positions)

        assert diversity is not None
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1  # 正規化された多様性指数は0-1の範囲

    def test_sector_mapping_unknown_symbol(self):
        """未知の銘柄のセクターマッピングテスト"""
        # 未知の銘柄を追加
        self.trade_manager.add_trade("0000", TradeType.BUY, 100, Decimal("1000"), persist_to_db=False)
        self.trade_manager.update_current_prices({"0000": Decimal("1100")})

        allocations = self.analyzer.get_sector_allocation()

        # 'Other' セクターが含まれることを確認
        sectors = [alloc.sector for alloc in allocations]
        assert "Other" in sectors

    def test_risk_free_rate_setting(self):
        """リスクフリーレート設定テスト"""
        custom_rate = 0.02  # 2%
        custom_analyzer = PortfolioAnalyzer(
            self.trade_manager, self.mock_stock_fetcher, risk_free_rate=custom_rate
        )

        assert custom_analyzer.risk_free_rate == custom_rate


class TestPortfolioMetrics:
    """PortfolioMetricsクラスのテスト"""

    def test_portfolio_metrics_creation(self):
        """PortfolioMetrics作成テスト"""
        metrics = PortfolioMetrics(
            total_value=Decimal("1000000"),
            total_cost=Decimal("950000"),
            total_pnl=Decimal("50000"),
            total_pnl_percent=Decimal("5.26"),
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            total_return=0.0526,
            win_ratio=0.65,
            concentration_risk=0.3,
            sector_diversity=0.8,
        )

        assert metrics.total_value == Decimal("1000000")
        assert metrics.total_cost == Decimal("950000")
        assert metrics.total_pnl == Decimal("50000")
        assert metrics.total_pnl_percent == Decimal("5.26")
        assert metrics.volatility == 0.15
        assert metrics.sharpe_ratio == 0.8
        assert metrics.max_drawdown == 0.05
        assert metrics.total_return == 0.0526
        assert metrics.win_ratio == 0.65
        assert metrics.concentration_risk == 0.3
        assert metrics.sector_diversity == 0.8


class TestSectorAllocation:
    """SectorAllocationクラスのテスト"""

    def test_sector_allocation_creation(self):
        """SectorAllocation作成テスト"""
        allocation = SectorAllocation(
            sector="Technology",
            value=Decimal("500000"),
            percentage=Decimal("50.0"),
            positions=3,
        )

        assert allocation.sector == "Technology"
        assert allocation.value == Decimal("500000")
        assert allocation.percentage == Decimal("50.0")
        assert allocation.positions == 3


class TestPerformanceReport:
    """PerformanceReportクラスのテスト"""

    def test_performance_report_creation(self):
        """PerformanceReport作成テスト"""
        metrics = PortfolioMetrics(
            total_value=Decimal("1000000"),
            total_cost=Decimal("950000"),
            total_pnl=Decimal("50000"),
            total_pnl_percent=Decimal("5.26"),
        )

        allocations = [
            SectorAllocation("Technology", Decimal("500000"), Decimal("50.0"), 2)
        ]

        top_performers = [("7203", Decimal("10.5"))]
        worst_performers = [("8306", Decimal("-2.1"))]

        report = PerformanceReport(
            date=datetime.now(),
            metrics=metrics,
            sector_allocations=allocations,
            top_performers=top_performers,
            worst_performers=worst_performers,
        )

        assert isinstance(report.date, datetime)
        assert isinstance(report.metrics, PortfolioMetrics)
        assert len(report.sector_allocations) == 1
        assert len(report.top_performers) == 1
        assert len(report.worst_performers) == 1

    def test_performance_report_to_dict(self):
        """PerformanceReport辞書変換テスト"""
        metrics = PortfolioMetrics(
            total_value=Decimal("1000000"),
            total_cost=Decimal("950000"),
            total_pnl=Decimal("50000"),
            total_pnl_percent=Decimal("5.26"),
        )

        report = PerformanceReport(
            date=datetime.now(),
            metrics=metrics,
            sector_allocations=[],
            top_performers=[],
            worst_performers=[],
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert "date" in report_dict
        assert "metrics" in report_dict
        assert "sector_allocations" in report_dict
        assert "top_performers" in report_dict
        assert "worst_performers" in report_dict

        # 日付がISO形式の文字列であることを確認
        assert isinstance(report_dict["date"], str)


if __name__ == "__main__":
    pytest.main([__file__])
