"""
ポートフォリオ管理システムのテスト
"""

from datetime import datetime
from decimal import Decimal

import pytest

from src.day_trade.automation.portfolio_manager import (
    PortfolioManager,
    PortfolioSummary,
    Position,
)
from src.day_trade.core.trade_manager import Trade, TradeType


class TestPosition:
    """ポジションクラスのテスト"""

    def test_position_initialization(self):
        """ポジション初期化のテスト"""
        position = Position("7203")

        assert position.symbol == "7203"
        assert position.quantity == 0
        assert position.average_price == Decimal("0")
        assert position.is_flat()
        assert not position.is_long()
        assert not position.is_short()

    def test_market_price_update(self):
        """市場価格更新のテスト"""
        position = Position("7203")
        position.quantity = 100
        position.average_price = Decimal("2500.0")

        position.update_market_price(Decimal("2600.0"))

        assert position.current_price == Decimal("2600.0")
        assert position.market_value == Decimal("260000.0")
        assert position.unrealized_pnl == Decimal("10000.0")  # (2600-2500) * 100

    def test_long_position_trade(self):
        """ロングポジション取引のテスト"""
        position = Position("7203")

        # 初回買い注文
        trade1 = Trade(
            id="t1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        position.add_trade(trade1)

        assert position.quantity == 100
        assert position.average_price == Decimal("2500.0")
        assert position.is_long()
        assert position.trades_count == 1
        assert position.total_commission == Decimal("25.0")

        # 追加買い注文（平均価格が変動）
        trade2 = Trade(
            id="t2",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=200,
            price=Decimal("2600.0"),
            timestamp=datetime.now(),
            commission=Decimal("50.0"),
            status="executed",
        )
        position.add_trade(trade2)

        expected_avg = (Decimal("2500.0") * 100 + Decimal("2600.0") * 200) / 300
        assert position.quantity == 300
        assert position.average_price == expected_avg
        assert position.trades_count == 2
        assert position.total_commission == Decimal("75.0")

    def test_position_close(self):
        """ポジションクローズのテスト"""
        position = Position("7203")

        # ロングポジション作成
        buy_trade = Trade(
            id="buy1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        position.add_trade(buy_trade)

        # 完全売却（利益確定）
        sell_trade = Trade(
            id="sell1",
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("2700.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        position.add_trade(sell_trade)

        assert position.quantity == 0
        assert position.is_flat()
        # 実現損益 = (2700 - 2500) * 100 - 50（手数料）= 19950
        expected_pnl = (Decimal("2700.0") - Decimal("2500.0")) * 100 - Decimal("50.0")
        assert position.realized_pnl == expected_pnl

    def test_partial_close(self):
        """部分決済のテスト"""
        position = Position("7203")

        # ロングポジション作成
        buy_trade = Trade(
            id="buy1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=200,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("50.0"),
            status="executed",
        )
        position.add_trade(buy_trade)

        # 部分売却
        sell_trade = Trade(
            id="sell1",
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=80,
            price=Decimal("2600.0"),
            timestamp=datetime.now(),
            commission=Decimal("20.0"),
            status="executed",
        )
        position.add_trade(sell_trade)

        assert position.quantity == 120  # 200 - 80
        assert position.is_long()
        # 部分決済の実現損益 = (2600 - 2500) * 80 - 70（手数料）= 7930
        expected_realized_pnl = (Decimal("2600.0") - Decimal("2500.0")) * 80 - Decimal(
            "70.0"
        )
        assert position.realized_pnl == expected_realized_pnl

    def test_short_position(self):
        """ショートポジションのテスト"""
        position = Position("7203")

        # 初回売り注文（ショート建て）
        short_trade = Trade(
            id="short1",
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        position.add_trade(short_trade)

        assert position.quantity == -100
        assert position.is_short()
        assert position.average_price == Decimal("2500.0")

        # ショートカバー（利益確定）
        cover_trade = Trade(
            id="cover1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2300.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        position.add_trade(cover_trade)

        assert position.quantity == 0
        assert position.is_flat()
        # ショート利益 = (2500 - 2300) * 100 - 50（手数料）= 19950
        expected_pnl = (Decimal("2500.0") - Decimal("2300.0")) * 100 - Decimal("50.0")
        assert position.realized_pnl == expected_pnl


class TestPortfolioManager:
    """ポートフォリオマネージャーのテスト"""

    @pytest.fixture
    def portfolio_manager(self):
        """ポートフォリオマネージャーインスタンス"""
        return PortfolioManager(initial_cash=Decimal("1000000"))

    def test_portfolio_manager_initialization(self, portfolio_manager):
        """ポートフォリオマネージャー初期化のテスト"""
        assert portfolio_manager.initial_cash == Decimal("1000000")
        assert portfolio_manager.current_cash == Decimal("1000000")
        assert len(portfolio_manager.positions) == 0
        assert len(portfolio_manager.trade_history) == 0
        assert portfolio_manager.max_equity == Decimal("1000000")

    def test_add_trade(self, portfolio_manager):
        """取引追加のテスト"""
        trade = Trade(
            id="t1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )

        initial_cash = portfolio_manager.current_cash
        portfolio_manager.add_trade(trade)

        # ポジション作成の確認
        assert "7203" in portfolio_manager.positions
        position = portfolio_manager.positions["7203"]
        assert position.quantity == 100
        assert position.average_price == Decimal("2500.0")

        # キャッシュ減少の確認
        expected_cash = initial_cash - (Decimal("2500.0") * 100 + Decimal("25.0"))
        assert portfolio_manager.current_cash == expected_cash

        # 履歴追加の確認
        assert len(portfolio_manager.trade_history) == 1
        assert portfolio_manager.trade_history[0] == trade

    def test_update_market_prices(self, portfolio_manager):
        """市場価格更新のテスト"""
        # 最初にポジションを作成
        trade = Trade(
            id="t1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        portfolio_manager.add_trade(trade)

        # 市場価格更新
        price_data = {"7203": Decimal("2600.0")}
        portfolio_manager.update_market_prices(price_data)

        position = portfolio_manager.positions["7203"]
        assert position.current_price == Decimal("2600.0")
        assert position.market_value == Decimal("260000.0")
        assert position.unrealized_pnl == Decimal("10000.0")

    def test_get_portfolio_summary(self, portfolio_manager):
        """ポートフォリオサマリー取得のテスト"""
        # 複数のポジションを作成
        trades = [
            Trade(
                "t1",
                "7203",
                TradeType.BUY,
                100,
                Decimal("2500.0"),
                datetime.now(),
                Decimal("25.0"),
                "executed",
            ),
            Trade(
                "t2",
                "6758",
                TradeType.BUY,
                200,
                Decimal("3000.0"),
                datetime.now(),
                Decimal("50.0"),
                "executed",
            ),
            Trade(
                "t3",
                "9984",
                TradeType.SELL,
                50,
                Decimal("8000.0"),
                datetime.now(),
                Decimal("40.0"),
                "executed",
            ),
        ]

        for trade in trades:
            portfolio_manager.add_trade(trade)

        # 市場価格更新
        price_data = {
            "7203": Decimal("2600.0"),
            "6758": Decimal("2900.0"),
            "9984": Decimal("8200.0"),
        }
        portfolio_manager.update_market_prices(price_data)

        summary = portfolio_manager.get_portfolio_summary()

        assert isinstance(summary, PortfolioSummary)
        assert summary.total_positions == 3
        assert summary.long_positions == 2
        assert summary.short_positions == 1

        # 市場価値計算の確認
        expected_market_value = (
            Decimal("2600.0") * 100  # 7203
            + Decimal("2900.0") * 200  # 6758
            + Decimal("8200.0") * 50  # 9984 (ショート)
        )
        assert summary.total_market_value == expected_market_value

    def test_risk_limits_check(self, portfolio_manager):
        """リスク制限チェックのテスト"""
        # 大きなポジションを作成（制限超過）
        large_trade = Trade(
            id="large",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=5000,  # 大量
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("250.0"),
            status="executed",
        )
        portfolio_manager.add_trade(large_trade)

        risk_check = portfolio_manager.check_risk_limits()

        # エクスポージャー制限違反の確認
        assert len(risk_check["violations"]) > 0
        assert risk_check["total_violations"] > 0
        assert risk_check["risk_score"] > 0

    def test_performance_metrics(self, portfolio_manager):
        """パフォーマンス指標のテスト"""
        # 取引を追加してエクイティカーブを構築
        trade = Trade(
            id="t1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        portfolio_manager.add_trade(trade)

        # 価格変動をシミュレート
        price_updates = [
            {"7203": Decimal("2600.0")},
            {"7203": Decimal("2550.0")},
            {"7203": Decimal("2700.0")},
        ]

        for prices in price_updates:
            portfolio_manager.update_market_prices(prices)

        metrics = portfolio_manager.get_performance_metrics(days=10)

        if metrics:  # エクイティカーブが十分にある場合
            assert "total_trades" in metrics
            assert "current_equity" in metrics
            assert "period_return" in metrics
            assert metrics["total_trades"] == 1

    def test_position_breakdown(self, portfolio_manager):
        """ポジション内訳のテスト"""
        # 複数ポジション作成
        trades = [
            Trade(
                "t1",
                "7203",
                TradeType.BUY,
                100,
                Decimal("2500.0"),
                datetime.now(),
                Decimal("25.0"),
                "executed",
            ),
            Trade(
                "t2",
                "6758",
                TradeType.SELL,
                50,
                Decimal("3000.0"),
                datetime.now(),
                Decimal("30.0"),
                "executed",
            ),
        ]

        for trade in trades:
            portfolio_manager.add_trade(trade)

        # 価格更新
        portfolio_manager.update_market_prices(
            {
                "7203": Decimal("2600.0"),
                "6758": Decimal("2900.0"),
            }
        )

        breakdown = portfolio_manager.get_position_breakdown()

        assert len(breakdown) == 2

        # ソート順の確認（損益順）
        first_position = breakdown[0]
        assert first_position["symbol"] in ["7203", "6758"]
        assert "unrealized_pnl" in first_position
        assert "unrealized_pnl_pct" in first_position
        assert "side" in first_position

    def test_close_position(self, portfolio_manager):
        """ポジションクローズのテスト"""
        # ポジション作成
        buy_trade = Trade(
            id="buy1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        portfolio_manager.add_trade(buy_trade)

        # ポジションクローズ
        close_trade = portfolio_manager.close_position("7203", Decimal("2600.0"))

        assert close_trade is not None
        assert close_trade.trade_type == TradeType.SELL
        assert close_trade.quantity == 100
        assert close_trade.price == Decimal("2600.0")

        # ポジションがフラットになることを確認
        position = portfolio_manager.positions["7203"]
        assert position.is_flat()

    def test_close_all_positions(self, portfolio_manager):
        """全ポジションクローズのテスト"""
        # 複数ポジション作成
        trades = [
            Trade(
                "t1",
                "7203",
                TradeType.BUY,
                100,
                Decimal("2500.0"),
                datetime.now(),
                Decimal("25.0"),
                "executed",
            ),
            Trade(
                "t2",
                "6758",
                TradeType.BUY,
                200,
                Decimal("3000.0"),
                datetime.now(),
                Decimal("50.0"),
                "executed",
            ),
        ]

        for trade in trades:
            portfolio_manager.add_trade(trade)

        # 全ポジションクローズ
        close_prices = {
            "7203": Decimal("2600.0"),
            "6758": Decimal("2950.0"),
        }
        closed_trades = portfolio_manager.close_all_positions(close_prices)

        assert len(closed_trades) == 2

        # 全ポジションがフラットになることを確認
        for symbol in ["7203", "6758"]:
            position = portfolio_manager.positions[symbol]
            assert position.is_flat()

    def test_daily_snapshot(self, portfolio_manager):
        """日次スナップショットのテスト"""
        # 取引を追加
        trade = Trade(
            id="t1",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            timestamp=datetime.now(),
            commission=Decimal("25.0"),
            status="executed",
        )
        portfolio_manager.add_trade(trade)

        # スナップショット作成
        portfolio_manager.create_daily_snapshot()

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in portfolio_manager.daily_snapshots

        snapshot = portfolio_manager.daily_snapshots[today]
        assert isinstance(snapshot, PortfolioSummary)


class TestMultipleScenarios:
    """複合シナリオのテスト"""

    @pytest.fixture
    def portfolio_manager(self):
        return PortfolioManager(initial_cash=Decimal("1000000"))

    def test_complex_trading_scenario(self, portfolio_manager):
        """複雑な取引シナリオのテスト"""

        # シナリオ1: 複数銘柄の買い建て
        initial_trades = [
            Trade(
                "t1",
                "7203",
                TradeType.BUY,
                100,
                Decimal("2500.0"),
                datetime.now(),
                Decimal("25.0"),
                "executed",
            ),
            Trade(
                "t2",
                "6758",
                TradeType.BUY,
                200,
                Decimal("3000.0"),
                datetime.now(),
                Decimal("50.0"),
                "executed",
            ),
            Trade(
                "t3",
                "9984",
                TradeType.BUY,
                50,
                Decimal("8000.0"),
                datetime.now(),
                Decimal("40.0"),
                "executed",
            ),
        ]

        for trade in initial_trades:
            portfolio_manager.add_trade(trade)

        # シナリオ2: 価格変動
        price_updates = [
            {
                "7203": Decimal("2600.0"),
                "6758": Decimal("2900.0"),
                "9984": Decimal("8200.0"),
            },
            {
                "7203": Decimal("2400.0"),
                "6758": Decimal("3100.0"),
                "9984": Decimal("7800.0"),
            },
        ]

        for prices in price_updates:
            portfolio_manager.update_market_prices(prices)

        # シナリオ3: 一部決済
        partial_sell = Trade(
            "sell1",
            "7203",
            TradeType.SELL,
            50,
            Decimal("2450.0"),
            datetime.now(),
            Decimal("20.0"),
            "executed",
        )
        portfolio_manager.add_trade(partial_sell)

        # シナリオ4: ショート建て
        short_trade = Trade(
            "short1",
            "4755",
            TradeType.SELL,
            100,
            Decimal("5000.0"),
            datetime.now(),
            Decimal("50.0"),
            "executed",
        )
        portfolio_manager.add_trade(short_trade)

        # 最終状態の確認
        summary = portfolio_manager.get_portfolio_summary()

        assert summary.total_positions == 4  # 7203(残), 6758, 9984, 4755(ショート)
        assert summary.long_positions == 3
        assert summary.short_positions == 1

        # 7203は部分決済されているはず
        position_7203 = portfolio_manager.positions["7203"]
        assert position_7203.quantity == 50  # 100 - 50
        assert position_7203.realized_pnl != Decimal("0")  # 部分決済により実現損益あり

        # 4755はショートポジション
        position_4755 = portfolio_manager.positions["4755"]
        assert position_4755.is_short()
        assert position_4755.quantity == -100
