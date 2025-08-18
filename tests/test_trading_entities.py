"""
DDD/トレーディングエンティティの包括的テストスイート

Issue #10: テストカバレッジ向上とエラーハンドリング強化
優先度: #2 (Priority Score: 195)
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from day_trade.domain.trading.entities import (
    Trade, Position, Portfolio, Order, OrderType, OrderStatus,
    TradingSession, MarketData, Symbol, TradingAccount
)
from day_trade.domain.common.value_objects import Money, Price, Quantity


class TestTrade:
    """Tradeエンティティのテスト"""

    def test_trade_creation(self):
        """取引作成テスト"""
        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )

        assert trade.symbol == "7203"
        assert trade.order_type == OrderType.BUY
        assert trade.quantity.value == 100
        assert trade.price.value == Decimal("1500")
        assert isinstance(trade.timestamp, datetime)
        assert trade.trade_id is not None

    def test_trade_total_value(self):
        """取引総額計算テスト"""
        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )

        total_value = trade.calculate_total_value()
        expected = Money(Decimal("150000"), "JPY")  # 100 * 1500

        assert total_value.amount == expected.amount
        assert total_value.currency == expected.currency

    def test_trade_with_commission(self):
        """手数料込み取引テスト"""
        commission = Money(Decimal("500"), "JPY")
        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now(),
            commission=commission
        )

        total_cost = trade.calculate_total_cost()
        expected = Money(Decimal("150500"), "JPY")  # 150000 + 500

        assert total_cost.amount == expected.amount

    def test_trade_is_buy_sell(self):
        """売買判定テスト"""
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )

        sell_trade = Trade(
            symbol="7203",
            order_type=OrderType.SELL,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )

        assert buy_trade.is_buy() is True
        assert buy_trade.is_sell() is False
        assert sell_trade.is_buy() is False
        assert sell_trade.is_sell() is True

    def test_trade_validation(self):
        """取引バリデーションテスト"""
        with pytest.raises(ValueError):
            Trade(
                symbol="",  # 空のシンボル
                order_type=OrderType.BUY,
                quantity=Quantity(100),
                price=Price(Decimal("1500"), "JPY"),
                timestamp=datetime.now()
            )

        with pytest.raises(ValueError):
            Trade(
                symbol="7203",
                order_type=OrderType.BUY,
                quantity=Quantity(0),  # 無効な数量
                price=Price(Decimal("1500"), "JPY"),
                timestamp=datetime.now()
            )


class TestPosition:
    """Positionエンティティのテスト"""

    def test_position_creation(self):
        """ポジション作成テスト"""
        position = Position(symbol="7203")

        assert position.symbol == "7203"
        assert position.quantity.value == 0
        assert position.average_cost.amount == Decimal("0")
        assert position.is_empty()

    def test_position_add_trade_buy(self):
        """買い取引追加テスト"""
        position = Position(symbol="7203")

        # 初回買い
        trade1 = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(trade1)

        assert position.quantity.value == 100
        assert position.average_cost.amount == Decimal("1500")

        # 追加買い
        trade2 = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(200),
            price=Price(Decimal("1800"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(trade2)

        assert position.quantity.value == 300
        # 平均コスト = (100*1500 + 200*1800) / 300 = 1700
        assert position.average_cost.amount == Decimal("1700")

    def test_position_add_trade_sell(self):
        """売り取引追加テスト"""
        position = Position(symbol="7203")

        # 買いポジション構築
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(buy_trade)

        # 一部売却
        sell_trade = Trade(
            symbol="7203",
            order_type=OrderType.SELL,
            quantity=Quantity(40),
            price=Price(Decimal("1600"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(sell_trade)

        assert position.quantity.value == 60
        assert position.average_cost.amount == Decimal("1500")  # 平均コストは変わらず

    def test_position_calculate_unrealized_pnl(self):
        """未実現損益計算テスト"""
        position = Position(symbol="7203")

        # 買いポジション構築
        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(trade)

        # 現在価格での未実現損益計算
        current_price = Price(Decimal("1600"), "JPY")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)

        expected = Money(Decimal("10000"), "JPY")  # (1600-1500) * 100
        assert unrealized_pnl.amount == expected.amount

    def test_position_calculate_realized_pnl(self):
        """実現損益計算テスト"""
        position = Position(symbol="7203")

        # 買いポジション構築
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(buy_trade)

        # 売却
        sell_trade = Trade(
            symbol="7203",
            order_type=OrderType.SELL,
            quantity=Quantity(40),
            price=Price(Decimal("1600"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(sell_trade)

        realized_pnl = position.calculate_realized_pnl()
        expected = Money(Decimal("4000"), "JPY")  # (1600-1500) * 40

        assert realized_pnl.amount == expected.amount

    def test_position_insufficient_quantity_sell(self):
        """不足数量売却エラーテスト"""
        position = Position(symbol="7203")

        # 少量の買いポジション
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(50),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(buy_trade)

        # 過剰な売却
        with pytest.raises(ValueError, match="insufficient quantity"):
            sell_trade = Trade(
                symbol="7203",
                order_type=OrderType.SELL,
                quantity=Quantity(100),  # 50より多い
                price=Price(Decimal("1600"), "JPY"),
                timestamp=datetime.now()
            )
            position.add_trade(sell_trade)

    def test_position_symbol_mismatch(self):
        """シンボル不一致エラーテスト"""
        position = Position(symbol="7203")

        with pytest.raises(ValueError, match="symbol mismatch"):
            trade = Trade(
                symbol="6758",  # 異なるシンボル
                order_type=OrderType.BUY,
                quantity=Quantity(100),
                price=Price(Decimal("1500"), "JPY"),
                timestamp=datetime.now()
            )
            position.add_trade(trade)


class TestPortfolio:
    """Portfolioエンティティのテスト"""

    def test_portfolio_creation(self):
        """ポートフォリオ作成テスト"""
        initial_cash = Money(Decimal("1000000"), "JPY")
        portfolio = Portfolio(cash_balance=initial_cash)

        assert portfolio.cash_balance.amount == Decimal("1000000")
        assert len(portfolio.positions) == 0
        assert portfolio.portfolio_id is not None

    def test_portfolio_add_position(self):
        """ポジション追加テスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000"), "JPY"))
        position = Position(symbol="7203")

        portfolio.add_position(position)

        assert len(portfolio.positions) == 1
        assert "7203" in portfolio.positions
        assert portfolio.positions["7203"] == position

    def test_portfolio_get_position(self):
        """ポジション取得テスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000"), "JPY"))

        # 存在しないポジション
        position = portfolio.get_position("7203")
        assert position is not None
        assert position.symbol == "7203"
        assert position.is_empty()

        # 自動的にポートフォリオに追加される
        assert "7203" in portfolio.positions

    def test_portfolio_calculate_total_value(self):
        """ポートフォリオ総価値計算テスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("500000"), "JPY"))

        # ポジション1
        position1 = Position(symbol="7203")
        trade1 = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position1.add_trade(trade1)
        portfolio.add_position(position1)

        # ポジション2
        position2 = Position(symbol="6758")
        trade2 = Trade(
            symbol="6758",
            order_type=OrderType.BUY,
            quantity=Quantity(200),
            price=Price(Decimal("2000"), "JPY"),
            timestamp=datetime.now()
        )
        position2.add_trade(trade2)
        portfolio.add_position(position2)

        # 現在価格
        market_prices = {
            "7203": Price(Decimal("1600"), "JPY"),
            "6758": Price(Decimal("2100"), "JPY")
        }

        total_value = portfolio.calculate_total_value(market_prices)

        # 現金 + ポジション価値
        # 500000 + (100*1600) + (200*2100) = 500000 + 160000 + 420000 = 1080000
        expected = Money(Decimal("1080000"), "JPY")
        assert total_value.amount == expected.amount

    def test_portfolio_calculate_unrealized_pnl(self):
        """ポートフォリオ未実現損益計算テスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("500000"), "JPY"))

        # ポジション作成
        position = Position(symbol="7203")
        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(trade)
        portfolio.add_position(position)

        # 現在価格
        market_prices = {"7203": Price(Decimal("1600"), "JPY")}

        unrealized_pnl = portfolio.calculate_unrealized_pnl(market_prices)
        expected = Money(Decimal("10000"), "JPY")  # (1600-1500) * 100

        assert unrealized_pnl.amount == expected.amount

    def test_portfolio_update_cash_balance(self):
        """現金残高更新テスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000"), "JPY"))

        # 現金増加
        portfolio.update_cash_balance(Money(Decimal("50000"), "JPY"))
        assert portfolio.cash_balance.amount == Decimal("1050000")

        # 現金減少
        portfolio.update_cash_balance(Money(Decimal("-30000"), "JPY"))
        assert portfolio.cash_balance.amount == Decimal("1020000")

    def test_portfolio_insufficient_cash(self):
        """現金不足エラーテスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("100000"), "JPY"))

        with pytest.raises(ValueError, match="insufficient cash"):
            portfolio.update_cash_balance(Money(Decimal("-200000"), "JPY"))


class TestOrder:
    """Orderエンティティのテスト"""

    def test_order_creation(self):
        """注文作成テスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        assert order.symbol == "7203"
        assert order.order_type == OrderType.BUY
        assert order.quantity.value == 100
        assert order.price.value == Decimal("1500")
        assert order.status == OrderStatus.PENDING
        assert order.order_id is not None
        assert isinstance(order.created_at, datetime)

    def test_order_execute(self):
        """注文執行テスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        execution_price = Price(Decimal("1505"), "JPY")
        trade = order.execute(execution_price)

        assert order.status == OrderStatus.FILLED
        assert order.executed_at is not None
        assert trade.symbol == "7203"
        assert trade.price.value == Decimal("1505")
        assert trade.quantity.value == 100

    def test_order_cancel(self):
        """注文キャンセルテスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        order.cancel()

        assert order.status == OrderStatus.CANCELLED
        assert order.cancelled_at is not None

    def test_order_already_executed(self):
        """既に執行済み注文の再執行エラーテスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        # 初回執行
        execution_price = Price(Decimal("1505"), "JPY")
        order.execute(execution_price)

        # 再執行を試行
        with pytest.raises(ValueError, match="already executed"):
            order.execute(execution_price)

    def test_order_already_cancelled(self):
        """既にキャンセル済み注文の執行エラーテスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        # キャンセル
        order.cancel()

        # 執行を試行
        with pytest.raises(ValueError, match="cannot execute cancelled order"):
            execution_price = Price(Decimal("1505"), "JPY")
            order.execute(execution_price)


class TestTradingSession:
    """TradingSessionエンティティのテスト"""

    def test_trading_session_creation(self):
        """取引セッション作成テスト"""
        start_time = datetime.now()
        session = TradingSession(
            session_name="test_session",
            start_time=start_time
        )

        assert session.session_name == "test_session"
        assert session.start_time == start_time
        assert session.end_time is None
        assert session.is_active() is True
        assert len(session.trades) == 0

    def test_trading_session_add_trade(self):
        """取引追加テスト"""
        session = TradingSession(
            session_name="test_session",
            start_time=datetime.now()
        )

        trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )

        session.add_trade(trade)

        assert len(session.trades) == 1
        assert session.trades[0] == trade

    def test_trading_session_end(self):
        """取引セッション終了テスト"""
        session = TradingSession(
            session_name="test_session",
            start_time=datetime.now()
        )

        assert session.is_active() is True

        end_time = datetime.now()
        session.end_session(end_time)

        assert session.end_time == end_time
        assert session.is_active() is False

    def test_trading_session_calculate_pnl(self):
        """セッション損益計算テスト"""
        session = TradingSession(
            session_name="test_session",
            start_time=datetime.now()
        )

        # 買い取引
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        session.add_trade(buy_trade)

        # 売り取引
        sell_trade = Trade(
            symbol="7203",
            order_type=OrderType.SELL,
            quantity=Quantity(100),
            price=Price(Decimal("1600"), "JPY"),
            timestamp=datetime.now()
        )
        session.add_trade(sell_trade)

        total_pnl = session.calculate_total_pnl()
        expected = Money(Decimal("10000"), "JPY")  # (1600-1500) * 100

        assert total_pnl.amount == expected.amount

    def test_trading_session_get_performance_metrics(self):
        """パフォーマンスメトリクス取得テスト"""
        session = TradingSession(
            session_name="test_session",
            start_time=datetime.now()
        )

        # 複数の取引を追加
        trades_data = [
            (OrderType.BUY, Decimal("1500")),
            (OrderType.SELL, Decimal("1600")),  # 利益
            (OrderType.BUY, Decimal("1550")),
            (OrderType.SELL, Decimal("1520")),  # 損失
        ]

        for order_type, price in trades_data:
            trade = Trade(
                symbol="7203",
                order_type=order_type,
                quantity=Quantity(100),
                price=Price(price, "JPY"),
                timestamp=datetime.now()
            )
            session.add_trade(trade)

        metrics = session.get_performance_metrics()

        assert "total_trades" in metrics
        assert "total_pnl" in metrics
        assert "winning_trades" in metrics
        assert "losing_trades" in metrics
        assert "win_rate" in metrics

        assert metrics["total_trades"] == 4
        assert metrics["winning_trades"] >= 0
        assert metrics["losing_trades"] >= 0


class TestMarketData:
    """MarketDataエンティティのテスト"""

    def test_market_data_creation(self):
        """マーケットデータ作成テスト"""
        timestamp = datetime.now()
        market_data = MarketData(
            symbol="7203",
            timestamp=timestamp,
            open_price=Price(Decimal("1500"), "JPY"),
            high_price=Price(Decimal("1520"), "JPY"),
            low_price=Price(Decimal("1480"), "JPY"),
            close_price=Price(Decimal("1510"), "JPY"),
            volume=Quantity(1000000)
        )

        assert market_data.symbol == "7203"
        assert market_data.timestamp == timestamp
        assert market_data.open_price.value == Decimal("1500")
        assert market_data.high_price.value == Decimal("1520")
        assert market_data.low_price.value == Decimal("1480")
        assert market_data.close_price.value == Decimal("1510")
        assert market_data.volume.value == 1000000

    def test_market_data_validation(self):
        """マーケットデータバリデーションテスト"""
        timestamp = datetime.now()

        with pytest.raises(ValueError, match="high price must be >= low price"):
            MarketData(
                symbol="7203",
                timestamp=timestamp,
                open_price=Price(Decimal("1500"), "JPY"),
                high_price=Price(Decimal("1480"), "JPY"),  # 高値 < 安値
                low_price=Price(Decimal("1520"), "JPY"),
                close_price=Price(Decimal("1510"), "JPY"),
                volume=Quantity(1000000)
            )

    def test_market_data_calculate_returns(self):
        """リターン計算テスト"""
        previous_data = MarketData(
            symbol="7203",
            timestamp=datetime.now() - timedelta(days=1),
            open_price=Price(Decimal("1500"), "JPY"),
            high_price=Price(Decimal("1520"), "JPY"),
            low_price=Price(Decimal("1480"), "JPY"),
            close_price=Price(Decimal("1500"), "JPY"),
            volume=Quantity(1000000)
        )

        current_data = MarketData(
            symbol="7203",
            timestamp=datetime.now(),
            open_price=Price(Decimal("1510"), "JPY"),
            high_price=Price(Decimal("1530"), "JPY"),
            low_price=Price(Decimal("1500"), "JPY"),
            close_price=Price(Decimal("1520"), "JPY"),
            volume=Quantity(1200000)
        )

        returns = current_data.calculate_returns(previous_data)

        # (1520 - 1500) / 1500 = 0.0133... ≈ 1.33%
        expected_returns = (Decimal("1520") - Decimal("1500")) / Decimal("1500")
        assert abs(returns - expected_returns) < Decimal("0.0001")


class TestTradingAccount:
    """TradingAccountエンティティのテスト"""

    def test_trading_account_creation(self):
        """取引口座作成テスト"""
        initial_balance = Money(Decimal("1000000"), "JPY")
        account = TradingAccount(
            account_id="TEST001",
            initial_balance=initial_balance
        )

        assert account.account_id == "TEST001"
        assert account.current_balance.amount == Decimal("1000000")
        assert account.initial_balance.amount == Decimal("1000000")
        assert len(account.transaction_history) == 0

    def test_trading_account_deposit(self):
        """入金テスト"""
        account = TradingAccount(
            account_id="TEST001",
            initial_balance=Money(Decimal("1000000"), "JPY")
        )

        deposit_amount = Money(Decimal("50000"), "JPY")
        account.deposit(deposit_amount)

        assert account.current_balance.amount == Decimal("1050000")
        assert len(account.transaction_history) == 1

        transaction = account.transaction_history[0]
        assert transaction["type"] == "deposit"
        assert transaction["amount"].amount == Decimal("50000")

    def test_trading_account_withdraw(self):
        """出金テスト"""
        account = TradingAccount(
            account_id="TEST001",
            initial_balance=Money(Decimal("1000000"), "JPY")
        )

        withdraw_amount = Money(Decimal("30000"), "JPY")
        account.withdraw(withdraw_amount)

        assert account.current_balance.amount == Decimal("970000")
        assert len(account.transaction_history) == 1

        transaction = account.transaction_history[0]
        assert transaction["type"] == "withdrawal"
        assert transaction["amount"].amount == Decimal("30000")

    def test_trading_account_insufficient_funds(self):
        """資金不足エラーテスト"""
        account = TradingAccount(
            account_id="TEST001",
            initial_balance=Money(Decimal("100000"), "JPY")
        )

        with pytest.raises(ValueError, match="insufficient funds"):
            withdraw_amount = Money(Decimal("200000"), "JPY")
            account.withdraw(withdraw_amount)

    def test_trading_account_calculate_total_returns(self):
        """総リターン計算テスト"""
        account = TradingAccount(
            account_id="TEST001",
            initial_balance=Money(Decimal("1000000"), "JPY")
        )

        # 取引による損益
        account.current_balance = Money(Decimal("1150000"), "JPY")

        total_returns = account.calculate_total_returns()
        expected = Decimal("0.15")  # 15%のリターン

        assert abs(total_returns - expected) < Decimal("0.0001")


class TestErrorScenarios:
    """エラーシナリオのテスト"""

    def test_invalid_trade_parameters(self):
        """無効な取引パラメータのテスト"""
        with pytest.raises(ValueError):
            Trade(
                symbol="",
                order_type=OrderType.BUY,
                quantity=Quantity(100),
                price=Price(Decimal("1500"), "JPY"),
                timestamp=datetime.now()
            )

    def test_position_concurrent_modification(self):
        """ポジション同時変更テスト"""
        position = Position(symbol="7203")

        # 買いポジション構築
        buy_trade = Trade(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY"),
            timestamp=datetime.now()
        )
        position.add_trade(buy_trade)

        # 同時に複数の売り注文（スレッドセーフティのテスト）
        def sell_partial():
            sell_trade = Trade(
                symbol="7203",
                order_type=OrderType.SELL,
                quantity=Quantity(30),
                price=Price(Decimal("1600"), "JPY"),
                timestamp=datetime.now()
            )
            position.add_trade(sell_trade)

        import threading
        threads = [threading.Thread(target=sell_partial) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 最終的な数量が正しいことを確認
        assert position.quantity.value >= 10  # 100 - 3*30 = 10


class TestIntegrationScenarios:
    """統合シナリオのテスト"""

    def test_complete_trading_workflow(self):
        """完全な取引ワークフローテスト"""
        # 1. 口座開設
        account = TradingAccount(
            account_id="INTEGRATION_TEST",
            initial_balance=Money(Decimal("1000000"), "JPY")
        )

        # 2. ポートフォリオ作成
        portfolio = Portfolio(cash_balance=account.current_balance)

        # 3. 注文作成
        order = Order(
            symbol="7203",
            order_type=OrderType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500"), "JPY")
        )

        # 4. 注文執行
        execution_price = Price(Decimal("1505"), "JPY")
        trade = order.execute(execution_price)

        # 5. ポジション更新
        position = portfolio.get_position("7203")
        position.add_trade(trade)

        # 6. 現金残高更新
        trade_cost = trade.calculate_total_cost()
        portfolio.update_cash_balance(Money(Decimal("0"), "JPY") - trade_cost)

        # 7. 結果検証
        assert order.status == OrderStatus.FILLED
        assert position.quantity.value == 100
        assert position.average_cost.value == Decimal("1505")
        assert portfolio.cash_balance.amount == Decimal("849500")  # 1000000 - 150500

    def test_portfolio_rebalancing(self):
        """ポートフォリオリバランステスト"""
        portfolio = Portfolio(cash_balance=Money(Decimal("1000000"), "JPY"))

        # 複数銘柄への投資
        symbols_and_prices = [
            ("7203", Decimal("1500")),
            ("6758", Decimal("2000")),
            ("9984", Decimal("3000"))
        ]

        for symbol, price in symbols_and_prices:
            trade = Trade(
                symbol=symbol,
                order_type=OrderType.BUY,
                quantity=Quantity(100),
                price=Price(price, "JPY"),
                timestamp=datetime.now()
            )

            position = portfolio.get_position(symbol)
            position.add_trade(trade)

            trade_cost = trade.calculate_total_cost()
            portfolio.update_cash_balance(Money(Decimal("0"), "JPY") - trade_cost)

        # 現在価格での評価
        market_prices = {
            "7203": Price(Decimal("1600"), "JPY"),
            "6758": Price(Decimal("2100"), "JPY"),
            "9984": Price(Decimal("2800"), "JPY")
        }

        total_value = portfolio.calculate_total_value(market_prices)
        unrealized_pnl = portfolio.calculate_unrealized_pnl(market_prices)

        # 検証
        assert len(portfolio.positions) == 3
        assert total_value.amount > Decimal("1000000")  # 利益が出ている
        assert unrealized_pnl.amount != Decimal("0")    # 未実現損益がある


if __name__ == "__main__":
    pytest.main([__file__, "-v"])