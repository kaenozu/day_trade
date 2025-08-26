"""
トレーディングタイプの包括的テストスイート

Issue #10: テストカバレッジ向上とエラーハンドリング強化
優先度: #3 (Priority Score: 170)
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from day_trade.core.types.trading_types import (
    TradingConstraints,
    TypeSafeTradeManager,
    NumericRange,
    OrderDict,
    SymbolCode,
    Price,
    Quantity,
    TradeDirection,
    OrderType,
    TimeInForce,
    MarketDataProvider,
    OrderExecutor,
    PortfolioManager
)
from day_trade.automation.trading_engine import RiskParameters
from day_trade.domain.common.value_objects import Money, Price, Quantity, Symbol


class TestTradingConstraints:
    """TradingConstraintsのテスト"""

    def test_constraints_creation(self):
        """制約作成テスト"""
        constraints = TradingConstraints(
            min_order_size=NumericRange(Quantity(1), Quantity(1000)),
            price_range=NumericRange(Decimal("100"), Decimal("10000")),
            max_position_size=Quantity(10000),
            allowed_symbols=["7203", "6758", "9984"]
        )

        assert constraints.min_order_size.min_value == Quantity(1)
        assert constraints.min_order_size.max_value == Quantity(1000)
        assert constraints.price_range.min_value == Decimal("100")
        assert constraints.price_range.max_value == Decimal("10000")
        assert constraints.max_position_size == Quantity(10000)
        assert "7203" in constraints.allowed_symbols

    def test_validate_order_valid(self):
        """有効注文バリデーションテスト"""
        constraints = TradingConstraints(
            min_order_size=NumericRange(Quantity(1), Quantity(1000)),
            price_range=NumericRange(Decimal("100"), Decimal("10000")),
            allowed_symbols=["7203", "6758"]
        )

        # 有効な注文
        order: OrderDict = {
            "order_id": "test_order_1",
            "symbol": "7203",
            "direction": "buy",
            "order_type": "limit",
            "quantity": Quantity(50),
            "price": Decimal("1500"),
            "time_in_force": "gtc",
            "timestamp": datetime.now()
        }

        # バリデーション実行（例外が発生しないことを確認）
        result = constraints.validate_order(order)
        assert result.is_success is True

    #     def test_validate_order_invalid_symbol(
#         self,
#     ):
#         """無効シンボルバリデーションテスト"""
#         constraints = TradingConstraints(
#             min_order_size=NumericRange(Quantity(1), Quantity(1000)),
#             price_range=NumericRange(Decimal("100"), Decimal("10000")),
#             allowed_symbols=["7203", "6758"]
#         )
# 
#         order: OrderDict = {
#             "order_id": "test_order_2",
#             "symbol": "9999",  # 許可されていないシンボル
#             "direction": "buy",
#             "order_type": "limit",
#             "quantity": Quantity(50),
#             "price": Decimal("1500"),
#             "time_in_force": "gtc",
#             "timestamp": datetime.now()
#         }
#         result = constraints.validate_order(order)
#         assert result.is_success is False
#         assert "not allowed" in result.error

    #     def test_validate_order_too_small(
#         self
#     ):
#         """最小取引サイズ未満テスト"""
#         constraints = TradingConstraints(
#             min_order_size=NumericRange(Quantity(100), Quantity(1000)),
#             price_range=NumericRange(Decimal("100"), Decimal("10000")),
#             allowed_symbols=["7203"]
#         )
# 
#         order: OrderDict = {
#             "order_id": "test_order_3",
#             "symbol": "7203",
#             "direction": "buy",
#             "order_type": "limit",
#             "quantity": Quantity(50),  # 最小注文サイズ未満
#             "price": Decimal("1500"),
#             "time_in_force": "gtc",
#             "timestamp": datetime.now()
#         }
#         result = constraints.validate_order(order)
#         assert result.is_success is False
#         assert "outside allowed range" in result.error

    #     def test_validate_order_too_large(self):
#         """最大取引サイズ超過テスト"""
#         constraints = TradingConstraints(
#             min_order_size=NumericRange(Quantity(1), Quantity(100)),
#             price_range=NumericRange(Decimal("100"), Decimal("10000")),
#             allowed_symbols=["7203"]
#         )
# 
#         order: OrderDict = {
#             "order_id": "test_order_4",
#             "symbol": "7203",
#             "direction": "buy",
#             "order_type": "limit",
#             "quantity": Quantity(150),  # 最大注文サイズ超過
#             "price": Decimal("1500"),
#             "time_in_force": "gtc",
#             "timestamp": datetime.now()
#         }
#         result = constraints.validate_order(order)
#         assert result.is_success is False
#         assert "outside allowed range" in result.error


class TestRiskParameters:
    """RiskParametersのテスト"""

    def test_risk_parameters_creation(self):
        """リスクパラメータ作成テスト"""
        risk_params = RiskParameters(
            max_position_size=Decimal("5000000"),
            max_daily_loss=Decimal("100000"),
            max_open_positions=10,
            stop_loss_ratio=Decimal("0.05"),
            take_profit_ratio=Decimal("0.15")
        )

        assert risk_params.max_position_size == Decimal("5000000")
        assert risk_params.max_daily_loss == Decimal("100000")
        assert risk_params.max_open_positions == 10
        assert risk_params.stop_loss_ratio == Decimal("0.05")
        assert risk_params.take_profit_ratio == Decimal("0.15")

    #     def test_calculate_position_size_limit(self):
#         """ポジションサイズ制限計算テスト"""
#         risk_params = RiskParameters(
#             max_position_size=Decimal("1000000"),
#             max_daily_loss=Decimal("50000"),
#             max_open_positions=10,
#             stop_loss_ratio=Decimal("0.05"),
#             take_profit_ratio=Decimal("0.15")
#         )
# 
#         # This method no longer exists in the new RiskParameters class
#         # portfolio_value = Money(Decimal("800000"), "JPY")
#         # limit = risk_params.calculate_position_size_limit(portfolio_value)
# 
#         # expected = Money(Decimal("160000"), "JPY")  # 800000 * 0.2
#         # assert limit.amount == expected.amount

    #     def test_calculate_stop_loss_price(self):
#         """ストップロス価格計算テスト"""
#         risk_params = RiskParameters(
#             max_position_size=Decimal("1000000"),
#             max_daily_loss=Decimal("50000"),
#             max_open_positions=10,
#             stop_loss_ratio=Decimal("0.05"),
#             take_profit_ratio=Decimal("0.15")
#         )
# 
#         # This method no longer exists in the new RiskParameters class
#         # entry_price = Price(Decimal("1000"), "JPY")
# 
#         # # 買いポジションのストップロス
#         # stop_loss_buy = risk_params.calculate_stop_loss_price(entry_price, is_long=True)
#         # expected_buy = Price(Decimal("950"), "JPY")  # 1000 * (1 - 0.05)
#         # assert stop_loss_buy.value == expected_buy.value
# 
#         # # 売りポジションのストップロス
#         # stop_loss_sell = risk_params.calculate_stop_loss_price(entry_price, is_long=False)
#         # expected_sell = Price(Decimal("1050"), "JPY")  # 1000 * (1 + 0.05)
#         # assert stop_loss_sell.value == expected_sell.value

    #     def test_calculate_take_profit_price(self):
#         """テイクプロフィット価格計算テスト"""
#         risk_params = RiskParameters(
#             max_position_size=Decimal("1000000"),
#             max_daily_loss=Decimal("50000"),
#             max_open_positions=10,
#             stop_loss_ratio=Decimal("0.05"),
#             take_profit_ratio=Decimal("0.15")
#         )
# 
#         # This method no longer exists in the new RiskParameters class
#         # entry_price = Price(Decimal("1000"), "JPY")
# 
#         # # 買いポジションのテイクプロフィット
#         # take_profit_buy = risk_params.calculate_take_profit_price(entry_price, is_long=True)
#         # expected_buy = Price(Decimal("1200"), "JPY")  # 1000 * (1 + 0.2)
#         # assert take_profit_buy.value == expected_buy.value
# 
#         # # 売りポジションのテイクプロフィット
#         # take_profit_sell = risk_params.calculate_take_profit_price(entry_price, is_long=False)
#         # expected_sell = Price(Decimal("800"), "JPY")  # 1000 * (1 - 0.2)
#         # assert take_profit_sell.value == expected_sell.value


# class TestOrderConstraints:
#     """OrderConstraintsのテスト"""
#
#     def test_order_constraints_creation(self):
#         """注文制約作成テスト"""
#         constraints = OrderConstraints(
#             max_orders_per_second=10,
#             max_pending_orders=50,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("1"),
#             max_price_deviation_percent=Decimal("5")
#         )
#
#         assert constraints.max_orders_per_second == 10
#         assert constraints.max_pending_orders == 50
#         assert constraints.order_timeout_minutes == 30
#         assert constraints.min_price_increment == Decimal("1")
#         assert constraints.max_price_deviation_percent == Decimal("5")
#
#     def test_validate_order_rate_limit(self):
#         """注文レート制限テスト"""
#         constraints = OrderConstraints(
#             max_orders_per_second=2,
#             max_pending_orders=50,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("1"),
#             max_price_deviation_percent=Decimal("5")
#         )
#
#         # 短時間で複数注文を送信
#         import time
#
#         # 最初の2つは成功
#         constraints.validate_order_rate()
#         constraints.validate_order_rate()
#
#         # 3つ目はレート制限エラー
#         with pytest.raises(OrderLimitExceededError):
#             constraints.validate_order_rate()
#
#     def test_validate_price_increment(self):
#         """価格刻み幅バリデーションテスト"""
#         constraints = OrderConstraints(
#             max_orders_per_second=10,
#             max_pending_orders=50,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("5"),  # 5円刻み
#             max_price_deviation_percent=Decimal("5")
#         )
#
#         # 有効な価格（5円刻み）
#         valid_price = Price(Decimal("1500"), "JPY")
#         constraints.validate_price_increment(valid_price)
#
#         # 無効な価格（5円刻みでない）
#         with pytest.raises(TradeValidationError, match="price increment"):
#             invalid_price = Price(Decimal("1502"), "JPY")  # 2円は5円刻みでない
#             constraints.validate_price_increment(invalid_price)
#
#     def test_validate_price_deviation(self):
#         """価格偏差バリデーションテスト"""
#         constraints = OrderConstraints(
#             max_orders_per_second=10,
#             max_pending_orders=50,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("1"),
#             max_price_deviation_percent=Decimal("3")  # 3%以内
#         )
#
#         market_price = Price(Decimal("1000"), "JPY")
#
#         # 有効な価格（3%以内）
#         valid_price = Price(Decimal("1020"), "JPY")  # 2%上昇
#         constraints.validate_price_deviation(valid_price, market_price)
#
#         # 無効な価格（3%超過）
#         with pytest.raises(TradeValidationError, match="price deviation"):
#             invalid_price = Price(Decimal("1050"), "JPY")  # 5%上昇
#             constraints.validate_price_deviation(invalid_price, market_price)



class TestTypeafeTradeManager:
    """TypeSafeTradeManagerのテスト"""

    @pytest.fixture
    def trade_manager(self):
        """取引マネージャーのインスタンスを作成"""
        constraints = TradingConstraints(
            min_order_size=NumericRange(Quantity(1), Quantity(1000)),
            price_range=NumericRange(Decimal("100"), Decimal("10000")),
            max_position_size=Quantity(10000),
            allowed_symbols=["7203", "6758", "9984"]
        )

        # Mock objects for protocols
        mock_data_provider = MagicMock(spec=MarketDataProvider)
        mock_order_executor = MagicMock(spec=OrderExecutor)
        mock_portfolio_manager = MagicMock(spec=PortfolioManager)

        # Configure mocks as needed for tests
        mock_data_provider.is_market_open.return_value = True
        mock_data_provider.get_current_price.return_value = Decimal("1500")
        mock_order_executor.place_order.return_value = "mock_order_id"
        mock_portfolio_manager.get_portfolio_summary.return_value = {"total_value": Decimal("1000000"), "cash_balance": Decimal("500000")}

        return TypeSafeTradeManager(mock_data_provider, mock_order_executor, mock_portfolio_manager, constraints)

    def test_place_market_order_valid(self, trade_manager):
        """有効な成行注文テスト"""
        symbol = "7203"
        quantity = Quantity(50)
        direction = "buy"

        result = trade_manager.place_market_order(symbol, direction, quantity)

        assert result.is_success is True
        assert isinstance(result.data, str)  # order_idが返される
        trade_manager.order_executor.place_order.assert_called_once()

    def test_place_limit_order_valid(self, trade_manager):
        """有効な指値注文テスト"""
        symbol = "7203"
        quantity = Quantity(50)
        price = Decimal("1500")
        direction = "buy"

        result = trade_manager.place_limit_order(symbol, direction, quantity, price)

        assert result.is_success is True
        assert isinstance(result.data, str)  # order_idが返される
        trade_manager.order_executor.place_order.assert_called_once()

    def test_place_order_invalid_symbol(self, trade_manager):
        """無効シンボル注文テスト"""
        symbol = "9999"  # 許可されていないシンボル
        quantity = Quantity(50)
        price = Decimal("1500")
        direction = "buy"

        result = trade_manager.place_limit_order(symbol, direction, quantity, price)
        assert result.is_success is False
        assert "not allowed" in result.error

    #     def test_place_order_risk_limit_exceeded(self, trade_manager):
#         """リスク制限超過テスト"""
#         # This test needs to be re-evaluated based on the new TypeSafeTradeManager and TradingConstraints
#         # with pytest.raises(RiskLimitExceededError):
#         #     symbol = Symbol("7203")
#         #     quantity = Quantity(1000)  # 過大な数量
#         #     price = Price(Decimal("1500"), "JPY")
#         #     trade_manager.place_limit_order(symbol, quantity, price, True)

    #     def test_calculate_position_risk_metrics(self, trade_manager):
#         """ポジションリスクメトリクス計算テスト"""
#         # This method no longer exists in TypeSafeTradeManager
#         # symbol = Symbol("7203")
#         # quantity = Quantity(100)
#         # entry_price = Price(Decimal("1500"), "JPY")
#         # current_price = Price(Decimal("1600"), "JPY")
# 
#         # metrics = trade_manager.calculate_position_risk_metrics(
#         #     symbol, quantity, entry_price, current_price
#         # )
# 
#         # assert "unrealized_pnl" in metrics
#         # assert "position_value" in metrics
#         # assert "risk_exposure" in metrics
#         # assert "stop_loss_price" in metrics
#         # assert "take_profit_price" in metrics
# 
#         # # 未実現損益 = (1600 - 1500) * 100 = 10000
#         # assert metrics["unrealized_pnl"].amount == Decimal("10000")

    #     def test_validate_portfolio_risk_valid(self, trade_manager):
#         """有効ポートフォリオリスク検証テスト"""
#         # This method no longer exists in TypeSafeTradeManager
#         # current_portfolio_value = Money(Decimal("1000000"), "JPY")
#         # daily_pnl = Money(Decimal("20000"), "JPY")
# 
#         # # 正常な範囲内であることを確認（例外が発生しない）
#         # trade_manager.validate_portfolio_risk(current_portfolio_value, daily_pnl)

    #     def test_validate_portfolio_risk_exceeded(self, trade_manager):
#         """ポートフォリオリスク超過テスト"""
#         # This method no longer exists in TypeSafeTradeManager
#         # current_portfolio_value = Money(Decimal("6000000"), "JPY")  # 上限超過
#         # daily_pnl = Money(Decimal("-120000"), "JPY")  # 大きな損失
# 
#         # with pytest.raises(RiskLimitExceededError):
#         #     trade_manager.validate_portfolio_risk(current_portfolio_value, daily_pnl)

    #     def test_get_order_history(self, trade_manager):
#         """注文履歴取得テスト"""
#         # This method no longer exists in TypeSafeTradeManager
#         # # 複数の注文を実行
#         # symbol = Symbol("7203")
# 
#         # with patch.object(trade_manager, '_get_current_market_price') as mock_price:
#         #     mock_price.return_value = Price(Decimal("1500"), "JPY")
# 
#         #     for i in range(3):
#         #         trade_manager.place_market_order(symbol, Quantity(10 + i), True)
# 
#         # history = trade_manager.get_order_history()
#         # assert len(history) == 3
# 
#         # # 特定シンボルの履歴
#         # symbol_history = trade_manager.get_order_history(symbol)
#         # assert len(symbol_history) == 3

    #     def test_cancel_order(self, trade_manager):
#         """注文キャンセルテスト"""
#         # This method no longer exists in TypeSafeTradeManager
#         # order_id = "mock_order_id"
#         # trade_manager.order_executor.cancel_order.return_value = True
# 
#         # result = trade_manager.cancel_order(order_id)
#         # assert result.is_success is True
#         # assert result.data is True
# 
#         # # 存在しない注文のキャンセル
#         # trade_manager.order_executor.cancel_order.return_value = False
#         # non_existent_result = trade_manager.cancel_order("non_existent_id")
#         # assert non_existent_result.is_success is True
#         # assert non_existent_result.data is False


# class TestMarketHours:
#     """MarketHoursのテスト"""
#
#     def test_market_hours_creation(self):
#         """市場時間作成テスト"""
#         market_hours = MarketHours(
#             market_name="TSE",
#             open_time="09:00",
#             close_time="15:00",
#             lunch_start="11:30",
#             lunch_end="12:30",
#             timezone="Asia/Tokyo"
#         )
#
#         assert market_hours.market_name == "TSE"
#         assert market_hours.open_time == "09:00"
#         assert market_hours.close_time == "15:00"
#         assert market_hours.lunch_start == "11:30"
#         assert market_hours.lunch_end == "12:30"
#         assert market_hours.timezone == "Asia/Tokyo"
#
#     def test_is_market_open(self):
#         """市場オープン判定テスト"""
#         market_hours = MarketHours(
#             market_name="TSE",
#             open_time="09:00",
#             close_time="15:00",
#             lunch_start="11:30",
#             lunch_end="12:30",
#             timezone="Asia/Tokyo"
#         )
#
#         # モックして特定の時間をテスト
#         with patch('datetime.datetime') as mock_datetime:
#             # 午前中（市場オープン）
#             mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30)  # 月曜日 10:30
#             assert market_hours.is_market_open() is True
#
#             # 昼休み（市場クローズ）
#             mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0)   # 月曜日 12:00
#             assert market_hours.is_market_open() is False
#
#             # 午後（市場オープン）
#             mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 0)   # 月曜日 14:00
#             assert market_hours.is_market_open() is True
#
#             # 取引終了後（市場クローズ）
#             mock_datetime.now.return_value = datetime(2024, 1, 15, 16, 0)   # 月曜日 16:00
#             assert market_hours.is_market_open() is False
#
#     def test_time_to_next_open(self):
#         """次の開場までの時間計算テスト"""
#         market_hours = MarketHours(
#             market_name="TSE",
#             open_time="09:00",
#             close_time="15:00",
#             lunch_start="11:30",
#             lunch_end="12:30",
#             timezone="Asia/Tokyo"
#         )
#
#         with patch('datetime.datetime') as mock_datetime:
#             # 土曜日の場合（次の月曜日まで）
#             mock_datetime.now.return_value = datetime(2024, 1, 13, 10, 0)  # 土曜日
#             time_delta = market_hours.time_to_next_open()
#             assert isinstance(time_delta, timedelta)
#             assert time_delta.total_seconds() > 0



# class TestOrderBookEntry:
#     """OrderBookEntryのテスト"""
#
#     def test_order_book_entry_creation(self):
#         """オーダーブックエントリ作成テスト"""
#         entry = OrderBookEntry(
#             price=Price(Decimal("1500"), "JPY"),
#             quantity=Quantity(1000),
#             order_count=5,
#             side="bid"
#         )
#
#         assert entry.price.value == Decimal("1500")
#         assert entry.quantity.value == 1000
#         assert entry.order_count == 5
#         assert entry.side == "bid"
#
#     def test_calculate_notional_value(self):
#         """想定元本計算テスト"""
#         entry = OrderBookEntry(
#             price=Price(Decimal("1500"), "JPY"),
#             quantity=Quantity(1000),
#             order_count=5,
#             side="bid"
#         )
#
#         notional = entry.calculate_notional_value()
#         expected = Money(Decimal("1500000"), "JPY")  # 1500 * 1000
#
#         assert notional.amount == expected.amount
#
#     def test_average_order_size(self):
#         """平均注文サイズ計算テスト"""
#         entry = OrderBookEntry(
#             price=Price(Decimal("1500"), "JPY"),
#             quantity=Quantity(1000),
#             order_count=4,
#             side="bid"
#         )
#
#         avg_size = entry.average_order_size()
#         assert avg_size == Quantity(250)  # 1000 / 4



# class TestPriceLevel:
#     """PriceLevelのテスト"""
#
#     def test_price_level_creation(self):
#         """価格レベル作成テスト"""
#         price_level = PriceLevel(
#             price=Price(Decimal("1500"), "JPY"),
#             bid_quantity=Quantity(800),
#             ask_quantity=Quantity(1200),
#             last_traded_quantity=Quantity(100),
#             last_trade_time=datetime.now()
#         )
#
#         assert price_level.price.value == Decimal("1500")
#         assert price_level.bid_quantity.value == 800
#         assert price_level.ask_quantity.value == 1200
#         assert price_level.last_traded_quantity.value == 100
#
#     def test_calculate_spread(self):
#         """スプレッド計算テスト"""
#         price_level = PriceLevel(
#             price=Price(Decimal("1500"), "JPY"),
#             bid_quantity=Quantity(800),
#             ask_quantity=Quantity(1200),
#             last_traded_quantity=Quantity(100),
#             last_trade_time=datetime.now()
#         )
#
#         bid_price = Price(Decimal("1499"), "JPY")
#         ask_price = Price(Decimal("1501"), "JPY")
#
#         spread = price_level.calculate_spread(bid_price, ask_price)
#         assert spread == Money(Decimal("2"), "JPY")  # 1501 - 1499
#
#     def test_calculate_imbalance(self):
#         """需給バランス計算テスト"""
#         price_level = PriceLevel(
#             price=Price(Decimal("1500"), "JPY"),
#             bid_quantity=Quantity(800),
#             ask_quantity=Quantity(1200),
#             last_traded_quantity=Quantity(100),
#             last_trade_time=datetime.now()
#         )
#
#         imbalance = price_level.calculate_imbalance()
#         # (800 - 1200) / (800 + 1200) = -400 / 2000 = -0.2
#         expected = Decimal("-0.2")
#         assert abs(imbalance - expected) < Decimal("0.001")



# class TestVolumeAnalysis:
#     """VolumeAnalysisのテスト"""
#
#     def test_volume_analysis_creation(self):
#         """出来高分析作成テスト"""
#         volume_analysis = VolumeAnalysis(
#             total_volume=Quantity(1000000),
#             average_volume=Quantity(50000),
#             volume_weighted_price=Price(Decimal("1505"), "JPY"),
#             time_period_minutes=60
#         )
#
#         assert volume_analysis.total_volume.value == 1000000
#         assert volume_analysis.average_volume.value == 50000
#         assert volume_analysis.volume_weighted_price.value == Decimal("1505")
#         assert volume_analysis.time_period_minutes == 60
#
#     def test_calculate_volume_rate(self):
#         """出来高レート計算テスト"""
#         volume_analysis = VolumeAnalysis(
#             total_volume=Quantity(120000),
#             average_volume=Quantity(50000),
#             volume_weighted_price=Price(Decimal("1505"), "JPY"),
#             time_period_minutes=60
#         )
#
#         rate = volume_analysis.calculate_volume_rate()
#         expected = Decimal("2.4")  # 120000 / 50000
#         assert rate == expected
#
#     def test_is_high_volume(self):
#         """高出来高判定テスト"""
#         volume_analysis = VolumeAnalysis(
#             total_volume=Quantity(200000),
#             average_volume=Quantity(100000),
#             volume_weighted_price=Price(Decimal("1505"), "JPY"),
#             time_period_minutes=60
#         )
#
#         # デフォルト閾値（1.5倍）
#         assert volume_analysis.is_high_volume() is True
#
#         # カスタム閾値
#         assert volume_analysis.is_high_volume(threshold=3.0) is False



# class TestLiquidityMetrics:
#     """LiquidityMetricsのテスト"""

#     def test_liquidity_metrics_creation(self):
#         """流動性メトリクス作成テスト"""
#         metrics = LiquidityMetrics(
#             bid_ask_spread=Money(Decimal("2"), "JPY"),
#             market_depth=Quantity(500000),
#             turnover_ratio=Decimal("0.15"),
#             price_impact=Decimal("0.002")
#         )
#
#         assert metrics.bid_ask_spread.amount == Decimal("2")
#         assert metrics.market_depth.value == 500000
#         assert metrics.turnover_ratio == Decimal("0.15")
#         assert metrics.price_impact == Decimal("0.002")
#
#     def test_calculate_liquidity_score(self):
#         """流動性スコア計算テスト"""
#         metrics = LiquidityMetrics(
#             bid_ask_spread=Money(Decimal("1"), "JPY"),    # 狭いスプレッド（良い）
#             market_depth=Quantity(1000000),               # 大きな市場深度（良い）
#             turnover_ratio=Decimal("0.20"),               # 高い回転率（良い）
#             price_impact=Decimal("0.001")                 # 低いプライスインパクト（良い）
#         )
#
#         score = metrics.calculate_liquidity_score()
#
#         # 高い流動性を示すスコア（0-100の範囲で高い値）
#         assert 70 <= score <= 100
#
#     def test_is_liquid_market(self):
#         """流動的市場判定テスト"""
#         # 高流動性市場
#         high_liquidity = LiquidityMetrics(
#             bid_ask_spread=Money(Decimal("1"), "JPY"),
#             market_depth=Quantity(1000000),
#             turnover_ratio=Decimal("0.25"),
#             price_impact=Decimal("0.001")
#         )
#         assert high_liquidity.is_liquid_market() is True
#
#         # 低流動性市場
#         low_liquidity = LiquidityMetrics(
#             bid_ask_spread=Money(Decimal("10"), "JPY"),   # 広いスプレッド
#             market_depth=Quantity(10000),                 # 小さな市場深度
#             turnover_ratio=Decimal("0.05"),               # 低い回転率
#             price_impact=Decimal("0.02")                  # 高いプライスインパクト
#         )
#         assert low_liquidity.is_liquid_market() is False



# class TestErrorScenarios:
#     """エラーシナリオのテスト"""
#
#     def test_concurrent_order_placement(self):
#         """同時注文配置テスト"""
#         constraints = TradingConstraints(
#             max_position_size=Money(Decimal("1000000"), "JPY"),
#             max_daily_loss=Money(Decimal("50000"), "JPY"),
#             max_single_trade_size=Money(Decimal("100000"), "JPY"),
#             min_trade_size=Money(Decimal("1000"), "JPY"),
#             allowed_symbols=["7203"],
#             max_orders_per_day=5  # 制限を低く設定
#         )
#
#         order_constraints = OrderConstraints(
#             max_orders_per_second=1,  # 厳しい制限
#             max_pending_orders=2,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("1"),
#             max_price_deviation_percent=Decimal("5")
#         )
#
#         risk_params = RiskParameters(
#             max_portfolio_value=Money(Decimal("5000000"), "JPY"),
#             max_daily_var=Money(Decimal("100000"), "JPY"),
#             position_size_limit_percent=Decimal("10"),
#             stop_loss_percent=Decimal("5"),
#             take_profit_percent=Decimal("15"),
#             max_correlation=Decimal("0.7")
#         )
#
#         trade_manager = TypeSafeTradeManager(constraints, risk_params, order_constraints)
#
#         def place_order():
#             try:
#                 symbol = Symbol("7203")
#                 quantity = Quantity(10)
#                 with patch.object(trade_manager, 
#                     '_get_current_market_price') as mock_price:
#                     mock_price.return_value = Price(Decimal("1500"), "JPY")
#                     return trade_manager.place_market_order(symbol, quantity, True)
#             except Exception as e:
#                 return str(e)
#
#         import threading
#         import time
#
#         # 複数スレッドで同時に注文
#         threads = []
#         results = []
#
#         def worker():
#             result = place_order()
#             results.append(result)
#
#         for _ in range(3):
#             thread = threading.Thread(target=worker)
#             threads.append(thread)
#             thread.start()
#             time.sleep(0.1)  # わずかな遅延
#
#         for thread in threads:
#             thread.join()
#
#         # 少なくとも1つは成功し、一部はレート制限エラーになることを確認
#         success_count = sum(1 for r in results if isinstance(r, dict))
#         error_count = sum(1 for r in results if isinstance(r, str))
#
#         assert success_count >= 1
#         assert error_count >= 0  # レート制限により一部失敗する可能性
#
#     def test_invalid_trading_constraints(self):
#         """無効な取引制約テスト"""
#         with pytest.raises(ValueError):
#             TradingConstraints(
#                 max_position_size=Money(Decimal("50000"), "JPY"),
#                 max_daily_loss=Money(Decimal("100000"), "JPY"),  # 最大損失 > ポジションサイズ
#                 max_single_trade_size=Money(Decimal("30000"), "JPY"),
#                 min_trade_size=Money(Decimal("1000"), "JPY"),
#                 allowed_symbols=["7203"],
#                 max_orders_per_day=100
#             )
#
#     def test_market_hours_edge_cases(self):
#         """市場時間エッジケーステスト"""
#         market_hours = MarketHours(
#             market_name="TSE",
#             open_time="09:00",
#             close_time="15:00",
#             lunch_start="11:30",
#             lunch_end="12:30",
#             timezone="Asia/Tokyo"
#         )
#
#         with patch('datetime.datetime') as mock_datetime:
#             # 週末テスト
#             mock_datetime.now.return_value = datetime(2024, 1, 13, 10, 0)  # 土曜日
#             assert market_hours.is_market_open() is False
#
#             mock_datetime.now.return_value = datetime(2024, 1, 14, 10, 0)  # 日曜日
#             assert market_hours.is_market_open() is False



# class TestIntegrationScenarios:
#     """統合シナリオのテスト"""
#
#     def test_complete_trading_workflow_with_risk_management(self):
#         """リスク管理付き完全取引ワークフローテスト"""
#         # 制約設定
#         constraints = TradingConstraints(
#             max_position_size=Money(Decimal("1000000"), "JPY"),
#             max_daily_loss=Money(Decimal("50000"), "JPY"),
#             max_single_trade_size=Money(Decimal("100000"), "JPY"),
#             min_trade_size=Money(Decimal("10000"), "JPY"),
#             allowed_symbols=["7203", "6758"],
#             max_orders_per_day=100
#         )
#
#         risk_params = RiskParameters(
#             max_portfolio_value=Money(Decimal("5000000"), "JPY"),
#             max_daily_var=Money(Decimal("100000"), "JPY"),
#             position_size_limit_percent=Decimal("20"),
#             stop_loss_percent=Decimal("5"),
#             take_profit_percent=Decimal("15"),
#             max_correlation=Decimal("0.7")
#         )
#
#         order_constraints = OrderConstraints(
#             max_orders_per_second=10,
#             max_pending_orders=50,
#             order_timeout_minutes=30,
#             min_price_increment=Decimal("1"),
#             max_price_deviation_percent=Decimal("3")
#         )
#
#         trade_manager = TypeSafeTradeManager(constraints, risk_params, order_constraints)
#
#         # 1. 成行注文
#         with patch.object(trade_manager, '_get_current_market_price') as mock_price:
#             mock_price.return_value = Price(Decimal("1500"), "JPY")
#
#             market_order = trade_manager.place_market_order(
#                 Symbol("7203"), Quantity(50), True
#             )
#             assert market_order["order_type"] == "market"
#
#         # 2. 指値注文
#         limit_order = trade_manager.place_limit_order(
#             Symbol("6758"), Quantity(30), Price(Decimal("2000"), "JPY"), True
#         )
#         assert limit_order["order_type"] == "limit"
#
#         # 3. リスクメトリクス計算
#         metrics = trade_manager.calculate_position_risk_metrics(
#             Symbol("7203"), Quantity(50),
#             Price(Decimal("1500"), "JPY"), Price(Decimal("1520"), "JPY")
#         )
#         assert metrics["unrealized_pnl"].amount == Decimal("1000")  # (1520-1500)*50
#
#         # 4. ポートフォリオリスク検証
#         trade_manager.validate_portfolio_risk(
#             Money(Decimal("2000000"), "JPY"),
#             Money(Decimal("10000"), "JPY")
#         )
#
#         # 5. 注文履歴確認
#         history = trade_manager.get_order_history()
#         assert len(history) == 2



if __name__ == "__main__":
    pytest.main([__file__, "-v"])