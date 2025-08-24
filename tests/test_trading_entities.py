"""
DDD/トレーディングエンティティの包括的テストスイート

Issue #10: テストカバレッジ向上とエラーハンドリング強化
優先度: #2 (Priority Score: 195)
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.day_trade.core.trade_manager import TradeManager
from src.day_trade.core.trade_models import Trade, Position, TradeStatus, BuyLot
from src.day_trade.domain.trading.entities import Portfolio, PositionId, TradeId # TradeId, PositionIdを追加
from src.day_trade.domain.common.value_objects import Money, Quantity, Symbol, Price # 追加
from src.day_trade.models.enums import TradeType


class TestTrade:
    """Tradeエンティティのテスト"""

    def test_trade_creation(self):
        """取引作成テスト"""
        trade_manager = TradeManager()
        trade_id = trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"), notes="Test trade")
        trade = trade_manager.get_trade_history()[0]

        assert trade.id == trade_id
        assert trade.symbol == "7203"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == Decimal("1500")
        assert isinstance(trade.timestamp, datetime)
        assert trade.notes == "Test trade"

    def test_trade_to_dict_from_dict(self):
        """Tradeのto_dictとfrom_dictメソッドのテスト"""
        trade_manager = TradeManager()
        trade_id = trade_manager.add_trade("7203", TradeType.SELL, 50, Decimal("1600"), commission=Decimal("100"), notes="Test sell trade")
        original_trade = trade_manager.get_trade_history()[0]

        trade_dict = original_trade.to_dict()
        restored_trade = Trade.from_dict(trade_dict)

        assert original_trade.id == restored_trade.id
        assert original_trade.symbol == restored_trade.symbol
        assert original_trade.trade_type == restored_trade.trade_type
        assert original_trade.quantity == restored_trade.quantity
        assert original_trade.price == restored_trade.price
        assert original_trade.timestamp == restored_trade.timestamp
        assert original_trade.commission == restored_trade.commission
        assert original_trade.status == restored_trade.status
        assert original_trade.notes == restored_trade.notes

    def test_trade_validation_invalid_quantity(self):
        """無効な数量での取引作成テスト"""
        trade_manager = TradeManager()
        with pytest.raises(ValueError, match="数量は正の整数である必要があります"):
            trade_manager.add_trade("7203", TradeType.BUY, 0, Decimal("1500"))

    def test_trade_validation_invalid_price(self):
        """無効な価格での取引作成テスト"""
        trade_manager = TradeManager()
        with pytest.raises(ValueError, match="価格は正数である必要があります"):
            trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("-100"))


class TestPosition:
    """Positionエンティティのテスト"""

    def setup_method(self):
        self.trade_manager = TradeManager()

    def test_position_creation(self):
        """ポジション作成テスト"""
        position = self.trade_manager.get_position("7203")
        assert position is None

    def test_position_add_trade_buy(self):
        """買い取引追加テスト"""
        # 初回買い
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))
        position = self.trade_manager.get_position("7203")
        assert position.quantity == 100
        assert position.average_price == Decimal("1500")

        # 追加買い
        self.trade_manager.add_trade("7203", TradeType.BUY, 200, Decimal("1800"))
        position = self.trade_manager.get_position("7203")
        assert position.quantity == 300
        # 平均コスト = (100*1500 + 200*1800) / 300 = 1700
        assert position.average_price == Decimal("1700")

    def test_position_add_trade_sell(self):
        """売り取引追加テスト"""
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))

        # 一部売却
        self.trade_manager.add_trade("7203", TradeType.SELL, 40, Decimal("1600"))
        position = self.trade_manager.get_position("7203")

        assert position.quantity == 60
        assert position.average_price == Decimal("1500")  # 平均コストは変わらず

    def test_position_calculate_unrealized_pnl(self):
        """未実現損益計算テスト"""
        position = Position(symbol="7203", quantity=0, average_price=Decimal("0"), total_cost=Decimal("0"))

        # 買いポジション構築
        trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("7203"),
            trade_type=TradeType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500")),
            timestamp=datetime.now(),
            commission=Money(Decimal("0"))
        )
        position.add_trade(trade)

        # 現在価格での未実現損益計算
        current_price = Price(Decimal("1600"))

        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        expected = Money(Decimal("10000"))  # (1600-1500) * 100
        assert unrealized_pnl.amount == expected.amount

    def test_position_calculate_realized_pnl(self):
        """実現損益計算テスト"""
        position = Position(symbol="7203", quantity=0, average_price=Decimal("0"), total_cost=Decimal("0"))

        # 買い取引
        buy_trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("7203"),
            trade_type=TradeType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("100")),
            timestamp=datetime.now(),
            commission=Money(Decimal("10"))
        )
        position.add_trade(buy_trade)

        # 売り取引
        sell_trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("7203"),
            trade_type=TradeType.SELL,
            quantity=Quantity(100),
            price=Price(Decimal("120")),
            timestamp=datetime.now(),
            commission=Money(Decimal("10"))
        )
        position.add_trade(sell_trade)

        realized_pnl = position.calculate_realized_pnl()
        # (120 - 100) * 100 - 10 - 10 = 2000 - 20 = 1980
        assert realized_pnl.amount == Decimal("1980")

    def test_position_insufficient_quantity_sell(self):
        """不足数量売却エラーテスト"""
        position = Position(symbol="7203", quantity=0, average_price=Decimal("0"), total_cost=Decimal("0"))

        # 少量の買いポジション
        buy_trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("7203"),
            trade_type=TradeType.BUY,
            quantity=Quantity(50),
            price=Price(Decimal("1500")),
            timestamp=datetime.now(),
            commission=Money(Decimal("0"))
        )
        position.add_trade(buy_trade)

        # 過剰な売却
        sell_trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("7203"),
            trade_type=TradeType.SELL,
            quantity=Quantity(100),  # 50より多い
            price=Price(Decimal("1600")),
            timestamp=datetime.now(),
            commission=Money(Decimal("0"))
        )
        with pytest.raises(ValueError, match="保有数量が不足しています"):
            position.add_trade(sell_trade)

    def test_position_symbol_mismatch(self):
        """シンボル不一致エラーテスト"""
        position = Position(symbol="7203", quantity=0, average_price=Decimal("0"), total_cost=Decimal("0"))

        trade = Trade(
            id=TradeId(uuid.uuid4()),
            symbol=Symbol("6758"),  # 異なるシンボル
            trade_type=TradeType.BUY,
            quantity=Quantity(100),
            price=Price(Decimal("1500")),
            timestamp=datetime.now(),
            commission=Money(Decimal("0"))
        )
        with pytest.raises(ValueError, match="異なる銘柄の取引を追加できません"):
            position.add_trade(trade)


class TestPortfolio:
    """Portfolioエンティティのテスト"""

    def setup_method(self):
        self.trade_manager = TradeManager()

    def test_portfolio_creation(self):
        """ポートフォリオ作成テスト"""
        assert len(self.trade_manager.get_all_positions()) == 0

    def test_portfolio_add_position(self):
        """ポジション追加テスト"""
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))
        assert len(self.trade_manager.get_all_positions()) == 1
        assert self.trade_manager.get_position("7203") is not None

    def test_portfolio_get_position(self):
        """ポジション取得テスト"""
        position = self.trade_manager.get_position("7203")
        assert position is None

        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))
        position = self.trade_manager.get_position("7203")
        assert position is not None
        assert position.symbol == "7203"
        assert position.quantity == 100

    def test_portfolio_calculate_total_value(self):
        """ポートフォリオ総価値計算テスト"""
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))
        self.trade_manager.add_trade("6758", TradeType.BUY, 200, Decimal("2000"))

        self.trade_manager.update_current_prices({
            "7203": Decimal("1600"),
            "6758": Decimal("2100")
        })

        summary = self.trade_manager.get_portfolio_summary()
        expected_value = 100 * 1600 + 200 * 2100
        assert Decimal(summary["total_market_value"]) == Decimal(expected_value)

    def test_portfolio_calculate_unrealized_pnl(self):
        """ポートフォリオ未実現損益計算テスト"""
        self.trade_manager.add_trade("7203", TradeType.BUY, 100, Decimal("1500"))
        self.trade_manager.update_current_prices({"7203": Decimal("1600")})
        summary = self.trade_manager.get_portfolio_summary()
        expected_pnl = (1600 - 1500) * 100
        assert Decimal(summary["total_unrealized_pnl"]) == Decimal(expected_pnl)

    def test_portfolio_insufficient_cash(self):
        """現金不足エラーテスト"""
        # This test needs to be re-evaluated as cash management is not yet implemented.



#
#
#
if __name__ == "__main__":
    pytest.main([__file__, "-v"])