"""
TradeManagerの100%カバレッジ最終攻撃テスト
実際のコード分析に基づく確実なカバレッジ向上
"""

import os
import tempfile
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    TradeStatus,
    BuyLot,
    Position,
    Trade,
    quantize_decimal,
    safe_decimal_conversion,
    mask_sensitive_info,
)


class TestTradeManagerPersistToDbFalse:
    """persist_to_db=Falseのパステスト（行1000-1022をカバー）"""

    def test_add_trade_memory_only_path(self):
        """メモリのみの取引追加パス"""
        manager = TradeManager()  # デフォルトでpersist_to_db=False

        # 買い取引
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        assert trade_id is not None
        assert len(manager.trades) == 1

        # 売り取引
        trade_id2 = manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        assert trade_id2 is not None
        assert len(manager.trades) == 2


class TestUpdatePositionPaths:
    """_update_positionメソッドのパステスト（行1039以降）"""

    def test_buy_new_position_creation(self):
        """新規ポジション作成パス（行1047-1048）"""
        manager = TradeManager()

        # 新規銘柄の買い取引
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 新規ポジションが作成されていることを確認
        assert "7203" in manager.positions
        position = manager.positions["7203"]
        assert position.quantity == 100
        assert len(position.buy_lots) == 1

    def test_buy_existing_position_addition(self):
        """既存ポジションへの追加パス（行1049以降）"""
        manager = TradeManager()

        # 最初の買い取引
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 同じ銘柄の追加買い取引
        manager.add_trade("7203", TradeType.BUY, 200, Decimal("2600"))

        # ポジションが統合されていることを確認
        position = manager.positions["7203"]
        assert position.quantity == 300
        assert len(position.buy_lots) == 2


class TestSellPositionFifoLogic:
    """売却時のFIFOロジックテスト"""

    def test_sell_fifo_partial(self):
        """FIFO部分売却テスト"""
        manager = TradeManager()

        # 複数回に分けて購入
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2400"))
        manager.add_trade("7203", TradeType.BUY, 200, Decimal("2500"))

        # 部分売却（FIFOで最古から売却）
        manager.add_trade("7203", TradeType.SELL, 150, Decimal("2700"))

        # 残ポジション確認
        position = manager.positions["7203"]
        assert position.quantity == 150
        # 最古のロットが部分的に消費され、新しいロットが残る
        assert len(position.buy_lots) >= 1

    def test_sell_fifo_complete(self):
        """FIFO完全売却テスト"""
        manager = TradeManager()

        # 購入
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 完全売却
        manager.add_trade("7203", TradeType.SELL, 100, Decimal("2600"))

        # ポジションが削除されていることを確認
        assert "7203" not in manager.positions


class TestPositionClassMethods:
    """Positionクラスのメソッドテスト"""

    def test_position_market_value(self):
        """市場価値計算テスト"""
        buy_lot = BuyLot(
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=None,
            trade_id="T001"
        )

        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250250"),
            current_price=Decimal("2600"),
            buy_lots=[buy_lot]
        )

        # 市場価値 = 数量 × 現在価格
        market_value = position.market_value()
        assert market_value == Decimal("260000")  # 100 * 2600

    def test_position_unrealized_pnl(self):
        """含み損益計算テスト"""
        buy_lot = BuyLot(
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=None,
            trade_id="T001"
        )

        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250250"),
            current_price=Decimal("2600"),
            buy_lots=[buy_lot]
        )

        # 含み損益 = 市場価値 - 総コスト
        unrealized_pnl = position.unrealized_pnl()
        assert unrealized_pnl == Decimal("9750")  # 260000 - 250250

    def test_position_unrealized_pnl_percent(self):
        """含み損益率計算テスト"""
        buy_lot = BuyLot(
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=None,
            trade_id="T001"
        )

        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500"),
            total_cost=Decimal("250250"),
            current_price=Decimal("2600"),
            buy_lots=[buy_lot]
        )

        # 含み損益率計算
        pnl_percent = position.unrealized_pnl_percent()
        assert isinstance(pnl_percent, Decimal)
        assert pnl_percent > Decimal("0")  # 利益が出ている


class TestBuyLotClassMethods:
    """BuyLotクラスのメソッドテスト"""

    def test_buy_lot_total_cost_per_share(self):
        """1株あたり総コスト計算テスト"""
        buy_lot = BuyLot(
            quantity=100,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=None,
            trade_id="T001"
        )

        # 1株あたり総コスト = (価格 × 数量 + 手数料) / 数量
        cost_per_share = buy_lot.total_cost_per_share()
        expected = (Decimal("2500") * 100 + Decimal("250")) / 100
        assert cost_per_share == expected

    def test_buy_lot_total_cost_per_share_zero_quantity(self):
        """数量ゼロでの1株あたり総コスト計算テスト"""
        buy_lot = BuyLot(
            quantity=0,
            price=Decimal("2500"),
            commission=Decimal("250"),
            timestamp=None,
            trade_id="T001"
        )

        # 数量ゼロの場合はゼロを返す
        cost_per_share = buy_lot.total_cost_per_share()
        assert cost_per_share == Decimal("0")


class TestQuantizeDecimalFunction:
    """quantize_decimal関数の完全テスト"""

    def test_quantize_various_precisions(self):
        """様々な精度でのquantize_decimalテスト"""
        # 2桁精度
        result = quantize_decimal(Decimal("123.456789"), 2)
        assert result == Decimal("123.46")

        # 4桁精度
        result = quantize_decimal(Decimal("123.456789"), 4)
        assert result == Decimal("123.4568")

        # 0桁精度（整数）
        result = quantize_decimal(Decimal("123.456789"), 0)
        assert result == Decimal("123")

        # 6桁精度
        result = quantize_decimal(Decimal("123.456789"), 6)
        assert result == Decimal("123.456789")


class TestMaskSensitiveInfoFunction:
    """mask_sensitive_info関数の完全テスト"""

    def test_mask_various_lengths(self):
        """様々な長さでのマスキングテスト"""
        # 長い文字列
        result = mask_sensitive_info("1234567890")
        assert result == "1*******0"
        assert result[0] == "1"
        assert result[-1] == "0"

        # 短い文字列
        result = mask_sensitive_info("ABC")
        assert result == "A*C"

        # 2文字
        result = mask_sensitive_info("AB")
        assert result == "**"

        # 1文字
        result = mask_sensitive_info("A")
        assert result == "*"

        # 空文字列
        result = mask_sensitive_info("")
        assert result == ""


class TestTradeManagerAnalyticsMethods:
    """TradeManagerの分析メソッドテスト"""

    def test_calculate_tax_various_scenarios(self):
        """様々なシナリオでの税金計算テスト"""
        manager = TradeManager(tax_rate=Decimal("0.20315"))

        # 利益の場合
        tax = manager.calculate_tax(Decimal("100000"))
        expected = Decimal("100000") * Decimal("0.20315")
        assert tax == expected

        # 損失の場合
        tax = manager.calculate_tax(Decimal("-50000"))
        assert tax == Decimal("0")

        # ゼロの場合
        tax = manager.calculate_tax(Decimal("0"))
        assert tax == Decimal("0")

        # 小さな利益
        tax = manager.calculate_tax(Decimal("100"))
        assert tax > Decimal("0")


class TestUpdateCurrentPriceMethod:
    """update_current_priceメソッドテスト"""

    def test_update_existing_position_price(self):
        """既存ポジションの価格更新テスト"""
        manager = TradeManager()

        # ポジション作成
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        # 価格更新
        manager.update_current_price("7203", Decimal("2600"))

        # 価格が更新されていることを確認
        position = manager.positions["7203"]
        assert position.current_price == Decimal("2600")

    def test_update_nonexistent_position_price(self):
        """存在しないポジションの価格更新テスト"""
        manager = TradeManager()

        # 存在しないポジションの価格更新（エラーにならない）
        manager.update_current_price("7203", Decimal("2500"))

        # 何も起こらない（エラーなし）
        assert "7203" not in manager.positions


class TestTradeToDict:
    """Trade.to_dictメソッドテスト"""

    def test_trade_to_dict_conversion(self):
        """取引の辞書変換テスト"""
        from datetime import datetime

        trade = Trade(
            id="T001",
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            commission=Decimal("250"),
            status=TradeStatus.EXECUTED,
            notes="テスト取引"
        )

        trade_dict = trade.to_dict()

        assert trade_dict["id"] == "T001"
        assert trade_dict["symbol"] == "7203"
        assert trade_dict["trade_type"] == "BUY"
        assert trade_dict["quantity"] == 100
        assert trade_dict["price"] == "2500"
        assert trade_dict["commission"] == "250"
        assert trade_dict["status"] == "executed"
        assert trade_dict["notes"] == "テスト取引"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])