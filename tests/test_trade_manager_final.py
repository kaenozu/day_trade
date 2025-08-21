"""
TradeManagerの最終的なカバレッジ向上テスト
実際に動作するテストのみを含む
"""

import os
import tempfile
from decimal import Decimal

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    quantize_decimal,
)


class TestBasicFunctionality:
    """基本機能テスト"""

    def test_quantize_decimal_function(self):
        """Decimal量子化関数テスト"""
        result = quantize_decimal(Decimal("123.456"), 2)
        assert result == Decimal("123.46")

    def test_trade_manager_creation(self):
        """TradeManager作成テスト"""
        manager = TradeManager()
        assert len(manager.trades) == 0
        assert len(manager.positions) == 0

    def test_buy_trade(self):
        """買い取引テスト"""
        manager = TradeManager()
        trade_id = manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        assert trade_id is not None
        assert len(manager.trades) == 1
        assert "7203" in manager.positions

    def test_buy_and_sell_trade(self):
        """買いと売り取引テスト"""
        manager = TradeManager()

        # 買い
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        assert manager.positions["7203"].quantity == 100

        # 部分売り
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))
        assert manager.positions["7203"].quantity == 50

        # 完全売り
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2700"))
        assert "7203" not in manager.positions

    def test_multiple_stocks(self):
        """複数銘柄テスト"""
        manager = TradeManager()

        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("6758", TradeType.BUY, 200, Decimal("1000"))

        assert len(manager.positions) == 2
        assert manager.positions["7203"].quantity == 100
        assert manager.positions["6758"].quantity == 200

    def test_trade_history(self):
        """取引履歴テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        history = manager.get_trade_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "7203"

    def test_realized_pnl_history(self):
        """実現損益履歴テスト"""
        manager = TradeManager()

        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
        manager.add_trade("7203", TradeType.SELL, 50, Decimal("2600"))

        pnl_history = manager.get_realized_pnl_history()
        assert len(pnl_history) > 0

    def test_portfolio_summary(self):
        """ポートフォリオサマリーテスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        summary = manager.get_portfolio_summary()
        assert "total_value" in summary
        assert "positions" in summary

    def test_error_handling_empty_symbol(self):
        """エラーハンドリング: 空銘柄コード"""
        manager = TradeManager()

        with pytest.raises(ValueError):
            manager.add_trade("", TradeType.BUY, 100, Decimal("2500"))

    def test_error_handling_negative_quantity(self):
        """エラーハンドリング: 負数量"""
        manager = TradeManager()

        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.BUY, -100, Decimal("2500"))

    def test_error_handling_sell_without_position(self):
        """エラーハンドリング: ポジション無しでの売却"""
        manager = TradeManager()

        with pytest.raises(ValueError):
            manager.add_trade("7203", TradeType.SELL, 100, Decimal("2500"))

    def test_export_csv(self):
        """CSV出力テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                manager.export_to_csv(tmp.name)
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_export_json(self):
        """JSON出力テスト"""
        manager = TradeManager()
        manager.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            try:
                manager.export_to_json(tmp.name)
                assert os.path.exists(tmp.name)
                assert os.path.getsize(tmp.name) > 0
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])