"""
TradeManagerの拡張テスト - 未カバー領域のテスト強化
"""

import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.day_trade.core.trade_manager import (
    TradeManager,
    TradeType,
    Position,
    RealizedPnL
)


class TestTradeManagerExtended:
    """TradeManagerの拡張テスト"""

    @pytest.fixture
    def trade_manager(self):
        """テスト用TradeManager"""
        return TradeManager(
            commission_rate=Decimal("0.001"),
            tax_rate=Decimal("0.2"),
            load_from_db=False,
        )

    def test_buy_stock_basic(self, trade_manager):
        """buy_stockメソッドの基本テスト"""
        result = trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            notes="テスト購入",
            persist_to_db=False
        )

        assert result["success"] is True
        assert result["symbol"] == "7203"
        assert result["quantity"] == 100
        assert result["price"] == 2500.0
        assert "trade_id" in result
        assert "commission" in result
        assert "position" in result

        # ポジション確認
        position = trade_manager.get_position("7203")
        assert position is not None
        assert position.quantity == 100

    def test_buy_stock_with_current_price(self, trade_manager):
        """buy_stockメソッドの現在価格更新テスト"""
        result = trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            current_market_price=Decimal("2550"),
            persist_to_db=False
        )

        assert result["success"] is True
        position = trade_manager.get_position("7203")
        assert position.current_price == Decimal("2550")

    def test_buy_stock_invalid_quantity(self, trade_manager):
        """buy_stockメソッドの無効数量テスト"""
        with pytest.raises(ValueError, match="購入数量は正数である必要があります"):
            trade_manager.buy_stock(
                symbol="7203",
                quantity=0,
                price=Decimal("2500"),
                persist_to_db=False
            )

        with pytest.raises(ValueError, match="購入数量は正数である必要があります"):
            trade_manager.buy_stock(
                symbol="7203",
                quantity=-100,
                price=Decimal("2500"),
                persist_to_db=False
            )

    def test_buy_stock_invalid_price(self, trade_manager):
        """buy_stockメソッドの無効価格テスト"""
        with pytest.raises(ValueError, match="購入価格は正数である必要があります"):
            trade_manager.buy_stock(
                symbol="7203",
                quantity=100,
                price=Decimal("0"),
                persist_to_db=False
            )

        with pytest.raises(ValueError, match="購入価格は正数である必要があります"):
            trade_manager.buy_stock(
                symbol="7203",
                quantity=100,
                price=Decimal("-2500"),
                persist_to_db=False
            )

    def test_sell_stock_basic(self, trade_manager):
        """sell_stockメソッドの基本テスト"""
        # 先に買いポジションを作成
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )

        # 売却実行
        result = trade_manager.sell_stock(
            symbol="7203",
            quantity=50,
            price=Decimal("2600"),
            notes="テスト売却",
            persist_to_db=False
        )

        assert result["success"] is True
        assert result["symbol"] == "7203"
        assert result["quantity"] == 50
        assert result["position_closed"] is False
        assert "realized_pnl" in result

        # ポジション確認
        position = trade_manager.get_position("7203")
        assert position is not None
        assert position.quantity == 50

    def test_sell_stock_complete_position(self, trade_manager):
        """sell_stockメソッドの完全売却テスト"""
        # 先に買いポジションを作成
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )

        # 完全売却実行
        result = trade_manager.sell_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2600"),
            persist_to_db=False
        )

        assert result["success"] is True
        assert result["position_closed"] is True

        # ポジションが削除されたことを確認
        position = trade_manager.get_position("7203")
        assert position is None

    def test_sell_stock_invalid_quantity(self, trade_manager):
        """sell_stockメソッドの無効数量テスト"""
        with pytest.raises(ValueError, match="売却数量は正数である必要があります"):
            trade_manager.sell_stock(
                symbol="7203",
                quantity=0,
                price=Decimal("2500"),
                persist_to_db=False
            )

    def test_sell_stock_no_position(self, trade_manager):
        """sell_stockメソッドのポジション未保有テスト"""
        with pytest.raises(ValueError, match="ポジションが存在しません"):
            trade_manager.sell_stock(
                symbol="7203",
                quantity=100,
                price=Decimal("2500"),
                persist_to_db=False
            )

    def test_sell_stock_insufficient_quantity(self, trade_manager):
        """sell_stockメソッドの保有数量不足テスト"""
        # 先に100株購入
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )

        # 150株売却を試行（数量不足）
        with pytest.raises(ValueError, match="売却数量.*が保有数量.*を超過しています"):
            trade_manager.sell_stock(
                symbol="7203",
                quantity=150,
                price=Decimal("2600"),
                persist_to_db=False
            )

    def test_execute_trade_order_buy(self, trade_manager):
        """execute_trade_orderメソッドの買い注文テスト"""
        trade_order = {
            "action": "buy",
            "symbol": "7203",
            "quantity": 100,
            "price": Decimal("2500"),
            "notes": "統一インターフェーステスト"
        }

        result = trade_manager.execute_trade_order(trade_order, persist_to_db=False)

        assert result["success"] is True
        assert result["symbol"] == "7203"
        assert result["quantity"] == 100

    def test_execute_trade_order_sell(self, trade_manager):
        """execute_trade_orderメソッドの売り注文テスト"""
        # 先に買いポジション作成
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )

        trade_order = {
            "action": "sell",
            "symbol": "7203",
            "quantity": 50,
            "price": Decimal("2600"),
            "current_market_price": Decimal("2580"),
            "notes": "統一インターフェーステスト"
        }

        result = trade_manager.execute_trade_order(trade_order, persist_to_db=False)

        assert result["success"] is True
        assert result["symbol"] == "7203"
        assert result["quantity"] == 50

    def test_execute_trade_order_invalid_action(self, trade_manager):
        """execute_trade_orderメソッドの無効アクションテスト"""
        trade_order = {
            "action": "invalid",
            "symbol": "7203",
            "quantity": 100,
            "price": Decimal("2500")
        }

        with pytest.raises(ValueError, match="無効な取引アクション"):
            trade_manager.execute_trade_order(trade_order, persist_to_db=False)

    def test_calculate_tax_implications_with_gains(self, trade_manager):
        """税務計算（利益あり）のテスト"""
        current_year = datetime.now().year

        # 利益の出る取引を実行
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )
        trade_manager.sell_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2700"),
            persist_to_db=False
        )

        tax_info = trade_manager.calculate_tax_implications(current_year)

        assert tax_info["year"] == current_year
        assert tax_info["total_trades"] == 1
        assert Decimal(tax_info["total_gain"]) > 0
        assert Decimal(tax_info["net_gain"]) > 0
        assert Decimal(tax_info["tax_due"]) > 0
        assert tax_info["winning_trades"] == 1
        assert tax_info["losing_trades"] == 0

    def test_calculate_tax_implications_with_losses(self, trade_manager):
        """税務計算（損失あり）のテスト"""
        current_year = datetime.now().year

        # 損失の出る取引を実行
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )
        trade_manager.sell_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2300"),
            persist_to_db=False
        )

        tax_info = trade_manager.calculate_tax_implications(current_year)

        assert tax_info["year"] == current_year
        assert tax_info["total_trades"] == 1
        assert Decimal(tax_info["total_gain"]) == 0
        assert Decimal(tax_info["total_loss"]) > 0
        assert Decimal(tax_info["net_gain"]) < 0
        assert Decimal(tax_info["tax_due"]) == 0  # 損失なので税金なし
        assert tax_info["winning_trades"] == 0
        assert tax_info["losing_trades"] == 1

    def test_calculate_tax_implications_no_trades(self, trade_manager):
        """税務計算（取引なし）のテスト"""
        current_year = datetime.now().year
        tax_info = trade_manager.calculate_tax_implications(current_year)

        assert tax_info["year"] == current_year
        assert tax_info["total_trades"] == 0
        assert Decimal(tax_info["total_gain"]) == 0
        assert Decimal(tax_info["total_loss"]) == 0
        assert Decimal(tax_info["net_gain"]) == 0
        assert Decimal(tax_info["tax_due"]) == 0

    @patch('src.day_trade.core.trade_manager.logger')
    def test_calculate_tax_implications_error_handling(self, mock_logger, trade_manager):
        """税務計算のエラー処理テスト"""
        # 無効な年を指定してエラーを発生させる
        invalid_year = "invalid_year"

        with pytest.raises(Exception):
            trade_manager.calculate_tax_implications(invalid_year)

        # エラーログが出力されることを確認
        mock_logger.error.assert_called_once()

    def test_load_from_json_basic(self, trade_manager):
        """load_from_jsonメソッドの基本テスト"""
        # 先にデータを作成
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )
        trade_manager.sell_stock(
            symbol="7203",
            quantity=50,
            price=Decimal("2600"),
            persist_to_db=False
        )

        # JSONに保存
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            json_path = tmp_file.name

        try:
            trade_manager.save_to_json(json_path)

            # 新しいマネージャーで読み込み
            new_manager = TradeManager(load_from_db=False)
            new_manager.load_from_json(json_path)

            # データが復元されていることを確認
            assert len(new_manager.trades) == len(trade_manager.trades)
            assert len(new_manager.positions) == len(trade_manager.positions)
            assert len(new_manager.realized_pnl) == len(trade_manager.realized_pnl)
            assert new_manager.commission_rate == trade_manager.commission_rate
            assert new_manager.tax_rate == trade_manager.tax_rate

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_load_from_json_invalid_file(self, trade_manager):
        """load_from_jsonメソッドの無効ファイルテスト"""
        # 存在しないファイル
        with pytest.raises(FileNotFoundError):
            trade_manager.load_from_json("nonexistent.json")

        # 無効なJSONファイル
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write("invalid json content")
            invalid_json_path = tmp_file.name

        try:
            with pytest.raises(json.JSONDecodeError):
                trade_manager.load_from_json(invalid_json_path)
        finally:
            if os.path.exists(invalid_json_path):
                os.unlink(invalid_json_path)

    @patch('src.day_trade.core.trade_manager.logger')
    def test_load_from_json_error_logging(self, mock_logger, trade_manager):
        """load_from_jsonメソッドのエラーログテスト"""
        # 存在しないファイルでエラーを発生
        try:
            trade_manager.load_from_json("nonexistent.json")
        except FileNotFoundError:
            pass

        # エラーログが出力されていることを確認
        mock_logger.error.assert_called_once()

    @patch('src.day_trade.core.trade_manager.logger')
    def test_save_to_json_error_logging(self, mock_logger, trade_manager):
        """save_to_jsonメソッドのエラーログテスト"""
        # 無効なパス（書き込み権限なし）でエラーを発生
        try:
            trade_manager.save_to_json("/invalid/path/test.json")
        except (OSError, PermissionError):
            pass

        # エラーログが出力されていることを確認
        mock_logger.error.assert_called_once()

    @patch('src.day_trade.core.trade_manager.logger')
    def test_export_to_csv_error_logging(self, mock_logger, trade_manager):
        """export_to_csvメソッドのエラーログテスト"""
        # 無効なパス（書き込み権限なし）でエラーを発生
        try:
            trade_manager.export_to_csv("/invalid/path/test.csv", "trades")
        except (OSError, PermissionError):
            pass

        # エラーログが出力されていることを確認
        mock_logger.error.assert_called_once()

    def test_position_to_dict_quantize(self, trade_manager):
        """Positionのto_dictメソッドでのquantize処理テスト"""
        position = Position(
            symbol="7203",
            quantity=100,
            average_price=Decimal("2500.123456"),
            total_cost=Decimal("250012.3456"),
            current_price=Decimal("2600.789012")
        )

        result = position.to_dict()

        # quantizeが正しく適用されていることを確認
        assert "unrealized_pnl_percent" in result
        # 小数点以下2桁に丸められていることを確認
        percent_str = result["unrealized_pnl_percent"]
        decimal_places = len(percent_str.split('.')[-1]) if '.' in percent_str else 0
        assert decimal_places <= 2

    def test_realized_pnl_to_dict_quantize(self, trade_manager):
        """RealizedPnLのto_dictメソッドでのquantize処理テスト"""
        realized_pnl = RealizedPnL(
            symbol="7203",
            quantity=100,
            buy_price=Decimal("2500.123"),
            sell_price=Decimal("2600.456"),
            buy_commission=Decimal("250.12"),
            sell_commission=Decimal("260.45"),
            pnl=Decimal("9769.785"),
            pnl_percent=Decimal("3.90791234"),
            buy_date=datetime.now(),
            sell_date=datetime.now()
        )

        result = realized_pnl.to_dict()

        # quantizeが正しく適用されていることを確認
        percent_str = result["pnl_percent"]
        decimal_places = len(percent_str.split('.')[-1]) if '.' in percent_str else 0
        assert decimal_places <= 2

    def test_database_error_rollback_buy_stock(self, trade_manager):
        """buy_stockでのデータベースエラー時のロールバックテスト"""
        # メモリ内データのバックアップを確認するため先にデータを作成
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()

        # データベースエラーをシミュレート
        with patch('src.day_trade.core.trade_manager.db_manager') as mock_db_manager:
            mock_db_manager.transaction_scope.side_effect = Exception("DB Error")

            with pytest.raises(Exception, match="DB Error"):
                trade_manager.buy_stock(
                    symbol="7203",
                    quantity=100,
                    price=Decimal("2500"),
                    persist_to_db=True
                )

            # メモリ内データが復元されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)

    def test_database_error_rollback_sell_stock(self, trade_manager):
        """sell_stockでのデータベースエラー時のロールバックテスト"""
        # 先に買いポジションを作成
        trade_manager.buy_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2500"),
            persist_to_db=False
        )

        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()
        initial_realized_pnl = trade_manager.realized_pnl.copy()

        # データベースエラーをシミュレート
        with patch('src.day_trade.core.trade_manager.db_manager') as mock_db_manager:
            mock_db_manager.transaction_scope.side_effect = Exception("DB Error")

            with pytest.raises(Exception, match="DB Error"):
                trade_manager.sell_stock(
                    symbol="7203",
                    quantity=50,
                    price=Decimal("2600"),
                    persist_to_db=True
                )

            # メモリ内データが復元されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)
            assert len(trade_manager.realized_pnl) == len(initial_realized_pnl)

    def test_get_earliest_buy_date_no_trades(self, trade_manager):
        """_get_earliest_buy_dateメソッドの取引なしテスト"""
        # プライベートメソッドを直接テスト
        result = trade_manager._get_earliest_buy_date("NONEXISTENT")

        # 現在時刻が返されることを確認（厳密な時刻比較は避ける）
        assert isinstance(result, datetime)

    def test_get_earliest_buy_date_with_trades(self, trade_manager):
        """_get_earliest_buy_dateメソッドの取引ありテスト"""
        # 異なる時刻で複数の買い取引を作成
        base_time = datetime(2023, 1, 15, 10, 0, 0)

        trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            timestamp=base_time,
            persist_to_db=False
        )

        trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=50,
            price=Decimal("2450"),
            timestamp=base_time.replace(hour=14),
            persist_to_db=False
        )

        earliest_date = trade_manager._get_earliest_buy_date("7203")
        assert earliest_date == base_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
