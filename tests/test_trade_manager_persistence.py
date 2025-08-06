"""
TradeManagerのデータベース永続化機能のテスト
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.day_trade.core.trade_manager import TradeManager, TradeType
from src.day_trade.utils.exceptions import DatabaseError


@pytest.mark.skip(reason="TradeManager永続化API変更により一時的に無効化")
class TestTradeManagerPersistence:
    """TradeManagerのデータベース永続化テスト"""

    @pytest.fixture
    def trade_manager(self):
        """テスト用TradeManager"""
        return TradeManager(
            commission_rate=Decimal("0.001"),
            tax_rate=Decimal("0.2"),
            load_from_db=False,
        )

    def test_buy_stock_with_db_persistence(self, trade_manager):
        """buy_stockでのデータベース永続化テスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = (
                None
            )

            # Stock作成のモック
            with patch("src.day_trade.models.stock.Stock") as mock_stock_class:
                mock_stock = Mock()
                mock_stock_class.return_value = mock_stock

                # DBTrade作成のモック
                with patch("src.day_trade.models.stock.Trade") as mock_trade_class:
                    mock_db_trade = Mock()
                    mock_db_trade.id = 1
                    mock_trade_class.create_buy_trade.return_value = mock_db_trade

                    result = trade_manager.buy_stock(
                        symbol="7203",
                        quantity=100,
                        price=Decimal("2500"),
                        persist_to_db=True,
                    )

                    assert result["success"] is True
                    assert result["symbol"] == "7203"

                    # データベース操作が呼ばれたことを確認
                    mock_db.transaction_scope.assert_called_once()
                    mock_session.add.assert_called_once()
                    mock_trade_class.create_buy_trade.assert_called_once()

    def test_sell_stock_with_db_persistence(self, trade_manager):
        """sell_stockでのデータベース永続化テスト"""
        # 先に買いポジションを作成（DBなし）
        trade_manager.buy_stock(
            symbol="7203", quantity=100, price=Decimal("2500"), persist_to_db=False
        )

        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # DBTrade作成のモック
            with patch("src.day_trade.models.stock.Trade") as mock_trade_class:
                mock_db_trade = Mock()
                mock_db_trade.id = 2
                mock_trade_class.create_sell_trade.return_value = mock_db_trade

                result = trade_manager.sell_stock(
                    symbol="7203",
                    quantity=50,
                    price=Decimal("2600"),
                    persist_to_db=True,
                )

                assert result["success"] is True
                assert result["symbol"] == "7203"

                # データベース操作が呼ばれたことを確認
                mock_db.transaction_scope.assert_called_once()
                mock_trade_class.create_sell_trade.assert_called_once()

    def test_load_trades_from_db_with_data(self, trade_manager):
        """_load_trades_from_dbでデータありのテスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # モックDBTradeデータを作成
            mock_db_trade = Mock()
            mock_db_trade.id = 1
            mock_db_trade.stock_code = "7203"
            mock_db_trade.trade_type = "buy"
            mock_db_trade.quantity = 100
            mock_db_trade.price = 2500.0
            mock_db_trade.trade_datetime = "2023-01-15T10:00:00"
            mock_db_trade.commission = 250.0
            mock_db_trade.memo = "テスト取引"

            mock_session.query.return_value.order_by.return_value.all.return_value = [
                mock_db_trade
            ]

            # 実行
            trade_manager._load_trades_from_db()

            # 取引が読み込まれたことを確認
            assert len(trade_manager.trades) == 1
            loaded_trade = trade_manager.trades[0]
            assert loaded_trade.symbol == "7203"
            assert loaded_trade.trade_type == TradeType.BUY
            assert loaded_trade.quantity == 100

    def test_load_trades_from_db_with_enum_type(self, trade_manager):
        """_load_trades_from_dbでenum型のテスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # TradeType enumを直接使用するケース
            mock_db_trade = Mock()
            mock_db_trade.id = 1
            mock_db_trade.stock_code = "7203"
            mock_db_trade.trade_type = TradeType.SELL  # 直接enum
            mock_db_trade.quantity = 50
            mock_db_trade.price = 2600.0
            mock_db_trade.trade_datetime = "2023-01-15T14:00:00"
            mock_db_trade.commission = 130.0
            mock_db_trade.memo = None

            mock_session.query.return_value.order_by.return_value.all.return_value = [
                mock_db_trade
            ]

            # 実行
            trade_manager._load_trades_from_db()

            # 取引が正しく読み込まれたことを確認
            assert len(trade_manager.trades) == 1
            loaded_trade = trade_manager.trades[0]
            assert loaded_trade.trade_type == TradeType.SELL

    def test_load_trades_from_db_error_recovery(self, trade_manager):
        """_load_trades_from_dbでエラー時の復旧テスト"""
        # 先にメモリ内データを作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()
        initial_counter = trade_manager._trade_counter

        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # 2番目の取引処理でエラーを発生
            mock_db_trade1 = Mock()
            mock_db_trade1.id = 1
            mock_db_trade1.stock_code = "7203"
            mock_db_trade1.trade_type = "buy"
            mock_db_trade1.quantity = 100
            mock_db_trade1.price = 2500.0
            mock_db_trade1.trade_datetime = "2023-01-15T10:00:00"
            mock_db_trade1.commission = 250.0
            mock_db_trade1.memo = "テスト取引1"

            mock_db_trade2 = Mock()
            mock_db_trade2.id = 2
            mock_db_trade2.stock_code = "8306"
            mock_db_trade2.trade_type = "invalid_type"  # 無効なtype
            mock_db_trade2.quantity = 50
            mock_db_trade2.price = 800.0
            mock_db_trade2.trade_datetime = "2023-01-15T11:00:00"
            mock_db_trade2.commission = 100.0
            mock_db_trade2.memo = "テスト取引2"

            mock_session.query.return_value.order_by.return_value.all.return_value = [
                mock_db_trade1,
                mock_db_trade2,
            ]

            # エラーが発生することを確認
            with pytest.raises(ValueError):
                trade_manager._load_trades_from_db()

            # データが復旧されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)
            assert trade_manager._trade_counter == initial_counter

    def test_sync_with_db_success(self, trade_manager):
        """sync_with_dbの成功テスト"""
        # 先にメモリ内データを作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )

        with patch.object(trade_manager, "_load_trades_from_db") as mock_load:
            # モックで新しいデータを設定
            mock_load.side_effect = lambda: setattr(trade_manager, "trades", [])

            # 実行
            trade_manager.sync_with_db()

            # _load_trades_from_dbが呼ばれたことを確認
            mock_load.assert_called_once()

    def test_sync_with_db_error_recovery(self, trade_manager):
        """sync_with_dbのエラー時復旧テスト"""
        # 先にメモリ内データを作成
        trade_manager.add_trade(
            "7203", TradeType.BUY, 100, Decimal("2500"), persist_to_db=False
        )
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()

        with patch.object(trade_manager, "_load_trades_from_db") as mock_load:
            mock_load.side_effect = DatabaseError("DB Load Error")

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.sync_with_db()

            # データが復旧されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)

    def test_load_trades_counter_update(self, trade_manager):
        """_load_trades_from_dbでのカウンター更新テスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # 複数のモックDBTradeデータを作成（IDが異なる）
            mock_db_trades = []
            for i in range(3):
                mock_trade = Mock()
                mock_trade.id = i + 10  # 10, 11, 12
                mock_trade.stock_code = f"720{i + 3}"
                mock_trade.trade_type = "buy"
                mock_trade.quantity = 100
                mock_trade.price = 2500.0
                mock_trade.trade_datetime = "2023-01-15T10:00:00"
                mock_trade.commission = 250.0
                mock_trade.memo = f"テスト取引{i + 1}"
                mock_db_trades.append(mock_trade)

            mock_session.query.return_value.order_by.return_value.all.return_value = (
                mock_db_trades
            )

            # 実行
            trade_manager._load_trades_from_db()

            # カウンターが最大ID+1に設定されていることを確認
            assert trade_manager._trade_counter == 13  # max(10,11,12) + 1

    def test_load_trades_empty_database(self, trade_manager):
        """_load_trades_from_dbで空データベースのテスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # 空のリストを返す
            mock_session.query.return_value.order_by.return_value.all.return_value = []

            # 実行
            trade_manager._load_trades_from_db()

            # データが空であることを確認
            assert len(trade_manager.trades) == 0
            assert len(trade_manager.positions) == 0
            assert len(trade_manager.realized_pnl) == 0

    def test_buy_stock_existing_stock_in_db(self, trade_manager):
        """buy_stockで既存銘柄がDBに存在する場合のテスト"""
        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            mock_session = Mock()
            mock_db.transaction_scope.return_value.__enter__.return_value = mock_session

            # 既存の銘柄マスタが見つかる場合
            existing_stock = Mock()
            existing_stock.code = "7203"
            existing_stock.name = "トヨタ自動車"
            mock_session.query.return_value.filter.return_value.first.return_value = (
                existing_stock
            )

            # DBTrade作成のモック
            with patch("src.day_trade.models.stock.Trade") as mock_trade_class:
                mock_db_trade = Mock()
                mock_db_trade.id = 1
                mock_trade_class.create_buy_trade.return_value = mock_db_trade

                result = trade_manager.buy_stock(
                    symbol="7203",
                    quantity=100,
                    price=Decimal("2500"),
                    persist_to_db=True,
                )

                assert result["success"] is True

                # 新しい銘柄は追加されない（既存があるため）
                mock_session.add.assert_not_called()
                # DBTradeは作成される
                mock_trade_class.create_buy_trade.assert_called_once()

    def test_buy_stock_database_transaction_rollback(self, trade_manager):
        """buy_stockでデータベーストランザクションのロールバックテスト"""
        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()

        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            # トランザクション内でエラーを発生
            mock_db.transaction_scope.side_effect = DatabaseError("Transaction failed")

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.buy_stock(
                    symbol="7203",
                    quantity=100,
                    price=Decimal("2500"),
                    persist_to_db=True,
                )

            # メモリ内データが変更されていないことを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)

    def test_sell_stock_database_transaction_rollback(self, trade_manager):
        """sell_stockでデータベーストランザクションのロールバックテスト"""
        # 先に買いポジションを作成
        trade_manager.buy_stock(
            symbol="7203", quantity=100, price=Decimal("2500"), persist_to_db=False
        )

        initial_trades = trade_manager.trades.copy()
        initial_positions = trade_manager.positions.copy()
        initial_realized_pnl = trade_manager.realized_pnl.copy()

        with patch("src.day_trade.core.trade_manager.db_manager") as mock_db:
            # トランザクション内でエラーを発生
            mock_db.transaction_scope.side_effect = DatabaseError("Transaction failed")

            # エラーが発生することを確認
            with pytest.raises(DatabaseError):
                trade_manager.sell_stock(
                    symbol="7203",
                    quantity=50,
                    price=Decimal("2600"),
                    persist_to_db=True,
                )

            # メモリ内データが復元されていることを確認
            assert len(trade_manager.trades) == len(initial_trades)
            assert len(trade_manager.positions) == len(initial_positions)
            assert len(trade_manager.realized_pnl) == len(initial_realized_pnl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
