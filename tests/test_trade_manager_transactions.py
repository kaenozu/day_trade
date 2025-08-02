"""
TradeManagerのトランザクション管理テスト
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.day_trade.core.trade_manager import TradeManager, TradeType, TradeStatus
from src.day_trade.models.database import db_manager, DatabaseConfig
from src.day_trade.models.stock import Stock, Trade


class TestTradeManagerTransactions:
    """TradeManagerのトランザクション管理テスト"""

    @pytest.fixture
    def test_db_manager(self):
        """テスト用データベースマネージャー"""
        config = DatabaseConfig.for_testing()
        test_manager = db_manager.__class__(config)
        test_manager.create_tables()
        return test_manager

    @pytest.fixture
    def trade_manager(self, test_db_manager):
        """テスト用TradeManager"""
        # 実際のdb_managerを一時的に置き換え
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            tm = TradeManager(
                commission_rate=Decimal("0.001"),
                tax_rate=Decimal("0.2"),
                load_from_db=False
            )
            yield tm

    def test_add_trade_with_database_persistence(self, trade_manager, test_db_manager):
        """データベース永続化を有効にした取引追加のテスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # 取引を追加（DB永続化有効）
            trade_id = trade_manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500"),
                notes="テスト取引",
                persist_to_db=True
            )

            # メモリ内のデータを検証
            assert len(trade_manager.trades) == 1
            assert trade_manager.trades[0].symbol == "7203"
            assert trade_manager.trades[0].trade_type == TradeType.BUY

            # データベース内のデータを検証
            with test_db_manager.session_scope() as session:
                # 銘柄マスタが作成されているか確認
                stock = session.query(Stock).filter(Stock.code == "7203").first()
                assert stock is not None
                assert stock.code == "7203"

                # 取引記録が作成されているか確認
                db_trade = session.query(Trade).filter(Trade.stock_code == "7203").first()
                assert db_trade is not None
                assert db_trade.trade_type == "buy"
                assert db_trade.quantity == 100
                assert db_trade.price == 2500.0

    def test_add_trade_memory_only(self, trade_manager):
        """メモリのみの取引追加のテスト"""
        # 取引を追加（DB永続化無効）
        trade_id = trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500"),
            notes="メモリのみテスト取引",
            persist_to_db=False
        )

        # メモリ内のデータを検証
        assert len(trade_manager.trades) == 1
        assert trade_manager.trades[0].symbol == "7203"

    def test_transaction_rollback_on_error(self, trade_manager, test_db_manager):
        """エラー時のトランザクションロールバックテスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # 銘柄作成でエラーを発生させるモック
            with patch.object(Stock, '__init__', side_effect=Exception("銘柄作成エラー")):
                with pytest.raises(Exception, match="銘柄作成エラー"):
                    trade_manager.add_trade(
                        symbol="7203",
                        trade_type=TradeType.BUY,
                        quantity=100,
                        price=Decimal("2500"),
                        persist_to_db=True
                    )

                # メモリ内にデータが残っていないことを確認
                assert len(trade_manager.trades) == 0

                # データベースにデータが残っていないことを確認
                with test_db_manager.session_scope() as session:
                    stock_count = session.query(Stock).count()
                    trade_count = session.query(Trade).count()
                    assert stock_count == 0
                    assert trade_count == 0

    def test_multiple_trades_atomic_transaction(self, trade_manager, test_db_manager):
        """複数取引のアトミック実行テスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # 複数の取引を順次追加
            trade_id1 = trade_manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500"),
                persist_to_db=True
            )

            trade_id2 = trade_manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=200,
                price=Decimal("2450"),
                persist_to_db=True
            )

            # メモリ内のデータを検証
            assert len(trade_manager.trades) == 2

            # ポジションが正しく計算されているか確認
            position = trade_manager.get_position("7203")
            assert position is not None
            assert position.quantity == 300  # 100 + 200

            # 平均価格の計算確認
            expected_avg_price = (
                (Decimal("2500") * 100 + Decimal("2450") * 200 +
                 trade_manager._calculate_commission(Decimal("2500"), 100) +
                 trade_manager._calculate_commission(Decimal("2450"), 200))
                / 300
            )
            assert abs(position.average_price - expected_avg_price) < Decimal("1")

            # データベースの一貫性を確認
            with test_db_manager.session_scope() as session:
                db_trades = session.query(Trade).filter(Trade.stock_code == "7203").all()
                assert len(db_trades) == 2
                assert sum(t.quantity for t in db_trades) == 300

    def test_load_trades_from_database(self, test_db_manager):
        """データベースからの取引読み込みテスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # まず直接データベースに取引データを作成
            with test_db_manager.transaction_scope() as session:
                stock = Stock(
                    code="7203",
                    name="トヨタ自動車",
                    market="東証一部",
                    sector="自動車",
                    industry="自動車"
                )
                session.add(stock)
                session.flush()

                db_trade = Trade.create_buy_trade(
                    session=session,
                    stock_code="7203",
                    quantity=100,
                    price=2500.0,
                    commission=250.0,
                    memo="DB直接作成取引"
                )

            # データベースから読み込むTradeManagerを作成
            tm = TradeManager(load_from_db=True)

            # 読み込まれたデータを検証
            assert len(tm.trades) == 1
            assert tm.trades[0].symbol == "7203"
            assert tm.trades[0].quantity == 100
            assert tm.trades[0].trade_type == TradeType.BUY

            # ポジションが正しく計算されているか確認
            position = tm.get_position("7203")
            assert position is not None
            assert position.quantity == 100

    def test_sync_with_database(self, trade_manager, test_db_manager):
        """データベース同期テスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # メモリ内に取引を追加（DB永続化あり）
            trade_id = trade_manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500"),
                persist_to_db=True
            )

            # 手動でメモリ内データを変更
            trade_manager.trades[0].notes = "手動変更"

            # データベースと同期
            trade_manager.sync_with_db()

            # データベースからの情報が優先されることを確認
            assert len(trade_manager.trades) == 1
            assert trade_manager.trades[0].notes != "手動変更"  # DB内容が復元される

    def test_concurrent_transaction_handling(self, test_db_manager):
        """並行トランザクション処理のテスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            tm1 = TradeManager(load_from_db=False)
            tm2 = TradeManager(load_from_db=False)

            # 2つのTradeManagerから同時に同じ銘柄の取引を追加
            trade_id1 = tm1.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500"),
                persist_to_db=True
            )

            trade_id2 = tm2.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=200,
                price=Decimal("2450"),
                persist_to_db=True
            )

            # データベースの一貫性を確認
            with test_db_manager.session_scope() as session:
                db_trades = session.query(Trade).filter(Trade.stock_code == "7203").all()
                assert len(db_trades) == 2

                # 銘柄マスタが重複作成されていないことを確認
                stocks = session.query(Stock).filter(Stock.code == "7203").all()
                assert len(stocks) == 1

    def test_commission_calculation_consistency(self, trade_manager, test_db_manager):
        """手数料計算の一貫性テスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # 手数料を明示的に指定
            explicit_commission = Decimal("200")
            trade_id = trade_manager.add_trade(
                symbol="7203",
                trade_type=TradeType.BUY,
                quantity=100,
                price=Decimal("2500"),
                commission=explicit_commission,
                persist_to_db=True
            )

            # メモリとデータベースで同じ手数料が記録されているか確認
            memory_trade = trade_manager.trades[0]
            assert memory_trade.commission == explicit_commission

            with test_db_manager.session_scope() as session:
                db_trade = session.query(Trade).filter(Trade.stock_code == "7203").first()
                assert Decimal(str(db_trade.commission)) == explicit_commission

    def test_error_logging_and_context(self, trade_manager, test_db_manager):
        """エラーログとコンテキスト情報のテスト"""
        with patch('src.day_trade.core.trade_manager.db_manager', test_db_manager):
            # ログモックを設定
            with patch('src.day_trade.core.trade_manager.log_error_with_context') as mock_log_error:
                with patch('src.day_trade.core.trade_manager.DBTrade.create_buy_trade',
                          side_effect=Exception("DB取引作成エラー")):

                    with pytest.raises(Exception):
                        trade_manager.add_trade(
                            symbol="7203",
                            trade_type=TradeType.BUY,
                            quantity=100,
                            price=Decimal("2500"),
                            persist_to_db=True
                        )

                    # エラーログが適切なコンテキスト情報と共に呼ばれたか確認
                    mock_log_error.assert_called_once()
                    call_args = mock_log_error.call_args
                    assert "symbol" in call_args[0][1]
                    assert "trade_type" in call_args[0][1]
                    assert call_args[0][1]["symbol"] == "7203"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
