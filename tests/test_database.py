"""
データベース機能のテスト
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from src.day_trade.models import Alert, PriceData, Stock, Trade, WatchlistItem
from src.day_trade.models.database import (
    Base,
    DatabaseConfig,
    DatabaseManager,
    TEST_DATABASE_URL,
)


@pytest.fixture(scope="function")
def test_db_manager():
    """テスト用インメモリデータベースマネージャー"""
    config = DatabaseConfig.for_testing()
    db_manager = DatabaseManager(config)
    db_manager.create_tables()
    yield db_manager
    db_manager.drop_tables()


@pytest.fixture(scope="function")
def populated_db_manager(test_db_manager):
    """データが投入されたテスト用データベースマネージャー"""
    with test_db_manager.session_scope() as session:
        # 銘柄データの追加
        stock1 = Stock(code="7203", name="トヨタ自動車", sector="輸送用機器")
        stock2 = Stock(code="9984", name="ソフトバンクグループ", sector="情報・通信業")
        session.add_all([stock1, stock2])
        session.flush()

        # 価格データの追加
        price1 = PriceData(
            stock_code="7203",
            datetime=datetime(2023, 1, 1, 9, 0, 0),
            open=100.0, high=105.0, low=99.0, close=103.0, volume=100000
        )
        price2 = PriceData(
            stock_code="7203",
            datetime=datetime(2023, 1, 2, 9, 0, 0),
            open=103.0, high=108.0, low=102.0, close=107.0, volume=120000
        )
        price3 = PriceData(
            stock_code="9984",
            datetime=datetime(2023, 1, 1, 9, 0, 0),
            open=5000.0, high=5100.0, low=4950.0, close=5050.0, volume=50000
        )
        session.add_all([price1, price2, price3])
        session.flush()

        # 取引データの追加
        trade1 = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime(2023, 1, 1, 10, 0, 0),
        )
        session.add(trade1)
        session.flush()

        # ウォッチリストアイテムの追加
        watchlist_item1 = WatchlistItem(
            stock_code="7203", group_name="MyList", memo="お気に入り"
        )
        session.add(watchlist_item1)
        session.flush()

        # アラートの追加
        alert1 = Alert(
            stock_code="7203",
            alert_type="price_above",
            threshold=105.0,
            is_active=True,
        )
        session.add(alert1)
        session.flush()

    yield test_db_manager


class TestDatabaseManager:
    """DatabaseManagerクラスのテスト"""

    def test_initialization(self, test_db_manager):
        """初期化テスト"""
        assert test_db_manager.engine is not None
        assert test_db_manager.session_factory is not None

    def test_create_and_drop_tables(self, test_db_manager):
        """テーブル作成・削除テスト"""
        # 既に作成されているので、ここではエラーにならないことを確認
        test_db_manager.create_tables()
        # 削除
        test_db_manager.drop_tables()
        # 再度作成
        test_db_manager.create_tables()

    def test_session_scope(self, test_db_manager):
        """セッションスコープテスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="1234", name="テスト銘柄")
            session.add(stock)
        # セッションがコミットされ、クローズされていることを確認
        with test_db_manager.session_scope() as session:
            retrieved_stock = session.query(Stock).filter_by(code="1234").first()
            assert retrieved_stock is not None
            assert retrieved_stock.name == "テスト銘柄"

    def test_session_scope_rollback_on_error(self, test_db_manager):
        """セッションスコープでのエラー時のロールバックテスト"""
        with pytest.raises(IntegrityError):
            with test_db_manager.session_scope() as session:
                stock1 = Stock(code="7203", name="トヨタ自動車")
                session.add(stock1)
                session.flush()  # ここでコミットはされないが、重複チェックのためにflush

                # 重複する銘柄を追加してIntegrityErrorを発生させる
                stock2 = Stock(code="7203", name="トヨタ自動車（重複）")
                session.add(stock2)
                session.flush()

        # ロールバックされていることを確認
        with test_db_manager.session_scope() as session:
            retrieved_stocks = session.query(Stock).filter_by(code="7203").all()
            assert len(retrieved_stocks) == 0

    def test_transaction_scope(self, test_db_manager):
        """トランザクションスコープテスト"""
        with test_db_manager.transaction_scope() as session:
            stock1 = Stock(code="A001", name="銘柄A")
            stock2 = Stock(code="B002", name="銘柄B")
            session.add_all([stock1, stock2])

        with test_db_manager.session_scope() as session:
            assert session.query(Stock).count() == 2

    def test_transaction_scope_rollback_on_error(self, test_db_manager):
        """トランザクションスコープでのエラー時のロールバックテスト"""
        with pytest.raises(IntegrityError):
            with test_db_manager.transaction_scope() as session:
                stock1 = Stock(code="C003", name="銘柄C")
                session.add(stock1)
                session.flush()

                # 重複する銘柄を追加
                stock2 = Stock(code="C003", name="銘柄C（重複）")
                session.add(stock2)
                session.flush()

        with test_db_manager.session_scope() as session:
            assert session.query(Stock).filter_by(code="C003").count() == 0

    def test_transaction_scope_retry_on_operational_error(self, test_db_manager):
        """OperationalError時のリトライテスト"""
        # OperationalErrorをモック
        with patch('src.day_trade.models.database.Session.begin') as mock_begin:
            mock_begin.side_effect = [
                OperationalError(None, None, "database is locked"),
                Mock(),  # 2回目で成功
            ]

            with test_db_manager.transaction_scope(retry_count=1) as session:
                stock = Stock(code="D004", name="銘柄D")
                session.add(stock)

            assert mock_begin.call_count == 2
            with test_db_manager.session_scope() as session:
                assert session.query(Stock).filter_by(code="D004").count() == 1

    def test_get_latest_prices(self, populated_db_manager):
        """最新価格取得テスト"""
        with populated_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["7203", "9984"])

            assert "7203" in latest_prices
            assert "9984" in latest_prices
            assert latest_prices["7203"].close == 107.0
            assert latest_prices["9984"].close == 5050.0

    def test_get_price_range(self, populated_db_manager):
        """期間指定価格取得テスト"""
        with populated_db_manager.session_scope() as session:
            prices = PriceData.get_price_range(
                session, "7203", datetime(2023, 1, 1), datetime(2023, 1, 1)
            )
            assert len(prices) == 1
            assert prices[0].close == 103.0

    def test_get_portfolio_summary(self, populated_db_manager):
        """ポートフォリオサマリー取得テスト"""
        with populated_db_manager.session_scope() as session:
            summary = Trade.get_portfolio_summary(session)

            assert summary["portfolio"]["7203"]["quantity"] == 100
            assert summary["portfolio"]["7203"]["total_cost"] == 10010.0
            assert summary["total_cost"] == 10010.0

    def test_get_recent_trades(self, populated_db_manager):
        """最近の取引履歴取得テスト"""
        with populated_db_manager.session_scope() as session:
            trades = Trade.get_recent_trades(session, days=1)
            assert len(trades) == 1
            assert trades[0].stock_code == "7203"

    def test_alembic_integration(self, test_db_manager):
        """Alembic統合テスト"""
        # alembic.iniとversionsディレクトリが存在することを確認
        assert os.path.exists("alembic.ini")
        assert os.path.exists("alembic/versions")

        # Alembic初期化（既に初期化済みの場合もあるのでエラーにならないことを確認）
        test_db_manager.init_alembic()

        # マイグレーション作成
        test_db_manager.migrate("test migration")

        # アップグレード
        test_db_manager.upgrade()

        # 現在のリビジョン取得
        current_rev = test_db_manager.current_revision()
        assert current_rev is not None

        # ダウングレード
        test_db_manager.downgrade()

    def test_bulk_insert(self, test_db_manager):
        """一括挿入テスト"""
        data_list = [
            {"code": "S001", "name": "銘柄S1"},
            {"code": "S002", "name": "銘柄S2"},
        ]
        test_db_manager.bulk_insert(Stock, data_list)

        with test_db_manager.session_scope() as session:
            assert session.query(Stock).filter(Stock.code.in_(["S001", "S002"])).count() == 2

    def test_bulk_update(self, test_db_manager):
        """一括更新テスト"""
        # データを挿入
        stock = Stock(code="U001", name="更新前銘柄")
        with test_db_manager.session_scope() as session:
            session.add(stock)

        # 更新データ
        update_data = [
            {"id": stock.id, "name": "更新後銘柄"},
        ]
        test_db_manager.bulk_update(Stock, update_data)

        with test_db_manager.session_scope() as session:
            updated_stock = session.query(Stock).filter_by(code="U001").first()
            assert updated_stock.name == "更新後銘柄"

    def test_atomic_operation(self, test_db_manager):
        """アトミック操作テスト"""
        def op1(session):
            session.add(Stock(code="A1", name="Test1"))

        def op2(session):
            session.add(Stock(code="A2", name="Test2"))

        test_db_manager.atomic_operation([op1, op2])

        with test_db_manager.session_scope() as session:
            assert session.query(Stock).filter(Stock.code.in_(["A1", "A2"])).count() == 2

    def test_execute_query(self, populated_db_manager):
        """生SQLクエリ実行テスト"""
        results = populated_db_manager.execute_query("SELECT code, name FROM stocks WHERE code = :code", {"code": "7203"})
        assert len(results) == 1
        assert results[0].code == "7203"
        assert results[0].name == "トヨタ自動車"

    def test_optimize_database(self, test_db_manager):
        """データベース最適化テスト"""
        # SQLiteの場合のみ実行されることを確認
        if test_db_manager.config.is_sqlite():
            test_db_manager.optimize_database()
            # エラーが発生しないことを確認
            assert True
        else:
            # SQLite以外の場合は何もしない
            assert True


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"


class TestDatabaseConfig:
    """DatabaseConfigクラスのテスト"""

    def test_for_testing(self):
        """テスト用設定生成テスト"""
        config = DatabaseConfig.for_testing()
        assert config.database_url == "sqlite:///:memory:"
        assert config.echo is False
        assert config.is_in_memory() is True

    def test_for_production(self):
        """本番用設定生成テスト"""
        config = DatabaseConfig.for_production()
        assert config.database_url.startswith("sqlite:///")
        assert config.echo is False
        assert config.pool_size == 10

    def test_is_sqlite(self):
        """SQLite判定テスト"""
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_sqlite() is True
        config = DatabaseConfig("postgresql://user:pass@host:port/db")
        assert config.is_sqlite() is False

    def test_is_in_memory(self):
        """インメモリ判定テスト"""
        config = DatabaseConfig("sqlite:///:memory:")
        assert config.is_in_in_memory() is True
        config = DatabaseConfig("sqlite:///test.db")
        assert config.is_in_memory() is False


class TestPriceDataModel:
    """PriceDataモデルのテスト"""

    def test_get_latest_prices_empty(self, test_db_manager):
        """空の最新価格取得テスト"""
        with test_db_manager.session_scope() as session:
            latest_prices = PriceData.get_latest_prices(session, ["UNKNOWN"])
            assert len(latest_prices) == 0


class TestTradeModel:
    """Tradeモデルのテスト"""

    def test_total_amount_buy(self):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="buy",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 10010.0

    def test_total_amount_sell(self):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code="7203",
            trade_type="sell",
            quantity=100,
            price=100.0,
            commission=10.0,
            trade_datetime=datetime.now(),
        )
        assert trade.total_amount == 9990.0

    def test_create_buy_trade(self, test_db_manager):
        """買い取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_buy_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "buy"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None

    def test_create_sell_trade(self, test_db_manager):
        """売り取引作成テスト"""
        with test_db_manager.session_scope() as session:
            trade = Trade.create_sell_trade(session, "7203", 100, 100.0, 10.0)
            assert trade.stock_code == "7203"
            assert trade.trade_type == "sell"
            assert trade.quantity == 100
            assert trade.price == 100.0
            assert trade.commission == 10.0
            assert trade.id is not None


class TestWatchlistItemModel:
    """WatchlistItemモデルのテスト"""

    def test_add_and_retrieve_watchlist_item(self, test_db_manager):
        """ウォッチリストアイテムの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="W001", name="ウォッチ銘柄")
            session.add(stock)
            session.flush()

            item = WatchlistItem(stock_code="W001", group_name="MyGroup", memo="テスト")
            session.add(item)

        with test_db_manager.session_scope() as session:
            retrieved_item = (
                session.query(WatchlistItem).filter_by(stock_code="W001").first()
            )
            assert retrieved_item is not None
            assert retrieved_item.group_name == "MyGroup"
            assert retrieved_item.memo == "テスト"
            assert retrieved_item.stock.name == "ウォッチ銘柄"


class TestAlertModel:
    """Alertモデルのテスト"""

    def test_add_and_retrieve_alert(self, test_db_manager):
        """アラートの追加と取得テスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="A001", name="アラート銘柄")
            session.add(stock)
            session.flush()

            alert = Alert(
                stock_code="A001",
                alert_type="price_above",
                threshold=150.0,
                is_active=True,
            )
            session.add(alert)

        with test_db_manager.session_scope() as session:
            retrieved_alert = (
                session.query(Alert).filter_by(stock_code="A001").first()
            )
            assert retrieved_alert is not None
            assert retrieved_alert.alert_type == "price_above"
            assert retrieved_alert.threshold == 150.0
            assert retrieved_alert.is_active is True
            assert retrieved_alert.stock.name == "アラート銘柄"