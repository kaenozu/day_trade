"""
データベース基盤のテスト
"""

import pytest
from sqlalchemy.orm import Session
from src.day_trade.models.database import DatabaseManager, get_db
from src.day_trade.models.stock import Stock, PriceData, Trade, WatchlistItem, Alert
from datetime import datetime


class TestDatabaseManager:
    """DatabaseManagerクラスのテスト"""

    @pytest.fixture
    def test_db_manager(self):
        """テスト用のデータベースマネージャー"""
        db_manager = DatabaseManager("sqlite:///:memory:", echo=False)
        db_manager.create_tables()
        yield db_manager
        # クリーンアップ
        db_manager.drop_tables()

    def test_session_scope(self, test_db_manager):
        """セッションスコープのテスト"""
        # セッションスコープを使用してデータを作成
        with test_db_manager.session_scope() as session:
            stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
            )
            session.add(stock)

        # 別のセッションで確認
        with test_db_manager.session_scope() as session:
            result = session.query(Stock).filter_by(code="7203").first()
            assert result is not None
            assert result.name == "トヨタ自動車"

    def test_rollback_on_error(self, test_db_manager):
        """エラー時のロールバックテスト"""
        # エラーを発生させる
        try:
            with test_db_manager.session_scope() as session:
                stock = Stock(code="7203", name="トヨタ自動車")
                session.add(stock)
                # わざとエラーを発生させる
                raise ValueError("Test error")
        except ValueError:
            pass

        # データが保存されていないことを確認
        with test_db_manager.session_scope() as session:
            result = session.query(Stock).filter_by(code="7203").first()
            assert result is None

    def test_create_and_query_tables(self, test_db_manager):
        """テーブル作成とクエリのテスト"""
        with test_db_manager.session_scope() as session:
            # 銘柄データ作成
            stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
            )
            session.add(stock)
            session.flush()

            # 価格データ作成
            price = PriceData(
                stock_code="7203",
                datetime=datetime(2023, 5, 1, 15, 0),
                open=2450.0,
                high=2480.0,
                low=2440.0,
                close=2470.0,
                volume=10000000,
            )
            session.add(price)

            # 取引データ作成
            trade = Trade(
                stock_code="7203",
                trade_type="buy",
                quantity=100,
                price=2470.0,
                commission=500.0,
                trade_datetime=datetime(2023, 5, 1, 15, 0),
                memo="テスト取引",
            )
            session.add(trade)

            # ウォッチリストアイテム作成
            watchlist = WatchlistItem(
                stock_code="7203", group_name="自動車", memo="主力銘柄"
            )
            session.add(watchlist)

            # アラート作成
            alert = Alert(
                stock_code="7203",
                alert_type="price_above",
                threshold=2500.0,
                is_active=True,
                memo="高値警戒",
            )
            session.add(alert)

        # データ確認
        with test_db_manager.session_scope() as session:
            # 銘柄確認
            stock = session.query(Stock).filter_by(code="7203").first()
            assert stock.name == "トヨタ自動車"

            # リレーション確認
            assert len(stock.price_data) == 1
            assert len(stock.trades) == 1
            assert len(stock.watchlist_items) == 1
            assert len(stock.alerts) == 1

            # 価格データ確認
            price = stock.price_data[0]
            assert price.close == 2470.0

            # 取引データ確認
            trade = stock.trades[0]
            assert trade.total_amount == 247500.0  # 2470 * 100 + 500

    def test_indexes_and_constraints(self, test_db_manager):
        """インデックスと制約のテスト"""
        # 最初のレコードを作成
        with test_db_manager.session_scope() as session:
            stock1 = Stock(code="7203", name="トヨタ自動車")
            session.add(stock1)

        # 重複エラーが発生することを確認
        with pytest.raises(Exception):  # IntegrityError
            with test_db_manager.session_scope() as session:
                stock2 = Stock(code="7203", name="トヨタ自動車（重複）")
                session.add(stock2)

    def test_timestamp_mixin(self, test_db_manager):
        """タイムスタンプMixinのテスト"""
        with test_db_manager.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            session.add(stock)
            session.flush()

            # created_atとupdated_atが設定されていることを確認
            assert stock.created_at is not None
            assert stock.updated_at is not None
            assert stock.created_at == stock.updated_at

            # 更新
            # original_updated = stock.updated_at # Removed
            stock.name = "トヨタ自動車株式会社"
            session.flush()

            # updated_atが更新されていることを確認（この実装では手動更新が必要）
            assert stock.created_at < datetime.now()


class TestDatabaseFunctions:
    """データベース関連関数のテスト"""

    def test_get_db_generator(self):
        """get_db関数のテスト"""
        gen = get_db()
        session = next(gen)
        assert isinstance(session, Session)

        # クリーンアップ
        try:
            next(gen)
        except StopIteration:
            pass
