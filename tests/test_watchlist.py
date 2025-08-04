"""
ウォッチリスト機能のテスト
"""

from unittest.mock import Mock

import pytest

from src.day_trade.core.watchlist import (
    AlertCondition,
    WatchlistManager,
)
from src.day_trade.models.database import DatabaseConfig, DatabaseManager
from src.day_trade.models.enums import AlertType
from src.day_trade.models.stock import Alert, Stock, WatchlistItem


@pytest.fixture
def test_db():
    """テスト用データベース"""
    config = DatabaseConfig.for_testing()
    db = DatabaseManager(config)
    # 全モデルを明示的にインポートしてテーブル作成
    db.create_tables()
    yield db
    db.drop_tables()


@pytest.fixture
def watchlist_manager(test_db):
    """ウォッチリストマネージャー"""
    # データベース設定を一時的に変更
    from src.day_trade.core import watchlist
    from src.day_trade.models import database

    original_db = database.db_manager
    database.db_manager = test_db
    # watchlistモジュールのdb_managerも変更
    original_watchlist_db = watchlist.db_manager
    watchlist.db_manager = test_db

    # モックフェッチャー
    mock_fetcher = Mock()
    mock_fetcher.get_company_info.return_value = {
        "name": "Test Company",
        "sector": "Test Sector",
        "industry": "Test Industry",
    }

    manager = WatchlistManager()
    manager.fetcher = mock_fetcher

    yield manager

    # 元の設定を復元
    database.db_manager = original_db
    watchlist.db_manager = original_watchlist_db


class TestWatchlistManager:
    """WatchlistManagerクラスのテスト"""

    def test_add_stock_success(self, watchlist_manager, test_db):
        """銘柄追加成功テスト"""
        # 銘柄をデータベースに追加
        with test_db.session_scope() as session:
            stock = Stock(
                code="7203", name="トヨタ自動車", sector="輸送用機器", market="プライム"
            )
            session.add(stock)

        result = watchlist_manager.add_stock("7203", "default", "テストメモ")
        assert result is True

    def test_add_stock_with_company_info_fetch(self, watchlist_manager):
        """企業情報取得付き銘柄追加テスト"""
        result = watchlist_manager.add_stock("9999", "default", "新規銘柄")
        assert result is True

    def test_remove_stock_success(self, watchlist_manager, test_db):
        """銘柄削除成功テスト"""
        # 事前に銘柄を追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            watchlist_item = WatchlistItem(stock_code="7203", group_name="default")
            session.add(stock)
            session.add(watchlist_item)

        result = watchlist_manager.remove_stock("7203", "default")
        assert result is True

    def test_get_watchlist(self, watchlist_manager, test_db):
        """ウォッチリスト取得テスト"""
        # 事前にデータを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            watchlist_item = WatchlistItem(
                stock_code="7203", group_name="default", memo="テスト"
            )
            session.add(stock)
            session.add(watchlist_item)

        result = watchlist_manager.get_watchlist()
        assert len(result) >= 1
        assert result[0]["stock_code"] == "7203"

    def test_get_watchlist_by_group(self, watchlist_manager, test_db):
        """グループ別ウォッチリスト取得テスト"""
        # 異なるグループのデータを追加
        with test_db.session_scope() as session:
            stock1 = Stock(code="7203", name="トヨタ自動車")
            stock2 = Stock(code="8306", name="三菱UFJ銀行")
            item1 = WatchlistItem(stock_code="7203", group_name="auto")
            item2 = WatchlistItem(stock_code="8306", group_name="bank")
            session.add_all([stock1, stock2, item1, item2])

        auto_result = watchlist_manager.get_watchlist("auto")
        bank_result = watchlist_manager.get_watchlist("bank")

        assert len(auto_result) == 1
        assert len(bank_result) == 1
        assert auto_result[0]["stock_code"] == "7203"
        assert bank_result[0]["stock_code"] == "8306"

    def test_get_groups(self, watchlist_manager, test_db):
        """グループ一覧取得テスト"""
        # 複数グループのデータを追加
        with test_db.session_scope() as session:
            stock1 = Stock(code="7203", name="トヨタ自動車")
            stock2 = Stock(code="8306", name="三菱UFJ銀行")
            item1 = WatchlistItem(stock_code="7203", group_name="auto")
            item2 = WatchlistItem(stock_code="8306", group_name="bank")
            session.add_all([stock1, stock2, item1, item2])

        groups = watchlist_manager.get_groups()
        assert len(groups) == 2
        assert "auto" in groups
        assert "bank" in groups

    def test_update_memo_success(self, watchlist_manager, test_db):
        """メモ更新成功テスト"""
        # 事前にデータを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            watchlist_item = WatchlistItem(
                stock_code="7203", group_name="default", memo="旧メモ"
            )
            session.add(stock)
            session.add(watchlist_item)

        result = watchlist_manager.update_memo("7203", "default", "新メモ")
        assert result is True

    def test_move_to_group_success(self, watchlist_manager, test_db):
        """グループ移動成功テスト"""
        # 事前にデータを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            watchlist_item = WatchlistItem(stock_code="7203", group_name="old_group")
            session.add(stock)
            session.add(watchlist_item)

        result = watchlist_manager.move_to_group("7203", "old_group", "new_group")
        assert result is True


class TestAlertFunctionality:
    """アラート機能のテスト"""

    def test_add_alert_success(self, watchlist_manager, test_db):
        """アラート追加成功テスト"""
        # 事前に銘柄を追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            session.add(stock)

        condition = AlertCondition(
            stock_code="7203",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            memo="高値警戒",
        )

        result = watchlist_manager.add_alert(condition)
        assert result is True

    def test_remove_alert_success(self, watchlist_manager, test_db):
        """アラート削除成功テスト"""
        # 事前にアラートを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            alert = Alert(
                stock_code="7203",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=3000.0,
                is_active=True,
            )
            session.add(stock)
            session.add(alert)

        result = watchlist_manager.remove_alert("7203", AlertType.PRICE_ABOVE, 3000.0)
        assert result is True

    def test_get_alerts(self, watchlist_manager, test_db):
        """アラート取得テスト"""
        # 事前にアラートを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            alert1 = Alert(
                stock_code="7203",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=3000.0,
                is_active=True,
            )
            alert2 = Alert(
                stock_code="7203",
                alert_type="price_below",
                threshold=2000.0,
                is_active=False,
            )
            session.add_all([stock, alert1, alert2])

        # アクティブなアラートのみ取得
        active_alerts = watchlist_manager.get_alerts(active_only=True)
        assert len(active_alerts) == 1

        # 全アラート取得
        all_alerts = watchlist_manager.get_alerts(active_only=False)
        assert len(all_alerts) == 2

    def test_toggle_alert_success(self, watchlist_manager, test_db):
        """アラート切り替え成功テスト"""
        # 事前にアラートを追加
        with test_db.session_scope() as session:
            stock = Stock(code="7203", name="トヨタ自動車")
            alert = Alert(
                stock_code="7203",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=3000.0,
                is_active=True,
            )
            session.add(stock)
            session.add(alert)
            session.flush()
            alert_id = alert.id

        result = watchlist_manager.toggle_alert(alert_id)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
