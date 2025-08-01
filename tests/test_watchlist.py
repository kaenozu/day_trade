"""
ウォッチリスト機能のテスト
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.day_trade.core.watchlist import (
    AlertCondition,
    AlertNotification,
    AlertType,
    WatchlistManager,
)


class TestWatchlistManager:
    """WatchlistManagerクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        # データベースをモック
        self.mock_db_manager = Mock()
        self.mock_session = Mock()
        self.mock_db_manager.session_scope.return_value.__enter__ = Mock(
            return_value=self.mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = Mock(
            return_value=None
        )

        # StockFetcherをモック
        self.mock_fetcher = Mock()

        with patch("src.day_trade.core.watchlist.db_manager", self.mock_db_manager):
            with patch(
                "src.day_trade.core.watchlist.StockFetcher",
                return_value=self.mock_fetcher,
            ):
                self.manager = WatchlistManager()

    def test_add_stock_success(self):
        """銘柄追加成功テスト"""
        # 既存アイテムが存在しない場合を設定
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        # 銘柄マスタに存在する場合を設定
        mock_stock = Mock()
        mock_stock.code = "7203"
        self.mock_session.query.return_value.filter.return_value.first.side_effect = [
            None,
            mock_stock,
        ]

        result = self.manager.add_stock("7203", "テスト", "テストメモ")

        assert result is True
        self.mock_session.add.assert_called_once()

    def test_add_stock_duplicate(self):
        """重複銘柄追加テスト"""
        # 既存アイテムが存在する場合を設定
        mock_existing = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_existing
        )

        result = self.manager.add_stock("7203", "テスト", "テストメモ")

        assert result is False
        self.mock_session.add.assert_not_called()

    def test_add_stock_with_company_info_fetch(self):
        """企業情報取得付き銘柄追加テスト"""
        # 既存アイテムなし、銘柄マスタにもなし
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        # 企業情報を設定
        company_info = {
            "name": "トヨタ自動車",
            "sector": "輸送用機器",
            "industry": "自動車",
        }
        self.mock_fetcher.get_company_info.return_value = company_info

        result = self.manager.add_stock("7203")

        assert result is True
        # 銘柄マスタとウォッチリストの両方に追加されることを確認
        assert self.mock_session.add.call_count == 2
        self.mock_fetcher.get_company_info.assert_called_once_with("7203")

    def test_remove_stock_success(self):
        """銘柄削除成功テスト"""
        mock_item = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_item
        )

        result = self.manager.remove_stock("7203", "default")

        assert result is True
        self.mock_session.delete.assert_called_once_with(mock_item)

    def test_remove_stock_not_found(self):
        """存在しない銘柄削除テスト"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        result = self.manager.remove_stock("7203", "default")

        assert result is False
        self.mock_session.delete.assert_not_called()

    def test_get_watchlist(self):
        """ウォッチリスト取得テスト"""
        # モックアイテムを作成
        mock_item1 = Mock()
        mock_item1.stock_code = "7203"
        mock_item1.group_name = "default"
        mock_item1.memo = "テストメモ"
        mock_item1.created_at = datetime.now()
        mock_item1.stock.name = "トヨタ自動車"

        mock_item2 = Mock()
        mock_item2.stock_code = "8306"
        mock_item2.group_name = "default"
        mock_item2.memo = ""
        mock_item2.created_at = datetime.now()
        mock_item2.stock.name = "三菱UFJ銀行"

        self.mock_session.query.return_value.join.return_value.all.return_value = [
            mock_item1,
            mock_item2,
        ]

        result = self.manager.get_watchlist()

        assert len(result) == 2
        assert result[0]["stock_code"] == "7203"
        assert result[0]["stock_name"] == "トヨタ自動車"
        assert result[1]["stock_code"] == "8306"
        assert result[1]["stock_name"] == "三菱UFJ銀行"

    def test_get_watchlist_by_group(self):
        """グループ別ウォッチリスト取得テスト"""
        mock_item = Mock()
        mock_item.stock_code = "7203"
        mock_item.group_name = "favorites"
        mock_item.memo = "お気に入り"
        mock_item.created_at = datetime.now()
        mock_item.stock.name = "トヨタ自動車"

        self.mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = [
            mock_item
        ]

        result = self.manager.get_watchlist("favorites")

        assert len(result) == 1
        assert result[0]["group_name"] == "favorites"
        # filterが呼ばれることを確認
        self.mock_session.query.return_value.join.return_value.filter.assert_called_once()

    def test_get_groups(self):
        """グループ一覧取得テスト"""
        mock_groups = [("default",), ("favorites",), ("tech",)]
        self.mock_session.query.return_value.distinct.return_value.all.return_value = (
            mock_groups
        )

        result = self.manager.get_groups()

        assert result == ["default", "favorites", "tech"]

    @patch("src.day_trade.core.watchlist.StockFetcher")
    def test_get_watchlist_with_prices(self, mock_fetcher_class):
        """価格情報付きウォッチリスト取得テスト"""
        # ウォッチリストのモックデータ
        watchlist_data = [
            {
                "stock_code": "7203",
                "stock_name": "トヨタ自動車",
                "group_name": "default",
                "memo": "テスト",
                "added_date": datetime.now(),
            }
        ]

        # 価格データのモック
        price_data = {
            "7203": {
                "current_price": 2500,
                "change": 50,
                "change_percent": 2.0,
                "volume": 1000000,
            }
        }

        # モックの設定
        self.manager.get_watchlist = Mock(return_value=watchlist_data)
        self.mock_fetcher.get_realtime_data.return_value = price_data

        result = self.manager.get_watchlist_with_prices()

        assert "7203" in result
        assert result["7203"]["current_price"] == 2500
        assert result["7203"]["stock_name"] == "トヨタ自動車"
        assert result["7203"]["change_percent"] == 2.0

    def test_update_memo_success(self):
        """メモ更新成功テスト"""
        mock_item = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_item
        )

        result = self.manager.update_memo("7203", "default", "新しいメモ")

        assert result is True
        assert mock_item.memo == "新しいメモ"

    def test_update_memo_not_found(self):
        """存在しない項目のメモ更新テスト"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        result = self.manager.update_memo("7203", "default", "新しいメモ")

        assert result is False

    def test_move_to_group_success(self):
        """グループ移動成功テスト"""
        mock_item = Mock()
        mock_item.group_name = "default"

        # 移動元アイテムは存在、移動先には存在しない
        self.mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_item,
            None,
        ]

        result = self.manager.move_to_group("7203", "default", "favorites")

        assert result is True
        assert mock_item.group_name == "favorites"

    def test_move_to_group_target_exists(self):
        """移動先に既存項目があるグループ移動テスト"""
        mock_item = Mock()
        mock_existing = Mock()

        # 移動元、移動先両方に存在
        self.mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_item,
            mock_existing,
        ]

        result = self.manager.move_to_group("7203", "default", "favorites")

        assert result is False


class TestAlertFunctionality:
    """アラート機能のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_db_manager = Mock()
        self.mock_session = Mock()
        self.mock_db_manager.session_scope.return_value.__enter__ = Mock(
            return_value=self.mock_session
        )
        self.mock_db_manager.session_scope.return_value.__exit__ = Mock(
            return_value=None
        )

        self.mock_fetcher = Mock()

        with patch("src.day_trade.core.watchlist.db_manager", self.mock_db_manager):
            with patch(
                "src.day_trade.core.watchlist.StockFetcher",
                return_value=self.mock_fetcher,
            ):
                self.manager = WatchlistManager()

    def test_add_alert_success(self):
        """アラート追加成功テスト"""
        # 既存アラートなし
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        condition = AlertCondition(
            stock_code="7203",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            memo="価格上昇アラート",
        )

        result = self.manager.add_alert(condition)

        assert result is True
        self.mock_session.add.assert_called_once()

    def test_add_alert_duplicate(self):
        """重複アラート追加テスト"""
        # 既存アラートあり
        mock_existing = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_existing
        )

        condition = AlertCondition(
            stock_code="7203", alert_type=AlertType.PRICE_ABOVE, threshold=3000.0
        )

        result = self.manager.add_alert(condition)

        assert result is False
        self.mock_session.add.assert_not_called()

    def test_remove_alert_success(self):
        """アラート削除成功テスト"""
        mock_alert = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_alert
        )

        result = self.manager.remove_alert("7203", AlertType.PRICE_ABOVE, 3000.0)

        assert result is True
        self.mock_session.delete.assert_called_once_with(mock_alert)

    def test_remove_alert_not_found(self):
        """存在しないアラート削除テスト"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        result = self.manager.remove_alert("7203", AlertType.PRICE_ABOVE, 3000.0)

        assert result is False
        self.mock_session.delete.assert_not_called()

    def test_get_alerts(self):
        """アラート一覧取得テスト"""
        mock_alert1 = Mock()
        mock_alert1.id = 1
        mock_alert1.stock_code = "7203"
        mock_alert1.alert_type = "price_above"
        mock_alert1.threshold = 3000.0
        mock_alert1.is_active = True
        mock_alert1.last_triggered = None
        mock_alert1.memo = "価格上昇"
        mock_alert1.created_at = datetime.now()
        mock_alert1.stock.name = "トヨタ自動車"

        mock_alert2 = Mock()
        mock_alert2.id = 2
        mock_alert2.stock_code = "8306"
        mock_alert2.alert_type = "price_below"
        mock_alert2.threshold = 700.0
        mock_alert2.is_active = True
        mock_alert2.last_triggered = None
        mock_alert2.memo = "価格下落"
        mock_alert2.created_at = datetime.now()
        mock_alert2.stock.name = "三菱UFJ銀行"

        self.mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = [
            mock_alert1,
            mock_alert2,
        ]

        result = self.manager.get_alerts()

        assert len(result) == 2
        assert result[0]["stock_code"] == "7203"
        assert result[0]["alert_type"] == "price_above"
        assert result[0]["threshold"] == 3000.0
        assert result[1]["stock_code"] == "8306"
        assert result[1]["alert_type"] == "price_below"

    def test_toggle_alert_success(self):
        """アラート切り替え成功テスト"""
        mock_alert = Mock()
        mock_alert.is_active = True
        mock_alert.stock_code = "7203"

        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_alert
        )

        result = self.manager.toggle_alert(1)

        assert result is True
        assert mock_alert.is_active is False

    def test_toggle_alert_not_found(self):
        """存在しないアラート切り替えテスト"""
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            None
        )

        result = self.manager.toggle_alert(999)

        assert result is False

    def test_check_alerts_price_above_triggered(self):
        """価格上昇アラートトリガーテスト"""
        # アクティブなアラートを設定
        alert_data = [
            {
                "id": 1,
                "stock_code": "7203",
                "stock_name": "トヨタ自動車",
                "alert_type": "price_above",
                "threshold": 2500.0,
                "is_active": True,
                "last_triggered": None,
                "memo": "テストアラート",
                "created_at": datetime.now(),
            }
        ]

        # 価格データを設定（閾値を超える）
        price_data = {
            "7203": {
                "current_price": 2600,  # 閾値2500を超える
                "change": 100,
                "change_percent": 4.0,
                "volume": 1000000,
            }
        }

        self.manager.get_alerts = Mock(return_value=alert_data)
        self.mock_fetcher.get_realtime_data.return_value = price_data

        # データベースアラートのモック
        mock_db_alert = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_db_alert
        )

        notifications = self.manager.check_alerts()

        assert len(notifications) == 1
        assert notifications[0].stock_code == "7203"
        assert notifications[0].alert_type == AlertType.PRICE_ABOVE
        assert notifications[0].current_value == 2600
        assert notifications[0].threshold == 2500.0

        # トリガー日時が更新されることを確認
        assert mock_db_alert.last_triggered is not None

    def test_check_alerts_price_below_not_triggered(self):
        """価格下落アラート非トリガーテスト"""
        alert_data = [
            {
                "id": 1,
                "stock_code": "7203",
                "stock_name": "トヨタ自動車",
                "alert_type": "price_below",
                "threshold": 2000.0,
                "is_active": True,
                "last_triggered": None,
                "memo": "テストアラート",
                "created_at": datetime.now(),
            }
        ]

        # 価格データを設定（閾値を下回らない）
        price_data = {
            "7203": {
                "current_price": 2500,  # 閾値2000を下回らない
                "change": 50,
                "change_percent": 2.0,
                "volume": 1000000,
            }
        }

        self.manager.get_alerts = Mock(return_value=alert_data)
        self.mock_fetcher.get_realtime_data.return_value = price_data

        notifications = self.manager.check_alerts()

        assert len(notifications) == 0

    def test_check_alerts_change_percent_up_triggered(self):
        """変化率上昇アラートトリガーテスト"""
        alert_data = [
            {
                "id": 1,
                "stock_code": "7203",
                "stock_name": "トヨタ自動車",
                "alert_type": "change_percent_up",
                "threshold": 3.0,
                "is_active": True,
                "last_triggered": None,
                "memo": "変化率アラート",
                "created_at": datetime.now(),
            }
        ]

        price_data = {
            "7203": {
                "current_price": 2600,
                "change": 100,
                "change_percent": 4.0,  # 閾値3.0を超える
                "volume": 1000000,
            }
        }

        self.manager.get_alerts = Mock(return_value=alert_data)
        self.mock_fetcher.get_realtime_data.return_value = price_data

        mock_db_alert = Mock()
        self.mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_db_alert
        )

        notifications = self.manager.check_alerts()

        assert len(notifications) == 1
        assert notifications[0].alert_type == AlertType.CHANGE_PERCENT_UP
        assert notifications[0].current_value == 4.0

    def test_get_watchlist_summary(self):
        """ウォッチリストサマリー取得テスト"""
        # グループとウォッチリストのモック
        self.manager.get_groups = Mock(return_value=["default", "favorites"])
        self.manager.get_watchlist = Mock(
            side_effect=[
                [{"stock_code": "7203"}, {"stock_code": "8306"}],  # defaultグループ
                [{"stock_code": "9984"}],  # favoritesグループ
            ]
        )
        self.manager.get_alerts = Mock(
            return_value=[
                {"id": 1},
                {"id": 2},
                {"id": 3},  # 3つのアクティブアラート
            ]
        )

        result = self.manager.get_watchlist_summary()

        assert result["total_groups"] == 2
        assert result["total_stocks"] == 3  # 2 + 1
        assert result["total_alerts"] == 3
        assert result["groups"] == ["default", "favorites"]

    @patch("pandas.DataFrame.to_csv")
    def test_export_watchlist_to_csv_success(self, mock_to_csv):
        """ウォッチリストCSVエクスポート成功テスト"""
        watchlist_data = {
            "7203": {
                "stock_name": "トヨタ自動車",
                "group_name": "default",
                "current_price": 2500,
                "change": 50,
                "change_percent": 2.0,
                "volume": 1000000,
                "memo": "テスト",
                "added_date": datetime.now(),
            }
        }

        self.manager.get_watchlist_with_prices = Mock(return_value=watchlist_data)

        result = self.manager.export_watchlist_to_csv("test.csv")

        assert result is True
        mock_to_csv.assert_called_once()

    def test_export_watchlist_to_csv_no_data(self):
        """データなしCSVエクスポートテスト"""
        self.manager.get_watchlist_with_prices = Mock(return_value={})

        result = self.manager.export_watchlist_to_csv("test.csv")

        assert result is False


class TestAlertType:
    """AlertTypeの基本テスト"""

    def test_alert_type_values(self):
        """AlertTypeの値テスト"""
        assert AlertType.PRICE_ABOVE.value == "price_above"
        assert AlertType.PRICE_BELOW.value == "price_below"
        assert AlertType.CHANGE_PERCENT_UP.value == "change_percent_up"
        assert AlertType.CHANGE_PERCENT_DOWN.value == "change_percent_down"
        assert AlertType.VOLUME_SPIKE.value == "volume_spike"


class TestAlertCondition:
    """AlertConditionの基本テスト"""

    def test_alert_condition_creation(self):
        """AlertCondition作成テスト"""
        condition = AlertCondition(
            stock_code="7203",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            memo="テストアラート",
        )

        assert condition.stock_code == "7203"
        assert condition.alert_type == AlertType.PRICE_ABOVE
        assert condition.threshold == 3000.0
        assert condition.memo == "テストアラート"

    def test_alert_condition_default_memo(self):
        """AlertConditionデフォルトメモテスト"""
        condition = AlertCondition(
            stock_code="7203", alert_type=AlertType.PRICE_ABOVE, threshold=3000.0
        )

        assert condition.memo == ""


class TestAlertNotification:
    """AlertNotificationの基本テスト"""

    def test_alert_notification_creation(self):
        """AlertNotification作成テスト"""
        notification = AlertNotification(
            stock_code="7203",
            stock_name="トヨタ自動車",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            current_value=3100.0,
            triggered_at=datetime.now(),
            memo="価格上昇アラート",
        )

        assert notification.stock_code == "7203"
        assert notification.stock_name == "トヨタ自動車"
        assert notification.alert_type == AlertType.PRICE_ABOVE
        assert notification.threshold == 3000.0
        assert notification.current_value == 3100.0
        assert notification.memo == "価格上昇アラート"


if __name__ == "__main__":
    pytest.main([__file__])
