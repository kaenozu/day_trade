"""
アラート機能のテスト
"""

import time
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest

from src.day_trade.core.alerts import (
    AlertCondition,
    AlertManager,
    AlertPriority,
    AlertTrigger,
    AlertType,
    NotificationHandler,
    NotificationMethod,
    create_change_alert,
    create_price_alert,
)


class TestAlertCondition:
    """AlertConditionクラスのテスト"""

    def test_alert_condition_creation(self):
        """アラート条件作成テスト"""
        condition = AlertCondition(
            alert_id="test_alert",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("3000"),
            comparison_operator=">",
            priority=AlertPriority.HIGH,
        )

        assert condition.alert_id == "test_alert"
        assert condition.symbol == "7203"
        assert condition.alert_type == AlertType.PRICE_ABOVE
        assert condition.condition_value == Decimal("3000")
        assert condition.comparison_operator == ">"
        assert condition.priority == AlertPriority.HIGH
        assert condition.is_active is True
        assert condition.cooldown_minutes == 60

    def test_alert_condition_custom_parameters(self):
        """カスタムパラメータテスト"""
        condition = AlertCondition(
            alert_id="custom_test",
            symbol="9984",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            custom_parameters={"param1": 100, "param2": "test"},
        )

        assert condition.custom_parameters == {"param1": 100, "param2": "test"}


class TestAlertTrigger:
    """AlertTriggerクラスのテスト"""

    def test_alert_trigger_creation(self):
        """アラート発火記録作成テスト"""
        trigger = AlertTrigger(
            alert_id="test_trigger",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("3100"),
            condition_value=Decimal("3000"),
            message="価格が3000円を上回りました",
            priority=AlertPriority.HIGH,
            current_price=Decimal("3100"),
            volume=1500000,
            change_percent=2.5,
        )

        assert trigger.alert_id == "test_trigger"
        assert trigger.symbol == "7203"
        assert trigger.alert_type == AlertType.PRICE_ABOVE
        assert trigger.current_value == Decimal("3100")
        assert trigger.condition_value == Decimal("3000")
        assert trigger.current_price == Decimal("3100")
        assert trigger.volume == 1500000
        assert trigger.change_percent == 2.5

    def test_alert_trigger_to_dict(self):
        """アラート発火記録の辞書変換テスト"""
        trigger = AlertTrigger(
            alert_id="dict_test",
            symbol="8306",
            trigger_time=datetime(2023, 6, 1, 10, 30, 0),
            alert_type=AlertType.PRICE_BELOW,
            current_value=Decimal("800"),
            condition_value=Decimal("850"),
            message="価格下落",
            priority=AlertPriority.MEDIUM,
        )

        trigger_dict = trigger.to_dict()

        assert trigger_dict["alert_id"] == "dict_test"
        assert trigger_dict["symbol"] == "8306"
        assert trigger_dict["trigger_time"] == "2023-06-01T10:30:00"
        assert trigger_dict["alert_type"] == "price_below"
        assert trigger_dict["current_value"] == "800"
        assert trigger_dict["priority"] == "medium"


class TestNotificationHandler:
    """NotificationHandlerクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.handler = NotificationHandler()
        self.sample_trigger = AlertTrigger(
            alert_id="notification_test",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="テスト通知",
            priority=AlertPriority.MEDIUM,
        )

    def test_console_notification(self, caplog):
        """コンソール通知テスト"""
        import logging
        caplog.set_level(logging.WARNING)

        self.handler._send_console_notification(self.sample_trigger)

        # ログ出力の確認
        assert len(caplog.records) > 0
        log_record = caplog.records[0]
        assert log_record.levelname == "WARNING"
        assert "Alert triggered - Console notification" in log_record.getMessage()

    def test_file_log_notification(self, tmp_path):
        """ファイルログ通知テスト"""
        import json
        import os

        # 一時ディレクトリに移動
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            self.handler._send_file_log_notification(self.sample_trigger)

            # ログファイルが作成されることを確認
            log_files = [
                f
                for f in os.listdir(".")
                if f.startswith("alerts_") and f.endswith(".log")
            ]
            assert len(log_files) == 1

            # ログ内容を確認
            with open(log_files[0], encoding="utf-8") as f:
                log_content = f.read().strip()
                log_data = json.loads(log_content)

            assert "timestamp" in log_data
            assert "alert" in log_data
            assert log_data["alert"]["alert_id"] == "notification_test"

        finally:
            os.chdir(original_cwd)

    def test_email_configuration(self):
        """メール設定テスト"""
        self.handler.configure_email(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="test@test.com",
            password="password",
            from_email="sender@test.com",
            to_emails=["recipient@test.com"],
        )

        assert self.handler.email_config["smtp_server"] == "smtp.test.com"
        assert self.handler.email_config["smtp_port"] == 587
        assert self.handler.email_config["username"] == "test@test.com"
        assert self.handler.email_config["to_emails"] == ["recipient@test.com"]
        assert NotificationMethod.EMAIL in self.handler.handlers

    def test_custom_handler(self):
        """カスタムハンドラーテスト"""
        custom_called = []

        def custom_handler(trigger):
            custom_called.append(trigger.alert_id)

        self.handler.add_custom_handler(NotificationMethod.WEBHOOK, custom_handler)
        self.handler.send_notification(
            self.sample_trigger, [NotificationMethod.WEBHOOK]
        )

        assert len(custom_called) == 1
        assert custom_called[0] == "notification_test"


class TestAlertManager:
    """AlertManagerクラスのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_stock_fetcher = Mock()
        self.mock_watchlist_manager = Mock()
        self.alert_manager = AlertManager(
            stock_fetcher=self.mock_stock_fetcher,
            watchlist_manager=self.mock_watchlist_manager,
        )

        # サンプルアラート条件
        self.sample_condition = AlertCondition(
            alert_id="test_condition",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("3000"),
            comparison_operator=">",
            priority=AlertPriority.MEDIUM,
        )

    def test_add_alert(self):
        """アラート追加テスト"""
        result = self.alert_manager.add_alert(self.sample_condition)

        assert result is True
        assert "test_condition" in self.alert_manager.alert_conditions
        assert (
            self.alert_manager.alert_conditions["test_condition"]
            == self.sample_condition
        )

    def test_remove_alert(self):
        """アラート削除テスト"""
        # まず追加
        self.alert_manager.add_alert(self.sample_condition)
        assert "test_condition" in self.alert_manager.alert_conditions

        # 削除
        result = self.alert_manager.remove_alert("test_condition")
        assert result is True
        assert "test_condition" not in self.alert_manager.alert_conditions

        # 存在しないアラートの削除
        result = self.alert_manager.remove_alert("non_existent")
        assert result is False

    def test_get_alerts(self):
        """アラート取得テスト"""
        # 複数のアラートを追加
        condition1 = AlertCondition(
            alert_id="alert1",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
        )
        condition2 = AlertCondition(
            alert_id="alert2",
            symbol="9984",
            alert_type=AlertType.PRICE_BELOW,
            condition_value=1500,
        )
        condition3 = AlertCondition(
            alert_id="alert3",
            symbol="7203",
            alert_type=AlertType.CHANGE_PERCENT_UP,
            condition_value=5,
        )

        self.alert_manager.add_alert(condition1)
        self.alert_manager.add_alert(condition2)
        self.alert_manager.add_alert(condition3)

        # 全アラート取得
        all_alerts = self.alert_manager.get_alerts()
        assert len(all_alerts) == 3

        # 特定銘柄のアラート取得
        toyota_alerts = self.alert_manager.get_alerts(symbol="7203")
        assert len(toyota_alerts) == 2
        assert all(alert.symbol == "7203" for alert in toyota_alerts)

    def test_compare_values(self):
        """値比較テスト"""
        # 大なり
        assert self.alert_manager._compare_values(100, 90, ">") is True
        assert self.alert_manager._compare_values(80, 90, ">") is False

        # 小なり
        assert self.alert_manager._compare_values(80, 90, "<") is True
        assert self.alert_manager._compare_values(100, 90, "<") is False

        # 以上
        assert self.alert_manager._compare_values(90, 90, ">=") is True
        assert self.alert_manager._compare_values(100, 90, ">=") is True
        assert self.alert_manager._compare_values(80, 90, ">=") is False

        # 以下
        assert self.alert_manager._compare_values(90, 90, "<=") is True
        assert self.alert_manager._compare_values(80, 90, "<=") is True
        assert self.alert_manager._compare_values(100, 90, "<=") is False

        # 等しい（誤差考慮）
        assert self.alert_manager._compare_values(90.0001, 90, "==") is True
        assert self.alert_manager._compare_values(90.1, 90, "==") is False

    def test_is_expired(self):
        """有効期限チェックテスト"""
        # 期限なし
        condition_no_expiry = AlertCondition(
            alert_id="no_expiry",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
        )
        assert self.alert_manager._is_expired(condition_no_expiry) is False

        # 未来の期限
        future_expiry = AlertCondition(
            alert_id="future_expiry",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
            expiry_date=datetime.now() + timedelta(hours=1),
        )
        assert self.alert_manager._is_expired(future_expiry) is False

        # 過去の期限
        past_expiry = AlertCondition(
            alert_id="past_expiry",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
            expiry_date=datetime.now() - timedelta(hours=1),
        )
        assert self.alert_manager._is_expired(past_expiry) is True

    def test_validate_condition(self):
        """条件検証テスト"""
        # 有効な条件
        valid_condition = AlertCondition(
            alert_id="valid",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
        )
        assert self.alert_manager._validate_condition(valid_condition) is True

        # IDなし
        no_id = AlertCondition(
            alert_id="",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
        )
        assert self.alert_manager._validate_condition(no_id) is False

        # 銘柄なし
        no_symbol = AlertCondition(
            alert_id="test",
            symbol="",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
        )
        assert self.alert_manager._validate_condition(no_symbol) is False

        # カスタム条件で関数なし
        custom_no_func = AlertCondition(
            alert_id="custom",
            symbol="7203",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="test",
        )
        assert self.alert_manager._validate_condition(custom_no_func) is False

    def test_should_check_condition(self):
        """条件チェック判定テスト"""
        # アクティブな条件
        active_condition = AlertCondition(
            alert_id="active",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
            is_active=True,
        )
        assert self.alert_manager._should_check_condition(active_condition) is True

        # 非アクティブな条件
        inactive_condition = AlertCondition(
            alert_id="inactive",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
            is_active=False,
        )
        assert self.alert_manager._should_check_condition(inactive_condition) is False

        # クールダウン中
        cooldown_condition = AlertCondition(
            alert_id="cooldown",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=3000,
            cooldown_minutes=60,
        )

        # 最初はチェック可能
        assert self.alert_manager._should_check_condition(cooldown_condition) is True

        # トリガー時間を記録
        self.alert_manager.last_trigger_times["cooldown"] = datetime.now()

        # クールダウン中はチェック不可
        assert self.alert_manager._should_check_condition(cooldown_condition) is False

        # クールダウン時間経過後はチェック可能
        self.alert_manager.last_trigger_times["cooldown"] = datetime.now() - timedelta(
            hours=2
        )
        assert self.alert_manager._should_check_condition(cooldown_condition) is True

    def test_evaluate_price_condition(self):
        """価格条件評価テスト"""
        # 価格上昇アラート
        price_above_condition = AlertCondition(
            alert_id="price_above",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("3000"),
            comparison_operator=">",
        )

        # 条件を満たす場合
        trigger = self.alert_manager._evaluate_condition(
            price_above_condition, Decimal("3100"), 1500000, 2.5, None
        )

        assert trigger is not None
        assert trigger.alert_id == "price_above"
        assert trigger.symbol == "7203"
        assert trigger.current_value == Decimal("3100")
        assert trigger.condition_value == Decimal("3000")
        assert "上回りました" in trigger.message

        # 条件を満たさない場合
        trigger = self.alert_manager._evaluate_condition(
            price_above_condition, Decimal("2900"), 1500000, -1.5, None
        )

        assert trigger is None

    def test_evaluate_change_percent_condition(self):
        """変化率条件評価テスト"""
        # 上昇率アラート
        change_up_condition = AlertCondition(
            alert_id="change_up",
            symbol="7203",
            alert_type=AlertType.CHANGE_PERCENT_UP,
            condition_value=5.0,
        )

        # 条件を満たす場合
        trigger = self.alert_manager._evaluate_condition(
            change_up_condition, Decimal("2500"), 1500000, 6.5, None
        )

        assert trigger is not None
        assert trigger.current_value == 6.5
        assert "上昇率" in trigger.message

        # 条件を満たさない場合
        trigger = self.alert_manager._evaluate_condition(
            change_up_condition, Decimal("2500"), 1500000, 3.0, None
        )

        assert trigger is None

    def test_evaluate_custom_condition(self):
        """カスタム条件評価テスト"""

        # カスタム関数
        def custom_func(symbol, price, volume, change_pct, historical_data, params):
            min_price = params.get("min_price", 0)
            return float(price) >= min_price

        custom_condition = AlertCondition(
            alert_id="custom",
            symbol="7203",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            custom_function=custom_func,
            custom_parameters={"min_price": 2800},
        )

        # 条件を満たす場合
        trigger = self.alert_manager._evaluate_condition(
            custom_condition, Decimal("3000"), 1500000, 2.0, None
        )

        assert trigger is not None
        assert trigger.current_value == "Custom"
        assert "カスタム条件" in trigger.message

        # 条件を満たさない場合
        trigger = self.alert_manager._evaluate_condition(
            custom_condition, Decimal("2700"), 1500000, 2.0, None
        )

        assert trigger is None

    def test_alert_history(self):
        """アラート履歴テスト"""
        # 履歴の追加
        trigger1 = AlertTrigger(
            alert_id="hist1",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=3100,
            condition_value=3000,
            message="テスト1",
            priority=AlertPriority.MEDIUM,
        )

        trigger2 = AlertTrigger(
            alert_id="hist2",
            symbol="9984",
            trigger_time=datetime.now() - timedelta(hours=2),
            alert_type=AlertType.PRICE_BELOW,
            current_value=1400,
            condition_value=1500,
            message="テスト2",
            priority=AlertPriority.HIGH,
        )

        self.alert_manager.alert_history.extend([trigger1, trigger2])

        # 全履歴取得
        all_history = self.alert_manager.get_alert_history(hours=24)
        assert len(all_history) == 2

        # 特定銘柄の履歴取得
        toyota_history = self.alert_manager.get_alert_history(symbol="7203", hours=24)
        assert len(toyota_history) == 1
        assert toyota_history[0].symbol == "7203"

        # 時間制限
        recent_history = self.alert_manager.get_alert_history(hours=1)
        assert len(recent_history) == 1  # trigger1のみ

    def test_monitoring_start_stop(self):
        """監視開始・停止テスト"""
        assert self.alert_manager.monitoring_active is False
        assert self.alert_manager.monitoring_thread is None

        # 監視開始
        self.alert_manager.start_monitoring(interval_seconds=1)
        assert self.alert_manager.monitoring_active is True
        assert self.alert_manager.monitoring_thread is not None
        assert self.alert_manager.monitoring_thread.is_alive()

        # 少し待つ
        time.sleep(0.1)

        # 監視停止
        self.alert_manager.stop_monitoring()
        assert self.alert_manager.monitoring_active is False

        # スレッドが終了するまで待つ
        time.sleep(1.2)
        assert not self.alert_manager.monitoring_thread.is_alive()


class TestHelperFunctions:
    """ヘルパー関数のテスト"""

    def test_create_price_alert(self):
        """価格アラート作成ヘルパーテスト"""
        # 上昇アラート
        alert_above = create_price_alert(
            alert_id="price_up",
            symbol="7203",
            target_price=Decimal("3000"),
            above=True,
            priority=AlertPriority.HIGH,
        )

        assert alert_above.alert_id == "price_up"
        assert alert_above.symbol == "7203"
        assert alert_above.alert_type == AlertType.PRICE_ABOVE
        assert alert_above.condition_value == Decimal("3000")
        assert alert_above.comparison_operator == ">"
        assert alert_above.priority == AlertPriority.HIGH
        assert "上昇" in alert_above.description

        # 下落アラート
        alert_below = create_price_alert(
            alert_id="price_down",
            symbol="9984",
            target_price=Decimal("1400"),
            above=False,
            priority=AlertPriority.MEDIUM,
        )

        assert alert_below.alert_type == AlertType.PRICE_BELOW
        assert alert_below.comparison_operator == "<"
        assert "下落" in alert_below.description

    def test_create_change_alert(self):
        """変化率アラート作成ヘルパーテスト"""
        # 上昇率アラート
        alert_up = create_change_alert(
            alert_id="change_up",
            symbol="7203",
            change_percent=5.0,
            up=True,
            priority=AlertPriority.MEDIUM,
        )

        assert alert_up.alert_id == "change_up"
        assert alert_up.symbol == "7203"
        assert alert_up.alert_type == AlertType.CHANGE_PERCENT_UP
        assert alert_up.condition_value == 5.0
        assert "上昇" in alert_up.description

        # 下落率アラート
        alert_down = create_change_alert(
            alert_id="change_down",
            symbol="9984",
            change_percent=-3.0,
            up=False,
            priority=AlertPriority.HIGH,
        )

        assert alert_down.alert_type == AlertType.CHANGE_PERCENT_DOWN
        assert "下落" in alert_down.description


class TestIntegration:
    """統合テスト"""

    def test_full_alert_workflow(self):
        """完全なアラートワークフローテスト"""
        # モックを設定
        mock_stock_fetcher = Mock()
        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 3100,
            "volume": 2000000,
            "change_percent": 4.2,
        }

        # 空のDataFrameを返すモック
        mock_stock_fetcher.get_historical_data.return_value = pd.DataFrame()

        alert_manager = AlertManager(stock_fetcher=mock_stock_fetcher)

        # アラート条件を追加
        condition = AlertCondition(
            alert_id="integration_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("3000"),
            comparison_operator=">",
            priority=AlertPriority.HIGH,
            cooldown_minutes=0,  # テスト用にクールダウンなし
        )

        alert_manager.add_alert(condition)

        # 通知ハンドラーをモック
        notification_called = []

        def mock_notification(trigger):
            notification_called.append(trigger.alert_id)

        alert_manager.notification_handler.add_custom_handler(
            NotificationMethod.CALLBACK, mock_notification
        )
        alert_manager.configure_notifications([NotificationMethod.CALLBACK])

        # アラートチェック実行
        alert_manager.check_all_alerts()

        # 通知が送信されることを確認
        assert len(notification_called) == 1
        assert notification_called[0] == "integration_test"

        # 履歴に記録されることを確認
        history = alert_manager.get_alert_history()
        assert len(history) == 1
        assert history[0].alert_id == "integration_test"
        assert history[0].symbol == "7203"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
