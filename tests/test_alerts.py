"""
アラート機能のテスト
"""

import time
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

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
        self.alert_manager = AlertManager(
            stock_fetcher=self.mock_stock_fetcher,
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

        # スレッドが終了するまで待つ（短縮版）
        time.sleep(0.1)
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


class TestEmailNotification:
    """メール通知機能の詳細テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.handler = NotificationHandler()
        self.sample_trigger = AlertTrigger(
            alert_id="email_test",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="メールテスト通知",
            priority=AlertPriority.HIGH,
            current_price=Decimal("2600"),
            volume=1500000,
            change_percent=4.17,
        )

    def test_email_notification_unavailable(self, monkeypatch):
        """メール機能が利用できない場合のテスト"""
        # EMAIL_AVAILABLEをFalseに設定
        monkeypatch.setattr("src.day_trade.core.alerts.EMAIL_AVAILABLE", False)

        with patch("src.day_trade.core.alerts.logger") as mock_logger:
            self.handler._send_email_notification(self.sample_trigger)
            mock_logger.warning.assert_called_with("メール機能が利用できません")

    def test_email_notification_incomplete_config(self, monkeypatch):
        """メール設定が不完全な場合のテスト"""
        # EMAIL_AVAILABLEを有効にする
        monkeypatch.setattr("src.day_trade.core.alerts.EMAIL_AVAILABLE", True)

        # 不完全なメール設定
        self.handler.email_config = {
            "smtp_server": "",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "from_email": "",
            "to_emails": [],
        }

        with patch("src.day_trade.core.alerts.logger") as mock_logger:
            self.handler._send_email_notification(self.sample_trigger)
            mock_logger.warning.assert_called_with("メール設定が不完全です")

    def test_email_notification_success(self, monkeypatch):
        """メール通知成功のテスト"""
        # EMAIL_AVAILABLEを有効にする
        monkeypatch.setattr("src.day_trade.core.alerts.EMAIL_AVAILABLE", True)

        # 完全なメール設定
        self.handler.email_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "test@example.com",
            "to_emails": ["recipient@example.com"],
        }

        # SMTPライブラリをモック
        # smtplibモジュール全体をモック
        import smtplib
        from email.mime.multipart import MIMEMultipart as MimeMultipart
        from email.mime.text import MIMEText as MimeText

        monkeypatch.setattr("src.day_trade.core.alerts.smtplib", smtplib)
        monkeypatch.setattr("src.day_trade.core.alerts.MimeMultipart", MimeMultipart)
        monkeypatch.setattr("src.day_trade.core.alerts.MimeText", MimeText)

        with patch.object(smtplib, "SMTP") as mock_smtp, patch(
            "src.day_trade.core.alerts.logger"
        ) as mock_logger:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            self.handler._send_email_notification(self.sample_trigger)

            # SMTP接続が正しく設定されることを確認
            mock_smtp.assert_called_once_with("smtp.example.com", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@example.com", "password")
            mock_server.send_message.assert_called_once()

            # 成功ログが出力されることを確認
            mock_logger.info.assert_called_with("アラートメールを送信: 7203")

    def test_email_notification_error(self, monkeypatch):
        """メール送信エラーのテスト"""
        # EMAIL_AVAILABLEを有効にする
        monkeypatch.setattr("src.day_trade.core.alerts.EMAIL_AVAILABLE", True)

        # 完全なメール設定
        self.handler.email_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "test@example.com",
            "to_emails": ["recipient@example.com"],
        }

        # SMTP接続でエラーを発生させる
        import smtplib

        monkeypatch.setattr("src.day_trade.core.alerts.smtplib", smtplib)

        with patch.object(smtplib, "SMTP") as mock_smtp, patch(
            "src.day_trade.core.alerts.logger"
        ) as mock_logger:
            mock_smtp.side_effect = Exception("SMTP connection failed")

            self.handler._send_email_notification(self.sample_trigger)

            # エラーログが出力されることを確認
            mock_logger.error.assert_called_once()
            error_call_args = mock_logger.error.call_args[0][0]
            assert "アラートメールの送信中にエラーが発生しました" in error_call_args
            # monkeypatchで置き換えたsmtplibが原因でエラーメッセージが変わるため、特定のエラー内容のチェックは行わない


class TestAlertManagerAdvanced:
    """AlertManagerの高度な機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.alert_manager = AlertManager()

    def test_email_configuration(self):
        """メール設定のテスト"""
        # NotificationHandlerのconfigure_emailメソッドを使用
        self.alert_manager.notification_handler.configure_email(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="test@gmail.com",
            password="app_password",
            from_email="test@gmail.com",
            to_emails=["alert@example.com"],
        )

        # 設定が反映されることを確認
        email_config = self.alert_manager.notification_handler.email_config
        assert email_config["smtp_server"] == "smtp.gmail.com"
        assert email_config["smtp_port"] == 587
        assert email_config["from_email"] == "test@gmail.com"
        assert len(email_config["to_emails"]) == 1

        # メール通知がハンドラーに追加されることを確認
        assert (
            NotificationMethod.EMAIL in self.alert_manager.notification_handler.handlers
        )

    def test_custom_alert_evaluation_function(self):
        """カスタムアラート評価関数のテスト"""

        # カスタム評価関数を定義
        def custom_evaluator(
            symbol, price, volume, change_pct, historical_data, params
        ):
            return params.get("custom_value", 0) > 50

        # カスタム条件を追加
        condition = AlertCondition(
            alert_id="custom_test",
            symbol="TEST",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            custom_function=custom_evaluator,
            custom_parameters={"custom_value": 60},
        )

        # AlertManagerにアラートを追加
        result = self.alert_manager.add_alert(condition)
        assert result is True

        # 追加されたアラートを確認
        alerts = self.alert_manager.get_alerts(symbol="TEST")
        assert len(alerts) == 1
        assert alerts[0].alert_id == "custom_test"

    def test_alert_cooldown_mechanism(self):
        """アラートクールダウン機能のテスト"""
        # クールダウン期間を短く設定
        condition = AlertCondition(
            alert_id="cooldown_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
            cooldown_minutes=1,  # 1分のクールダウン
        )

        self.alert_manager.add_alert(condition)

        # 最初のトリガーを追加
        trigger_time = datetime.now()
        self.alert_manager.last_trigger_times["cooldown_test"] = trigger_time

        # クールダウン期間内での重複チェック
        assert self.alert_manager._should_check_condition(condition) is False

        # クールダウン期間外での重複チェック（2分後）
        past_trigger_time = datetime.now() - timedelta(minutes=2)
        self.alert_manager.last_trigger_times["cooldown_test"] = past_trigger_time
        assert self.alert_manager._should_check_condition(condition) is True

    def test_alert_history_management(self):
        """アラート履歴管理のテスト"""
        # テスト用のトリガーを作成
        trigger1 = AlertTrigger(
            alert_id="history_test_1",
            symbol="7203",
            trigger_time=datetime.now() - timedelta(hours=2),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="履歴テスト1",
            priority=AlertPriority.MEDIUM,
        )

        trigger2 = AlertTrigger(
            alert_id="history_test_2",
            symbol="9984",
            trigger_time=datetime.now() - timedelta(hours=1),
            alert_type=AlertType.PRICE_BELOW,
            current_value=Decimal("15000"),
            condition_value=Decimal("15500"),
            message="履歴テスト2",
            priority=AlertPriority.HIGH,
        )

        # 履歴に追加
        self.alert_manager.alert_history.extend([trigger1, trigger2])

        # 履歴取得（24時間以内）
        history = self.alert_manager.get_alert_history(hours=24)
        assert len(history) >= 2

        # 特定銘柄の履歴取得
        symbol_history = self.alert_manager.get_alert_history(symbol="7203", hours=24)
        assert len(symbol_history) >= 1
        assert all(h.symbol == "7203" for h in symbol_history)

        # 短時間範囲での履歴取得（0.5時間以内）
        recent_history = self.alert_manager.get_alert_history(hours=0.5)
        # trigger2のみが含まれるはず（1時間前だが、0.5時間以内ではない）
        assert len(recent_history) == 0

    def test_alert_conditions_export(self):
        """アラート設定のエクスポート機能テスト"""
        # テスト用アラート条件を追加
        conditions = [
            AlertCondition(
                alert_id="export_test_1",
                symbol="7203",
                alert_type=AlertType.PRICE_ABOVE,
                condition_value=Decimal("2600"),
                priority=AlertPriority.HIGH,
                description="エクスポートテスト1",
            ),
            AlertCondition(
                alert_id="export_test_2",
                symbol="9984",
                alert_type=AlertType.PRICE_BELOW,
                condition_value=Decimal("15000"),
                priority=AlertPriority.MEDIUM,
                description="エクスポートテスト2",
            ),
        ]

        for condition in conditions:
            self.alert_manager.add_alert(condition)

        # 一時ファイルにエクスポート
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            export_file = tmp_file.name

        try:
            self.alert_manager.export_alerts_config(export_file)

            # ファイルが作成されることを確認
            assert os.path.exists(export_file)

            # エクスポートされた内容を確認
            import json

            with open(export_file, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "export_test_1" in exported_data
            assert "export_test_2" in exported_data
            assert exported_data["export_test_1"]["symbol"] == "7203"
            assert exported_data["export_test_2"]["symbol"] == "9984"

        finally:
            # 一時ファイルを削除
            if os.path.exists(export_file):
                os.unlink(export_file)


class TestNotificationHandlerAdvanced:
    """NotificationHandlerの高度な機能テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.handler = NotificationHandler()
        self.sample_trigger = AlertTrigger(
            alert_id="advanced_test",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="高度テスト通知",
            priority=AlertPriority.CRITICAL,
            current_price=Decimal("2600"),
            volume=2000000,
            change_percent=5.5,
        )

    def test_custom_notification_handler(self):
        """カスタム通知ハンドラーのテスト"""
        custom_calls = []

        def custom_handler(trigger):
            custom_calls.append(f"Custom: {trigger.symbol} - {trigger.message}")

        # カスタムハンドラーを追加
        self.handler.add_custom_handler(NotificationMethod.CALLBACK, custom_handler)

        # 通知送信
        self.handler.send_notification(
            self.sample_trigger, [NotificationMethod.CALLBACK]
        )

        # カスタムハンドラーが呼ばれることを確認
        assert len(custom_calls) == 1
        assert "7203" in custom_calls[0]
        assert "高度テスト通知" in custom_calls[0]

    def test_multiple_notification_methods(self):
        """複数通知方法の同時使用テスト"""
        with patch("src.day_trade.core.alerts.logger") as mock_logger:
            methods = [NotificationMethod.CONSOLE, NotificationMethod.FILE_LOG]
            self.handler.send_notification(self.sample_trigger, methods)

            # コンソール通知のログが出力されることを確認
            mock_logger.warning.assert_called()
            warning_call_args = mock_logger.warning.call_args[0][0]
            assert "Alert triggered - Console notification" in warning_call_args

    def test_notification_error_handling(self):
        """通知エラーハンドリングのテスト"""

        # エラーを発生させるカスタムハンドラー
        def error_handler(trigger):
            raise Exception("Notification error")

        self.handler.add_custom_handler(NotificationMethod.CALLBACK, error_handler)

        # エラーが発生しても他の処理が継続されることを確認
        with patch("src.day_trade.core.alerts.logger") as mock_logger:
            self.handler.send_notification(
                self.sample_trigger, [NotificationMethod.CALLBACK]
            )

            # エラーログが出力されることを確認
            mock_logger.error.assert_called()


class TestAlertManagerBulkOperations:
    """アラートマネージャーの一括操作テスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_check_all_alerts_with_no_conditions(self):
        """アラート条件がない場合の一括チェックテスト"""
        # アラート条件がない状態で一括チェックを実行
        self.alert_manager.check_all_alerts()
        # エラーが発生しないことを確認
        assert len(self.alert_manager.alert_history) == 0

    def test_bulk_data_fetch_fallback(self):
        """一括データ取得のフォールバックテスト"""
        symbols = ["7203", "9984"]

        # get_bulk_current_pricesメソッドが存在しない場合のフォールバックテスト
        with patch.object(
            self.alert_manager.stock_fetcher, "get_current_price"
        ) as mock_get_price:
            mock_get_price.return_value = {
                "current_price": 2500.0,
                "volume": 1000000,
                "change_percent": 2.5,
            }

            bulk_data = self.alert_manager._fetch_bulk_market_data(symbols)

            # 個別取得が呼び出されることを確認
            assert mock_get_price.call_count == len(symbols)
            assert len(bulk_data) == len(symbols)

    def test_monitoring_loop_error_handling(self):
        """監視ループのエラーハンドリングテスト"""
        with patch.object(self.alert_manager, "check_all_alerts") as mock_check:
            mock_check.side_effect = Exception("監視エラー")

            # 監視を短時間開始
            self.alert_manager.start_monitoring(interval_seconds=0.1)

            # 短時間待機して監視を停止
            import time

            time.sleep(0.2)
            self.alert_manager.stop_monitoring()

            # エラーが発生しても監視が継続されることを確認
            assert mock_check.call_count >= 1

    def test_notification_methods_configuration(self):
        """通知方法の設定テスト"""
        # デフォルトの通知方法を確認
        assert self.alert_manager.default_notification_methods == [
            NotificationMethod.CONSOLE
        ]

        # 通知方法を変更
        new_methods = [NotificationMethod.EMAIL, NotificationMethod.FILE_LOG]
        self.alert_manager.configure_notifications(new_methods)

        # 設定が反映されることを確認
        assert self.alert_manager.default_notification_methods == new_methods

    def test_condition_validation_edge_cases(self):
        """条件検証のエッジケーステスト"""
        # 空のalert_id
        invalid_condition1 = AlertCondition(
            alert_id="",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )
        assert self.alert_manager.add_alert(invalid_condition1) is False

        # 空のsymbol
        invalid_condition2 = AlertCondition(
            alert_id="test_empty_symbol",
            symbol="",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )
        assert self.alert_manager.add_alert(invalid_condition2) is False

        # カスタム条件でcustom_functionがない場合
        invalid_condition3 = AlertCondition(
            alert_id="test_no_custom_func",
            symbol="7203",
            alert_type=AlertType.CUSTOM_CONDITION,
            condition_value="custom",
            custom_function=None,
        )
        assert self.alert_manager.add_alert(invalid_condition3) is False

    def test_alert_trigger_dict_conversion(self):
        """アラートトリガーの辞書変換テスト"""
        trigger = AlertTrigger(
            alert_id="dict_test",
            symbol="7203",
            trigger_time=datetime(2024, 1, 15, 10, 30, 45),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="辞書変換テスト",
            priority=AlertPriority.HIGH,
            current_price=Decimal("2600"),
            volume=1500000,
            change_percent=4.2,
        )

        result = trigger.to_dict()

        assert result["alert_id"] == "dict_test"
        assert result["symbol"] == "7203"
        assert result["trigger_time"] == "2024-01-15T10:30:45"
        assert result["alert_type"] == AlertType.PRICE_ABOVE.value
        assert result["current_value"] == "2600"
        assert result["condition_value"] == "2500"
        assert result["message"] == "辞書変換テスト"
        assert result["priority"] == AlertPriority.HIGH.value
        assert result["current_price"] == "2600"
        assert result["volume"] == 1500000
        assert result["change_percent"] == 4.2

    def test_alert_expiry_handling(self):
        """アラートの有効期限処理テスト"""
        # 期限切れのアラート条件
        expired_condition = AlertCondition(
            alert_id="expired_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
            expiry_date=datetime.now() - timedelta(hours=1),  # 1時間前に期限切れ
        )

        self.alert_manager.add_alert(expired_condition)

        # 期限切れチェック
        assert self.alert_manager._is_expired(expired_condition) is True
        assert self.alert_manager._should_check_condition(expired_condition) is False

        # 有効なアラート条件
        valid_condition = AlertCondition(
            alert_id="valid_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
            expiry_date=datetime.now() + timedelta(hours=1),  # 1時間後に期限切れ
        )

        self.alert_manager.add_alert(valid_condition)

        # 有効チェック
        assert self.alert_manager._is_expired(valid_condition) is False
        assert self.alert_manager._should_check_condition(valid_condition) is True


class TestAlertStrategyIntegration:
    """アラート戦略統合テスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_evaluate_condition_with_strategy(self):
        """戦略パターンを使用した条件評価テスト"""
        condition = AlertCondition(
            alert_id="strategy_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
            comparison_operator=">",
        )

        # 戦略が存在しない場合のテスト
        with patch.object(
            self.alert_manager.strategy_factory, "get_strategy"
        ) as mock_get_strategy:
            mock_get_strategy.return_value = None

            result = self.alert_manager._evaluate_condition(
                condition=condition,
                current_price=Decimal("2600"),
                volume=1000000,
                change_percent=4.0,
                historical_data=None,
            )

            assert result is None

    def test_evaluate_condition_strategy_error(self):
        """戦略実行中のエラーテスト"""
        condition = AlertCondition(
            alert_id="strategy_error_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )

        # 戦略実行時にエラーを発生させる
        from unittest.mock import Mock

        mock_strategy = Mock()
        mock_strategy.evaluate.side_effect = Exception("戦略エラー")

        with patch.object(
            self.alert_manager.strategy_factory, "get_strategy"
        ) as mock_get_strategy:
            mock_get_strategy.return_value = mock_strategy

            result = self.alert_manager._evaluate_condition(
                condition=condition,
                current_price=Decimal("2600"),
                volume=1000000,
                change_percent=4.0,
                historical_data=None,
            )

            assert result is None

    def test_check_symbol_alerts_with_data_no_current_data(self):
        """不正なデータでのアラートチェックテスト"""
        condition = AlertCondition(
            alert_id="invalid_data_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )

        # 不正なマーケットデータ
        invalid_market_data = {
            "current_data": None,  # current_dataがNone
            "historical_data": None,
        }

        # エラーが発生しないことを確認
        self.alert_manager._check_symbol_alerts_with_data(
            symbol="7203", conditions=[condition], market_data=invalid_market_data
        )

        # アラート履歴に何も追加されないことを確認
        assert len(self.alert_manager.alert_history) == 0

    def test_check_symbol_alerts_data_processing_error(self):
        """データ処理エラーのテスト"""
        condition = AlertCondition(
            alert_id="data_error_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )

        # stock_fetcher.get_current_priceがNoneを返す場合
        with patch.object(
            self.alert_manager.stock_fetcher, "get_current_price"
        ) as mock_get_price:
            mock_get_price.return_value = None

            self.alert_manager._check_symbol_alerts("7203", [condition])

            # アラート履歴に何も追加されないことを確認
            assert len(self.alert_manager.alert_history) == 0


class TestAlertManagerCompareValues:
    """値比較メソッドのテスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_compare_values_all_operators(self):
        """全ての比較演算子のテスト"""
        # 大なり
        assert self.alert_manager._compare_values(10, 5, ">") is True
        assert self.alert_manager._compare_values(5, 10, ">") is False

        # 小なり
        assert self.alert_manager._compare_values(5, 10, "<") is True
        assert self.alert_manager._compare_values(10, 5, "<") is False

        # 以上
        assert self.alert_manager._compare_values(10, 10, ">=") is True
        assert self.alert_manager._compare_values(11, 10, ">=") is True
        assert self.alert_manager._compare_values(9, 10, ">=") is False

        # 以下
        assert self.alert_manager._compare_values(10, 10, "<=") is True
        assert self.alert_manager._compare_values(9, 10, "<=") is True
        assert self.alert_manager._compare_values(11, 10, "<=") is False

        # 等しい（小数点誤差考慮）
        assert self.alert_manager._compare_values(10.0, 10.0, "==") is True
        assert (
            self.alert_manager._compare_values(10.0001, 10.0, "==") is True
        )  # 誤差範囲内
        assert (
            self.alert_manager._compare_values(10.01, 10.0, "==") is False
        )  # 誤差範囲外

    def test_compare_values_invalid_data(self):
        """無効なデータでの比較テスト"""
        # 文字列を数値に変換できない場合
        assert self.alert_manager._compare_values("invalid", 10, ">") is False
        assert self.alert_manager._compare_values(10, "invalid", ">") is False
        assert self.alert_manager._compare_values(None, 10, ">") is False

    def test_compare_values_decimal_support(self):
        """Decimal型サポートのテスト"""
        from decimal import Decimal

        assert (
            self.alert_manager._compare_values(Decimal("10.5"), Decimal("10.0"), ">")
            is True
        )
        assert self.alert_manager._compare_values(Decimal("10.0"), "10.5", "<") is True


class TestAlertHistoryManagement:
    """アラート履歴制限のテスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_alert_history_limit(self):
        """アラート履歴制限機能のテスト"""
        # 1001個のアラートトリガーを作成
        triggers = []
        for i in range(1001):
            trigger = AlertTrigger(
                alert_id=f"limit_test_{i}",
                symbol="7203",
                trigger_time=datetime.now(),
                alert_type=AlertType.PRICE_ABOVE,
                current_value=Decimal("2500"),
                condition_value=Decimal("2400"),
                message=f"履歴制限テスト {i}",
                priority=AlertPriority.MEDIUM,
            )
            triggers.append(trigger)

        # 履歴に大量のトリガーを追加
        self.alert_manager.alert_history = triggers

        # _handle_alert_triggerを呼び出して履歴制限をテスト
        new_trigger = AlertTrigger(
            alert_id="new_trigger",
            symbol="7203",
            trigger_time=datetime.now(),
            alert_type=AlertType.PRICE_ABOVE,
            current_value=Decimal("2600"),
            condition_value=Decimal("2500"),
            message="新しいトリガー",
            priority=AlertPriority.HIGH,
        )

        self.alert_manager._handle_alert_trigger(new_trigger)

        # 履歴が1000件に制限されることを確認
        assert len(self.alert_manager.alert_history) == 1000
        # 最新のトリガーが含まれることを確認
        assert self.alert_manager.alert_history[-1].alert_id == "new_trigger"


class TestAlertManagerErrorHandling:
    """AlertManagerのエラーハンドリングテスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_add_alert_exception_handling(self):
        """add_alertメソッドの例外処理テスト"""
        # _validate_conditionでエラーを発生させる
        with patch.object(self.alert_manager, "_validate_condition") as mock_validate:
            mock_validate.side_effect = Exception("検証エラー")

            condition = AlertCondition(
                alert_id="exception_test",
                symbol="7203",
                alert_type=AlertType.PRICE_ABOVE,
                condition_value=Decimal("2500"),
            )

            result = self.alert_manager.add_alert(condition)
            assert result is False

    def test_check_all_alerts_bulk_error_fallback(self):
        """一括データ取得エラー時のフォールバック処理テスト"""
        condition = AlertCondition(
            alert_id="fallback_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )
        self.alert_manager.add_alert(condition)

        # _fetch_bulk_market_dataでエラーを発生させる
        with patch.object(
            self.alert_manager, "_fetch_bulk_market_data"
        ) as mock_bulk_fetch, patch.object(
            self.alert_manager, "_check_symbol_alerts"
        ) as mock_check_symbol:
            mock_bulk_fetch.side_effect = Exception("一括取得エラー")

            self.alert_manager.check_all_alerts()

            # フォールバック処理が呼ばれることを確認
            mock_check_symbol.assert_called_once()

    def test_check_symbol_alerts_exception_handling(self):
        """_check_symbol_alertsの例外処理テスト"""
        condition = AlertCondition(
            alert_id="symbol_error_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )

        with patch.object(
            self.alert_manager.stock_fetcher, "get_current_price"
        ) as mock_get_price:
            mock_get_price.side_effect = Exception("データ取得エラー")

            # 例外が発生してもプログラムが停止しないことを確認
            self.alert_manager._check_symbol_alerts("7203", [condition])

            # アラート履歴に何も追加されないことを確認
            assert len(self.alert_manager.alert_history) == 0


class TestAlertExportImportErrorHandling:
    """アラートエクスポート・エラーハンドリングテスト"""

    def setup_method(self):
        self.alert_manager = AlertManager()

    def test_export_alerts_config_write_error(self):
        """エクスポート時の書き込みエラーテスト"""
        condition = AlertCondition(
            alert_id="export_error_test",
            symbol="7203",
            alert_type=AlertType.PRICE_ABOVE,
            condition_value=Decimal("2500"),
        )
        self.alert_manager.add_alert(condition)

        # 書き込み権限がないディレクトリを指定してエラーを発生させる
        invalid_path = "/invalid/path/export.json"

        # エラーが発生してもプログラムが停止しないことを確認
        self.alert_manager.export_alerts_config(invalid_path)

        # エクスポートに失敗してもアラート条件は残ることを確認
        assert len(self.alert_manager.get_alerts()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
