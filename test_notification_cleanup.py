#!/usr/bin/env python3
"""
Issue #445: 不要な通知パラメータ削除のテスト
テストケース：削除された通知チャネルが正しく除外されていることを確認
"""

import unittest
from enum import Enum

try:
    from src.day_trade.monitoring.alert_engine import NotificationChannel as AlertNotificationChannel
    from src.day_trade.risk_management.factories.alert_factory import NotificationChannelType
    from src.day_trade.risk_management.interfaces.alert_interfaces import NotificationChannel as InterfaceNotificationChannel
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)


class TestNotificationChannelCleanup(unittest.TestCase):
    """通知チャンネル削除テスト"""

    def test_alert_engine_notification_channels(self):
        """alert_engine.pyの通知チャンネル確認"""
        # 必要なチャンネルのみ残っていることを確認
        expected_channels = {"EMAIL", "SLACK", "WEBHOOK"}
        actual_channels = {member.name for member in AlertNotificationChannel}

        self.assertEqual(actual_channels, expected_channels)

        # 不要なチャンネルが削除されていることを確認
        removed_channels = {"SMS", "DISCORD"}
        for channel in removed_channels:
            self.assertNotIn(channel, actual_channels, f"{channel} should be removed")

    def test_alert_factory_notification_channel_types(self):
        """alert_factory.pyの通知チャンネルタイプ確認"""
        # 必要なチャンネルのみ残っていることを確認
        expected_types = {"EMAIL", "SLACK", "WEBHOOK"}
        actual_types = {member.name for member in NotificationChannelType}

        self.assertEqual(actual_types, expected_types)

        # 不要なチャンネルが削除されていることを確認
        removed_types = {"SMS", "DISCORD", "TEAMS", "TELEGRAM", "PUSH_NOTIFICATION", "PLUGIN"}
        for channel_type in removed_types:
            self.assertNotIn(channel_type, actual_types, f"{channel_type} should be removed")

    def test_interface_notification_channels(self):
        """interfaces.pyの通知チャンネル確認"""
        # 必要なチャンネルのみ残っていることを確認
        expected_channels = {"EMAIL", "SLACK", "WEBHOOK"}
        actual_channels = {member.name for member in InterfaceNotificationChannel}

        self.assertEqual(actual_channels, expected_channels)

        # 不要なチャンネルが削除されていることを確認
        removed_channels = {"SMS", "DASHBOARD", "PUSH_NOTIFICATION"}
        for channel in removed_channels:
            self.assertNotIn(channel, actual_channels, f"{channel} should be removed")

    def test_notification_channel_values(self):
        """通知チャンネルの値が正しいことを確認"""
        # alert_engine
        self.assertEqual(AlertNotificationChannel.EMAIL.value, "email")
        self.assertEqual(AlertNotificationChannel.SLACK.value, "slack")
        self.assertEqual(AlertNotificationChannel.WEBHOOK.value, "webhook")

        # alert_factory
        self.assertEqual(NotificationChannelType.EMAIL.value, "email")
        self.assertEqual(NotificationChannelType.SLACK.value, "slack")
        self.assertEqual(NotificationChannelType.WEBHOOK.value, "webhook")

        # interfaces
        self.assertEqual(InterfaceNotificationChannel.EMAIL.value, "email")
        self.assertEqual(InterfaceNotificationChannel.SLACK.value, "slack")
        self.assertEqual(InterfaceNotificationChannel.WEBHOOK.value, "webhook")


class TestNotificationSystemIntegration(unittest.TestCase):
    """通知システム統合テスト"""

    def test_notification_system_import(self):
        """通知システムの正常インポート確認"""
        try:
            from src.day_trade.monitoring.notification_system import NotificationSystem
            system = NotificationSystem()
            self.assertIsInstance(system, NotificationSystem)
        except Exception as e:
            self.fail(f"NotificationSystem import failed: {e}")

    def test_alert_engine_import(self):
        """アラートエンジンの正常インポート確認"""
        try:
            from src.day_trade.monitoring.alert_engine import IntelligentAlertEngine
            engine = IntelligentAlertEngine()
            self.assertIsInstance(engine, IntelligentAlertEngine)
        except Exception as e:
            self.fail(f"IntelligentAlertEngine import failed: {e}")

    def test_alert_factory_import(self):
        """アラートファクトリーの正常インポート確認"""
        try:
            from src.day_trade.risk_management.factories.alert_factory import AlertChannelFactory
            factory = AlertChannelFactory()
            self.assertIsInstance(factory, AlertChannelFactory)
        except Exception as e:
            self.fail(f"AlertChannelFactory import failed: {e}")


def print_cleanup_summary():
    """削除内容のサマリー表示"""
    print("\n" + "="*60)
    print("Issue #445: 不要な通知パラメータ削除完了サマリー")
    print("="*60)
    print("\n■ 削除された通知チャンネル:")
    print("  - SMS通知（未実装で不要）")
    print("  - Discord通知（SlackやEmailと機能重複）")
    print("  - Teams通知（使用されていない）")
    print("  - Telegram通知（使用されていない）")
    print("  - Push通知（使用されていない）")
    print("  - Dashboard通知（使用されていない）")

    print("\n■ 保持された通知チャンネル:")
    print("  - Email通知（重要な通知手段）")
    print("  - Slack通知（チーム通知に必要）")
    print("  - Webhook通知（外部統合に必要）")

    print("\n■ 変更されたファイル:")
    files = [
        "src/day_trade/monitoring/alert_engine.py",
        "src/day_trade/monitoring/notification_system.py",
        "src/day_trade/risk_management/factories/alert_factory.py",
        "src/day_trade/risk_management/config/unified_config.py",
        "src/day_trade/risk_management/interfaces/alert_interfaces.py",
        "src/day_trade/realtime/alert_system.py"
    ]
    for file in files:
        print(f"  - {file}")

    print("\n■ 効果:")
    print("  - 不要な通知設定パラメータを削除")
    print("  - 設定ファイルの簡素化")
    print("  - メンテナンス性の向上")
    print("  - 実際に使用される機能への集中")
    print("="*60)


if __name__ == "__main__":
    print_cleanup_summary()
    print("\n通知チャンネル削除のテストを実行します...")
    unittest.main()